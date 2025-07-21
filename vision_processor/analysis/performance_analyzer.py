"""Performance Analyzer for Model Comparison
========================================

Analyzes model performance metrics including inference time, success rates,
memory usage, and overall extraction quality.
"""

import statistics
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from ..extraction.production_extractor import ExtractionResult


@dataclass
class ModelPerformanceMetrics:
    """Comprehensive performance metrics for a single model."""

    model_name: str

    # Time metrics
    total_inference_time: float
    average_inference_time: float
    median_inference_time: float
    min_inference_time: float
    max_inference_time: float
    inference_time_std: float

    # Success metrics
    total_documents: int
    successful_extractions: int
    success_rate: float

    # Field extraction metrics
    total_fields_extracted: int
    average_fields_per_document: float
    median_fields_per_document: float
    max_fields_extracted: int
    min_fields_extracted: int

    # Confidence metrics
    average_confidence_score: float
    median_confidence_score: float
    high_confidence_count: int  # confidence > 0.8
    low_confidence_count: int  # confidence < 0.4

    # Core field performance
    core_fields_detection_rate: float
    required_fields_detection_rate: float

    # Output quality metrics
    structured_output_rate: float
    fallback_usage_rate: float

    # Processing quality
    processing_notes_summary: Dict[str, int]


@dataclass
class CrossModelComparison:
    """Comparison metrics between multiple models."""

    models_compared: List[str]
    winner_by_metric: Dict[str, str]
    performance_gaps: Dict[str, float]
    statistical_significance: Dict[str, bool]

    # Detailed comparisons
    success_rate_comparison: Dict[str, float]
    speed_comparison: Dict[str, float]
    field_extraction_comparison: Dict[str, float]
    confidence_comparison: Dict[str, float]


class PerformanceAnalyzer:
    """Analyzes model performance across multiple dimensions."""

    def __init__(self):
        """Initialize performance analyzer."""
        self.model_results: Dict[str, List[ExtractionResult]] = {}

    def add_results(self, model_name: str, results: List[ExtractionResult]):
        """Add extraction results for a model.

        Args:
            model_name: Name of the model
            results: List of extraction results
        """
        self.model_results[model_name] = results

    def analyze_model_performance(self, model_name: str) -> Optional[ModelPerformanceMetrics]:
        """Analyze performance for a single model.

        Args:
            model_name: Name of the model to analyze

        Returns:
            ModelPerformanceMetrics or None if model not found
        """
        if model_name not in self.model_results:
            return None

        results = self.model_results[model_name]
        if not results:
            return None

        # Time metrics - handle both dict and object formats
        inference_times = []
        successful_results = []
        field_counts = []
        confidence_scores = []
        core_field_counts = []
        required_field_counts = []

        for r in results:
            if isinstance(r, dict):
                # Dictionary format from conversion
                inference_times.append(r.get("extraction_time", 0))
                if r.get("successful", False):
                    successful_results.append(r)
                field_counts.append(r.get("field_count", 0))
                confidence_scores.append(r.get("confidence_score", 0))
                # For dict format, we don't have core/required field breakdown
                core_field_counts.append(0)
                required_field_counts.append(0)
            else:
                # Object format
                inference_times.append(r.extraction_time)
                if r.is_successful:
                    successful_results.append(r)
                field_counts.append(r.field_count)
                confidence_scores.append(r.confidence_score)
                core_field_counts.append(len(r.core_fields_found))
                required_field_counts.append(len(r.required_fields_found))

        # Calculate core fields detection rate - simplified for dict format
        core_fields_detection_rate = 0.0
        required_fields_detection_rate = 0.0

        # For object format, calculate properly
        if results and not isinstance(results[0], dict):
            total_possible_core_fields = sum(
                len(r.core_fields_found) + len(r.missing_core_fields) for r in results
            )
            core_fields_found = sum(len(r.core_fields_found) for r in results)
            core_fields_detection_rate = (
                core_fields_found / total_possible_core_fields if total_possible_core_fields > 0 else 0
            )

            total_possible_required_fields = sum(
                len(r.required_fields_found) + len(r.missing_required_fields) for r in results
            )
            required_fields_found = sum(len(r.required_fields_found) for r in results)
            required_fields_detection_rate = (
                required_fields_found / total_possible_required_fields
                if total_possible_required_fields > 0
                else 0
            )

        # Output quality metrics - handle both dict and object formats
        structured_outputs = 0
        fallback_usages = 0
        for r in results:
            if isinstance(r, dict):
                if r.get("is_structured", False):
                    structured_outputs += 1
                # Dict format doesn't track fallback usage
            else:
                if r.has_structured_output:
                    structured_outputs += 1
                if r.raw_markdown_fallback_used:
                    fallback_usages += 1

        # Confidence categorization
        high_confidence = sum(1 for score in confidence_scores if score > 0.8)
        low_confidence = sum(1 for score in confidence_scores if score < 0.4)

        # Processing notes summary - handle both dict and object formats
        processing_notes_summary = {}
        for result in results:
            if isinstance(result, dict):
                # Dict format doesn't have processing notes
                continue
            else:
                for note in result.processing_notes:
                    processing_notes_summary[note] = processing_notes_summary.get(note, 0) + 1

        return ModelPerformanceMetrics(
            model_name=model_name,
            total_inference_time=sum(inference_times),
            average_inference_time=statistics.mean(inference_times),
            median_inference_time=statistics.median(inference_times),
            min_inference_time=min(inference_times),
            max_inference_time=max(inference_times),
            inference_time_std=statistics.stdev(inference_times) if len(inference_times) > 1 else 0,
            total_documents=len(results),
            successful_extractions=len(successful_results),
            success_rate=len(successful_results) / len(results),
            total_fields_extracted=sum(field_counts),
            average_fields_per_document=statistics.mean(field_counts),
            median_fields_per_document=statistics.median(field_counts),
            max_fields_extracted=max(field_counts),
            min_fields_extracted=min(field_counts),
            average_confidence_score=statistics.mean(confidence_scores),
            median_confidence_score=statistics.median(confidence_scores),
            high_confidence_count=high_confidence,
            low_confidence_count=low_confidence,
            core_fields_detection_rate=core_fields_detection_rate,
            required_fields_detection_rate=required_fields_detection_rate,
            structured_output_rate=structured_outputs / len(results),
            fallback_usage_rate=fallback_usages / len(results),
            processing_notes_summary=processing_notes_summary,
        )

    def compare_models(self) -> Optional[CrossModelComparison]:
        """Compare performance across all analyzed models.

        Returns:
            CrossModelComparison or None if insufficient data
        """
        if len(self.model_results) < 2:
            return None

        model_names = list(self.model_results.keys())
        model_metrics = {}

        # Get metrics for each model
        for model_name in model_names:
            metrics = self.analyze_model_performance(model_name)
            if metrics:
                model_metrics[model_name] = metrics

        if len(model_metrics) < 2:
            return None

        # Determine winners by metric
        winner_by_metric = {
            "success_rate": max(model_metrics.keys(), key=lambda m: model_metrics[m].success_rate),
            "average_speed": min(
                model_metrics.keys(), key=lambda m: model_metrics[m].average_inference_time
            ),
            "field_extraction": max(
                model_metrics.keys(), key=lambda m: model_metrics[m].average_fields_per_document
            ),
            "confidence": max(
                model_metrics.keys(), key=lambda m: model_metrics[m].average_confidence_score
            ),
            "core_fields": max(
                model_metrics.keys(), key=lambda m: model_metrics[m].core_fields_detection_rate
            ),
            "structured_output": max(
                model_metrics.keys(), key=lambda m: model_metrics[m].structured_output_rate
            ),
        }

        # Calculate performance gaps
        performance_gaps = {}
        for metric, _winner in winner_by_metric.items():
            if metric == "success_rate":
                values = [model_metrics[m].success_rate for m in model_names]
                performance_gaps[metric] = max(values) - min(values)
            elif metric == "average_speed":
                values = [model_metrics[m].average_inference_time for m in model_names]
                performance_gaps[metric] = max(values) - min(values)
            elif metric == "field_extraction":
                values = [model_metrics[m].average_fields_per_document for m in model_names]
                performance_gaps[metric] = max(values) - min(values)
            elif metric == "confidence":
                values = [model_metrics[m].average_confidence_score for m in model_names]
                performance_gaps[metric] = max(values) - min(values)
            elif metric == "core_fields":
                values = [model_metrics[m].core_fields_detection_rate for m in model_names]
                performance_gaps[metric] = max(values) - min(values)
            elif metric == "structured_output":
                values = [model_metrics[m].structured_output_rate for m in model_names]
                performance_gaps[metric] = max(values) - min(values)

        # Create detailed comparisons
        success_rate_comparison = {m: model_metrics[m].success_rate for m in model_names}
        speed_comparison = {m: model_metrics[m].average_inference_time for m in model_names}
        field_extraction_comparison = {m: model_metrics[m].average_fields_per_document for m in model_names}
        confidence_comparison = {m: model_metrics[m].average_confidence_score for m in model_names}

        # Simplified statistical significance (would need proper statistical tests for production)
        statistical_significance = {metric: gap > 0.1 for metric, gap in performance_gaps.items()}

        return CrossModelComparison(
            models_compared=model_names,
            winner_by_metric=winner_by_metric,
            performance_gaps=performance_gaps,
            statistical_significance=statistical_significance,
            success_rate_comparison=success_rate_comparison,
            speed_comparison=speed_comparison,
            field_extraction_comparison=field_extraction_comparison,
            confidence_comparison=confidence_comparison,
        )

    def create_performance_dataframe(self) -> pd.DataFrame:
        """Create DataFrame with performance data for all models.

        Returns:
            DataFrame with comprehensive performance data
        """
        all_data = []

        for model_name, results in self.model_results.items():
            for result in results:
                if isinstance(result, dict):
                    # Handle dictionary format
                    row = {
                        "model": model_name,
                        "image": result.get("image_name", ""),
                        "inference_time": result.get("extraction_time", 0),
                        "field_count": result.get("field_count", 0),
                        "is_successful": result.get("successful", False),
                        "confidence_score": result.get("confidence_score", 0),
                        "core_fields_found": 0,  # Not available in dict format
                        "required_fields_found": 0,  # Not available in dict format
                        "has_structured_output": result.get("is_structured", False),
                        "fallback_used": False,  # Not available in dict format
                        "categories_count": 0,  # Not available in dict format
                    }
                else:
                    # Handle object format
                    row = {
                        "model": model_name,
                        "image": result.image_name,
                        "inference_time": result.extraction_time,
                        "field_count": result.field_count,
                        "is_successful": result.is_successful,
                        "confidence_score": result.confidence_score,
                        "core_fields_found": len(result.core_fields_found),
                        "required_fields_found": len(result.required_fields_found),
                        "has_structured_output": result.has_structured_output,
                        "fallback_used": result.raw_markdown_fallback_used,
                        "categories_count": len(result.fields_by_category),
                    }

                # Add category-specific field counts - only for object format
                if not isinstance(result, dict):
                    for category, fields in result.fields_by_category.items():
                        row[f"{category}_fields"] = len(fields)

                all_data.append(row)

        return pd.DataFrame(all_data)

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get high-level performance summary.

        Returns:
            Dictionary with performance summary
        """
        summary = {
            "total_models": len(self.model_results),
            "total_documents_processed": sum(len(results) for results in self.model_results.values()),
            "models_analyzed": list(self.model_results.keys()),
        }

        # Add per-model summaries
        model_summaries = {}
        for model_name in self.model_results.keys():
            metrics = self.analyze_model_performance(model_name)
            if metrics:
                model_summaries[model_name] = {
                    "success_rate": f"{metrics.success_rate:.1%}",
                    "avg_inference_time": f"{metrics.average_inference_time:.2f}s",
                    "avg_fields_extracted": f"{metrics.average_fields_per_document:.1f}",
                    "avg_confidence": f"{metrics.average_confidence_score:.2f}",
                }

        summary["model_summaries"] = model_summaries

        # Add comparison if multiple models
        if len(self.model_results) >= 2:
            comparison = self.compare_models()
            if comparison:
                summary["best_performers"] = comparison.winner_by_metric
                summary["performance_gaps"] = {
                    metric: f"{gap:.3f}" for metric, gap in comparison.performance_gaps.items()
                }

        return summary

    def identify_performance_issues(self) -> Dict[str, List[str]]:
        """Identify potential performance issues for each model.

        Returns:
            Dictionary mapping model names to list of issues
        """
        issues = {}

        for model_name in self.model_results.keys():
            model_issues = []
            metrics = self.analyze_model_performance(model_name)

            if not metrics:
                model_issues.append("No metrics available")
                issues[model_name] = model_issues
                continue

            # Check for performance issues
            if metrics.success_rate < 0.5:
                model_issues.append(f"Low success rate: {metrics.success_rate:.1%}")

            if metrics.average_inference_time > 10.0:
                model_issues.append(f"Slow inference: {metrics.average_inference_time:.1f}s average")

            if metrics.average_fields_per_document < 3:
                model_issues.append(
                    f"Low field extraction: {metrics.average_fields_per_document:.1f} fields/doc"
                )

            if metrics.average_confidence_score < 0.6:
                model_issues.append(f"Low confidence: {metrics.average_confidence_score:.2f} average")

            if metrics.core_fields_detection_rate < 0.3:
                model_issues.append(f"Poor core field detection: {metrics.core_fields_detection_rate:.1%}")

            if metrics.structured_output_rate < 0.7:
                model_issues.append(f"Low structured output rate: {metrics.structured_output_rate:.1%}")

            if metrics.fallback_usage_rate > 0.5:
                model_issues.append(f"High fallback usage: {metrics.fallback_usage_rate:.1%}")

            if not model_issues:
                model_issues.append("No performance issues detected")

            issues[model_name] = model_issues

        return issues
