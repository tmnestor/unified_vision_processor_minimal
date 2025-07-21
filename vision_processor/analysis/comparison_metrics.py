"""Comparison Metrics for Model Evaluation
======================================

Advanced metrics for comparing model performance including F1 scores,
precision, recall, and custom business metrics for Australian tax documents.
"""

import statistics
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from ..config.production_schema import PRODUCTION_SCHEMA, FieldCategory
from ..extraction.production_extractor import ExtractionResult


@dataclass
class F1Metrics:
    """F1 score metrics for field evaluation."""

    field_name: str
    precision: float
    recall: float
    f1_score: float
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    support: int  # Number of true instances


@dataclass
class ModelComparisonMetrics:
    """Comprehensive comparison metrics between models."""

    model_names: List[str]

    # Overall performance
    overall_f1_scores: Dict[str, float]
    macro_averaged_f1: Dict[str, float]
    weighted_f1: Dict[str, float]

    # Field-specific F1 scores
    field_f1_scores: Dict[str, Dict[str, F1Metrics]]

    # Category-based metrics
    category_performance: Dict[str, Dict[str, float]]

    # Business-specific metrics
    ato_compliance_scores: Dict[str, float]
    critical_field_performance: Dict[str, float]

    # Statistical significance
    significant_differences: Dict[str, bool]
    confidence_intervals: Dict[str, Tuple[float, float]]

    # Ranking and recommendations
    model_rankings: Dict[str, int]  # By metric
    performance_recommendations: Dict[str, List[str]]


class ComparisonMetrics:
    """Calculates advanced comparison metrics for model evaluation."""

    def __init__(self):
        """Initialize comparison metrics calculator."""
        self.model_results: Dict[str, List[ExtractionResult]] = {}
        self.production_schema = PRODUCTION_SCHEMA
        self.ground_truth = self._create_ground_truth()

    def add_results(self, model_name: str, results: List[ExtractionResult]):
        """Add extraction results for a model.

        Args:
            model_name: Name of the model
            results: List of extraction results
        """
        self.model_results[model_name] = results

    def _create_ground_truth(self) -> Dict[str, Dict[str, bool]]:
        """Create ground truth labels for evaluation.

        This is a simplified ground truth. In production, this would come from
        human annotations or verified data.

        Returns:
            Dictionary mapping image names to field presence ground truth
        """
        # Simplified ground truth based on common document patterns
        # In production, this would be loaded from annotated data
        ground_truth = {}

        # Define expected fields for different document types
        common_fields = ["date_a_li", "supplier_a_pgs", "total_a_li", "subtotal_a_li", "tax_a_li"]

        # ABN typically only in business invoices
        abn_documents = ["image39.png", "image76.png", "image71.png"]

        # Create ground truth for each document
        # This is simplified - real ground truth would be manually annotated
        for i in range(1, 100):  # Assuming image01.png to image99.png
            image_name = f"image{i:02d}.png"
            ground_truth[image_name] = {}

            # Set common fields as present
            for field in common_fields:
                ground_truth[image_name][field] = True

            # ABN only in specific documents
            ground_truth[image_name]["supplierABN_a_pgs"] = image_name in abn_documents

            # Other fields have varying presence
            ground_truth[image_name]["quantity_a_li"] = i % 3 == 0  # Every 3rd document
            ground_truth[image_name]["desc_a_li"] = i % 2 == 0  # Every 2nd document

        return ground_truth

    def calculate_f1_metrics(self, model_name: str, field_name: str) -> Optional[F1Metrics]:
        """Calculate F1 metrics for a specific model and field.

        Args:
            model_name: Name of the model
            field_name: Name of the field

        Returns:
            F1Metrics or None if insufficient data
        """
        if model_name not in self.model_results:
            return None

        results = self.model_results[model_name]
        if not results:
            return None

        # Collect predictions and ground truth
        y_true = []
        y_pred = []

        for result in results:
            image_name = result.image_name

            # Get ground truth for this image/field combination
            if image_name in self.ground_truth and field_name in self.ground_truth[image_name]:
                true_label = self.ground_truth[image_name][field_name]
            else:
                # Default assumption - field should be present
                true_label = field_name in PRODUCTION_SCHEMA.get_core_fields()

            # Get prediction
            pred_label = field_name in result.extracted_fields

            y_true.append(true_label)
            y_pred.append(pred_label)

        if not y_true:
            return None

        # Calculate metrics
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # Calculate confusion matrix components
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        support = sum(y_true)

        return F1Metrics(
            field_name=field_name,
            precision=precision,
            recall=recall,
            f1_score=f1,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            true_negatives=tn,
            support=support,
        )

    def calculate_model_f1_scores(self, model_name: str) -> Dict[str, F1Metrics]:
        """Calculate F1 scores for all fields for a specific model.

        Args:
            model_name: Name of the model

        Returns:
            Dictionary mapping field names to F1Metrics
        """
        field_metrics = {}

        # Focus on core fields and high-priority fields
        important_fields = (
            PRODUCTION_SCHEMA.get_core_fields()
            + PRODUCTION_SCHEMA.get_required_fields()
            + PRODUCTION_SCHEMA.get_fields_by_category(FieldCategory.FINANCIAL)
            + PRODUCTION_SCHEMA.get_fields_by_category(FieldCategory.SUPPLIER)
        )

        # Remove duplicates
        important_fields = list(set(important_fields))

        for field_name in important_fields:
            metrics = self.calculate_f1_metrics(model_name, field_name)
            if metrics:
                field_metrics[field_name] = metrics

        return field_metrics

    def compare_models(self) -> Optional[ModelComparisonMetrics]:
        """Compare all models using comprehensive metrics.

        Returns:
            ModelComparisonMetrics or None if insufficient data
        """
        if len(self.model_results) < 2:
            return None

        model_names = list(self.model_results.keys())

        # Calculate F1 scores for each model
        all_f1_scores = {}
        field_f1_scores = {}

        for model_name in model_names:
            model_f1_scores = self.calculate_model_f1_scores(model_name)
            all_f1_scores[model_name] = model_f1_scores

            # Store field-specific metrics
            for field_name, metrics in model_f1_scores.items():
                if field_name not in field_f1_scores:
                    field_f1_scores[field_name] = {}
                field_f1_scores[field_name][model_name] = metrics

        # Calculate overall F1 scores
        overall_f1_scores = {}
        macro_averaged_f1 = {}
        weighted_f1 = {}

        for model_name in model_names:
            model_metrics = all_f1_scores[model_name]
            if model_metrics:
                f1_scores = [m.f1_score for m in model_metrics.values()]
                supports = [m.support for m in model_metrics.values()]

                overall_f1_scores[model_name] = statistics.mean(f1_scores) if f1_scores else 0
                macro_averaged_f1[model_name] = statistics.mean(f1_scores) if f1_scores else 0

                # Weighted F1
                if supports and sum(supports) > 0:
                    weighted_f1[model_name] = sum(
                        f1 * support for f1, support in zip(f1_scores, supports, strict=False)
                    ) / sum(supports)
                else:
                    weighted_f1[model_name] = 0

        # Calculate category-based performance
        category_performance = self._calculate_category_performance(all_f1_scores)

        # Calculate ATO compliance scores
        ato_compliance_scores = self._calculate_ato_compliance_scores()

        # Calculate critical field performance
        critical_field_performance = self._calculate_critical_field_performance(all_f1_scores)

        # Determine statistical significance (simplified)
        significant_differences = self._test_statistical_significance(overall_f1_scores)

        # Calculate confidence intervals (simplified)
        confidence_intervals = self._calculate_confidence_intervals(all_f1_scores)

        # Create model rankings
        model_rankings = self._create_model_rankings(
            overall_f1_scores, ato_compliance_scores, critical_field_performance
        )

        # Generate performance recommendations
        performance_recommendations = self._generate_performance_recommendations(all_f1_scores, model_names)

        return ModelComparisonMetrics(
            model_names=model_names,
            overall_f1_scores=overall_f1_scores,
            macro_averaged_f1=macro_averaged_f1,
            weighted_f1=weighted_f1,
            field_f1_scores=field_f1_scores,
            category_performance=category_performance,
            ato_compliance_scores=ato_compliance_scores,
            critical_field_performance=critical_field_performance,
            significant_differences=significant_differences,
            confidence_intervals=confidence_intervals,
            model_rankings=model_rankings,
            performance_recommendations=performance_recommendations,
        )

    def _calculate_category_performance(
        self, all_f1_scores: Dict[str, Dict[str, F1Metrics]]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate performance by field category."""
        category_performance = {}

        for category in FieldCategory:
            category_fields = PRODUCTION_SCHEMA.get_fields_by_category(category)
            category_performance[category.value] = {}

            for model_name, model_metrics in all_f1_scores.items():
                # Get F1 scores for fields in this category
                category_f1_scores = [
                    metrics.f1_score
                    for field_name, metrics in model_metrics.items()
                    if field_name in category_fields
                ]

                if category_f1_scores:
                    category_performance[category.value][model_name] = statistics.mean(category_f1_scores)
                else:
                    category_performance[category.value][model_name] = 0.0

        return category_performance

    def _calculate_ato_compliance_scores(self) -> Dict[str, float]:
        """Calculate ATO compliance scores based on critical fields."""
        compliance_scores = {}

        # Define ATO critical fields
        ato_critical_fields = ["supplierABN_a_pgs", "total_a_li", "tax_a_li", "date_a_li", "supplier_a_pgs"]

        for model_name, results in self.model_results.items():
            if not results:
                compliance_scores[model_name] = 0.0
                continue

            total_compliance = 0
            total_documents = len(results)

            for result in results:
                # Check compliance for this document
                document_compliance = 0
                for field in ato_critical_fields:
                    if field in result.extracted_fields:
                        # Validate the field value
                        value = result.extracted_fields[field]
                        if PRODUCTION_SCHEMA.validate_field_value(field, str(value)):
                            document_compliance += 1

                # Compliance score for this document (0-1)
                document_score = document_compliance / len(ato_critical_fields)
                total_compliance += document_score

            compliance_scores[model_name] = total_compliance / total_documents

        return compliance_scores

    def _calculate_critical_field_performance(
        self, all_f1_scores: Dict[str, Dict[str, F1Metrics]]
    ) -> Dict[str, float]:
        """Calculate performance on critical fields."""
        critical_field_performance = {}

        critical_fields = [
            field_name
            for field_name in PRODUCTION_SCHEMA.get_all_fields()
            if PRODUCTION_SCHEMA.get_field_definition(field_name).ato_compliance_level == "critical"
        ]

        for model_name, model_metrics in all_f1_scores.items():
            critical_f1_scores = [
                metrics.f1_score
                for field_name, metrics in model_metrics.items()
                if field_name in critical_fields
            ]

            if critical_f1_scores:
                critical_field_performance[model_name] = statistics.mean(critical_f1_scores)
            else:
                critical_field_performance[model_name] = 0.0

        return critical_field_performance

    def _test_statistical_significance(self, overall_f1_scores: Dict[str, float]) -> Dict[str, bool]:
        """Test statistical significance of differences (simplified).

        In production, would use proper statistical tests like paired t-tests.
        """
        significant_differences = {}

        if len(overall_f1_scores) < 2:
            return significant_differences

        model_names = list(overall_f1_scores.keys())

        for i, model1 in enumerate(model_names):
            for model2 in model_names[i + 1 :]:
                comparison_key = f"{model1}_vs_{model2}"

                # Simplified significance test - difference > 0.05
                difference = abs(overall_f1_scores[model1] - overall_f1_scores[model2])
                significant_differences[comparison_key] = difference > 0.05

        return significant_differences

    def _calculate_confidence_intervals(
        self, all_f1_scores: Dict[str, Dict[str, F1Metrics]]
    ) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals for F1 scores (simplified)."""
        confidence_intervals = {}

        for model_name, model_metrics in all_f1_scores.items():
            f1_scores = [metrics.f1_score for metrics in model_metrics.values()]

            if len(f1_scores) > 1:
                mean_f1 = statistics.mean(f1_scores)
                std_f1 = statistics.stdev(f1_scores)

                # 95% confidence interval (simplified)
                margin = 1.96 * std_f1 / np.sqrt(len(f1_scores))
                confidence_intervals[model_name] = (mean_f1 - margin, mean_f1 + margin)
            else:
                # Single value - no confidence interval
                mean_f1 = f1_scores[0] if f1_scores else 0
                confidence_intervals[model_name] = (mean_f1, mean_f1)

        return confidence_intervals

    def _create_model_rankings(
        self,
        overall_f1_scores: Dict[str, float],
        ato_compliance_scores: Dict[str, float],
        critical_field_performance: Dict[str, float],
    ) -> Dict[str, int]:
        """Create model rankings based on different metrics."""
        rankings = {}

        # Rank by overall F1 score
        f1_ranking = sorted(overall_f1_scores.items(), key=lambda x: x[1], reverse=True)
        rankings["overall_f1"] = {model: rank + 1 for rank, (model, _) in enumerate(f1_ranking)}

        # Rank by ATO compliance
        compliance_ranking = sorted(ato_compliance_scores.items(), key=lambda x: x[1], reverse=True)
        rankings["ato_compliance"] = {model: rank + 1 for rank, (model, _) in enumerate(compliance_ranking)}

        # Rank by critical field performance
        critical_ranking = sorted(critical_field_performance.items(), key=lambda x: x[1], reverse=True)
        rankings["critical_fields"] = {model: rank + 1 for rank, (model, _) in enumerate(critical_ranking)}

        # Overall ranking (weighted combination)
        overall_scores = {}
        for model in overall_f1_scores.keys():
            weighted_score = (
                overall_f1_scores[model] * 0.4
                + ato_compliance_scores[model] * 0.4
                + critical_field_performance[model] * 0.2
            )
            overall_scores[model] = weighted_score

        overall_ranking = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)
        rankings["overall"] = {model: rank + 1 for rank, (model, _) in enumerate(overall_ranking)}

        return rankings

    def _generate_performance_recommendations(
        self, all_f1_scores: Dict[str, Dict[str, F1Metrics]], model_names: List[str]
    ) -> Dict[str, List[str]]:
        """Generate performance improvement recommendations."""
        recommendations = {}

        for model_name in model_names:
            model_recommendations = []
            model_metrics = all_f1_scores[model_name]

            # Identify weak fields
            weak_fields = [
                field_name for field_name, metrics in model_metrics.items() if metrics.f1_score < 0.5
            ]

            if weak_fields:
                model_recommendations.append(
                    f"Focus on improving extraction for: {', '.join(weak_fields[:3])}"
                )

            # Precision vs Recall analysis
            low_precision_fields = [
                field_name
                for field_name, metrics in model_metrics.items()
                if metrics.precision < 0.6 and metrics.recall > 0.7
            ]

            if low_precision_fields:
                model_recommendations.append(
                    f"Reduce false positives for: {', '.join(low_precision_fields[:2])}"
                )

            low_recall_fields = [
                field_name
                for field_name, metrics in model_metrics.items()
                if metrics.recall < 0.6 and metrics.precision > 0.7
            ]

            if low_recall_fields:
                model_recommendations.append(
                    f"Improve detection rate for: {', '.join(low_recall_fields[:2])}"
                )

            # ATO compliance specific
            ato_critical_fields = ["supplierABN_a_pgs", "total_a_li", "tax_a_li"]
            weak_ato_fields = [
                field
                for field in ato_critical_fields
                if field in model_metrics and model_metrics[field].f1_score < 0.8
            ]

            if weak_ato_fields:
                model_recommendations.append(
                    f"Critical for ATO compliance: improve {', '.join(weak_ato_fields)}"
                )

            if not model_recommendations:
                model_recommendations.append("Performance is satisfactory across all metrics")

            recommendations[model_name] = model_recommendations

        return recommendations

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary.

        Returns:
            Dictionary with metrics summary
        """
        comparison = self.compare_models()
        if not comparison:
            return {"error": "Insufficient data for comparison"}

        summary = {
            "models_compared": comparison.model_names,
            "overall_performance": {
                model: f"{score:.3f}" for model, score in comparison.overall_f1_scores.items()
            },
            "ato_compliance": {
                model: f"{score:.3f}" for model, score in comparison.ato_compliance_scores.items()
            },
            "best_performers": {
                "overall_f1": max(
                    comparison.overall_f1_scores.keys(), key=lambda x: comparison.overall_f1_scores[x]
                ),
                "ato_compliance": max(
                    comparison.ato_compliance_scores.keys(),
                    key=lambda x: comparison.ato_compliance_scores[x],
                ),
                "critical_fields": max(
                    comparison.critical_field_performance.keys(),
                    key=lambda x: comparison.critical_field_performance[x],
                ),
            },
            "significant_differences": comparison.significant_differences,
            "recommendations": comparison.performance_recommendations,
        }

        return summary
