"""Field Analyzer for Production Schema Analysis
===========================================

Analyzes field-level extraction performance, validation results, and
field-specific metrics using the production schema.
"""

import statistics
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from ..config.production_schema import PRODUCTION_SCHEMA, FieldCategory, FieldType
from ..extraction.production_extractor import ExtractionResult


@dataclass
class FieldPerformanceMetrics:
    """Performance metrics for a specific field."""

    field_name: str
    field_type: FieldType
    field_category: FieldCategory

    # Detection metrics
    total_documents: int
    detected_count: int
    detection_rate: float

    # Validation metrics
    valid_values_count: int
    invalid_values_count: int
    validation_rate: float

    # Value analysis
    unique_values: Set[str]
    most_common_values: List[Tuple[str, int]]
    value_patterns: List[str]

    # Quality metrics
    average_value_length: float
    empty_or_na_count: int

    # Model comparison
    model_performance: Dict[str, float]  # model_name -> detection_rate


@dataclass
class CategoryAnalysis:
    """Analysis for a field category."""

    category: FieldCategory
    field_count: int
    total_detections: int
    average_detection_rate: float

    # Performance by model
    model_performance: Dict[str, float]

    # Field distribution
    best_performing_fields: List[str]
    worst_performing_fields: List[str]

    # Category-specific insights
    critical_fields_performance: float
    required_fields_performance: float


class FieldAnalyzer:
    """Analyzes field-level extraction performance and patterns."""

    def __init__(self):
        """Initialize field analyzer."""
        self.model_results: Dict[str, List[ExtractionResult]] = {}
        self.production_schema = PRODUCTION_SCHEMA

    def add_results(self, model_name: str, results: List[ExtractionResult]):
        """Add extraction results for a model.

        Args:
            model_name: Name of the model
            results: List of extraction results
        """
        self.model_results[model_name] = results

    def analyze_field_performance(self, field_name: str) -> Optional[FieldPerformanceMetrics]:
        """Analyze performance for a specific field across all models.

        Args:
            field_name: Name of the field to analyze

        Returns:
            FieldPerformanceMetrics or None if field not found
        """
        field_definition = self.production_schema.get_field_definition(field_name)
        if not field_definition:
            return None

        # Collect all field values across all models and documents
        all_values = []
        model_detections = defaultdict(int)
        model_totals = defaultdict(int)
        valid_values = 0
        invalid_values = 0

        for model_name, results in self.model_results.items():
            for result in results:
                model_totals[model_name] += 1

                if field_name in result.extracted_fields:
                    value = result.extracted_fields[field_name]
                    all_values.append(str(value))
                    model_detections[model_name] += 1

                    # Validate value
                    if self.production_schema.validate_field_value(field_name, str(value)):
                        valid_values += 1
                    else:
                        invalid_values += 1

        total_documents = sum(model_totals.values())
        if total_documents == 0:
            return None

        # Calculate detection rates by model
        model_performance = {}
        for model_name in self.model_results.keys():
            if model_totals[model_name] > 0:
                model_performance[model_name] = model_detections[model_name] / model_totals[model_name]

        # Analyze values
        unique_values = set(all_values)
        value_counts = defaultdict(int)
        for value in all_values:
            value_counts[value] += 1

        most_common_values = sorted(value_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        # Calculate average value length
        average_value_length = statistics.mean([len(v) for v in all_values]) if all_values else 0

        # Count empty or N/A values
        invalid_indicators = ["", "N/A", "NA", "NOT AVAILABLE", "NOT FOUND", "NONE", "-"]
        empty_or_na_count = sum(1 for v in all_values if v.upper() in invalid_indicators)

        # Extract value patterns (for pattern analysis)
        value_patterns = self._extract_value_patterns(all_values, field_definition.field_type)

        return FieldPerformanceMetrics(
            field_name=field_name,
            field_type=field_definition.field_type,
            field_category=field_definition.category,
            total_documents=total_documents,
            detected_count=len(all_values),
            detection_rate=len(all_values) / total_documents,
            valid_values_count=valid_values,
            invalid_values_count=invalid_values,
            validation_rate=valid_values / len(all_values) if all_values else 0,
            unique_values=unique_values,
            most_common_values=most_common_values,
            value_patterns=value_patterns,
            average_value_length=average_value_length,
            empty_or_na_count=empty_or_na_count,
            model_performance=model_performance,
        )

    def analyze_category_performance(self, category: FieldCategory) -> Optional[CategoryAnalysis]:
        """Analyze performance for all fields in a category.

        Args:
            category: Field category to analyze

        Returns:
            CategoryAnalysis or None if no fields in category
        """
        category_fields = self.production_schema.get_fields_by_category(category)
        if not category_fields:
            return None

        # Analyze each field in the category
        field_metrics = {}
        for field_name in category_fields:
            metrics = self.analyze_field_performance(field_name)
            if metrics:
                field_metrics[field_name] = metrics

        if not field_metrics:
            return None

        # Calculate category-level metrics
        total_detections = sum(metrics.detected_count for metrics in field_metrics.values())
        detection_rates = [metrics.detection_rate for metrics in field_metrics.values()]
        average_detection_rate = statistics.mean(detection_rates)

        # Model performance aggregation
        model_performance = defaultdict(list)
        for metrics in field_metrics.values():
            for model_name, rate in metrics.model_performance.items():
                model_performance[model_name].append(rate)

        model_avg_performance = {
            model: statistics.mean(rates) for model, rates in model_performance.items()
        }

        # Identify best and worst performing fields
        sorted_fields = sorted(field_metrics.items(), key=lambda x: x[1].detection_rate, reverse=True)
        best_performing_fields = [name for name, _ in sorted_fields[:3]]
        worst_performing_fields = [name for name, _ in sorted_fields[-3:]]

        # Category-specific analysis
        critical_fields = [
            name
            for name, metrics in field_metrics.items()
            if self.production_schema.get_field_definition(name).ato_compliance_level == "critical"
        ]
        critical_fields_performance = (
            statistics.mean([field_metrics[name].detection_rate for name in critical_fields])
            if critical_fields
            else 0
        )

        required_fields = [
            name
            for name, metrics in field_metrics.items()
            if self.production_schema.get_field_definition(name).is_required
        ]
        required_fields_performance = (
            statistics.mean([field_metrics[name].detection_rate for name in required_fields])
            if required_fields
            else 0
        )

        return CategoryAnalysis(
            category=category,
            field_count=len(field_metrics),
            total_detections=total_detections,
            average_detection_rate=average_detection_rate,
            model_performance=model_avg_performance,
            best_performing_fields=best_performing_fields,
            worst_performing_fields=worst_performing_fields,
            critical_fields_performance=critical_fields_performance,
            required_fields_performance=required_fields_performance,
        )

    def _extract_value_patterns(self, values: List[str], field_type: FieldType) -> List[str]:
        """Extract common patterns from field values.

        Args:
            values: List of field values
            field_type: Type of field for pattern recognition

        Returns:
            List of common patterns found
        """
        patterns = []

        if not values:
            return patterns

        if field_type == FieldType.DATE:
            # Date patterns
            date_patterns = set()
            for value in values:
                if "/" in value:
                    date_patterns.add("DD/MM/YYYY format")
                elif "-" in value:
                    date_patterns.add("DD-MM-YYYY format")
            patterns.extend(list(date_patterns))

        elif field_type == FieldType.CURRENCY:
            # Currency patterns
            currency_patterns = set()
            for value in values:
                if value.startswith("$"):
                    currency_patterns.add("$X.XX format")
                elif "AUD" in value:
                    currency_patterns.add("AUD X.XX format")
            patterns.extend(list(currency_patterns))

        elif field_type == FieldType.ABN:
            # ABN patterns
            abn_patterns = set()
            for value in values:
                if " " in value:
                    abn_patterns.add("XX XXX XXX XXX format")
                else:
                    abn_patterns.add("11-digit string format")
            patterns.extend(list(abn_patterns))

        elif field_type == FieldType.PHONE:
            # Phone patterns
            phone_patterns = set()
            for value in values:
                if "(" in value and ")" in value:
                    phone_patterns.add("(XX) XXXX XXXX format")
                elif "-" in value:
                    phone_patterns.add("XX-XXXX-XXXX format")
                elif " " in value:
                    phone_patterns.add("XX XXXX XXXX format")
            patterns.extend(list(phone_patterns))

        # General patterns
        if values:
            # Length patterns
            lengths = [len(v) for v in values]
            if lengths:
                avg_length = statistics.mean(lengths)
                patterns.append(f"Average length: {avg_length:.1f} characters")

        return patterns

    def create_field_detection_matrix(self) -> pd.DataFrame:
        """Create matrix showing field detection rates by model.

        Returns:
            DataFrame with fields as rows and models as columns
        """
        all_fields = self.production_schema.get_all_fields()
        model_names = list(self.model_results.keys())

        matrix_data = []

        for field_name in all_fields:
            row = {"field": field_name}

            field_metrics = self.analyze_field_performance(field_name)
            if field_metrics:
                for model_name in model_names:
                    row[model_name] = field_metrics.model_performance.get(model_name, 0.0)
            else:
                for model_name in model_names:
                    row[model_name] = 0.0

            matrix_data.append(row)

        return pd.DataFrame(matrix_data).set_index("field")

    def identify_problematic_fields(self, threshold: float = 0.3) -> Dict[str, List[str]]:
        """Identify fields with poor detection rates.

        Args:
            threshold: Detection rate threshold below which fields are considered problematic

        Returns:
            Dictionary mapping issue types to lists of field names
        """
        issues = {
            "low_detection": [],
            "poor_validation": [],
            "inconsistent_across_models": [],
            "high_empty_values": [],
        }

        for field_name in self.production_schema.get_all_fields():
            metrics = self.analyze_field_performance(field_name)
            if not metrics:
                continue

            # Low detection rate
            if metrics.detection_rate < threshold:
                issues["low_detection"].append(field_name)

            # Poor validation rate
            if metrics.validation_rate < 0.7 and metrics.detected_count > 0:
                issues["poor_validation"].append(field_name)

            # Inconsistent across models
            if len(metrics.model_performance) > 1:
                detection_rates = list(metrics.model_performance.values())
                if max(detection_rates) - min(detection_rates) > 0.5:
                    issues["inconsistent_across_models"].append(field_name)

            # High empty/N/A values
            if metrics.detected_count > 0 and metrics.empty_or_na_count / metrics.detected_count > 0.3:
                issues["high_empty_values"].append(field_name)

        return issues

    def get_field_extraction_summary(self) -> Dict[str, Any]:
        """Get comprehensive field extraction summary.

        Returns:
            Dictionary with field extraction summary
        """
        all_fields = self.production_schema.get_all_fields()
        total_fields = len(all_fields)

        # Analyze all fields
        field_analyses = {}
        detection_rates = []

        for field_name in all_fields:
            metrics = self.analyze_field_performance(field_name)
            if metrics:
                field_analyses[field_name] = metrics
                detection_rates.append(metrics.detection_rate)

        # Calculate summary statistics
        overall_detection_rate = statistics.mean(detection_rates) if detection_rates else 0

        # Analyze by category
        category_summaries = {}
        for category in FieldCategory:
            category_analysis = self.analyze_category_performance(category)
            if category_analysis:
                category_summaries[category.value] = {
                    "field_count": category_analysis.field_count,
                    "detection_rate": category_analysis.average_detection_rate,
                    "best_fields": category_analysis.best_performing_fields,
                    "worst_fields": category_analysis.worst_performing_fields,
                }

        # Identify top and bottom performers
        sorted_fields = sorted(field_analyses.items(), key=lambda x: x[1].detection_rate, reverse=True)

        top_performers = [(name, metrics.detection_rate) for name, metrics in sorted_fields[:5]]
        bottom_performers = [(name, metrics.detection_rate) for name, metrics in sorted_fields[-5:]]

        return {
            "total_fields_analyzed": len(field_analyses),
            "overall_detection_rate": overall_detection_rate,
            "category_performance": category_summaries,
            "top_performing_fields": top_performers,
            "bottom_performing_fields": bottom_performers,
            "problematic_fields": self.identify_problematic_fields(),
            "field_type_distribution": self._get_field_type_distribution(),
        }

    def _get_field_type_distribution(self) -> Dict[str, int]:
        """Get distribution of field types in the schema."""
        type_counts = defaultdict(int)

        for field_name in self.production_schema.get_all_fields():
            field_definition = self.production_schema.get_field_definition(field_name)
            if field_definition:
                type_counts[field_definition.field_type.value] += 1

        return dict(type_counts)

    def generate_field_recommendations(self) -> Dict[str, List[str]]:
        """Generate recommendations for improving field extraction.

        Returns:
            Dictionary with recommendations by category
        """
        recommendations = {
            "prompt_improvements": [],
            "validation_enhancements": [],
            "pattern_additions": [],
            "model_specific": [],
        }

        problematic_fields = self.identify_problematic_fields()

        # Prompt improvement recommendations
        if problematic_fields["low_detection"]:
            recommendations["prompt_improvements"].append(
                f"Consider adding explicit field requests for: {', '.join(problematic_fields['low_detection'][:5])}"
            )

        # Validation enhancement recommendations
        if problematic_fields["poor_validation"]:
            recommendations["validation_enhancements"].append(
                f"Improve validation rules for: {', '.join(problematic_fields['poor_validation'][:5])}"
            )

        # Pattern addition recommendations
        for field_name in problematic_fields["low_detection"][:3]:
            field_definition = self.production_schema.get_field_definition(field_name)
            if field_definition and not field_definition.extraction_patterns:
                recommendations["pattern_additions"].append(
                    f"Add extraction patterns for {field_name} ({field_definition.field_type.value})"
                )

        # Model-specific recommendations
        if len(self.model_results) > 1:
            for field_name in problematic_fields["inconsistent_across_models"][:3]:
                recommendations["model_specific"].append(
                    f"Investigate model-specific issues with {field_name} extraction"
                )

        return recommendations
