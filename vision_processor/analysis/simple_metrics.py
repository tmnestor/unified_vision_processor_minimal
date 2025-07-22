"""Information Extraction Capability Metrics
==========================================

Advanced metrics focusing on Information Extraction Capability:
- Extraction completeness (how much information is captured)
- Information density (quality of extracted data)
- Critical field coverage (essential business information)
- Relative extraction performance between models
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

# All fields equally weighted for extraction capability
FIELD_WEIGHTS = {
    "DATE": 1.0,
    "SUPPLIER": 1.0,
    "ABN": 1.0,
    "GST": 1.0,
    "TOTAL": 1.0,
    "SUBTOTAL": 1.0,
    "ITEMS": 1.0,
    "QUANTITIES": 1.0,
    "PRICES": 1.0,
    "RECEIPT_NUMBER": 1.0,
    "PAYMENT_METHOD": 1.0,
    "DOCUMENT_TYPE": 1.0,
    "BUSINESS_ADDRESS": 1.0,
    "BUSINESS_PHONE": 1.0,
    "PAYER_NAME": 1.0,
    "PAYER_ADDRESS": 1.0,
    "PAYER_PHONE": 1.0,
    "PAYER_EMAIL": 1.0,
    "BANK_ACCOUNT_NUMBER": 1.0,
}

@dataclass
class InformationExtractionMetrics:
    """Advanced metrics focused on Information Extraction Capability."""

    model_name: str
    total_images: int
    successful_extractions: int
    total_fields_extracted: int
    total_core_fields_found: int
    weighted_extraction_score: float
    critical_field_coverage: float
    information_density: float
    extraction_completeness: float
    average_inference_time: float

    @property
    def success_rate(self) -> float:
        """Calculate basic success rate."""
        if self.total_images == 0:
            return 0.0
        return self.successful_extractions / self.total_images

    @property
    def avg_fields_per_image(self) -> float:
        """Average fields extracted per image."""
        if self.total_images == 0:
            return 0.0
        return self.total_fields_extracted / self.total_images

    @property
    def avg_core_fields_per_image(self) -> float:
        """Average core fields found per image."""
        if self.total_images == 0:
            return 0.0
        return self.total_core_fields_found / self.total_images

    @property
    def information_extraction_capability(self) -> float:
        """
        Information Extraction Capability based purely on field extraction volume.

        Since all fields are equally important:
        - More fields extracted = higher capability
        - Success rate multiplier ensures failed extractions are penalized
        """
        return self.avg_fields_per_image * self.success_rate


class InformationExtractionCalculator:
    """Calculate Information Extraction Capability metrics from results."""

    def __init__(self):
        self.results: Dict[str, List[Dict]] = {}

    def add_results(self, model_name: str, results: List[Dict]):
        """Add results for a model."""
        self.results[model_name] = results

    def _calculate_weighted_extraction_score(self, extracted_fields: List[str]) -> float:
        """Calculate weighted extraction score based on field importance."""
        if not extracted_fields:
            return 0.0

        total_weight = 0.0
        max_possible_weight = sum(FIELD_WEIGHTS.values())

        for field in extracted_fields:
            field_upper = field.upper()
            if field_upper in FIELD_WEIGHTS:
                total_weight += FIELD_WEIGHTS[field_upper]

        return total_weight / max_possible_weight

    def _calculate_critical_field_coverage(self, extracted_fields: List[str]) -> float:
        """Calculate coverage of critical fields (weight >= 0.8)."""
        critical_fields = [f for f, w in FIELD_WEIGHTS.items() if w >= 0.8]
        if not critical_fields:
            return 1.0

        found_critical = 0
        for field in extracted_fields:
            if field.upper() in critical_fields:
                found_critical += 1

        return found_critical / len(critical_fields)

    def _calculate_information_density(self, extracted_fields: List[str], total_fields: int) -> float:
        """Calculate information density - quality vs quantity of extracted fields."""
        if total_fields == 0:
            return 0.0

        # Ratio of valuable fields (in our weight list) to total fields extracted
        valuable_fields = sum(1 for f in extracted_fields if f.upper() in FIELD_WEIGHTS)
        return valuable_fields / total_fields

    def _calculate_extraction_completeness(self, extracted_fields: List[str]) -> float:
        """Calculate how complete the extraction is compared to maximum possible."""
        max_possible_fields = len(FIELD_WEIGHTS)
        relevant_fields = sum(1 for f in extracted_fields if f.upper() in FIELD_WEIGHTS)
        return relevant_fields / max_possible_fields

    def calculate_metrics(self, model_name: str) -> Optional[InformationExtractionMetrics]:
        """Calculate comprehensive Information Extraction Capability metrics."""
        if model_name not in self.results:
            return None

        results = self.results[model_name]
        if not results:
            return None

        total_images = len(results)
        successful = sum(1 for r in results if r.get('successful', False))
        total_fields = sum(r.get('field_count', 0) for r in results)
        total_core_fields = sum(r.get('core_fields_found', 0) for r in results)
        total_time = sum(r.get('extraction_time', 0) for r in results)
        avg_time = total_time / total_images if total_images > 0 else 0

        # Calculate advanced metrics
        weighted_scores = []
        critical_coverages = []
        information_densities = []
        extraction_completenesses = []

        for result in results:
            extracted_fields = result.get('extracted_fields', [])
            if isinstance(extracted_fields, dict):
                extracted_fields = list(extracted_fields.keys())
            elif not isinstance(extracted_fields, list):
                extracted_fields = []

            field_count = result.get('field_count', len(extracted_fields))

            weighted_scores.append(self._calculate_weighted_extraction_score(extracted_fields))
            critical_coverages.append(self._calculate_critical_field_coverage(extracted_fields))
            information_densities.append(self._calculate_information_density(extracted_fields, field_count))
            extraction_completenesses.append(self._calculate_extraction_completeness(extracted_fields))

        # Average the advanced metrics
        avg_weighted_score = sum(weighted_scores) / len(weighted_scores) if weighted_scores else 0
        avg_critical_coverage = sum(critical_coverages) / len(critical_coverages) if critical_coverages else 0
        avg_information_density = sum(information_densities) / len(information_densities) if information_densities else 0
        avg_extraction_completeness = sum(extraction_completenesses) / len(extraction_completenesses) if extraction_completenesses else 0

        return InformationExtractionMetrics(
            model_name=model_name,
            total_images=total_images,
            successful_extractions=successful,
            total_fields_extracted=total_fields,
            total_core_fields_found=total_core_fields,
            weighted_extraction_score=avg_weighted_score,
            critical_field_coverage=avg_critical_coverage,
            information_density=avg_information_density,
            extraction_completeness=avg_extraction_completeness,
            average_inference_time=avg_time
        )

    def compare_models(self) -> Dict[str, Dict]:
        """Advanced comparison focused on Information Extraction Capability."""
        comparison = {}

        for model_name in self.results:
            metrics = self.calculate_metrics(model_name)
            if metrics:
                comparison[model_name] = {
                    'extraction_capability': f"{metrics.information_extraction_capability:.3f}",
                    'weighted_score': f"{metrics.weighted_extraction_score:.3f}",
                    'critical_coverage': f"{metrics.critical_field_coverage:.1%}",
                    'info_density': f"{metrics.information_density:.3f}",
                    'completeness': f"{metrics.extraction_completeness:.1%}",
                    'success_rate': f"{metrics.success_rate:.1%}",
                    'avg_fields': f"{metrics.avg_fields_per_image:.1f}",
                    'avg_time': f"{metrics.average_inference_time:.1f}s"
                }

        return comparison

    def get_extraction_capability_ranking(self) -> List[tuple[str, float]]:
        """Get models ranked by Information Extraction Capability."""
        rankings = []

        for model_name in self.results:
            metrics = self.calculate_metrics(model_name)
            if metrics:
                rankings.append((model_name, metrics.information_extraction_capability))

        # Sort by extraction capability (descending)
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings

    def get_detailed_analysis(self) -> Dict:
        """Get detailed analysis of Information Extraction Capability."""
        analysis = {
            'models_tested': list(self.results.keys()),
            'total_images': sum(len(r) for r in self.results.values()),
            'field_weights_used': FIELD_WEIGHTS,
            'metrics_breakdown': {}
        }

        # Calculate relative performance
        all_metrics = {}
        for model_name in self.results:
            metrics = self.calculate_metrics(model_name)
            if metrics:
                all_metrics[model_name] = metrics

        # Find best performer by Information Extraction Capability
        ranking = self.get_extraction_capability_ranking()
        if ranking:
            best_model, best_score = ranking[0]
            analysis['best_performer'] = {
                'model': best_model,
                'extraction_capability': f"{best_score:.3f}",
                'superiority_analysis': self._analyze_superiority(all_metrics, best_model)
            }

        # Detailed breakdown for each model
        for model_name, metrics in all_metrics.items():
            analysis['metrics_breakdown'][model_name] = {
                'overall_capability': metrics.information_extraction_capability,
                'component_scores': {
                    'weighted_extraction': metrics.weighted_extraction_score,
                    'critical_coverage': metrics.critical_field_coverage,
                    'information_density': metrics.information_density,
                    'extraction_completeness': metrics.extraction_completeness
                },
                'traditional_metrics': {
                    'success_rate': metrics.success_rate,
                    'avg_fields_per_image': metrics.avg_fields_per_image,
                    'avg_core_fields_per_image': metrics.avg_core_fields_per_image,
                    'avg_inference_time': metrics.average_inference_time
                }
            }

        return analysis

    def _analyze_superiority(self, all_metrics: Dict[str, InformationExtractionMetrics], best_model: str) -> Dict:
        """Analyze why one model is superior in Information Extraction Capability."""
        if best_model not in all_metrics or len(all_metrics) < 2:
            return {}

        best = all_metrics[best_model]
        superiority = {
            'strengths': [],
            'comparison_with_others': {}
        }

        # Identify strengths
        if best.critical_field_coverage >= 0.8:
            superiority['strengths'].append(f"Excellent critical field coverage ({best.critical_field_coverage:.1%})")
        if best.information_density >= 0.7:
            superiority['strengths'].append(f"High information density ({best.information_density:.3f})")
        if best.weighted_extraction_score >= 0.6:
            superiority['strengths'].append(f"Strong weighted extraction ({best.weighted_extraction_score:.3f})")

        # Compare with other models
        for other_model, other_metrics in all_metrics.items():
            if other_model != best_model:
                capability_diff = best.information_extraction_capability - other_metrics.information_extraction_capability
                superiority['comparison_with_others'][other_model] = {
                    'capability_advantage': f"{capability_diff:.3f}",
                    'percentage_better': f"{(capability_diff / other_metrics.information_extraction_capability * 100):.1f}%" if other_metrics.information_extraction_capability > 0 else "significantly better",
                    'key_advantages': self._identify_key_advantages(best, other_metrics)
                }

        return superiority

    def _identify_key_advantages(self, best: InformationExtractionMetrics, other: InformationExtractionMetrics) -> List[str]:
        """Identify specific advantages of one model over another."""
        advantages = []

        if best.critical_field_coverage - other.critical_field_coverage >= 0.1:
            advantages.append(f"Better critical field coverage (+{(best.critical_field_coverage - other.critical_field_coverage):.1%})")

        if best.information_density - other.information_density >= 0.1:
            advantages.append(f"Higher information density (+{(best.information_density - other.information_density):.3f})")

        if best.weighted_extraction_score - other.weighted_extraction_score >= 0.1:
            advantages.append(f"Superior weighted extraction (+{(best.weighted_extraction_score - other.weighted_extraction_score):.3f})")

        if best.extraction_completeness - other.extraction_completeness >= 0.1:
            advantages.append(f"More complete extraction (+{(best.extraction_completeness - other.extraction_completeness):.1%})")

        return advantages

    def create_dataframe(self, comparison_results):
        """Create a pandas DataFrame from results for CSV export.

        Args:
            comparison_results: ComparisonResults object

        Returns:
            pandas.DataFrame with flattened results
        """
        try:
            import pandas as pd
        except ImportError:
            print("⚠️  pandas not available, cannot create DataFrame")
            return None

        rows = []

        # Extract data from comparison results
        if hasattr(comparison_results, 'model_results'):
            for model_name, model_data in comparison_results.model_results.items():
                if hasattr(model_data, 'extraction_results'):
                    for result in model_data.extraction_results:
                        row = {
                            'model': model_name,
                            'image': getattr(result, 'image_name', 'unknown'),
                            'successful': getattr(result, 'successful', False),
                            'field_count': len(getattr(result, 'extracted_fields', {})),
                            'core_fields_found': getattr(result, 'core_fields_found', 0),
                            'extraction_time': getattr(result, 'extraction_time', 0.0),
                        }
                        rows.append(row)

        if rows:
            return pd.DataFrame(rows)
        else:
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=[
                'model', 'image', 'successful', 'field_count',
                'core_fields_found', 'extraction_time'
            ])
