"""Simple Metrics for Model Comparison
=====================================

Simplified metrics focusing on what actually matters:
- Success rate
- Fields extracted
- Core fields found
- Processing time
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class SimpleModelMetrics:
    """Simple metrics for a model."""

    model_name: str
    total_images: int
    successful_extractions: int
    total_fields_extracted: int
    total_core_fields_found: int
    average_inference_time: float

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
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


class SimpleMetricsCalculator:
    """Calculate simple metrics from extraction results."""

    def __init__(self):
        self.results: Dict[str, List[Dict]] = {}

    def add_results(self, model_name: str, results: List[Dict]):
        """Add results for a model."""
        self.results[model_name] = results

    def calculate_metrics(self, model_name: str) -> Optional[SimpleModelMetrics]:
        """Calculate metrics for a model."""
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

        return SimpleModelMetrics(
            model_name=model_name,
            total_images=total_images,
            successful_extractions=successful,
            total_fields_extracted=total_fields,
            total_core_fields_found=total_core_fields,
            average_inference_time=avg_time
        )

    def compare_models(self) -> Dict[str, Dict]:
        """Simple comparison of all models."""
        comparison = {}

        for model_name in self.results:
            metrics = self.calculate_metrics(model_name)
            if metrics:
                comparison[model_name] = {
                    'success_rate': f"{metrics.success_rate:.1%}",
                    'avg_fields': f"{metrics.avg_fields_per_image:.1f}",
                    'avg_core_fields': f"{metrics.avg_core_fields_per_image:.1f}",
                    'avg_time': f"{metrics.average_inference_time:.1f}s"
                }

        return comparison

    def get_summary(self) -> Dict:
        """Get simple summary of all models."""
        summary = {
            'models_tested': list(self.results.keys()),
            'total_images': sum(len(r) for r in self.results.values()),
            'comparison': self.compare_models()
        }

        # Find best performing model
        best_success_rate = 0
        best_model = None

        for model_name in self.results:
            metrics = self.calculate_metrics(model_name)
            if metrics and metrics.success_rate > best_success_rate:
                best_success_rate = metrics.success_rate
                best_model = model_name

        if best_model:
            summary['best_performer'] = {
                'model': best_model,
                'success_rate': f"{best_success_rate:.1%}"
            }

        return summary
