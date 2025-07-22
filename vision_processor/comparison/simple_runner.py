"""Simple Comparison Runner
=========================

Simplified model comparison runner using the refactored components.
"""

import gc
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd
from rich.console import Console
from rich.progress import track

from ..analysis.simple_metrics import SimpleMetricsCalculator
from ..config.model_registry import get_model_registry
from ..extraction.dynamic_extractor import DynamicFieldExtractor
from ..utils.simple_repetition_control import SimpleRepetitionCleaner

console = Console()


@dataclass
class SimpleComparisonResults:
    """Simple comparison results."""

    models_tested: List[str]
    total_images: int
    extraction_results: Dict[str, List[Dict]]
    metrics_summary: Dict
    execution_time: float


class SimpleComparisonRunner:
    """Simplified comparison runner."""

    def __init__(self, datasets_path: str, output_dir: str, max_tokens: int = 256):
        """Initialize runner.

        Args:
            datasets_path: Path to dataset directory
            output_dir: Output directory for results
            max_tokens: Maximum tokens for model generation
        """
        self.datasets_path = Path(datasets_path)
        self.output_dir = Path(output_dir)
        self.max_tokens = max_tokens
        self.console = console

        # Create components
        self.extractor = DynamicFieldExtractor(min_fields_for_success=3)
        self.repetition_cleaner = SimpleRepetitionCleaner()
        self.metrics_calculator = SimpleMetricsCalculator()
        self.model_registry = get_model_registry()

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_comparison(self, model_names: List[str]) -> SimpleComparisonResults:
        """Run model comparison.

        Args:
            model_names: List of models to compare

        Returns:
            SimpleComparisonResults with all results
        """
        start_time = time.time()

        # Step 1: Load dataset
        self.console.print("📁 Loading Dataset", style="bold blue")
        image_paths = self._load_dataset()
        if not image_paths:
            raise ValueError("No images found in dataset")

        self.console.print(f"✅ Found {len(image_paths)} images")

        # Step 2: Run extractions
        extraction_results = {}

        for model_name in model_names:
            self.console.print(f"\n🤖 Processing with {model_name.upper()}", style="bold cyan")

            try:
                results = self._process_model(model_name, image_paths)
                extraction_results[model_name] = results

                # Add to metrics calculator
                self.metrics_calculator.add_results(model_name, results)

                # Print quick summary
                successful = sum(1 for r in results if r.get('successful', False))
                self.console.print(
                    f"✅ {model_name}: {successful}/{len(results)} successful extractions"
                )

            except Exception as e:
                self.console.print(f"❌ Error with {model_name}: {e}", style="red")
                extraction_results[model_name] = []

        # Step 3: Calculate metrics
        self.console.print("\n📊 Calculating Metrics", style="bold green")
        metrics_summary = self.metrics_calculator.get_summary()

        # Step 4: Export results
        self._export_results(extraction_results, metrics_summary)

        # Create results object
        results = SimpleComparisonResults(
            models_tested=model_names,
            total_images=len(image_paths),
            extraction_results=extraction_results,
            metrics_summary=metrics_summary,
            execution_time=time.time() - start_time
        )

        # Print final summary
        self._print_summary(results)

        return results

    def _load_dataset(self) -> List[Path]:
        """Load dataset images."""
        image_extensions = {'.png', '.jpg', '.jpeg'}
        image_paths = []

        for ext in image_extensions:
            image_paths.extend(self.datasets_path.glob(f'*{ext}'))

        return sorted(image_paths)

    def _process_model(self, model_name: str, image_paths: List[Path]) -> List[Dict]:
        """Process all images with a model."""
        # Create model
        model_config = self.model_registry.get_model_config(model_name)
        if not model_config:
            raise ValueError(f"Unknown model: {model_name}")

        model = self.model_registry.create_model(model_name)

        results = []

        # Process each image
        for i, image_path in enumerate(track(image_paths, description=f"Processing {model_name}")):
            try:
                # Get model response
                response = model.process_image(
                    str(image_path),
                    "Extract all fields and values from this document.",
                    max_new_tokens=self.max_tokens
                )

                # Clean response
                cleaned_response = self.repetition_cleaner.clean_response(response.raw_text)

                # Extract fields
                extraction_result = self.extractor.extract_fields(
                    cleaned_response,
                    image_path.name,
                    model_name,
                    response.processing_time
                )

                # Convert to simple format
                result = self._convert_to_simple_format(extraction_result)
                results.append(result)

            except Exception as e:
                # Handle errors gracefully
                results.append({
                    'image_name': image_path.name,
                    'model_name': model_name,
                    'successful': False,
                    'error': str(e),
                    'field_count': 0,
                    'core_fields_found': 0,
                    'extraction_time': 0
                })

            # Periodic cleanup
            if (i + 1) % 5 == 0:
                gc.collect()

        # Cleanup model
        del model
        gc.collect()

        return results

    def _convert_to_simple_format(self, extraction_result) -> Dict:
        """Convert extraction result to simple format."""
        # Define core fields
        CORE_FIELDS = {
            "DATE", "TOTAL", "GST", "ABN", "SUPPLIER_NAME",
            "INVOICE_NUMBER", "AMOUNT", "DESCRIPTION",
            "BSB", "ACCOUNT_NUMBER", "BUSINESS_NAME", "RECEIPT_NUMBER"
        }

        # Count core fields
        core_fields_found = 0
        if extraction_result.extracted_fields:
            for field_name in extraction_result.extracted_fields:
                if field_name.upper() in CORE_FIELDS:
                    core_fields_found += 1

        return {
            'image_name': extraction_result.image_name,
            'model_name': extraction_result.model_name,
            'successful': extraction_result.is_successful,
            'field_count': extraction_result.field_count,
            'core_fields_found': core_fields_found,
            'extraction_time': extraction_result.processing_time,
            'fields': extraction_result.extracted_fields,
            'confidence_score': extraction_result.extraction_score / 10.0
        }

    def _export_results(self, extraction_results: Dict, metrics_summary: Dict):
        """Export results to files."""
        # Export JSON results
        json_path = self.output_dir / 'simple_comparison_results.json'
        with json_path.open('w') as f:
            json.dump({
                'extraction_results': extraction_results,
                'metrics_summary': metrics_summary
            }, f, indent=2)

        self.console.print(f"✅ JSON results: {json_path}")

        # Export CSV for analysis
        csv_data = []
        for model_name, results in extraction_results.items():
            for result in results:
                csv_data.append({
                    'model': model_name,
                    'image': result['image_name'],
                    'field_count': result['field_count'],
                    'core_fields_found': result['core_fields_found'],
                    'successful': result['successful'],
                    'extraction_time': result['extraction_time'],
                    'confidence_score': result.get('confidence_score', 0)
                })

        df = pd.DataFrame(csv_data)
        csv_path = self.output_dir / 'simple_results.csv'
        df.to_csv(csv_path, index=False)

        self.console.print(f"✅ CSV results: {csv_path}")

    def _print_summary(self, results: SimpleComparisonResults):
        """Print results summary."""
        self.console.print("\n📊 COMPARISON SUMMARY", style="bold green")
        self.console.print("=" * 50)

        self.console.print(f"Total images: {results.total_images}")
        self.console.print(f"Models tested: {', '.join(results.models_tested)}")
        self.console.print(f"Execution time: {results.execution_time:.1f}s")

        if 'comparison' in results.metrics_summary:
            self.console.print("\nModel Performance:")
            for model, metrics in results.metrics_summary['comparison'].items():
                self.console.print(f"  {model}:")
                for key, value in metrics.items():
                    self.console.print(f"    {key}: {value}")

        if 'best_performer' in results.metrics_summary:
            best = results.metrics_summary['best_performer']
            self.console.print(
                f"\n🏆 Best performer: {best['model']} ({best['success_rate']})"
            )
