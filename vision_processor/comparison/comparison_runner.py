"""Comparison Runner - Main Orchestrator
====================================

Orchestrates the complete model comparison pipeline including dataset validation,
model loading, extraction, analysis, and reporting generation.
"""

import gc
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image
from rich.console import Console
from rich.progress import track

from ..analysis.comparison_metrics import ComparisonMetrics
from ..analysis.field_analyzer import FieldAnalyzer
from ..analysis.performance_analyzer import PerformanceAnalyzer
from ..config.model_registry import get_model_registry
from ..config.production_config import ProductionConfig
from ..extraction.dynamic_extractor import DynamicExtractionResult, DynamicFieldExtractor
from ..extraction.production_extractor import ExtractionResult
from .model_validator import ModelValidator


@dataclass
class DatasetInfo:
    """Information about the dataset being processed."""

    dataset_path: Path
    total_images: int
    image_files: List[Path]
    verified_images: List[Path]
    missing_images: List[Path]

    def get_validation_summary(self) -> Dict[str, Any]:
        """Get dataset validation summary."""
        return {
            "total_expected": self.total_images,
            "found": len(self.verified_images),
            "missing": len(self.missing_images),
            "validation_rate": len(self.verified_images) / self.total_images
            if self.total_images > 0
            else 0,
        }


@dataclass
class ComparisonResults:
    """Complete comparison results."""

    # Configuration
    config: ProductionConfig
    dataset_info: DatasetInfo
    models_tested: List[str]

    # Extraction results
    extraction_results: Dict[str, List[ExtractionResult]]

    # Analysis results
    performance_analysis: Optional[Any] = None
    field_analysis: Optional[Any] = None
    comparison_metrics: Optional[Any] = None

    # Timing information
    total_execution_time: float = 0.0
    model_execution_times: Dict[str, float] = None

    # Success metrics
    overall_success_rate: float = 0.0
    model_success_rates: Dict[str, float] = None

    def __post_init__(self):
        if self.model_execution_times is None:
            self.model_execution_times = {}
        if self.model_success_rates is None:
            self.model_success_rates = {}


class ComparisonRunner:
    """Main orchestrator for model comparison pipeline."""

    def __init__(self, config: ProductionConfig):
        """Initialize comparison runner.

        Args:
            config: Production configuration
        """
        self.config = config
        self.console = Console()

        # Initialize components
        self.model_registry = get_model_registry()
        self.model_validator = ModelValidator(self.model_registry)
        # Use dynamic extractor for original script compatibility
        self.extractor = DynamicFieldExtractor(
            min_fields_for_success=config.extraction.min_total_fields,
        )

        # Initialize analyzers
        self.performance_analyzer = PerformanceAnalyzer()
        self.field_analyzer = FieldAnalyzer()
        self.comparison_metrics = ComparisonMetrics()

        # Results storage
        self.results: Optional[ComparisonResults] = None

    def run_comparison(self) -> ComparisonResults:
        """Run complete model comparison pipeline.

        Returns:
            ComparisonResults with complete analysis
        """
        start_time = time.time()

        self.console.print("🏆 PRODUCTION MODEL COMPARISON PIPELINE", style="bold blue")
        self.console.print(f"📋 Models: {', '.join(self.config.models)}")
        self.console.print(f"📁 Dataset: {self.config.datasets_path}")
        self.console.print(f"📊 Output: {self.config.output_dir}")

        # Step 1: Validate configuration and environment
        self._validate_environment()

        # Step 2: Discover and validate dataset
        dataset_info = self._discover_and_validate_dataset()

        # Step 3: Validate models
        valid_models = self._validate_models()

        if not valid_models:
            raise RuntimeError("No valid models found for comparison")

        # Step 4: Run extraction for each model
        extraction_results = self._run_extractions(valid_models, dataset_info.verified_images)

        # Step 5: Perform comprehensive analysis
        analysis_results = self._run_analysis(extraction_results)

        # Step 6: Calculate timing and success metrics
        total_time = time.time() - start_time
        success_metrics = self._calculate_success_metrics(extraction_results)

        # Create results object
        self.results = ComparisonResults(
            config=self.config,
            dataset_info=dataset_info,
            models_tested=valid_models,
            extraction_results=extraction_results,
            performance_analysis=analysis_results["performance"],
            field_analysis=analysis_results["field"],
            comparison_metrics=analysis_results["metrics"],
            total_execution_time=total_time,
            model_execution_times=success_metrics["execution_times"],
            overall_success_rate=success_metrics["overall_success_rate"],
            model_success_rates=success_metrics["model_success_rates"],
        )

        # Step 7: Print summary
        self._print_completion_summary()

        return self.results

    def _validate_environment(self):
        """Validate environment and configuration."""
        self.console.print("\n🔍 ENVIRONMENT VALIDATION", style="bold yellow")

        # Validate configuration
        if not self.config.validate():
            raise RuntimeError("Configuration validation failed")

        self.console.print("✅ Configuration validated")

        # Print GPU information
        try:
            import torch

            if torch.cuda.is_available():
                self.console.print(f"✅ CUDA Available: {torch.cuda.get_device_name(0)}")
                memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
                self.console.print(f"   GPU Memory: {memory_gb:.1f}GB")
            else:
                self.console.print("⚠️  CUDA not available - using CPU")
        except ImportError:
            self.console.print("⚠️  PyTorch not available")

    def _discover_and_validate_dataset(self) -> DatasetInfo:
        """Discover and validate dataset images."""
        self.console.print("\n📊 DATASET DISCOVERY", style="bold yellow")

        dataset_path = Path(self.config.datasets_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

        # Discover all PNG images
        image_files = list(dataset_path.glob("*.png"))
        self.console.print(f"📁 Found {len(image_files)} PNG images in {dataset_path}")

        # Verify images can be opened
        verified_images = []
        missing_images = []

        for image_path in image_files:
            try:
                with Image.open(image_path) as img:
                    # Basic validation - ensure image can be loaded
                    img.verify()
                verified_images.append(image_path)
            except Exception as e:
                self.console.print(f"⚠️  Invalid image: {image_path.name} - {e}")
                missing_images.append(image_path)

        dataset_info = DatasetInfo(
            dataset_path=dataset_path,
            total_images=len(image_files),
            image_files=image_files,
            verified_images=verified_images,
            missing_images=missing_images,
        )

        # Print validation summary
        validation = dataset_info.get_validation_summary()
        self.console.print(f"✅ Verified: {validation['found']}/{validation['total_expected']} images")

        if validation["found"] == 0:
            raise RuntimeError("No valid images found in dataset")
        elif validation["found"] < validation["total_expected"]:
            self.console.print("⚠️  Some images failed validation but proceeding", style="yellow")

        return dataset_info

    def _validate_models(self) -> List[str]:
        """Validate that requested models can be loaded."""
        self.console.print("\n🤖 MODEL VALIDATION", style="bold yellow")

        valid_models = []

        for model_name in self.config.models:
            self.console.print(f"🔍 Validating {model_name}...")

            # Check if model is registered
            if not self.model_registry.get_model(model_name):
                self.console.print(f"❌ Model not registered: {model_name}")
                continue

            # Get model path
            model_path = getattr(self.config.model_paths, model_name, None)
            if not model_path:
                self.console.print(f"❌ No model path configured for: {model_name}")
                continue

            # Validate model
            if self.model_validator.validate_model_loading(model_name, model_path, self.config.processing):
                valid_models.append(model_name)
                self.console.print(f"✅ {model_name} validation passed")
            else:
                self.console.print(f"❌ {model_name} validation failed")

        self.console.print(f"\n📋 Valid models: {valid_models}")
        return valid_models

    def _run_extractions(
        self, model_names: List[str], image_paths: List[Path]
    ) -> Dict[str, List[ExtractionResult]]:
        """Run extractions for all models on all images."""
        self.console.print("\n🔥 EXTRACTION PIPELINE", style="bold yellow")

        extraction_results = {}

        for model_name in model_names:
            self.console.print(f"\n{'=' * 50}")
            self.console.print(f"🤖 PROCESSING WITH {model_name.upper()}", style="bold cyan")
            self.console.print(f"{'=' * 50}")

            model_results = []
            model_start_time = time.time()

            # Create model instance
            try:
                model_path = getattr(self.config.model_paths, model_name)
                model = self.model_registry.create_model(
                    model_name, self.config.processing, model_path=model_path
                )

                # Load model
                with model:
                    self.console.print(f"✅ {model_name} loaded successfully")

                    # Get model-specific prompt
                    prompts = self.config.get_prompts()
                    prompt = prompts.get(model_name, prompts.get("default", ""))

                    # Process each image
                    for i, image_path in enumerate(
                        track(image_paths, description=f"Processing with {model_name}")
                    ):
                        try:
                            # Load image
                            image = Image.open(image_path).convert("RGB")

                            # Run model inference
                            response = model.process_image(
                                image, prompt, max_new_tokens=self.config.processing.max_tokens
                            )

                            # Extract fields using dynamic extractor (original script logic)
                            extraction_result = self.extractor.extract_fields(
                                response.raw_text, image_path.name, model_name, response.processing_time
                            )

                            # Note: DynamicExtractionResult already has these fields populated

                            model_results.append(extraction_result)

                            # Print progress
                            status = "✅" if extraction_result.is_successful else "❌"
                            fields_str = f"{extraction_result.field_count} fields"
                            time_str = f"{response.processing_time:.1f}s"

                            self.console.print(
                                f"   {i + 1:2d}. {image_path.name:<15} {status} {time_str} | {fields_str}"
                            )

                            # Cleanup every few images
                            if (i + 1) % 5 == 0:
                                gc.collect()

                        except Exception as e:
                            self.console.print(
                                f"   {i + 1:2d}. {image_path.name:<15} ❌ Error: {str(e)[:30]}..."
                            )
                            # Create error result
                            error_result = DynamicExtractionResult(
                                image_name=image_path.name,
                                model_name=model_name,
                                processing_time=0.0,
                                raw_response=f"Error: {e}",
                                cleaned_response="",
                                extracted_fields={},
                                field_count=0,
                                is_successful=False,
                                extraction_score=0,
                                using_raw_markdown=False,
                                processing_notes=[f"Processing error: {e}"],
                            )
                            model_results.append(error_result)

            except Exception as e:
                self.console.print(f"❌ Failed to load {model_name}: {e}")
                continue

            finally:
                # Cleanup model
                gc.collect()
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass

            # Calculate model summary
            model_time = time.time() - model_start_time
            successful_extractions = sum(1 for r in model_results if r.is_successful)
            success_rate = successful_extractions / len(model_results) if model_results else 0
            avg_fields = (
                sum(r.field_count for r in model_results) / len(model_results) if model_results else 0
            )

            self.console.print(f"\n📊 {model_name.upper()} Summary:")
            self.console.print(f"   ⏱️  Total time: {model_time:.1f}s")
            self.console.print(
                f"   ✅ Success rate: {successful_extractions}/{len(model_results)} ({success_rate:.1%})"
            )
            self.console.print(f"   📊 Avg fields: {avg_fields:.1f}")

            extraction_results[model_name] = model_results

        return extraction_results

    def _run_analysis(self, extraction_results: Dict[str, List[ExtractionResult]]) -> Dict[str, Any]:
        """Run comprehensive analysis on extraction results."""
        self.console.print("\n📊 COMPREHENSIVE ANALYSIS", style="bold green")

        # Add results to analyzers
        for model_name, results in extraction_results.items():
            self.performance_analyzer.add_results(model_name, results)
            self.field_analyzer.add_results(model_name, results)
            self.comparison_metrics.add_results(model_name, results)

        analysis_results = {}

        # Performance analysis
        self.console.print("📈 Running performance analysis...")
        analysis_results["performance"] = {
            "comparison": self.performance_analyzer.compare_models(),
            "issues": self.performance_analyzer.identify_performance_issues(),
            "summary": self.performance_analyzer.get_performance_summary(),
        }

        # Field analysis
        self.console.print("🏷️  Running field analysis...")
        analysis_results["field"] = {
            "summary": self.field_analyzer.get_field_extraction_summary(),
            "problematic_fields": self.field_analyzer.identify_problematic_fields(),
            "recommendations": self.field_analyzer.generate_field_recommendations(),
        }

        # Comparison metrics
        self.console.print("🔢 Calculating comparison metrics...")
        analysis_results["metrics"] = {
            "comparison": self.comparison_metrics.compare_models(),
            "summary": self.comparison_metrics.get_metrics_summary(),
        }

        self.console.print("✅ Analysis complete")
        return analysis_results

    def _calculate_success_metrics(
        self, extraction_results: Dict[str, List[ExtractionResult]]
    ) -> Dict[str, Any]:
        """Calculate overall success metrics."""
        model_success_rates = {}
        model_execution_times = {}
        total_successful = 0
        total_documents = 0

        for model_name, results in extraction_results.items():
            if results:
                successful = sum(1 for r in results if r.is_successful)
                model_success_rates[model_name] = successful / len(results)
                model_execution_times[model_name] = sum(r.extraction_time for r in results)

                total_successful += successful
                total_documents += len(results)

        overall_success_rate = total_successful / total_documents if total_documents > 0 else 0

        return {
            "model_success_rates": model_success_rates,
            "execution_times": model_execution_times,
            "overall_success_rate": overall_success_rate,
        }

    def _print_completion_summary(self):
        """Print completion summary."""
        if not self.results:
            return

        self.console.print(f"\n{'=' * 70}")
        self.console.print("🏆 COMPARISON COMPLETE", style="bold green")
        self.console.print(f"{'=' * 70}")

        self.console.print(f"⏱️  Total execution time: {self.results.total_execution_time:.1f}s")
        self.console.print(f"📊 Overall success rate: {self.results.overall_success_rate:.1%}")
        self.console.print(f"🤖 Models compared: {len(self.results.models_tested)}")
        self.console.print(f"📁 Images processed: {len(self.results.dataset_info.verified_images)}")

        # Model-specific summaries
        self.console.print("\n📋 Model Performance Summary:")
        for model_name in self.results.models_tested:
            success_rate = self.results.model_success_rates.get(model_name, 0)
            exec_time = self.results.model_execution_times.get(model_name, 0)

            self.console.print(f"   {model_name}: {success_rate:.1%} success, {exec_time:.1f}s total")

        # Best performers
        if self.results.comparison_metrics and self.results.comparison_metrics["summary"]:
            best_performers = self.results.comparison_metrics["summary"].get("best_performers", {})
            if best_performers:
                self.console.print("\n🥇 Best Performers:")
                for metric, model in best_performers.items():
                    self.console.print(f"   {metric}: {model}")

    def get_results(self) -> Optional[ComparisonResults]:
        """Get comparison results.

        Returns:
            ComparisonResults or None if comparison hasn't been run
        """
        return self.results

    def export_dataframe(self) -> Optional[Any]:
        """Export results as pandas DataFrame.

        Returns:
            DataFrame with all extraction results or None
        """
        if not self.results:
            return None

        return self.performance_analyzer.create_performance_dataframe()
