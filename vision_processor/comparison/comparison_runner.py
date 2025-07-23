"""Comparison Runner - Main Orchestrator
====================================

Orchestrates the complete model comparison pipeline including dataset validation,
model loading, extraction, analysis, and reporting generation.
"""

import gc
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image
from rich.console import Console
from rich.progress import track

from ..config.model_registry import get_model_registry
from ..config.simple_config import SimpleConfig
from ..utils.memory_monitor import MemoryMonitor
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
    config: SimpleConfig
    dataset_info: DatasetInfo
    models_tested: List[str]

    # Extraction results
    extraction_results: Dict[str, List[Dict[str, Any]]]

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

    def __init__(self, config: SimpleConfig):
        """Initialize comparison runner.

        Args:
            config: Production configuration
        """
        self.config = config
        self.console = Console()

        # Initialize memory monitor
        self.memory_monitor = MemoryMonitor(self.console)

        # Initialize components
        self.model_registry = get_model_registry()
        self.model_validator = ModelValidator(self.model_registry)
        # Pure model comparison - no complex extraction components needed

        # Results storage
        self.results: Optional[ComparisonResults] = None

    def run_comparison(self) -> ComparisonResults:
        """Run complete model comparison pipeline.

        Returns:
            ComparisonResults with complete analysis
        """
        start_time = time.time()

        self.console.print("🏆 SIMPLE MODEL COMPARISON PIPELINE", style="bold blue")
        # Model details will be shown when they're loaded

        # Take initial memory snapshot
        self.memory_monitor.take_snapshot("Startup")
        self.memory_monitor.print_current_usage("🚀 Initial State")

        # Step 1: Validate configuration and environment
        self._validate_environment()
        self.memory_monitor.take_snapshot("Environment Validated")

        # Step 2: Discover and validate dataset
        dataset_info = self._discover_and_validate_dataset()
        self.memory_monitor.take_snapshot("Dataset Loaded")

        # Step 3: Validate models
        valid_models = self._validate_models()
        self.memory_monitor.take_snapshot("Models Validated")

        if not valid_models:
            raise RuntimeError("No valid models found for comparison")

        # Step 4: Run extraction for each model
        extraction_results = self._run_extractions(
            valid_models, dataset_info.verified_images
        )
        self.memory_monitor.cleanup_and_measure("All Models Processed")

        # Step 5: Perform comprehensive analysis
        analysis_results = self._run_analysis(extraction_results)
        self.memory_monitor.take_snapshot("Analysis Complete")

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

        # Step 7: Print summary with memory analysis
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
                self.console.print(
                    f"✅ CUDA Available: {torch.cuda.get_device_name(0)}"
                )
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
        self.console.print(
            f"✅ Verified: {validation['found']}/{validation['total_expected']} images"
        )

        if validation["found"] == 0:
            raise RuntimeError("No valid images found in dataset")
        elif validation["found"] < validation["total_expected"]:
            self.console.print(
                "⚠️  Some images failed validation but proceeding", style="yellow"
            )

        return dataset_info

    def _validate_models(self) -> List[str]:
        """Validate that requested models can be loaded."""
        self.console.print("\n🤖 MODEL VALIDATION", style="bold yellow")

        valid_models = []

        for model_name in self.config.models_list:
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

            # V100 COMPATIBLE: Skip heavy validation to avoid loading both models simultaneously
            # Just do basic path and registration validation
            try:
                # Check if path exists and is accessible
                model_path_obj = Path(model_path)
                if not model_path_obj.exists():
                    self.console.print(f"❌ Model path does not exist: {model_path}")
                    continue

                # Basic registration check already passed above
                valid_models.append(model_name)
                self.console.print(
                    f"✅ {model_name} validation passed (V100 lightweight mode)"
                )

            except Exception as e:
                self.console.print(f"❌ {model_name} validation failed: {e}")

        self.console.print(f"\n📋 Valid models: {valid_models}")
        return valid_models

    def _run_extractions(
        self, model_names: List[str], image_paths: List[Path]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Run extractions for all models on all images."""
        self.console.print("\n🔥 EXTRACTION PIPELINE", style="bold yellow")

        extraction_results = {}

        for model_name in model_names:
            self.console.print(f"\n{'=' * 50}")
            self.console.print(
                f"🤖 PROCESSING WITH {model_name.upper()}", style="bold cyan"
            )
            self.console.print(f"{'=' * 50}")

            model_results = []
            model_start_time = time.time()

            # Create model instance
            try:
                model_path = getattr(self.config.model_paths, model_name)
                model = self.model_registry.create_model(
                    model_name,
                    self.config.processing,
                    model_path=model_path,
                    config=self.config,  # Pass full config so model can access device_config
                )

                # Explicitly load model (V100-compatible sequential loading)
                model.load_model()
                self.console.print(f"✅ {model_name} loaded successfully")
                self.memory_monitor.print_current_usage(f"📦 {model_name} Loaded")

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
                            image,
                            prompt,
                            max_new_tokens=self.config.processing.max_tokens,
                        )

                        # DEBUG: Print actual response length and content to trace truncation
                        self.console.print(
                            f"DEBUG: Raw response length: {len(response.raw_text)} chars",
                            style="red",
                        )
                        self.console.print(
                            f"DEBUG: Last 50 chars: '{response.raw_text[-50:]}'",
                            style="red",
                        )

                        # DEBUG: Show full raw response to understand format
                        if (
                            len(response.raw_text) < 3000
                        ):  # Only show if reasonable length
                            self.console.print(
                                "DEBUG: Full raw response:", style="yellow"
                            )
                            self.console.print(response.raw_text, style="dim yellow")

                        # Pure model comparison: minimal processing to preserve raw outputs
                        analysis_dict = {
                            "img_name": image_path.name,
                            "raw_response": response.raw_text,  # Pure model output
                            "model_name": model_name,
                            "processing_time": response.processing_time,
                            "response_length": len(response.raw_text),
                            "successful": True,  # Always true for raw comparison
                            "timestamp": datetime.now().isoformat(),
                        }

                        model_results.append(analysis_dict)

                        # Print what the model sees with better formatting - ALL key-value pairs
                        self.console.print(
                            f"\n🔍 {model_name.upper()} sees in {image_path.name}:"
                        )

                        # Clean response to remove markdown artifacts and repetition
                        clean_text = response.raw_text
                        # Remove markdown code blocks
                        if "```" in clean_text:
                            clean_text = clean_text.split("```")[0].strip()
                        # Take only the first occurrence if there's repetition
                        if "**Answer:**" in clean_text:
                            # For numbered lists, take the content after the first **Answer:**
                            parts = clean_text.split("**Answer:**")
                            if len(parts) > 1:
                                clean_text = parts[1].strip()
                                # If there's another **Answer:** (repetition), take only up to it
                                if "**Answer:**" in clean_text:
                                    clean_text = clean_text.split("**Answer:**")[
                                        0
                                    ].strip()

                        # Parse key-value pairs - handle the 20-field format
                        lines = clean_text.strip().split("\n")

                        # Expected keys in order
                        expected_keys = [
                            "DATE",
                            "SUPPLIER",
                            "ABN",
                            "GST",
                            "TOTAL",
                            "SUBTOTAL",
                            "SUPPLIER_WEBSITE",
                            "ITEMS",
                            "QUANTITIES",
                            "PRICES",
                            "RECEIPT_NUMBER",
                            "PAYMENT_METHOD",
                            "DOCUMENT_TYPE",
                            "BUSINESS_ADDRESS",
                            "BUSINESS_PHONE",
                            "PAYER_NAME",
                            "PAYER_ADDRESS",
                            "PAYER_PHONE",
                            "PAYER_EMAIL",
                            "BANK_ACCOUNT_NUMBER",
                        ]

                        # Extract key-value pairs from response
                        extracted_pairs = {}

                        # Try to parse structured format
                        # Check if it's a numbered list format (e.g., "1. DATE: value 2. SUPPLIER: value")
                        full_text = " ".join(lines)
                        numbered_pattern = (
                            r"\d+\.\s*([A-Z_]+):\s*([^0-9]*?)(?=\s*\d+\.\s*[A-Z_]+:|$)"
                        )
                        numbered_matches = re.findall(numbered_pattern, full_text)

                        if numbered_matches:
                            # Numbered list format
                            for key, value in numbered_matches:
                                key = key.strip().upper()
                                value = value.strip()
                                if key in expected_keys:
                                    extracted_pairs[key] = value
                        elif len(lines) == 1 or (
                            len(lines) > 0
                            and all(
                                key in lines[0]
                                for key in ["DATE:", "SUPPLIER:", "ABN:"]
                            )
                        ):
                            # Single line format - parse all key-value pairs from the first line
                            text = lines[0] if lines else ""

                            # Extract each expected key in order
                            for i, key in enumerate(expected_keys):
                                key_pattern = f"{key}:"
                                if key_pattern in text:
                                    start_idx = text.find(key_pattern) + len(
                                        key_pattern
                                    )
                                    # Find the next key or end of string
                                    next_key_idx = len(text)
                                    for next_key in expected_keys[i + 1 :]:
                                        next_pattern = f" {next_key}:"
                                        if next_pattern in text[start_idx:]:
                                            next_key_idx = text.find(
                                                next_pattern, start_idx
                                            )
                                            break
                                    value = text[start_idx:next_key_idx].strip()
                                    extracted_pairs[key] = value
                        else:
                            # Multi-line format - parse line by line
                            for line in lines:
                                if ":" in line:
                                    try:
                                        # Remove leading number if present (e.g., "1. DATE:" becomes "DATE:")
                                        line = re.sub(r"^\d+\.\s*", "", line)
                                        key, value = line.split(":", 1)
                                        key = key.strip().upper()
                                        value = value.strip()
                                        if key in expected_keys:
                                            extracted_pairs[key] = value
                                    except ValueError:
                                        pass

                        # Display all expected fields (show N/A for missing ones)
                        for key in expected_keys:
                            value = extracted_pairs.get(key, "N/A")
                            # Truncate very long values for display
                            if len(value) > 100:
                                value = value[:97] + "..."
                            self.console.print(
                                f"   {key:18}: {value}", style="dim cyan"
                            )

                        # Print progress - raw model comparison
                        status = "✅"  # Always successful for raw comparison
                        response_str = f"{len(response.raw_text)} chars"
                        time_str = f"{response.processing_time:.1f}s"

                        self.console.print(
                            f"   {i + 1:2d}. {image_path.name:<15} {status} {time_str} | {response_str}"
                        )

                        # Cleanup and memory check every few images
                        if (i + 1) % 5 == 0:
                            gc.collect()
                            # Take memory snapshot periodically
                            if (i + 1) % 10 == 0:  # Every 10 images
                                self.memory_monitor.take_snapshot(
                                    f"{model_name} - Image {i + 1}"
                                )

                    except Exception as e:
                        self.console.print(
                            f"   {i + 1:2d}. {image_path.name:<15} ❌ Error: {str(e)[:30]}..."
                        )
                        # Create error result in original dictionary format
                        error_result = {
                            "img_name": image_path.name,
                            "response": f"Error: {e}",
                            "is_structured": False,
                            "extraction_score": 0,
                            "successful": False,
                            "extraction_time": 0.0,  # Fixed field name
                            "doc_type": "UNKNOWN",
                            "image_name": image_path.name,
                            "model_name": model_name,
                            "field_count": 0,
                            "is_successful": False,
                            "confidence_score": 0.0,
                        }
                        model_results.append(error_result)

            except Exception as e:
                self.console.print(f"❌ Failed to load {model_name}: {e}")
                continue

            finally:
                # Explicit model cleanup for V100 compatibility (matching original script)
                if "model" in locals() and model is not None:
                    try:
                        self.console.print(f"\n🧹 Cleaning up {model_name.upper()}")
                        model.unload_model()
                        del model
                    except Exception as cleanup_error:
                        self.console.print(
                            f"⚠️  Cleanup warning for {model_name}: {cleanup_error}"
                        )

                # Aggressive memory cleanup for V100 16GB limit
                self._cleanup_gpu_memory()
                self.memory_monitor.cleanup_and_measure(f"🧹 {model_name} Cleaned Up")

            # Calculate model summary
            model_time = time.time() - model_start_time
            successful_extractions = sum(
                1 for r in model_results if r.get("successful", False)
            )
            success_rate = (
                successful_extractions / len(model_results) if model_results else 0
            )
            avg_fields = (
                sum(r.get("extraction_score", 0) for r in model_results)
                / len(model_results)
                if model_results
                else 0
            )

            self.console.print(f"\n📊 {model_name.upper()} Summary:")
            self.console.print(f"   ⏱️  Total time: {model_time:.1f}s")
            self.console.print(
                f"   ✅ Success rate: {successful_extractions}/{len(model_results)} ({success_rate:.1%})"
            )
            self.console.print(f"   📊 Avg fields: {avg_fields:.1f}")

            extraction_results[model_name] = model_results

        return extraction_results

    def _run_analysis(
        self, extraction_results: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Run simple analysis on raw model outputs for document understanding comparison."""
        self.console.print(
            "\n📊 RAW MODEL DOCUMENT UNDERSTANDING COMPARISON", style="bold green"
        )

        # Simple raw response comparison
        analysis_results = {}

        # Basic statistics for each model
        for model_name, results in extraction_results.items():
            total_responses = len(results)
            avg_response_length = (
                sum(r.get("response_length", 0) for r in results) / total_responses
                if total_responses > 0
                else 0
            )
            avg_processing_time = (
                sum(r.get("processing_time", 0) for r in results) / total_responses
                if total_responses > 0
                else 0
            )

            self.console.print(
                f"📝 {model_name.upper()}: {total_responses} responses, avg {avg_response_length:.0f} chars, {avg_processing_time:.1f}s"
            )

        analysis_results["performance"] = {
            "comparison": "Raw model output comparison - no complex metrics",
            "models": list(extraction_results.keys()),
            "total_images": len(next(iter(extraction_results.values()), [])),
        }

        analysis_results["field"] = {
            "summary": "Raw output analysis - no field extraction performed",
        }

        analysis_results["metrics"] = {
            "summary": {
                "analysis_type": "Pure document understanding comparison",
                "focus": "Raw model responses without regex processing",
            },
        }

        self.console.print("✅ Information Extraction Capability analysis complete")
        return analysis_results

    def _calculate_success_metrics(
        self, extraction_results: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Calculate overall success metrics."""
        model_success_rates = {}
        model_execution_times = {}
        total_successful = 0
        total_documents = 0

        for model_name, results in extraction_results.items():
            if results:
                successful = sum(1 for r in results if r.get("successful", False))
                model_success_rates[model_name] = successful / len(results)
                model_execution_times[model_name] = sum(
                    r.get("extraction_time", 0.0) for r in results
                )

                total_successful += successful
                total_documents += len(results)

        overall_success_rate = (
            total_successful / total_documents if total_documents > 0 else 0
        )

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

        self.console.print(
            f"⏱️  Total execution time: {self.results.total_execution_time:.1f}s"
        )
        self.console.print(
            f"📊 Overall success rate: {self.results.overall_success_rate:.1%}"
        )
        self.console.print(f"🤖 Models compared: {len(self.results.models_tested)}")
        self.console.print(
            f"📁 Images processed: {len(self.results.dataset_info.verified_images)}"
        )

        # Model-specific summaries
        self.console.print("\n📋 Model Performance Summary:")
        for model_name in self.results.models_tested:
            success_rate = self.results.model_success_rates.get(model_name, 0)
            exec_time = self.results.model_execution_times.get(model_name, 0)
            num_images = len(self.results.dataset_info.verified_images)
            avg_time_per_image = exec_time / num_images if num_images > 0 else 0

            self.console.print(
                f"   {model_name}: {success_rate:.1%} success, {exec_time:.1f}s total, {avg_time_per_image:.1f}s per image"
            )

        # Best performers with detailed explanations
        if self.results.comparison_metrics and self.results.comparison_metrics.get(
            "summary"
        ):
            summary = self.results.comparison_metrics["summary"]
            best_performers = summary.get("best_performers", {})
            performance_explanations = summary.get("performance_explanations", {})

            if best_performers:
                self.console.print("\n🥇 Best Performers:")
                for metric, model in best_performers.items():
                    self.console.print(f"   {metric}: {model}")

                # Add detailed explanations if available
                if performance_explanations:
                    self.console.print("\n📊 Detailed Performance Explanations:")
                    self.console.print("-" * 50)

                    for category, details in performance_explanations.items():
                        category_display = category.replace("_", " ").title()
                        self.console.print(f"\n🎯 [bold]{category_display}[/bold]:")
                        self.console.print(
                            f"   [green]Winner[/green]: {details['winner']} ([cyan]Score: {details['score']}[/cyan])"
                        )
                        self.console.print(
                            f"   [yellow]Why[/yellow]: {details['explanation']}"
                        )

            # Processing time comparison
            self.console.print("\n⏱️  Processing Speed Comparison:")
            for model_name in self.results.models_tested:
                exec_time = self.results.model_execution_times.get(model_name, 0)
                num_images = len(self.results.dataset_info.verified_images)
                avg_time_per_image = exec_time / num_images if num_images > 0 else 0
                self.console.print(
                    f"   {model_name}: {avg_time_per_image:.1f}s per image"
                )

        # Memory Usage Analysis
        self._print_memory_summary()

    def _print_memory_summary(self):
        """Print comprehensive memory usage summary."""
        self.console.print(f"\n{'=' * 70}")
        self.console.print("💾 MEMORY USAGE ANALYSIS", style="bold cyan")
        self.console.print(f"{'=' * 70}")

        # Print peak usage
        self.memory_monitor.print_peak_usage()

        # Print timeline
        self.memory_monitor.print_memory_timeline()

        # Print summary statistics
        summary = self.memory_monitor.get_memory_summary()
        if summary:
            self.console.print("\n📈 [bold]Memory Statistics:[/bold]")
            self.console.print(
                f"   📊 Peak Process Memory: {summary.get('peak_process_memory_gb', 0):.1f}GB"
            )
            self.console.print(
                f"   📊 Average Process Memory: {summary.get('avg_process_memory_gb', 0):.1f}GB"
            )

            if "peak_gpu_memory_gb" in summary and summary["gpu_total_memory_gb"]:
                self.console.print(
                    f"   🎮 Peak GPU Memory: {summary['peak_gpu_memory_gb']:.1f}GB / {summary['gpu_total_memory_gb']:.1f}GB"
                )
                self.console.print(
                    f"   🎮 Average GPU Memory: {summary['avg_gpu_memory_gb']:.1f}GB"
                )
                gpu_utilization = (
                    summary["peak_gpu_memory_gb"] / summary["gpu_total_memory_gb"]
                ) * 100
                self.console.print(
                    f"   🎮 Peak GPU Utilization: {gpu_utilization:.1f}%"
                )

            self.console.print(
                f"   ⏱️  Monitoring Duration: {summary.get('monitoring_duration_s', 0):.1f}s"
            )
            self.console.print(
                f"   📸 Total Snapshots: {summary.get('total_snapshots', 0)}"
            )

    def get_results(self) -> Optional[ComparisonResults]:
        """Get comparison results.

        Returns:
            ComparisonResults or None if comparison hasn't been run
        """
        return self.results

    def _cleanup_gpu_memory(self):
        """Aggressive memory cleanup for V100 16GB limit (matching original script)."""
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:
            pass

    def export_dataframe(self) -> Optional[Any]:
        """Export results as simple data structure.

        Returns:
            Raw comparison results or None
        """
        if not self.results:
            return None

        return self.results
