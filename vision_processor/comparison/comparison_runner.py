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
from typing import Any

from PIL import Image
from rich.console import Console
from rich.progress import track

from ..config import ConfigManager
from ..config.model_registry import get_model_registry
from ..exceptions import ImageProcessingError, ModelInferenceError, ModelLoadError
from ..utils.memory_monitor import MemoryMonitor
from .model_validator import ModelValidator


@dataclass
class DatasetInfo:
    """Information about the dataset being processed."""

    dataset_path: Path
    total_images: int
    image_files: list[Path]
    verified_images: list[Path]
    missing_images: list[Path]

    def get_validation_summary(self) -> dict[str, Any]:
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
    config: ConfigManager
    dataset_info: DatasetInfo
    models_tested: list[str]

    # Extraction results
    extraction_results: dict[str, list[dict[str, Any]]]

    # Analysis results
    performance_analysis: Any | None = None
    field_analysis: Any | None = None
    comparison_metrics: Any | None = None

    # Timing information
    total_execution_time: float = 0.0
    model_execution_times: dict[str, float] | None = None

    # Success metrics
    overall_success_rate: float = 0.0
    model_success_rates: dict[str, float] | None = None

    def __post_init__(self):
        if self.model_execution_times is None:
            self.model_execution_times = {}
        if self.model_success_rates is None:
            self.model_success_rates = {}


class ComparisonRunner:
    """Main orchestrator for model comparison pipeline."""

    def __init__(self, config: ConfigManager):
        """Initialize comparison runner.

        Args:
            config: Production configuration
        """
        self.config = config
        self.console = Console()

        # Initialize memory monitor
        self.memory_monitor = MemoryMonitor(self.console)

        # Initialize components
        self.model_registry = get_model_registry(self.config)
        self.model_validator = ModelValidator(self.model_registry)
        # Pure model comparison - no complex extraction components needed

        # Results storage
        self.results: ComparisonResults | None = None

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

        # Configuration is validated during ConfigManager initialization
        self.console.print("✅ Configuration validated (during initialization)")

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

    def _validate_models(self) -> list[str]:
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
        self, model_names: list[str], image_paths: list[Path]
    ) -> dict[str, list[dict[str, Any]]]:
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

                        # Show debug output based on YAML configuration
                        debug_mode = self.config.defaults.debug_mode

                        if debug_mode:
                            # DEBUG: Print actual response length and content
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
                                len(response.raw_text) < 4000
                            ):  # Only show if reasonable length
                                self.console.print(
                                    "DEBUG: Full raw response:", style="yellow"
                                )
                                # Show first 2000 chars if longer
                                if len(response.raw_text) > 2000:
                                    self.console.print(
                                        response.raw_text[:2000] + "...[TRUNCATED]",
                                        style="dim yellow",
                                    )
                                else:
                                    self.console.print(
                                        response.raw_text, style="dim yellow"
                                    )

                        # Pure model comparison: minimal processing to preserve raw outputs
                        analysis_dict = {
                            "img_name": image_path.name,
                            "raw_response": response.raw_text,  # Pure model output
                            "model_name": model_name,
                            "processing_time": response.processing_time,
                            "response_length": len(response.raw_text),
                            "successful": True,  # Always true for raw comparison
                            "timestamp": datetime.now().isoformat(),
                            "extracted_fields": {},  # Initialize empty dict for parsed fields
                        }

                        model_results.append(analysis_dict)

                        # Print what the model sees with better formatting - ALL key-value pairs
                        self.console.print(
                            f"\n🔍 {model_name.upper()} sees in {image_path.name}:"
                        )

                        # Check post-processing configuration with debug
                        try:
                            post_processing_config = (
                                self.config.get_legacy_config_dict().get(
                                    "post_processing", {}
                                )
                            )
                        except Exception as e:
                            self.console.print(
                                f"❌ DEBUG: Error accessing post_processing config: {e}"
                            )
                            self.console.print(
                                f"❌ DEBUG: ConfigManager methods: {[m for m in dir(self.config) if not m.startswith('_')]}"
                            )
                            post_processing_config = {}
                        post_processing_enabled = post_processing_config.get(
                            "enabled", True
                        )
                        smart_mode = post_processing_config.get("smart_mode", False)

                        # Smart detection: check if response needs cleaning
                        needs_processing = (
                            "**" in response.raw_text
                            or "```" in response.raw_text
                            or "Answer:" in response.raw_text
                            or len(response.raw_text) > 2000
                        )  # Very long responses usually need cleaning

                        if not post_processing_enabled or (
                            smart_mode and not needs_processing
                        ):
                            # 🚀 SMART BYPASS: Response is clean, use simple parsing
                            processing_type = (
                                "disabled"
                                if not post_processing_enabled
                                else "clean (smart mode)"
                            )
                            self.console.print(
                                f"📋 Clean model output ({processing_type}):"
                            )

                            # Simple parsing for clean KEY: value format (no complex cleaning needed)
                            extracted_fields = {}
                            raw_text = response.raw_text.strip()

                            # Split on spaces and look for KEY: value patterns
                            parts = raw_text.split()
                            i = 0
                            while i < len(parts) - 1:
                                part = parts[i]
                                if part.endswith(":"):
                                    key = part[:-1]  # Remove the colon
                                    # Collect value(s) until next key or end
                                    value_parts = []
                                    j = i + 1
                                    while j < len(parts) and not parts[j].endswith(":"):
                                        value_parts.append(parts[j])
                                        j += 1
                                    value = (
                                        " ".join(value_parts) if value_parts else "N/A"
                                    )
                                    extracted_fields[key] = value
                                    i = j
                                else:
                                    i += 1

                            # Store parsed fields
                            analysis_dict["extracted_fields"] = extracted_fields

                            # Display parsed fields (matching full post-processing format)
                            if extracted_fields:
                                sorted_pairs = sorted(extracted_fields.items())
                                for key, value in sorted_pairs:
                                    # Clean up values - remove trailing asterisks and whitespace
                                    value = value.rstrip("*").strip()
                                    self.console.print(
                                        f"   {key:20}: {value}", style="dim green"
                                    )
                            else:
                                self.console.print(
                                    "   No key-value pairs found", style="dim red"
                                )

                            # Simple status display with key count
                            status = "✅"
                            key_count = len(extracted_fields)
                            response_str = f"{len(response.raw_text)} chars"
                            time_str = f"{response.processing_time:.1f}s"
                            keys_str = f"{key_count} keys"
                            self.console.print(
                                f"   {i + 1:2d}. {image_path.name:<15} {status} {time_str} | {keys_str} | {response_str}"
                            )
                            continue  # Skip all the post-processing below

                        # 🔧 FULL POST-PROCESSING: Handle messy responses (markdown, explanations, etc.)
                        processing_type = (
                            "full (messy response detected)"
                            if smart_mode
                            else "full (always enabled)"
                        )
                        self.console.print(
                            f"🔧 Processing response ({processing_type}):"
                        )

                        # Clean response to remove markdown artifacts and repetition
                        clean_text = response.raw_text

                        # Apply cleaning based on content patterns
                        match (("```" in clean_text), ("**Answer:**" in clean_text)):
                            case (True, _):
                                # Remove markdown code blocks
                                clean_text = clean_text.split("```")[0].strip()
                            case (_, True):
                                # Handle Answer: sections
                                parts = clean_text.split("**Answer:**")
                                if len(parts) > 1:
                                    clean_text = parts[1].strip()
                                    # Remove subsequent Answer: sections (repetition)
                                    if "**Answer:**" in clean_text:
                                        clean_text = clean_text.split("**Answer:**")[
                                            0
                                        ].strip()
                            case _:
                                pass

                        # Remove markdown formatting
                        clean_text = clean_text.replace("**", "")  # Remove bold markers
                        clean_text = re.sub(
                            r"\*\s*", "", clean_text
                        )  # Remove bullet points

                        # Stop at repetition markers and explanation sections
                        end_markers = [
                            "**END OF",
                            "END OF DOCUMENT",
                            "END OF OUTPUT",
                            "END OF FILE",
                            "Note:",
                            "NOTE:",
                            "Answer:",
                            "Final Answer:",
                            "The output is exactly",
                            "The above output",
                            "Please ensure",
                            "I apologize",
                            "I hope it is correct",
                        ]
                        for marker in end_markers:
                            if marker in clean_text:
                                clean_text = clean_text.split(marker)[0].strip()
                                break

                        # Remove repetitive N/A values (Llama bug where it repeats N/A hundreds of times)
                        # Look for pattern like "... N/A N/A N/A N/A N/A ..." and truncate
                        na_repetition_pattern = (
                            r"(\b(N/A\s*){10,})"  # 10+ consecutive N/A values
                        )
                        if re.search(na_repetition_pattern, clean_text):
                            # Find where the repetition starts and truncate there
                            match = re.search(na_repetition_pattern, clean_text)
                            if match:
                                clean_text = clean_text[: match.start()].strip()

                        # Parse key-value pairs - handle generic format
                        lines = clean_text.strip().split("\n")

                        # DEBUG: Show parsing info (only if debug mode enabled)
                        if debug_mode:
                            self.console.print(
                                f"DEBUG: Parsing {len(lines)} lines", style="yellow"
                            )
                            if len(lines) > 0:
                                self.console.print(
                                    f"DEBUG: First line: '{lines[0][:100]}...'",
                                    style="yellow",
                                )

                            # Show what clean_text looks like before parsing
                            self.console.print(
                                "DEBUG: Clean text being passed to parser:",
                                style="cyan",
                            )
                            self.console.print(
                                f"'{clean_text[:200]}{'...' if len(clean_text) > 200 else ''}'",
                                style="dim cyan",
                            )

                        # Extract key-value pairs using robust AWK-style parsing
                        extracted_pairs = self._extract_keyvalue_pairs_robust(
                            clean_text
                        )

                        if debug_mode:
                            self.console.print(
                                f"DEBUG: Extracted {len(extracted_pairs)} pairs using robust parser",
                                style="yellow",
                            )
                            if len(extracted_pairs) == 0:
                                self.console.print(
                                    "DEBUG: No pairs extracted. First 500 chars of clean_text:",
                                    style="red",
                                )
                                self.console.print(clean_text[:500], style="dim red")

                        # Display all extracted fields
                        if extracted_pairs:
                            # Sort by key for consistent display
                            sorted_pairs = sorted(extracted_pairs.items())
                            for key, value in sorted_pairs:
                                # Clean up values - remove trailing asterisks and whitespace
                                value = value.rstrip("*").strip()

                                # Handle OCR fallback in TRANSACTIONS field
                                if key == "TRANSACTIONS" and "<OCR/>" in value:
                                    # Extract just the marker for display
                                    value = "[OCR output returned - model provided full document instead of transaction list]"
                                    self.console.print(
                                        f"   {key:20}: {value}", style="dim yellow"
                                    )
                                else:
                                    # NO TRUNCATION - show complete extracted data for fair comparison
                                    self.console.print(
                                        f"   {key:20}: {value}", style="dim cyan"
                                    )

                            # Store the extracted fields in the analysis_dict
                            analysis_dict["extracted_fields"] = extracted_pairs
                        else:
                            self.console.print(
                                "   No structured data extracted", style="dim red"
                            )

                        # Print progress - raw model comparison with key count
                        status = "✅"  # Always successful for raw comparison
                        key_count = len(extracted_pairs) if extracted_pairs else 0
                        response_str = f"{len(response.raw_text)} chars"
                        time_str = f"{response.processing_time:.1f}s"
                        keys_str = f"{key_count} keys"

                        self.console.print(
                            f"   {i + 1:2d}. {image_path.name:<15} {status} {time_str} | {keys_str} | {response_str}"
                        )

                        # Cleanup and memory check every few images
                        if (i + 1) % 5 == 0:
                            gc.collect()
                            # Take memory snapshot periodically
                            if (i + 1) % 10 == 0:  # Every 10 images
                                self.memory_monitor.take_snapshot(
                                    f"{model_name} - Image {i + 1}"
                                )

                    except (ImageProcessingError, ModelInferenceError) as e:
                        self.console.print(
                            f"   {i + 1:2d}. {image_path.name:<15} ❌ {type(e).__name__}: {str(e)[:30]}..."
                        )
                    except Exception as e:
                        self.console.print(
                            f"   {i + 1:2d}. {image_path.name:<15} ❌ Unexpected error: {str(e)[:30]}..."
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

            except ModelLoadError as e:
                self.console.print(f"❌ Failed to load {model_name}: {e.message}")
                continue
            except Exception as e:
                self.console.print(f"❌ Unexpected error loading {model_name}: {e}")
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
                sum(float(r.get("extraction_score", 0) or 0) for r in model_results)
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
        self, extraction_results: dict[str, list[dict[str, Any]]]
    ) -> dict[str, Any]:
        """Run analysis comparing extracted fields between models."""
        self.console.print("\n📊 MODEL FIELD EXTRACTION COMPARISON", style="bold green")

        analysis_results = {}

        # Get expected fields from config (parses from prompt if not explicitly defined)
        expected_fields = self.config.get_expected_fields()
        if not expected_fields:
            # Fallback to hardcoded list if parsing fails
            expected_fields = [
                "DOCUMENT_TYPE",
                "SUPPLIER",
                "ABN",
                "PAYER_NAME",
                "PAYER_ADDRESS",
                "PAYER_PHONE",
                "PAYER_EMAIL",
                "INVOICE_DATE",
                "DUE_DATE",
                "GST",
                "TOTAL",
                "SUBTOTAL",
                "SUPPLIER_WEBSITE",
                "ITEMS",
                "QUANTITIES",
                "PRICES",
                "BUSINESS_ADDRESS",
                "BUSINESS_PHONE",
                "BANK_NAME",
                "BSB_NUMBER",
                "BANK_ACCOUNT_NUMBER",
                "ACCOUNT_HOLDER",
                "STATEMENT_PERIOD",
                "OPENING_BALANCE",
                "CLOSING_BALANCE",
                "TRANSACTIONS",
            ]

        # Analyze field extraction by model
        model_field_stats: dict[str, dict[str, Any]] = {}

        for model_name, results in extraction_results.items():
            total_fields_extracted = 0
            field_extraction_counts = {field: 0 for field in expected_fields}
            fields_with_values = {field: 0 for field in expected_fields}

            for result in results:
                extracted = result.get("extracted_fields", {})
                total_fields_extracted += len(extracted)

                for field in expected_fields:
                    if field in extracted:
                        field_extraction_counts[field] += 1
                        if extracted[field] not in ["N/A", "n/a", ""]:
                            fields_with_values[field] += 1

            avg_fields_per_doc = total_fields_extracted / len(results) if results else 0

            model_field_stats[model_name] = {
                "total_docs": len(results),
                "avg_fields_extracted": avg_fields_per_doc,
                "field_extraction_rates": {
                    field: count / len(results) if results else 0
                    for field, count in field_extraction_counts.items()
                },
                "field_value_rates": {
                    field: count / len(results) if results else 0
                    for field, count in fields_with_values.items()
                },
            }

            # Print summary with emphasis on the crucial metric
            self.console.print(f"\n📝 {model_name.upper()} Field Extraction:")
            self.console.print(
                f"   [bold cyan]Average fields extracted: {avg_fields_per_doc:.1f}/26 ({(avg_fields_per_doc / 26) * 100:.1f}%)[/bold cyan]"
            )

            # Show completion quality using YAML config thresholds
            quality_thresholds = self.config.get_legacy_config_dict().get(
                "quality_thresholds",
                {"excellent": 24, "good": 20, "fair": 15, "poor": 0},
            )

            if avg_fields_per_doc >= quality_thresholds["excellent"]:
                quality = "Excellent - Near complete extraction"
                style = "bold green"
            elif avg_fields_per_doc >= quality_thresholds["good"]:
                quality = "Good - Most fields extracted"
                style = "green"
            elif avg_fields_per_doc >= quality_thresholds["fair"]:
                quality = "Fair - Moderate extraction"
                style = "yellow"
            else:
                quality = "Poor - Many fields missing"
                style = "red"
            self.console.print(f"   Quality: [{style}]{quality}[/{style}]")

            # Show top extracted fields
            top_fields = sorted(
                fields_with_values.items(), key=lambda x: x[1], reverse=True
            )[:5]
            self.console.print("   Top extracted fields with values:")
            for field, count in top_fields:
                rate = (count / len(results)) * 100 if results else 0
                self.console.print(f"     - {field}: {rate:.1f}%")

        # Compare models on ALL expected fields (why limit to just 5?)
        self.console.print("\n🔍 Complete Field-by-Field Comparison:")
        self.console.print(
            "   (Showing all fields with >0% extraction by at least one model)"
        )

        # Find fields that at least one model extracted with some success
        active_fields = []
        for field in expected_fields:
            has_extractions = any(
                stats["field_value_rates"][field] > 0
                for stats in model_field_stats.values()
            )
            if has_extractions:
                active_fields.append(field)

        for field in active_fields:
            self.console.print(f"\n   {field}:")
            for model_name, stats in model_field_stats.items():
                rate = stats["field_value_rates"][field] * 100
                if rate > 0:
                    self.console.print(
                        f"     {model_name}: {rate:.1f}% extraction rate"
                    )
                else:
                    self.console.print(
                        f"     {model_name}: [dim]0% extraction rate[/dim]"
                    )

        # Calculate and display performance metrics
        self.console.print("\n⏱️  Processing Speed Comparison:")
        processing_times = {}
        for model_name, results in extraction_results.items():
            total_responses = len(results)
            avg_processing_time = (
                sum(r.get("processing_time", 0) for r in results) / total_responses
                if total_responses > 0
                else 0
            )
            processing_times[model_name] = avg_processing_time

            # Determine speed rating using YAML config thresholds
            speed_thresholds = self.config.get_legacy_config_dict().get(
                "speed_thresholds", {"very_fast": 2.0, "fast": 5.0, "moderate": 10.0}
            )

            if avg_processing_time < speed_thresholds["very_fast"]:
                speed_rating = "Very Fast"
                speed_style = "bold green"
            elif avg_processing_time < speed_thresholds["fast"]:
                speed_rating = "Fast"
                speed_style = "green"
            elif avg_processing_time < speed_thresholds["moderate"]:
                speed_rating = "Moderate"
                speed_style = "yellow"
            else:
                speed_rating = "Slow"
                speed_style = "red"

            self.console.print(
                f"   {model_name.upper()}: [{speed_style}]{avg_processing_time:.1f}s per document ({speed_rating})[/{speed_style}]"
            )

        # Show speed advantage
        if len(processing_times) == 2:
            models = list(processing_times.items())
            faster = min(models, key=lambda x: x[1])
            slower = max(models, key=lambda x: x[1])
            if slower[1] > 0:
                speed_advantage = ((slower[1] - faster[1]) / slower[1]) * 100
                self.console.print(
                    f"   [cyan]{faster[0].upper()} is {speed_advantage:.1f}% faster[/cyan]"
                )

        analysis_results["performance"] = {
            "comparison": "Field-by-field extraction comparison",
            "models": list(extraction_results.keys()),
            "total_images": len(next(iter(extraction_results.values()), [])),
        }

        analysis_results["field"] = {
            "summary": "Detailed field extraction analysis",
            "model_stats": model_field_stats,
            "expected_fields": expected_fields,
        }

        analysis_results["metrics"] = {
            "summary": {
                "analysis_type": "Structured field extraction comparison",
                "focus": "Comparing extraction rates for 26 defined fields",
            },
        }

        # Determine the winner based on both crucial metrics
        self.console.print("\n🏆 OVERALL COMPARISON RESULTS:")

        # Field extraction winner
        model_scores = {
            model: stats["avg_fields_extracted"]
            for model, stats in model_field_stats.items()
        }

        field_winner = max(model_scores.items(), key=lambda x: x[1])
        field_loser = min(model_scores.items(), key=lambda x: x[1])

        self.console.print("\n   📊 Field Extraction Quality:")
        if field_winner[1] - field_loser[1] < 1.0:  # Less than 1 field difference
            self.console.print(
                f"      [yellow]Too close to call! Both models extracted ~{field_winner[1]:.1f} fields on average[/yellow]"
            )
        else:
            self.console.print(
                f"      [bold green]{field_winner[0].upper()} wins with {field_winner[1]:.1f}/26 fields[/bold green]"
            )
            self.console.print(
                f"      [dim]{field_loser[0].upper()} extracted {field_loser[1]:.1f}/26 fields[/dim]"
            )
            advantage = (
                ((field_winner[1] - field_loser[1]) / field_loser[1]) * 100
                if field_loser[1] > 0
                else 0
            )
            self.console.print(
                f"      [cyan]Advantage: {advantage:.1f}% more fields extracted[/cyan]"
            )

        # Speed winner
        self.console.print("\n   ⚡ Processing Speed:")
        speed_winner = None
        if len(processing_times) >= 2:
            speed_winner = min(processing_times.items(), key=lambda x: x[1])
            speed_loser = max(processing_times.items(), key=lambda x: x[1])
            self.console.print(
                f"      [bold green]{speed_winner[0].upper()} wins at {speed_winner[1]:.1f}s per document[/bold green]"
            )
            self.console.print(
                f"      [dim]{speed_loser[0].upper()} takes {speed_loser[1]:.1f}s per document[/dim]"
            )

        self.console.print("\n✅ Field extraction comparison complete")
        return analysis_results

    def _calculate_success_metrics(
        self, extraction_results: dict[str, list[dict[str, Any]]]
    ) -> dict[str, Any]:
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
                    r.get("processing_time", 0.0) for r in results
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
            success_rate = (
                self.results.model_success_rates.get(model_name, 0)
                if self.results.model_success_rates
                else 0
            )
            exec_time = (
                self.results.model_execution_times.get(model_name, 0)
                if self.results.model_execution_times
                else 0
            )
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
                exec_time = (
                    self.results.model_execution_times.get(model_name, 0)
                    if self.results.model_execution_times
                    else 0
                )
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

    def get_results(self) -> ComparisonResults | None:
        """Get comparison results.

        Returns:
            ComparisonResults or None if comparison hasn't been run
        """
        return self.results

    def _extract_keyvalue_pairs_robust(self, text: str) -> dict[str, str]:
        """Extract key-value pairs using robust AWK-style parsing from backup."""

        if not text or not text.strip():
            return {}

        extracted_pairs = {}

        # First, try to split on newlines (multi-line format)
        lines = text.strip().split("\n")

        # Debug: show what we're parsing (only if debug mode enabled)
        # NOTE: Debug output controlled by YAML config defaults.debug_mode
        if hasattr(self, "config") and self.config.defaults.debug_mode:
            print(
                f"DEBUG _extract_keyvalue_pairs_robust: Processing {len(lines)} lines"
            )
            if len(lines) > 0:
                print(f"DEBUG: First 3 lines: {lines[:3]}")

        # If we have only 1 line and it contains multiple field patterns,
        # it's likely a single-line response that needs field-boundary splitting
        if len(lines) == 1 and len(lines[0]) > 200:  # Likely single-line response
            single_line = lines[0]
            if hasattr(self, "config") and self.config.defaults.debug_mode:
                print(
                    f"DEBUG: Detected single-line response ({len(single_line)} chars), splitting on field boundaries"
                )

            # Get expected fields from the prompt/config
            expected_fields = [
                "DOCUMENT_TYPE",
                "SUPPLIER",
                "ABN",
                "PAYER_NAME",
                "PAYER_ADDRESS",
                "PAYER_PHONE",
                "PAYER_EMAIL",
                "INVOICE_DATE",
                "DUE_DATE",
                "GST",
                "TOTAL",
                "SUBTOTAL",
                "SUPPLIER_WEBSITE",
                "ITEMS",
                "QUANTITIES",
                "PRICES",
                "BUSINESS_ADDRESS",
                "BUSINESS_PHONE",
                "BANK_NAME",
                "BSB_NUMBER",
                "BANK_ACCOUNT_NUMBER",
                "ACCOUNT_HOLDER",
                "STATEMENT_PERIOD",
                "OPENING_BALANCE",
                "CLOSING_BALANCE",
                "TRANSACTIONS",
            ]

            # Much simpler approach: just insert newlines before each field
            reconstructed_text = single_line

            # Insert newline before each field (except the first one)
            for field in expected_fields:
                # Look for " FIELD:" pattern and replace with "\nFIELD:"
                pattern = f" {field}:"
                reconstructed_text = reconstructed_text.replace(pattern, f"\n{field}:")

            # Split into lines
            lines = [
                line.strip() for line in reconstructed_text.split("\n") if line.strip()
            ]

            if hasattr(self, "config") and self.config.defaults.debug_mode:
                print(
                    f"DEBUG: Reconstructed {len(lines)} field lines from single-line response"
                )
                print(
                    f"DEBUG: First few reconstructed lines: {lines[:3] if lines else 'None'}"
                )

        # Now process lines normally (whether original multi-line or reconstructed)
        for line in lines:
            line = line.strip()

            # Skip empty lines, comments, and instruction lines
            if (
                not line
                or line.startswith("#")
                or line.startswith("NOTE:")
                or line.startswith("Note:")
            ):
                continue

            # Skip instruction/format lines and answer sections
            if any(
                skip in line.lower()
                for skip in [
                    "example:",
                    "format:",
                    "output:",
                    "instruction:",
                    "required output format",
                    "answer:",
                    "final answer:",
                ]
            ):
                continue

            # Look for KEY: VALUE pattern
            if ":" in line and len(line.split(":", 1)) == 2:
                # Remove numbered list prefixes
                line = re.sub(r"^\d+\.\s*", "", line)

                try:
                    key, value = line.split(":", 1)
                    key = key.strip().upper()
                    value = value.strip()

                    # Remove markdown formatting from key
                    key = key.replace("**", "").replace("*", "")

                    # Simple value cleaning (no complex truncation needed after line reconstruction)

                    # Clean value
                    value = value.replace("**", "").replace("*", "").strip()
                    value = value.strip('"').strip("'").strip()

                    # Validate key format - must be alphabetic (with underscores/spaces)
                    if (
                        key
                        and value
                        and key.replace("_", "").replace(" ", "").isalpha()
                        and not key.isdigit()
                        and len(key) > 1
                    ):  # At least 2 characters
                        # Normalize key format
                        key = key.replace(" ", "_")

                        # Keep ALL values, including N/A (for fair comparison)
                        extracted_pairs[key] = value

                        # Debug
                        if hasattr(self, "config") and self.config.defaults.debug_mode:
                            if len(extracted_pairs) <= 5:  # Show first few extractions
                                print(f"DEBUG: Extracted {key}: {value[:50]}...")

                except ValueError:
                    continue

        if hasattr(self, "config") and self.config.defaults.debug_mode:
            print(f"DEBUG: Final extraction count: {len(extracted_pairs)} pairs")
        return extracted_pairs

    def _count_keyvalue_pairs(self, text: str) -> int:
        """Count the number of key-value pairs in processed text (from backup)."""
        if not text or not text.strip():
            return 0

        lines = text.strip().split("\n")
        count = 0
        for line in lines:
            line = line.strip()
            if ":" in line and len(line.split(":", 1)) == 2:
                key, value = line.split(":", 1)
                if key.strip() and value.strip():
                    count += 1

        return count

    def _has_valid_keyvalue_pairs(self, text: str) -> bool:
        """Check if text contains valid key-value pairs after conversion (from backup)."""
        if not text or not text.strip():
            return False

        lines = text.split("\n")
        valid_pairs = 0

        for line in lines:
            line = line.strip()
            if line and ":" in line and len(line.split(":", 1)) == 2:
                field, value = line.split(":", 1)
                field = field.strip()
                value = value.strip()
                # Valid if field is alphabetic and value is not empty
                if field and value and field.replace("_", "").isalpha():
                    valid_pairs += 1

        # Consider successful if we have at least 3 valid key-value pairs
        return valid_pairs >= 3

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

    def export_dataframe(self) -> Any | None:
        """Export results as pandas DataFrame for downstream processing.

        Returns:
            pandas DataFrame with extraction results or None
        """
        if not self.results:
            return None

        try:
            import pandas as pd
        except ImportError:
            self.console.print("⚠️  pandas not available, returning raw results")
            return self.results

        # Flatten extraction results into DataFrame format
        rows = []

        for model_name, model_results in self.results.extraction_results.items():
            for result in model_results:
                # Base row data
                row = {
                    "model_name": model_name,
                    "image_name": result.get("img_name", ""),
                    "processing_time": result.get("processing_time", 0.0),
                    "response_length": result.get("response_length", 0),
                    "successful": result.get("successful", False),
                    "timestamp": result.get("timestamp", ""),
                    "field_count": len(result.get("extracted_fields", {})),
                }

                # Add extracted fields as columns
                extracted_fields = result.get("extracted_fields", {})
                expected_fields = self.config.get_expected_fields()
                for field_name in expected_fields:
                    row[f"field_{field_name}"] = extracted_fields.get(field_name, "N/A")

                rows.append(row)

        results_dataframe = pd.DataFrame(rows)

        # Add summary columns
        if not results_dataframe.empty:
            # Calculate fields with actual values (not N/A)
            field_columns = [
                col for col in results_dataframe.columns if col.startswith("field_")
            ]
            results_dataframe["fields_with_values"] = results_dataframe[
                field_columns
            ].apply(
                lambda row: sum(1 for val in row if val not in ["N/A", "n/a", ""]),
                axis=1,
            )

            # Add performance rating based on field extraction
            quality_thresholds = self.config.get_legacy_config_dict().get(
                "quality_thresholds", {"excellent": 12, "good": 8, "fair": 5, "poor": 0}
            )

            def get_quality_rating(field_count):
                if field_count >= quality_thresholds["excellent"]:
                    return "excellent"
                elif field_count >= quality_thresholds["good"]:
                    return "good"
                elif field_count >= quality_thresholds["fair"]:
                    return "fair"
                else:
                    return "poor"

            results_dataframe["quality_rating"] = results_dataframe[
                "fields_with_values"
            ].apply(get_quality_rating)

        return results_dataframe
