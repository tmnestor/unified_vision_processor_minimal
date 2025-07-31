"""Evaluation Module for Vision Model Key-Value Extraction Performance.

This module provides comprehensive evaluation tools for comparing vision model
performance on document extraction tasks.
"""

import csv
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Legacy SimpleConfig import removed - now using ConfigManager
from ..extraction.extraction_manager import SimpleExtractionManager


class ExtractionEvaluator:
    """Evaluates key-value extraction performance against ground truth."""

    def __init__(
        self,
        ground_truth_csv: str,
        images_dir: str,
        output_dir: str,
        config_manager=None,
    ):
        """Initialize evaluator.

        Args:
            ground_truth_csv: Path to CSV with ground truth data
            images_dir: Directory containing test images
            output_dir: Directory to save evaluation results
            config_manager: Optional ConfigManager for getting model display names
        """
        self.ground_truth_csv = ground_truth_csv
        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir)
        self.console = Console()
        self.config_manager = config_manager

        # Create output directory
        self.output_dir.mkdir(exist_ok=True)

        # Load ground truth
        self.ground_truth = self._load_ground_truth()

        # Dynamic extraction fields from model_comparison.yaml
        self.extraction_fields = self._load_extraction_fields()

    def _get_model_display_name(self, model_type: str) -> str:
        """Get the display name for a model type.

        Args:
            model_type: Model type key (e.g., 'llama', 'internvl')

        Returns:
            Display name for the model (e.g., 'Llama-3.2-11B-Vision-Instruct', 'InternVL3-8B')
        """
        if self.config_manager:
            return self.config_manager.get_model_display_name(model_type)
        else:
            # Fallback to uppercase if no config manager available
            return model_type.upper()

    def _load_ground_truth(self) -> Dict[str, Dict[str, Any]]:
        """Load ground truth data from CSV."""
        ground_truth = {}

        with Path(self.ground_truth_csv).open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_file = row.pop("image_file")
                # Convert numeric values
                for key, value in row.items():
                    if key in ["GST", "TOTAL", "SUBTOTAL"] and value:
                        try:
                            row[key] = float(value)
                        except ValueError:
                            pass
                ground_truth[image_file] = row

        self.console.print(f"✅ Loaded ground truth for {len(ground_truth)} images")
        return ground_truth

    def _load_extraction_fields(self) -> List[str]:
        """Load extraction fields dynamically from model_comparison.yaml."""
        try:
            # Look for model_comparison.yaml in current directory and parent directories
            config_path = Path("model_comparison.yaml")
            if not config_path.exists():
                config_path = Path("..") / "model_comparison.yaml"
            if not config_path.exists():
                config_path = Path("../../model_comparison.yaml")

            if not config_path.exists():
                self.console.print(
                    "⚠️ model_comparison.yaml not found, using fallback fields",
                    style="yellow",
                )
                # Fallback to basic fields if YAML not found
                return [
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
                    "DESCRIPTIONS",
                ]

            # Use unified config loading - no more raw YAML!
            from ..config import ConfigManager

            config_manager = ConfigManager.get_global_instance(str(config_path))

            # Get fields from single source of truth (extraction_prompt)
            unique_fields = config_manager.get_expected_fields()

            self.console.print(
                f"✅ Loaded {len(unique_fields)} extraction fields from model_comparison.yaml"
            )
            return unique_fields

        except Exception as e:
            self.console.print(
                f"⚠️ Error loading extraction fields: {e}", style="yellow"
            )
            # Return fallback fields
            return [
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
                "DESCRIPTIONS",
            ]

    def _calculate_field_accuracy(
        self, extracted: str, ground_truth: str, field_type: str
    ) -> float:
        """Calculate accuracy for a specific field."""
        if not ground_truth:  # No ground truth available
            return 1.0 if not extracted else 0.0

        if not extracted:  # Nothing extracted but ground truth exists
            return 0.0

        # Normalize for comparison
        extracted = str(extracted).strip().lower()
        ground_truth = str(ground_truth).strip().lower()

        if field_type in ["GST", "TOTAL", "SUBTOTAL"]:
            # Numeric comparison with tolerance
            try:
                ext_num = float(re.sub(r"[^\d.]", "", extracted))
                gt_num = float(re.sub(r"[^\d.]", "", ground_truth))
                return 1.0 if abs(ext_num - gt_num) < 0.01 else 0.0
            except ValueError:
                return 1.0 if extracted == ground_truth else 0.0

        elif field_type in ["ITEMS", "QUANTITIES", "PRICES"]:
            # List comparison (pipe-separated)
            ext_items = [item.strip() for item in extracted.split("|")]
            gt_items = [item.strip() for item in ground_truth.split("|")]

            if len(ext_items) != len(gt_items):
                return 0.0

            matches = sum(
                1
                for e, g in zip(ext_items, gt_items, strict=True)
                if e.lower() == g.lower()
            )
            return matches / len(gt_items) if gt_items else 0.0

        elif field_type == "DATE":
            # Date comparison (flexible format)
            # Extract date components
            ext_date = re.sub(r"[^\d-/]", "", extracted)
            gt_date = re.sub(r"[^\d-/]", "", ground_truth)
            return 1.0 if ext_date == gt_date else 0.0

        else:
            # String comparison with fuzzy matching
            if extracted == ground_truth:
                return 1.0
            elif extracted in ground_truth or ground_truth in extracted:
                return 0.8
            else:
                return 0.0

    def _evaluate_single_image(
        self, manager: SimpleExtractionManager, image_file: str
    ) -> Dict[str, Any]:
        """Evaluate extraction for a single image using the working extraction manager."""
        image_path = self.images_dir / image_file

        if not image_path.exists():
            return {"error": f"Image not found: {image_path}"}

        # Get ground truth
        gt_data = self.ground_truth.get(image_file, {})

        try:
            # Extract using the working SimpleExtractionManager
            result = manager.process_document(str(image_path))
            self.console.print("🔍 DEBUG: Extraction completed, processing result...")

            extracted_data = result.extracted_fields
            self.console.print(
                f"🔍 DEBUG: Extracted {len(extracted_data)} fields: {list(extracted_data.keys())}"
            )

            # Calculate field-wise accuracy
            field_accuracies = {}
            for field in self.extraction_fields:
                gt_value = gt_data.get(field, "")
                ext_value = extracted_data.get(field, "")
                try:
                    field_accuracies[field] = self._calculate_field_accuracy(
                        ext_value, gt_value, field
                    )
                except Exception as field_error:
                    self.console.print(
                        f"🔍 DEBUG: Error calculating accuracy for {field}: {field_error}"
                    )
                    field_accuracies[field] = 0.0

            # Overall accuracy
            total_fields = len([f for f in self.extraction_fields if gt_data.get(f)])
            overall_accuracy = (
                sum(field_accuracies.values()) / len(field_accuracies)
                if field_accuracies
                else 0.0
            )

            self.console.print(
                f"🔍 DEBUG: Calculated overall accuracy: {overall_accuracy:.1%}"
            )

            result_dict = {
                "image_file": image_file,
                "processing_time": result.processing_time,
                "response_length": 0,  # Not directly available from result
                "extracted_fields": len(extracted_data),
                "total_gt_fields": total_fields,
                "field_accuracies": field_accuracies,
                "overall_accuracy": overall_accuracy,
                "raw_response": str(
                    extracted_data
                ),  # Show the extracted fields instead
                "extracted_data": extracted_data,
                "ground_truth": gt_data,
                "confidence": result.model_confidence,
            }

            self.console.print(
                "🔍 DEBUG: Successfully created result dict, returning..."
            )
            return result_dict

        except Exception as e:
            self.console.print(f"🔍 DEBUG: Exception in _evaluate_single_image: {e}")
            import traceback

            self.console.print(f"🔍 DEBUG: Traceback: {traceback.format_exc()}")
            return {"error": str(e), "image_file": image_file}

    def evaluate_model(
        self, model_type: str, test_images: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Evaluate a specific model on all test images using working extraction manager."""
        self.console.print(
            f"\n🔬 Evaluating {self._get_model_display_name(model_type)} model..."
        )

        # Setup extraction manager using ConfigManager
        from ..config import ConfigManager

        config = ConfigManager()
        config.set_model_type(model_type)
        manager = SimpleExtractionManager(config)

        # Get test images
        if test_images is None:
            test_images = list(self.ground_truth.keys())

        results = []
        total_time = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task(
                f"Processing {len(test_images)} images...", total=None
            )

            for image_file in test_images:
                progress.update(task, description=f"Processing {image_file}...")

                result = self._evaluate_single_image(manager, image_file)
                results.append(result)

                if "processing_time" in result:
                    total_time += result["processing_time"]

        # Calculate aggregate metrics
        successful_results = [r for r in results if "error" not in r]
        failed_results = [r for r in results if "error" in r]

        if successful_results:
            avg_accuracy = sum(r["overall_accuracy"] for r in successful_results) / len(
                successful_results
            )
            avg_processing_time = total_time / len(successful_results)
            avg_fields_extracted = sum(
                r["extracted_fields"] for r in successful_results
            ) / len(successful_results)

            # Field-wise accuracy
            field_wise_accuracy = {}
            for field in self.extraction_fields:
                field_scores = [
                    r["field_accuracies"].get(field, 0.0) for r in successful_results
                ]
                field_wise_accuracy[field] = sum(field_scores) / len(field_scores)

        else:
            avg_accuracy = 0.0
            avg_processing_time = 0.0
            avg_fields_extracted = 0.0
            field_wise_accuracy = {}

        return {
            "model_type": model_type,
            "total_images": len(test_images),
            "successful": len(successful_results),
            "failed": len(failed_results),
            "success_rate": len(successful_results) / len(test_images),
            "avg_accuracy": avg_accuracy,
            "avg_processing_time": avg_processing_time,
            "avg_fields_extracted": avg_fields_extracted,
            "field_wise_accuracy": field_wise_accuracy,
            "total_processing_time": total_time,
            "detailed_results": results,
        }

    def compare_models(
        self,
        models: Optional[List[str]] = None,
        test_images: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Compare multiple models side by side using working extraction managers."""
        if models is None:
            models = ["internvl3", "llama32_vision"]
        self.console.print("🏁 Starting Model Comparison Evaluation")
        self.console.print("📝 Using model-specific prompts from model_comparison.yaml")

        comparison_results = {}

        for model_type in models:
            try:
                results = self.evaluate_model(model_type, test_images)
                comparison_results[model_type] = results

                # Save individual results
                output_file = self.output_dir / f"{model_type}_results.json"
                with output_file.open("w") as f:
                    json.dump(results, f, indent=2, default=str)

                self.console.print(
                    f"✅ {model_type}: {results['avg_accuracy']:.1%} accuracy"
                )

            except Exception as e:
                self.console.print(f"❌ {model_type} evaluation failed: {e}")
                comparison_results[model_type] = {"error": str(e)}

        # Save comparison results
        comparison_file = self.output_dir / "comparison_results.json"
        with comparison_file.open("w") as f:
            json.dump(comparison_results, f, indent=2, default=str)

        return comparison_results

    def generate_report(
        self, comparison_results: Dict[str, Any], generate_visualizations: bool = True
    ) -> None:
        """Generate a comprehensive evaluation report with optional visualizations."""
        self.console.print("\n📊 EVALUATION REPORT")
        self.console.print("=" * 60)

        # Overview table
        overview_table = Table(title="Model Performance Overview")
        overview_table.add_column("Model", style="cyan")
        overview_table.add_column("Avg Accuracy", justify="center")
        overview_table.add_column("Avg Speed (s)", justify="center")
        overview_table.add_column("Non-N/A Fields/Image", justify="center")
        overview_table.add_column("VRAM Usage", justify="center")

        for model_type, results in comparison_results.items():
            if "error" not in results:
                # Calculate non-N/A fields (fields with actual values, not "N/A")
                non_na_fields = self._calculate_non_na_fields(results)

                # Get VRAM usage from enhanced results
                vram_usage = "N/A"
                if (
                    "model_estimated_vram" in comparison_results
                    and model_type in comparison_results["model_estimated_vram"]
                ):
                    vram_gb = comparison_results["model_estimated_vram"][model_type]
                    vram_usage = f"{vram_gb:.1f}GB"

                overview_table.add_row(
                    self._get_model_display_name(model_type),
                    f"{results['avg_accuracy']:.1%}",
                    f"{results['avg_processing_time']:.1f}s",
                    f"{non_na_fields:.1f}",
                    vram_usage,
                )
            else:
                overview_table.add_row(
                    self._get_model_display_name(model_type),
                    "FAILED",
                    "N/A",
                    "N/A",
                    "N/A",
                )

        self.console.print(overview_table)

        # Field-wise accuracy comparison
        self.console.print("\n📋 Field-wise Accuracy Comparison")
        field_table = Table()
        field_table.add_column("Field", style="cyan")

        working_models = [
            (model, results)
            for model, results in comparison_results.items()
            if "error" not in results
        ]

        for model, _ in working_models:
            field_table.add_column(
                self._get_model_display_name(model), justify="center"
            )

        # Get all fields
        all_fields = set()
        for _, results in working_models:
            all_fields.update(results["field_wise_accuracy"].keys())

        for field in sorted(all_fields):
            row = [field]
            for _, results in working_models:
                accuracy = results["field_wise_accuracy"].get(field, 0.0)
                row.append(f"{accuracy:.1%}")
            field_table.add_row(*row)

        self.console.print(field_table)

        # Winner analysis
        if len(working_models) >= 2:
            best_model = max(working_models, key=lambda x: x[1]["avg_accuracy"])
            fastest_model = min(
                working_models, key=lambda x: x[1]["avg_processing_time"]
            )

            self.console.print("\n🏆 WINNERS:")
            self.console.print(
                f"🎯 Best Accuracy: {self._get_model_display_name(best_model[0])} ({best_model[1]['avg_accuracy']:.1%})"
            )
            self.console.print(
                f"⚡ Fastest: {self._get_model_display_name(fastest_model[0])} ({fastest_model[1]['avg_processing_time']:.1f}s)"
            )

        # Generate visualizations if requested
        if generate_visualizations and working_models:
            self.console.print("\n🎨 GENERATING VISUALIZATIONS")
            self.console.print("=" * 40)

            try:
                from ..analysis.dynamic_visualizations import DynamicModelVisualizer
                from ..config import ConfigManager

                # Use the existing config manager that was passed to the evaluator
                config = self.config_manager or ConfigManager("model_comparison.yaml")

                # Debug: Show what methods are available on ConfigManager
                config_methods = [
                    method for method in dir(config) if not method.startswith("_")
                ]
                self.console.print(
                    f"📝 ConfigManager methods: {config_methods[:10]}...", style="dim"
                )

                visualizer = DynamicModelVisualizer(
                    config, str(self.output_dir / "visualizations")
                )

                # Add memory data to comparison results for VRAM visualization
                comparison_results_with_memory = self._add_memory_data_to_results(
                    comparison_results, config, working_models
                )

                # Generate all visualizations
                viz_paths = visualizer.generate_all_visualizations(
                    comparison_results_with_memory
                )

                if viz_paths:
                    self.console.print(
                        f"✅ Generated {len(viz_paths)} visualizations:", style="green"
                    )
                    for path in viz_paths:
                        self.console.print(f"   📊 {Path(path).name}", style="cyan")

                    # Create summary report with visualizations
                    summary_path = visualizer.create_summary_report(
                        comparison_results, viz_paths
                    )
                    self.console.print(
                        f"📄 Visual report: {Path(summary_path).name}",
                        style="bold green",
                    )

            except ImportError as e:
                self.console.print(
                    f"⚠️ Visualization dependencies missing: {e}", style="yellow"
                )
                self.console.print(
                    "💡 Install with: pip install matplotlib seaborn", style="blue"
                )
            except Exception as e:
                self.console.print(
                    f"❌ Error generating visualizations: {e}", style="red"
                )

        # Save detailed report
        report_file = self.output_dir / "evaluation_report.md"
        self._save_markdown_report(comparison_results, report_file)
        self.console.print(f"\n📄 Detailed report saved: {report_file}")

    def _save_markdown_report(
        self, comparison_results: Dict[str, Any], output_file: Path
    ) -> None:
        """Save detailed markdown report."""
        with output_file.open("w") as f:
            f.write("# Vision Model Key-Value Extraction Evaluation Report\n\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## Overview\n\n")
            f.write(
                "| Model | Avg Accuracy | Avg Speed | Non-N/A Fields/Image | VRAM Usage |\n"
            )
            f.write(
                "|-------|-------------|-----------|---------------------|------------|\n"
            )

            for model_type, results in comparison_results.items():
                if "error" not in results:
                    # Calculate non-N/A fields and get VRAM usage
                    non_na_fields = self._calculate_non_na_fields(results)
                    vram_usage = "N/A"
                    if (
                        "model_estimated_vram" in comparison_results
                        and model_type in comparison_results["model_estimated_vram"]
                    ):
                        vram_gb = comparison_results["model_estimated_vram"][model_type]
                        vram_usage = f"{vram_gb:.1f}GB"

                    f.write(
                        f"| {self._get_model_display_name(model_type)} | "
                        f"{results['avg_accuracy']:.1%} | {results['avg_processing_time']:.1f}s | "
                        f"{non_na_fields:.1f} | {vram_usage} |\n"
                    )

            f.write("\n## Field-wise Accuracy\n\n")
            working_models = [
                (model, results)
                for model, results in comparison_results.items()
                if "error" not in results
            ]

            if working_models:
                # Header
                f.write("| Field |")
                for model, _ in working_models:
                    f.write(f" {self._get_model_display_name(model)} |")
                f.write("\n|-------|")
                for _ in working_models:
                    f.write("---------|")
                f.write("\n")

                # Get all fields
                all_fields = set()
                for _, results in working_models:
                    all_fields.update(results["field_wise_accuracy"].keys())

                for field in sorted(all_fields):
                    f.write(f"| {field} |")
                    for _, results in working_models:
                        accuracy = results["field_wise_accuracy"].get(field, 0.0)
                        f.write(f" {accuracy:.1%} |")
                    f.write("\n")

            f.write("\n## Detailed Results\n\n")
            for model_type, results in comparison_results.items():
                f.write(f"### {self._get_model_display_name(model_type)}\n\n")
                if "error" in results:
                    f.write(f"**Error**: {results['error']}\n\n")
                else:
                    f.write(
                        f"- **Images Processed**: {results['successful']}/{results['total_images']}\n"
                    )
                    f.write(f"- **Overall Accuracy**: {results['avg_accuracy']:.1%}\n")
                    f.write(
                        f"- **Average Processing Time**: {results['avg_processing_time']:.1f}s\n"
                    )
                    f.write(
                        f"- **Total Processing Time**: {results['total_processing_time']:.1f}s\n\n"
                    )

    def _calculate_non_na_fields(self, model_results: Dict[str, Any]) -> float:
        """Calculate average number of non-N/A fields per image for a model.

        Args:
            model_results: Results dictionary for a single model

        Returns:
            Average number of fields with actual values (not "N/A") per image
        """
        if "extracted_data" not in model_results:
            # Fallback: use field-wise accuracy data if available
            if "field_wise_accuracy" in model_results:
                # Count fields with >0% accuracy (fields that were found in at least some images)
                non_zero_fields = sum(
                    1
                    for accuracy in model_results["field_wise_accuracy"].values()
                    if accuracy > 0
                )
                return float(non_zero_fields)
            return 0.0

        # Calculate from actual extracted data
        total_non_na_fields = 0
        total_images = len(model_results["extracted_data"])

        if total_images == 0:
            return 0.0

        for image_data in model_results["extracted_data"]:
            if isinstance(image_data, dict):
                # Count fields that have values other than "N/A", "", or None
                non_na_count = sum(
                    1
                    for value in image_data.values()
                    if value and str(value).strip().upper() not in ["N/A", ""]
                )
                total_non_na_fields += non_na_count

        return total_non_na_fields / total_images

    def _add_memory_data_to_results(
        self, comparison_results: Dict[str, Any], config, working_models: List[tuple]
    ) -> Dict[str, Any]:
        """Add memory data to comparison results for VRAM visualization.

        First attempts to read stored memory data from comparison_results_full.json,
        falls back to basic estimates only if no stored data is available.

        Args:
            comparison_results: Original comparison results
            config: ConfigManager instance
            working_models: List of (model_name, results) tuples

        Returns:
            Enhanced comparison results with memory data
        """
        # Create enhanced results with memory data
        enhanced_results = comparison_results.copy()
        model_names = [model_name for model_name, _ in working_models]

        # First, try to load stored memory data from ComparisonRunner output
        stored_memory_data = self._load_stored_memory_data(config)

        if stored_memory_data:
            # Use stored memory data from previous comparison run
            model_estimated_vram = {}
            for model_name in model_names:
                if model_name in stored_memory_data:
                    model_estimated_vram[model_name] = stored_memory_data[model_name]
                    self.console.print(
                        f"✅ Loaded stored VRAM data for {self._get_model_display_name(model_name)}: {stored_memory_data[model_name]:.1f}GB",
                        style="green",
                    )

            if model_estimated_vram:
                enhanced_results["model_estimated_vram"] = model_estimated_vram
                enhanced_results["models_tested"] = model_names
                self.console.print(
                    f"✅ Using stored VRAM data for {len(model_estimated_vram)} models",
                    style="green",
                )
                return enhanced_results

        # Fallback: No stored data available, skip VRAM visualization
        self.console.print(
            "⚠️ No stored memory data found - VRAM visualization will be skipped",
            style="yellow",
        )
        self.console.print(
            "💡 Run model comparison first to generate memory data",
            style="blue",
        )

        return enhanced_results

    def _load_stored_memory_data(self, config) -> Dict[str, float]:
        """Load stored memory data from comparison_results_full.json.

        KFP Compatibility: Reads from persistent storage configured in output_dir.
        In KFP environments, this MUST be mounted persistent volume (e.g., NFS).

        Args:
            config: ConfigManager instance to get output directory

        Returns:
            Dictionary mapping model names to VRAM usage in GB, empty if not found
        """
        try:
            # Look for stored comparison results in output directory
            output_dir = Path(config.output_dir)
            results_file = output_dir / "comparison_results_full.json"

            if not results_file.exists():
                self.console.print(
                    f"📁 No stored results found at: {results_file}",
                    style="dim",
                )
                return {}

            # Load and parse JSON file
            with results_file.open("r") as f:
                stored_results = json.load(f)

            # Extract memory data
            memory_data = stored_results.get("model_estimated_vram", {})

            if memory_data:
                self.console.print(
                    f"📁 Found stored memory data: {results_file}",
                    style="green",
                )
                return memory_data
            else:
                self.console.print(
                    "📁 Stored results found but no memory data available",
                    style="yellow",
                )
                return {}

        except Exception as e:
            self.console.print(
                f"⚠️ Error loading stored memory data: {e}",
                style="yellow",
            )
            return {}
