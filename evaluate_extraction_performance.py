#!/usr/bin/env python3
"""Evaluation Script for InternVL vs Llama-Vision Key-Value Extraction Performance.

This script compares the extraction performance of both models against ground truth data
and provides comprehensive metrics and analysis.
"""

import csv
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from vision_processor.config.model_factory import ModelFactory
from vision_processor.config.simple_config import SimpleConfig


class ExtractionEvaluator:
    """Evaluates key-value extraction performance against ground truth."""

    def __init__(
        self,
        ground_truth_csv: str,
        images_dir: str = "datasets",
        output_dir: str = "evaluation_results",
    ):
        """Initialize evaluator.

        Args:
            ground_truth_csv: Path to CSV with ground truth data
            images_dir: Directory containing test images
            output_dir: Directory to save evaluation results
        """
        self.ground_truth_csv = ground_truth_csv
        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir)
        self.console = Console()

        # Create output directory
        self.output_dir.mkdir(exist_ok=True)

        # Load ground truth
        self.ground_truth = self._load_ground_truth()

        # Key extraction patterns
        self.key_patterns = {
            "DATE": r"DATE:\s*([^\n\r|]+)",
            "STORE": r"STORE:\s*([^\n\r|]+)",
            "ABN": r"ABN:\s*([^\n\r|]+)",
            "GST": r"GST:\s*\$?([0-9.]+)",
            "TOTAL": r"TOTAL:\s*\$?([0-9.]+)",
            "SUBTOTAL": r"SUBTOTAL:\s*\$?([0-9.]+)",
            "ITEMS": r"ITEMS:\s*([^\n\r]+)",
            "QUANTITIES": r"QUANTITIES:\s*([^\n\r]+)",
            "PRICES": r"PRICES:\s*([^\n\r]+)",
            "RECEIPT_NUMBER": r"RECEIPT(?:_NUMBER)?:\s*([^\n\r|]+)",
            "PAYMENT_METHOD": r"PAYMENT(?:_METHOD)?:\s*([^\n\r|]+)",
            "TIME": r"TIME:\s*([^\n\r|]+)",
        }

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

    def _extract_key_values(self, response_text: str) -> Dict[str, str]:
        """Extract key-value pairs from model response using patterns."""
        extracted = {}

        for key, pattern in self.key_patterns.items():
            match = re.search(pattern, response_text, re.IGNORECASE | re.MULTILINE)
            if match:
                value = match.group(1).strip()
                # Clean common artifacts
                value = re.sub(r"[<>]", "", value)  # Remove < >
                value = re.sub(r"\s+", " ", value)  # Normalize whitespace
                extracted[key] = value

        return extracted

    def _calculate_field_accuracy(self, extracted: str, ground_truth: str, field_type: str) -> float:
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

            matches = sum(1 for e, g in zip(ext_items, gt_items, strict=True) if e.lower() == g.lower())
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

    def _evaluate_single_image(self, model, image_file: str, prompt: str) -> Dict[str, Any]:
        """Evaluate extraction for a single image."""
        image_path = self.images_dir / image_file

        if not image_path.exists():
            return {"error": f"Image not found: {image_path}"}

        # Get ground truth
        gt_data = self.ground_truth.get(image_file, {})

        try:
            # Extract using model
            start_time = time.time()
            response = model.process_image(str(image_path), prompt)
            processing_time = time.time() - start_time

            # Extract key-values from response
            extracted_data = self._extract_key_values(response.raw_text)

            # Calculate field-wise accuracy
            field_accuracies = {}
            for field in self.key_patterns.keys():
                gt_value = gt_data.get(field, "")
                ext_value = extracted_data.get(field, "")
                field_accuracies[field] = self._calculate_field_accuracy(ext_value, gt_value, field)

            # Overall accuracy
            total_fields = len([f for f in self.key_patterns.keys() if gt_data.get(f)])
            overall_accuracy = (
                sum(field_accuracies.values()) / len(field_accuracies) if field_accuracies else 0.0
            )

            return {
                "image_file": image_file,
                "processing_time": processing_time,
                "response_length": len(response.raw_text),
                "extracted_fields": len(extracted_data),
                "total_gt_fields": total_fields,
                "field_accuracies": field_accuracies,
                "overall_accuracy": overall_accuracy,
                "raw_response": response.raw_text[:500] + "...",
                "extracted_data": extracted_data,
                "ground_truth": gt_data,
                "confidence": getattr(response, "confidence", 0.0),
            }

        except Exception as e:
            return {"error": str(e), "image_file": image_file}

    def evaluate_model(
        self, model_type: str, prompt: str, test_images: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Evaluate a specific model on all test images."""
        self.console.print(f"\n🔬 Evaluating {model_type.upper()} model...")

        # Setup model
        config = SimpleConfig()
        config.switch_model(model_type)  # Properly switch model type and path
        model = ModelFactory.create_model(config)

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
            task = progress.add_task(f"Processing {len(test_images)} images...", total=None)

            for image_file in test_images:
                progress.update(task, description=f"Processing {image_file}...")

                result = self._evaluate_single_image(model, image_file, prompt)
                results.append(result)

                if "processing_time" in result:
                    total_time += result["processing_time"]

        # Calculate aggregate metrics
        successful_results = [r for r in results if "error" not in r]
        failed_results = [r for r in results if "error" in r]

        if successful_results:
            avg_accuracy = sum(r["overall_accuracy"] for r in successful_results) / len(successful_results)
            avg_processing_time = total_time / len(successful_results)
            avg_fields_extracted = sum(r["extracted_fields"] for r in successful_results) / len(
                successful_results
            )

            # Field-wise accuracy
            field_wise_accuracy = {}
            for field in self.key_patterns.keys():
                field_scores = [r["field_accuracies"].get(field, 0.0) for r in successful_results]
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
        prompt: str,
        models: Optional[List[str]] = None,
        test_images: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Compare multiple models side by side."""
        if models is None:
            models = ["internvl3", "llama32_vision"]
        self.console.print("🏁 Starting Model Comparison Evaluation")
        self.console.print(f"📝 Prompt: {prompt[:100]}...")

        comparison_results = {}

        for model_type in models:
            try:
                results = self.evaluate_model(model_type, prompt, test_images)
                comparison_results[model_type] = results

                # Save individual results
                output_file = self.output_dir / f"{model_type}_results.json"
                with output_file.open("w") as f:
                    json.dump(results, f, indent=2, default=str)

                self.console.print(f"✅ {model_type}: {results['avg_accuracy']:.1%} accuracy")

            except Exception as e:
                self.console.print(f"❌ {model_type} evaluation failed: {e}")
                comparison_results[model_type] = {"error": str(e)}

        # Save comparison results
        comparison_file = self.output_dir / "comparison_results.json"
        with comparison_file.open("w") as f:
            json.dump(comparison_results, f, indent=2, default=str)

        return comparison_results

    def generate_report(self, comparison_results: Dict[str, Any]) -> None:
        """Generate a comprehensive evaluation report."""
        self.console.print("\n📊 EVALUATION REPORT")
        self.console.print("=" * 60)

        # Overview table
        overview_table = Table(title="Model Performance Overview")
        overview_table.add_column("Model", style="cyan")
        overview_table.add_column("Success Rate", justify="center")
        overview_table.add_column("Avg Accuracy", justify="center")
        overview_table.add_column("Avg Speed (s)", justify="center")
        overview_table.add_column("Fields/Image", justify="center")

        for model_type, results in comparison_results.items():
            if "error" not in results:
                overview_table.add_row(
                    model_type.upper(),
                    f"{results['success_rate']:.1%}",
                    f"{results['avg_accuracy']:.1%}",
                    f"{results['avg_processing_time']:.1f}s",
                    f"{results['avg_fields_extracted']:.1f}",
                )
            else:
                overview_table.add_row(model_type.upper(), "FAILED", "N/A", "N/A", "N/A")

        self.console.print(overview_table)

        # Field-wise accuracy comparison
        self.console.print("\n📋 Field-wise Accuracy Comparison")
        field_table = Table()
        field_table.add_column("Field", style="cyan")

        working_models = [
            (model, results) for model, results in comparison_results.items() if "error" not in results
        ]

        for model, _ in working_models:
            field_table.add_column(model.upper(), justify="center")

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
            fastest_model = min(working_models, key=lambda x: x[1]["avg_processing_time"])

            self.console.print("\n🏆 WINNERS:")
            self.console.print(
                f"🎯 Best Accuracy: {best_model[0].upper()} ({best_model[1]['avg_accuracy']:.1%})"
            )
            self.console.print(
                f"⚡ Fastest: {fastest_model[0].upper()} ({fastest_model[1]['avg_processing_time']:.1f}s)"
            )

        # Save detailed report
        report_file = self.output_dir / "evaluation_report.md"
        self._save_markdown_report(comparison_results, report_file)
        self.console.print(f"\n📄 Detailed report saved: {report_file}")

    def _save_markdown_report(self, comparison_results: Dict[str, Any], output_file: Path) -> None:
        """Save detailed markdown report."""
        with output_file.open("w") as f:
            f.write("# Vision Model Key-Value Extraction Evaluation Report\n\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## Overview\n\n")
            f.write("| Model | Success Rate | Avg Accuracy | Avg Speed | Fields/Image |\n")
            f.write("|-------|-------------|-------------|-----------|-------------|\n")

            for model_type, results in comparison_results.items():
                if "error" not in results:
                    f.write(
                        f"| {model_type.upper()} | {results['success_rate']:.1%} | "
                        f"{results['avg_accuracy']:.1%} | {results['avg_processing_time']:.1f}s | "
                        f"{results['avg_fields_extracted']:.1f} |\n"
                    )

            f.write("\n## Field-wise Accuracy\n\n")
            working_models = [
                (model, results) for model, results in comparison_results.items() if "error" not in results
            ]

            if working_models:
                # Header
                f.write("| Field |")
                for model, _ in working_models:
                    f.write(f" {model.upper()} |")
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
                f.write(f"### {model_type.upper()}\n\n")
                if "error" in results:
                    f.write(f"**Error**: {results['error']}\n\n")
                else:
                    f.write(f"- **Images Processed**: {results['successful']}/{results['total_images']}\n")
                    f.write(f"- **Overall Accuracy**: {results['avg_accuracy']:.1%}\n")
                    f.write(f"- **Average Processing Time**: {results['avg_processing_time']:.1f}s\n")
                    f.write(f"- **Total Processing Time**: {results['total_processing_time']:.1f}s\n\n")


def main():
    """Main evaluation function."""
    console = Console()

    # Setup
    ground_truth_csv = "evaluation_ground_truth.csv"
    images_dir = "datasets"

    if not Path(ground_truth_csv).exists():
        console.print(f"❌ Ground truth file not found: {ground_truth_csv}")
        console.print("Please create the ground truth CSV file first.")
        return 1

    if not Path(images_dir).exists():
        console.print(f"❌ Images directory not found: {images_dir}")
        return 1

    # Create evaluator
    evaluator = ExtractionEvaluator(ground_truth_csv, images_dir)

    # Evaluation prompt
    prompt = """<|image|>Extract information from this Australian business document and return in KEY-VALUE format.

CRITICAL: Use EXACT KEY-VALUE format below - NO JSON, NO other formats.

DATE: [document date in DD/MM/YYYY format]
STORE: [business name in capitals]
ABN: [Australian Business Number if visible]
GST: [GST amount]
TOTAL: [total amount including GST]
SUBTOTAL: [subtotal before GST if visible]
ITEMS: [item1 | item2 | item3]
QUANTITIES: [qty1 | qty2 | qty3]
PRICES: [price1 | price2 | price3]
RECEIPT_NUMBER: [receipt/invoice number if visible]
PAYMENT_METHOD: [payment method if visible]
TIME: [time if visible]

Return ONLY the key-value pairs above. Skip any keys where information is not available."""

    try:
        # Run comparison
        results = evaluator.compare_models(prompt)

        # Generate report
        evaluator.generate_report(results)

        console.print("\n🎉 Evaluation completed successfully!")
        console.print(f"📁 Results saved in: {evaluator.output_dir}")

        return 0

    except KeyboardInterrupt:
        console.print("\n⚠️ Evaluation interrupted by user")
        return 1
    except Exception as e:
        console.print(f"\n❌ Evaluation failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
