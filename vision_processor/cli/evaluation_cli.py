"""Evaluation CLI for Vision Model Performance Comparison."""

import csv
import json
import time
from pathlib import Path

import typer
from rich.console import Console

from ..evaluation import ExtractionEvaluator

app = typer.Typer(
    name="evaluation",
    help="Vision Model Evaluation Tools",
    rich_markup_mode="rich",
)
console = Console()


@app.command()
def compare(
    ground_truth_csv: str = typer.Argument(..., help="Path to ground truth CSV file"),
    images_dir: str = typer.Option("datasets", help="Directory containing test images"),
    models: str = typer.Option(
        "internvl3,llama32_vision", help="Models to compare (comma-separated)"
    ),
    output_dir: str = typer.Option(
        "evaluation_results", help="Output directory for results"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
) -> None:
    """Compare extraction performance between vision models."""

    # Validate inputs
    if not Path(ground_truth_csv).exists():
        console.print(f"[red]❌ Ground truth file not found: {ground_truth_csv}[/red]")
        raise typer.Exit(1) from None

    if not Path(images_dir).exists():
        console.print(f"[red]❌ Images directory not found: {images_dir}[/red]")
        raise typer.Exit(1) from None

    try:
        # Parse models
        model_list = [m.strip() for m in models.split(",")]

        # Create evaluator
        evaluator = ExtractionEvaluator(
            ground_truth_csv=ground_truth_csv,
            images_dir=images_dir,
            output_dir=output_dir,
        )

        console.print(f"🔬 Evaluating {len(model_list)} models on ground truth data")
        console.print(f"📁 Images: {images_dir}")
        console.print(f"📊 Ground truth: {ground_truth_csv}")
        console.print(f"💾 Results: {output_dir}")

        # Run comparison
        results = evaluator.compare_models(models=model_list)

        # Generate report
        evaluator.generate_report(results)

        console.print("\\n🎉 Evaluation completed successfully!")
        console.print(f"📄 View results: {Path(output_dir) / 'evaluation_report.md'}")

    except Exception as e:
        console.print(f"[red]❌ Evaluation failed: {e}[/red]")
        if verbose:
            import traceback

            console.print(traceback.format_exc())
        raise typer.Exit(1) from None


@app.command()
def benchmark(
    images_dir: str = typer.Argument(
        ..., help="Directory containing images to benchmark"
    ),
    model: str = typer.Option("internvl3", help="Model to benchmark"),
    output_file: str = typer.Option(
        "benchmark_results.json", help="Output file for results"
    ),
    iterations: int = typer.Option(3, help="Number of iterations per image"),
) -> None:
    """Benchmark processing speed and consistency for a single model."""

    if not Path(images_dir).exists():
        console.print(f"[red]❌ Images directory not found: {images_dir}[/red]")
        raise typer.Exit(1) from None

    try:
        # Legacy SimpleConfig import removed - now using ConfigManager
        from ..extraction.extraction_manager import SimpleExtractionManager

        console.print(f"⚡ Benchmarking {model} performance")
        console.print(f"📁 Images: {images_dir}")
        console.print(f"🔄 Iterations: {iterations}")

        # Setup model
        from ..config import ConfigManager
        config = ConfigManager()
        config.set_model_type(model)
        manager = SimpleExtractionManager(config)

        # Find images
        image_files = list(Path(images_dir).glob("*.png")) + list(
            Path(images_dir).glob("*.jpg")
        )
        console.print(f"📸 Found {len(image_files)} images")

        benchmark_results = []

        for image_file in image_files:
            console.print(f"\\n🔍 Benchmarking {image_file.name}...")

            image_results = []
            for i in range(iterations):
                start_time = time.time()
                result = manager.process_document(str(image_file))
                processing_time = time.time() - start_time

                image_results.append(
                    {
                        "iteration": i + 1,
                        "processing_time": processing_time,
                        "fields_extracted": len(result.extracted_fields),
                        "confidence": result.model_confidence,
                    }
                )

                console.print(
                    f"  Iteration {i + 1}: {processing_time:.2f}s ({len(result.extracted_fields)} fields)"
                )

            # Calculate statistics
            times = [r["processing_time"] for r in image_results]
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)

            benchmark_results.append(
                {
                    "image_file": image_file.name,
                    "iterations": image_results,
                    "avg_processing_time": avg_time,
                    "min_processing_time": min_time,
                    "max_processing_time": max_time,
                    "consistency": (max_time - min_time)
                    / avg_time,  # Lower is more consistent
                }
            )

            console.print(
                f"  Average: {avg_time:.2f}s (range: {min_time:.2f}s - {max_time:.2f}s)"
            )

        # Save results
        with Path(output_file).open("w") as f:
            json.dump(
                {
                    "model": model,
                    "total_images": len(image_files),
                    "iterations_per_image": iterations,
                    "results": benchmark_results,
                    "summary": {
                        "avg_processing_time": sum(
                            r["avg_processing_time"] for r in benchmark_results
                        )
                        / len(benchmark_results),
                        "total_processing_time": sum(
                            r["avg_processing_time"] for r in benchmark_results
                        )
                        * iterations,
                    },
                },
                f,
                indent=2,
                default=str,
            )

        console.print(f"\\n💾 Benchmark results saved: {output_file}")

    except Exception as e:
        console.print(f"[red]❌ Benchmark failed: {e}[/red]")
        raise typer.Exit(1) from None


@app.command()
def validate_ground_truth(
    ground_truth_csv: str = typer.Argument(..., help="Path to ground truth CSV file"),
    images_dir: str = typer.Option("datasets", help="Directory containing test images"),
) -> None:
    """Validate ground truth data against available images."""

    if not Path(ground_truth_csv).exists():
        console.print(f"[red]❌ Ground truth file not found: {ground_truth_csv}[/red]")
        raise typer.Exit(1) from None

    if not Path(images_dir).exists():
        console.print(f"[red]❌ Images directory not found: {images_dir}[/red]")
        raise typer.Exit(1) from None

    try:
        # Load ground truth
        with Path(ground_truth_csv).open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            ground_truth_data = list(reader)

        # Find available images
        image_files = set()
        for ext in ["*.png", "*.jpg", "*.jpeg"]:
            image_files.update(Path(images_dir).glob(ext))
        image_names = {img.name for img in image_files}

        console.print(f"📊 Ground truth entries: {len(ground_truth_data)}")
        console.print(f"📸 Available images: {len(image_names)}")

        # Check for mismatches
        missing_images = []
        orphaned_images = image_names.copy()

        for entry in ground_truth_data:
            image_file = entry.get("image_file", "")
            if image_file in image_names:
                orphaned_images.discard(image_file)
                console.print(f"✅ {image_file}")
            else:
                missing_images.append(image_file)
                console.print(f"❌ {image_file} (not found)")

        # Report results
        if missing_images:
            console.print(f"\\n⚠️  Missing images ({len(missing_images)}):")
            for img in missing_images:
                console.print(f"  - {img}")

        if orphaned_images:
            console.print(f"\\n📸 Unused images ({len(orphaned_images)}):")
            for img in orphaned_images:
                console.print(f"  - {img}")

        if not missing_images and not orphaned_images:
            console.print("\\n🎉 All ground truth entries have matching images!")

    except Exception as e:
        console.print(f"[red]❌ Validation failed: {e}[/red]")
        raise typer.Exit(1) from None


if __name__ == "__main__":
    app()
