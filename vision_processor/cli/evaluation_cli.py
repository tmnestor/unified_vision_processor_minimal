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
    help="""
Vision Model Evaluation Tools

[bold cyan]Typical Workflow:[/bold cyan]
1. [green]validate-ground-truth[/green] - Check your CSV and images match
2. [blue]compare[/blue] - Run model comparison against ground truth  
3. [magenta]visualize[/magenta] - Generate charts and reports from results

[bold cyan]Quick Start:[/bold cyan]
  evaluation_cli compare ground_truth.csv  # Compare all models
  evaluation_cli visualize results.json    # Generate charts from results
""",
    rich_markup_mode="rich",
)
console = Console()


@app.command()
def compare(
    ground_truth_csv: str = typer.Argument(..., help="Path to ground truth CSV file"),
    images_dir: str = typer.Option(
        None, help="Directory containing test images (default from config)"
    ),
    models: str = typer.Option(
        None, help="Models to compare (comma-separated, default from config)"
    ),
    output_dir: str = typer.Option(
        None, help="Output directory for results (default from config)"
    ),
    visualizations: bool = typer.Option(
        True,
        "--visualizations/--no-visualizations",
        help="Generate dynamic visualizations (charts, heatmaps)",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
    debug: bool = typer.Option(False, "--debug", help="Enable debug output"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Minimize output"),
) -> None:
    """[Step 2] Compare extraction performance between vision models against ground truth.
    
    Runs model inference on all images and compares results against ground truth CSV.
    Generates JSON results file and optional visualizations.
    """

    # Validate ground truth file first
    if not Path(ground_truth_csv).exists():
        console.print(f"[red]❌ Ground truth file not found: {ground_truth_csv}[/red]")
        raise typer.Exit(1) from None

    # Simple path validation - just check if paths exist directly
    if images_dir and not Path(images_dir).exists():
        console.print(f"[red]❌ Images directory not found: {images_dir}[/red]")
        raise typer.Exit(1) from None

    try:
        # Apply logging overrides via ConfigManager (evaluator will inherit these settings)
        from ..config import ConfigManager
        from ..utils.path_resolver import PathResolver

        config = ConfigManager()
        path_resolver = PathResolver(config)

        # Apply CLI logging overrides
        if debug:
            config.defaults.debug_mode = True
            config.defaults.verbose_mode = True  # Debug implies verbose
        elif verbose:
            config.defaults.verbose_mode = True

        if quiet:
            config.defaults.console_output = False
            config.defaults.verbose_mode = False
            config.defaults.debug_mode = False

        # Resolve images directory
        if images_dir is None:
            # Default to ground truth CSV directory if not specified
            images_dir = str(Path(ground_truth_csv).parent)
            console.print(
                f"📁 Using images directory: {images_dir} (same as ground truth CSV)"
            )
        else:
            # Use provided path - convert relative to absolute
            images_dir = str(Path(images_dir).resolve())
            console.print(f"📁 Using specified images directory: {images_dir}")

        output_dir = path_resolver.resolve_output_path(output_dir)

        # Parse models - use config default if not specified
        if models is None:
            # Get default models from config
            default_models = config.defaults.models
            model_list = [m.strip() for m in default_models.split(",")]
        else:
            model_list = [m.strip() for m in models.split(",")]

        # Create evaluator
        evaluator = ExtractionEvaluator(
            ground_truth_csv=ground_truth_csv,
            images_dir=images_dir,
            output_dir=output_dir,
            config_manager=config,
        )

        console.print(f"🔬 Evaluating {len(model_list)} models on ground truth data")
        console.print(f"📁 Images: {images_dir}")
        console.print(f"📊 Ground truth: {ground_truth_csv}")
        console.print(f"💾 Results: {output_dir}")

        # Run comparison
        results = evaluator.compare_models(models=model_list)

        # Generate report with optional visualizations
        evaluator.generate_report(results, generate_visualizations=visualizations)

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
    model: str = typer.Option(None, help="Model to benchmark (default from config)"),
    output_file: str = typer.Option(
        None,
        help="Output file for results (default: benchmark_results.json in config output_dir)",
    ),
    iterations: int = typer.Option(3, help="Number of iterations per image"),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
    debug: bool = typer.Option(False, "--debug", help="Enable debug output"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Minimize output"),
) -> None:
    """[Optional] Benchmark processing speed and consistency for a single model.
    
    Tests model performance across multiple iterations to measure speed and consistency.
    Useful for performance tuning and hardware optimization.
    """

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
        from ..utils.path_resolver import PathResolver

        config = ConfigManager()
        path_resolver = PathResolver(config)

        # Apply CLI logging overrides
        if debug:
            config.defaults.debug_mode = True
            config.defaults.verbose_mode = True  # Debug implies verbose
        elif verbose:
            config.defaults.verbose_mode = True

        if quiet:
            config.defaults.console_output = False
            config.defaults.verbose_mode = False
            config.defaults.debug_mode = False

        # Resolve output file path using utility
        output_file = path_resolver.resolve_output_path(
            output_file, "benchmark_results.json" if not output_file else None
        )

        # Use config default model if not specified
        if model is None:
            # Get first model from config defaults
            default_models = config.defaults.models
            model = default_models.split(",")[0].strip()

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
                            float(r.get("avg_processing_time", 0) or 0)
                            for r in benchmark_results
                        )
                        / len(benchmark_results),
                        "total_processing_time": sum(
                            float(r.get("avg_processing_time", 0) or 0)
                            for r in benchmark_results
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
    images_dir: str = typer.Option(
        None, help="Directory containing test images (default from config)"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
    debug: bool = typer.Option(False, "--debug", help="Enable debug output"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Minimize output"),
) -> None:
    """[Step 1] Validate ground truth data against available images.
    
    Checks that every image listed in ground truth CSV exists in the images directory.
    Reports missing images and unused image files. Run this before comparison.
    """

    if not Path(ground_truth_csv).exists():
        console.print(f"[red]❌ Ground truth file not found: {ground_truth_csv}[/red]")
        raise typer.Exit(1) from None

    try:
        # Apply logging overrides via ConfigManager
        from ..config import ConfigManager
        from ..utils.path_resolver import PathResolver

        config = ConfigManager()
        path_resolver = PathResolver(config)

        # Apply CLI logging overrides
        if debug:
            config.defaults.debug_mode = True
            config.defaults.verbose_mode = True  # Debug implies verbose
        elif verbose:
            config.defaults.verbose_mode = True

        if quiet:
            config.defaults.console_output = False
            config.defaults.verbose_mode = False
            config.defaults.debug_mode = False

        # Resolve images directory
        if images_dir is None:
            # Default to ground truth CSV directory if not specified
            images_dir = str(Path(ground_truth_csv).parent)
            console.print(
                f"📁 Using images directory: {images_dir} (same as ground truth CSV)"
            )
        else:
            # Use provided path - convert relative to absolute
            images_dir = str(Path(images_dir).resolve())
            console.print(f"📁 Using specified images directory: {images_dir}")

        # Validate images directory exists
        if not Path(images_dir).exists():
            console.print(f"[red]❌ Images directory not found: {images_dir}[/red]")
            raise typer.Exit(1) from None

        # Load ground truth
        with Path(ground_truth_csv).open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            ground_truth_data = list(reader)

        # Find available images
        image_files: set[Path] = set()
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


@app.command()
def visualize(
    results_file: str = typer.Argument(
        ..., help="Path to comparison results JSON file (from compare command)"
    ),
    output_dir: str = typer.Option(
        None, help="Output directory for visualizations (default from config)"
    ),
    ground_truth_csv: str = typer.Option(
        None, help="Path to ground truth CSV (required for accuracy charts)"
    ),
    images_dir: str = typer.Option(
        None, help="Directory containing images (default from config)"
    ),
    format: str = typer.Option(
        "all", help="Visualization format: all, charts, html, or heatmaps"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
    debug: bool = typer.Option(False, "--debug", help="Enable debug output"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Minimize output"),
) -> None:
    """[Step 3] Generate visualizations from comparison results.
    
    Creates charts, heatmaps, and HTML reports from JSON results produced by compare command.
    Can generate different visualization types and formats.
    
    [bold cyan]Examples:[/bold cyan]
      visualize results.json                           # Basic charts from results
      visualize results.json --ground-truth-csv gt.csv # Full accuracy analysis
      visualize results.json --format heatmaps         # Only accuracy heatmaps
    """
    
    if not Path(results_file).exists():
        console.print(f"[red]❌ Results file not found: {results_file}[/red]")
        console.print("[yellow]💡 Run 'compare' command first to generate results[/yellow]")
        raise typer.Exit(1) from None

    try:
        import json

        from ..analysis.dynamic_visualizations import DynamicVisualizationGenerator
        from ..config import ConfigManager

        # Load configuration
        config = ConfigManager()
        
        # Apply CLI logging overrides
        if debug:
            config.defaults.debug_mode = True
            config.defaults.verbose_mode = True
        elif verbose:
            config.defaults.verbose_mode = True
        if quiet:
            config.defaults.console_output = False
            config.defaults.verbose_mode = False
            config.defaults.debug_mode = False
        
        # Use config defaults if not provided
        effective_output_dir = output_dir or config.defaults.output_dir
        effective_images_dir = images_dir or config.defaults.datasets_path
        
        console.print("🎨 [bold]GENERATING VISUALIZATIONS[/bold]")
        console.print(f"📊 Results: {results_file}")
        console.print(f"💾 Output: {effective_output_dir}")
        
        # Load comparison results
        with Path(results_file).open('r') as f:
            comparison_results = json.load(f)
        
        console.print(f"✅ Loaded results for {len(comparison_results)} models")
        
        # Initialize visualization generator
        viz_generator = DynamicVisualizationGenerator(
            output_dir=effective_output_dir,
            config_manager=config
        )
        
        # Generate visualizations based on format
        viz_paths = []
        
        if format in ["all", "charts"]:
            console.print("📈 Generating performance charts...")
            chart_path = viz_generator.create_model_performance_dashboard(comparison_results)
            if chart_path:
                viz_paths.append(chart_path)
        
        if format in ["all", "heatmaps"] and ground_truth_csv:
            console.print("🔥 Generating accuracy heatmaps...")
            heatmap_path = viz_generator.create_field_accuracy_heatmap(comparison_results)
            if heatmap_path:
                viz_paths.append(heatmap_path)
        
        if format in ["all", "html"]:
            console.print("🌐 Generating HTML report...")
            html_path = viz_generator.create_interactive_html_report(
                comparison_results, viz_paths
            )
            if html_path:
                console.print(f"📄 Interactive report: {html_path}")
        
        console.print(f"\n🎉 Generated {len(viz_paths)} visualizations!")
        
        if viz_paths:
            console.print("\n📊 [bold]Generated Files:[/bold]")
            for path in viz_paths:
                console.print(f"  📈 {Path(path).name}")
        
        if not ground_truth_csv and format in ["all", "heatmaps"]:
            console.print("\n[yellow]💡 Add --ground-truth-csv for accuracy heatmaps[/yellow]")

    except Exception as e:
        console.print(f"[red]❌ Visualization failed: {e}[/red]")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        raise typer.Exit(1) from None


if __name__ == "__main__":
    app()
