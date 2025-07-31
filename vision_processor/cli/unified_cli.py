#!/usr/bin/env python3
"""
Unified Vision Processor CLI
===========================

Single, clean entry point for all vision processing commands.
Consolidates functionality from model_comparison.py, evaluation_cli.py,
extract_cli.py, and batch_to_csv_cli.py into a unified interface.

Usage: python -m vision_processor <command>

Based on the CLI consolidation plan to eliminate interface confusion
and provide a consistent, maintainable command structure.
"""

import json
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from ..config.config_manager import ConfigManager
from ..exceptions import (
    ConfigurationError,
    ImageProcessingError,
    ModelLoadError,
    VisionProcessorError,
)

# Create main app
app = typer.Typer(
    help="""
[bold blue]Unified Vision Document Processing CLI[/bold blue]

Single entry point for all vision processing functionality.

[bold cyan]Core Workflows:[/bold cyan]
  [green]compare[/green]     - Model comparison with auto ground truth detection
  [green]extract[/green]     - Single image extraction
  [green]batch[/green]       - Batch process directory

[bold cyan]Evaluation:[/bold cyan]
  [blue]evaluate[/blue]      - Ground truth evaluation
  [blue]benchmark[/blue]     - Performance testing
  [blue]validate-ground-truth[/blue] - CSV/image validation

[bold cyan]System:[/bold cyan]
  [yellow]check[/yellow]      - Environment validation
  [yellow]models[/yellow]     - List/validate models
  [yellow]config[/yellow]     - Show configuration
  [yellow]schema[/yellow]     - Show field schema

[bold cyan]Utilities:[/bold cyan]
  [magenta]convert[/magenta]   - Batch results to CSV

All commands use model_comparison.yaml as single source of truth.
CLI parameters override YAML values with fail-fast error handling.
""",
    rich_markup_mode="rich",
)

console = Console()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _display_table(extracted_fields: dict) -> None:
    """Display results in table format."""
    from rich.table import Table

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="green")

    for field, value in extracted_fields.items():
        table.add_row(field, str(value))

    console.print("\n📋 [bold blue]Extracted Fields[/bold blue]")
    console.print(table)


def _display_clean_format(extracted_fields: dict) -> None:
    """Display results in clean YAML-like format."""
    console.print("\n📋 [bold blue]Extracted Fields[/bold blue]")

    max_field_length = (
        max(len(field) for field in extracted_fields.keys()) if extracted_fields else 0
    )

    for field, value in extracted_fields.items():
        padded_field = field.ljust(max_field_length)
        console.print(f"[cyan]{padded_field}[/cyan]: [green]{value}[/green]")


def _display_comparison_summary(results, config) -> None:
    """Display comprehensive comparison summary like old model_comparison.py."""
    from rich.table import Table

    console.print("\n" + "=" * 80)
    console.print("🏁 [bold green]COMPARISON COMPLETE[/bold green]")
    console.print("=" * 80)

    # Extract key metrics from results
    models_tested = getattr(results, "models_tested", [])
    total_execution_time = getattr(results, "total_execution_time", 0)
    dataset_info = getattr(results, "dataset_info", None)
    model_success_rates = getattr(results, "model_success_rates", {})
    model_execution_times = getattr(results, "model_execution_times", {})

    # Overall summary
    total_images = dataset_info.total_images if dataset_info else 0
    verified_images = len(dataset_info.verified_images) if dataset_info else 0
    overall_success_rate = (
        sum(model_success_rates.values()) / len(model_success_rates)
        if model_success_rates
        else 0
    )

    console.print(
        f"⏱️  Total execution time: [bold cyan]{total_execution_time:.1f}s[/bold cyan]"
    )
    console.print(
        f"✅ Overall success rate: [bold green]{overall_success_rate:.1%}[/bold green]"
    )
    console.print(f"🤖 Models compared: [bold blue]{len(models_tested)}[/bold blue]")
    console.print(f"📸 Images processed: [bold blue]{verified_images}[/bold blue]")

    # Model Performance Summary
    console.print("\n📊 [bold cyan]Model Performance Summary:[/bold cyan]")

    for model in models_tested:
        success_rate = model_success_rates.get(model, 0)
        exec_time = model_execution_times.get(model, 0)
        time_per_image = exec_time / verified_images if verified_images > 0 else 0

        model_display = (
            config.get_model_display_name(model)
            if hasattr(config, "get_model_display_name")
            else model
        )

        console.print(
            f"  {model}: [green]{success_rate:.1%}[/green] success, [yellow]{exec_time:.1f}s[/yellow] total, [cyan]{time_per_image:.1f}s[/cyan] per image"
        )

    # Processing Speed Comparison
    if len(models_tested) >= 2:
        console.print("\n⚡ [bold cyan]Processing Speed Comparison:[/bold cyan]")

        # Sort models by speed (fastest first)
        sorted_models = sorted(
            models_tested, key=lambda m: model_execution_times.get(m, float("inf"))
        )

        for model in sorted_models:
            exec_time = model_execution_times.get(model, 0)
            time_per_image = exec_time / verified_images if verified_images > 0 else 0
            console.print(
                f"  {model}: [yellow]{time_per_image:.1f}s[/yellow] per image"
            )

    # Additional details table
    if len(models_tested) >= 2:
        table = Table(title="Detailed Model Comparison")
        table.add_column("Model", style="cyan")
        table.add_column("Success Rate", justify="center")
        table.add_column("Total Time", justify="center")
        table.add_column("Time/Image", justify="center")
        table.add_column("Status", justify="center")

        for model in models_tested:
            success_rate = model_success_rates.get(model, 0)
            exec_time = model_execution_times.get(model, 0)
            time_per_image = exec_time / verified_images if verified_images > 0 else 0

            # Determine status
            if success_rate >= 0.9:
                status = "[green]Excellent[/green]"
            elif success_rate >= 0.7:
                status = "[yellow]Good[/yellow]"
            else:
                status = "[red]Needs Work[/red]"

            table.add_row(
                model.upper(),
                f"{success_rate:.1%}",
                f"{exec_time:.1f}s",
                f"{time_per_image:.1f}s",
                status,
            )

        console.print("\n")
        console.print(table)

    # Field-wise extraction analysis
    _display_fieldwise_extraction_table(results, config)


def _display_fieldwise_extraction_table(results, config) -> None:
    """Display field-wise extraction performance table like old model_comparison.py."""
    from rich.table import Table

    console.print("\n📋 [bold cyan]Field-wise Extraction Performance:[/bold cyan]")

    # Extract field analysis data
    field_analysis = getattr(results, "field_analysis", None)
    models_tested = getattr(results, "models_tested", [])

    if not field_analysis or "model_stats" not in field_analysis:
        console.print("  ⚠️ No field-wise data available", style="yellow")
        return

    # Get expected fields from config
    expected_fields = (
        config.get_expected_fields() if hasattr(config, "get_expected_fields") else []
    )

    if not expected_fields:
        console.print("  ⚠️ No expected fields found in configuration", style="yellow")
        return

    # Create field-wise table
    table = Table(title="Field Extraction Rates by Model")
    table.add_column("Field", style="cyan", width=20)

    for model in models_tested:
        model_display = (
            config.get_model_display_name(model)
            if hasattr(config, "get_model_display_name")
            else model.upper()
        )
        table.add_column(model_display, justify="center", width=12)

    # Process each field
    for field in expected_fields:
        row_data = [field]
        field_rates = []

        for model in models_tested:
            if model in field_analysis.model_stats:
                model_stats = field_analysis.model_stats[model]

                # Try to get field extraction rate
                rate = 0.0
                if (
                    hasattr(model_stats, "field_value_rates")
                    and field in model_stats.field_value_rates
                ):
                    rate = model_stats.field_value_rates[field]
                elif (
                    hasattr(model_stats, "field_extraction_rates")
                    and field in model_stats.field_extraction_rates
                ):
                    rate = model_stats.field_extraction_rates[field]

                field_rates.append(rate)

                # Color code based on performance
                if rate >= 0.9:
                    rate_str = f"[green]{rate:.1%}[/green]"
                elif rate >= 0.7:
                    rate_str = f"[yellow]{rate:.1%}[/yellow]"
                elif rate >= 0.5:
                    rate_str = f"[orange1]{rate:.1%}[/orange1]"
                else:
                    rate_str = f"[red]{rate:.1%}[/red]"

                row_data.append(rate_str)
            else:
                row_data.append("[dim]N/A[/dim]")
                field_rates.append(0.0)

        table.add_row(*row_data)

    console.print("\n")
    console.print(table)

    # Add legend
    console.print("\n📊 [bold]Performance Legend:[/bold]")
    console.print(
        "  [green]■[/green] Excellent (≥90%)  [yellow]■[/yellow] Good (70-89%)  [orange1]■[/orange1] Fair (50-69%)  [red]■[/red] Poor (<50%)"
    )


# =============================================================================
# CORE WORKFLOW COMMANDS
# =============================================================================


@app.command()
def compare(
    datasets_path: Optional[str] = typer.Option(
        None, help="Path to datasets directory (overrides YAML)"
    ),
    output_dir: Optional[str] = typer.Option(
        None, help="Output directory (overrides YAML)"
    ),
    models: Optional[str] = typer.Option(
        None, help="Models to compare: llama,internvl (overrides YAML)"
    ),
    ground_truth_csv: Optional[str] = typer.Option(
        None, help="Ground truth CSV - enables evaluation mode"
    ),
    max_tokens: Optional[int] = typer.Option(
        None, help="Max tokens for model responses (overrides YAML)"
    ),
    quantization: Optional[bool] = typer.Option(
        None, help="Enable 8-bit quantization (overrides YAML)"
    ),
    trust_remote_code: Optional[bool] = typer.Option(
        None, help="Allow execution of remote code (overrides YAML)"
    ),
    llama_path: Optional[str] = typer.Option(None, help="Custom path to Llama model"),
    internvl_path: Optional[str] = typer.Option(
        None, help="Custom path to InternVL model"
    ),
    config_path: str = typer.Option(
        "model_comparison.yaml", help="Path to configuration file"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    debug: bool = typer.Option(False, "--debug", help="Debug output"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Minimal output"),
) -> None:
    """Run model comparison with smart ground truth detection.

    Auto-detects ground truth presence and switches between:
    - Standard comparison mode (no ground truth)
    - Evaluation mode (with ground truth CSV)
    """
    console.print("🔄 [bold blue]Model Comparison[/bold blue]")

    try:
        # Load YAML configuration
        config_file = Path(config_path)
        defaults = {}
        if config_file.exists():
            import yaml

            with config_file.open("r") as f:
                yaml_config = yaml.safe_load(f) or {}
                defaults = yaml_config.get("defaults", {})
        else:
            console.print(
                f"❌ Configuration file not found: {config_path}", style="bold red"
            )
            console.print(
                f"💡 Expected location: {config_file.absolute()}", style="yellow"
            )
            raise typer.Exit(1) from None

        # Apply CLI overrides (CLI parameters override YAML values)
        effective_datasets_path = datasets_path or defaults.get("datasets_path")
        effective_output_dir = output_dir or defaults.get("output_dir")
        effective_models = models or defaults.get("models", "llama,internvl")
        effective_max_tokens = max_tokens or defaults.get("max_tokens", 256)
        effective_quantization = (
            quantization
            if quantization is not None
            else defaults.get("quantization", True)
        )
        effective_trust_remote_code = (
            trust_remote_code
            if trust_remote_code is not None
            else defaults.get("trust_remote_code", True)
        )

        # Validate required paths
        if not effective_datasets_path:
            console.print("❌ No datasets_path configured", style="bold red")
            console.print(
                "💡 Set datasets_path in model_comparison.yaml OR use --datasets-path",
                style="yellow",
            )
            raise typer.Exit(1) from None

        if not effective_output_dir:
            console.print("❌ No output_dir configured", style="bold red")
            console.print(
                "💡 Set output_dir in model_comparison.yaml OR use --output-dir",
                style="yellow",
            )
            raise typer.Exit(1) from None

        models_list = [m.strip() for m in effective_models.split(",")]

        # Create and configure ConfigManager
        config = ConfigManager.get_global_instance(config_path)

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

        # Check if we have ground truth - if so, route to evaluation mode
        if ground_truth_csv:
            console.print(f"🎯 Ground truth detected: {ground_truth_csv}")
            console.print("📊 Running in evaluation mode...")

            # Import and use evaluation logic
            from ..evaluation import ExtractionEvaluator

            evaluator = ExtractionEvaluator(
                config_path=config_path,
                ground_truth_csv=ground_truth_csv,
                images_dir=effective_datasets_path,
                models=models_list,
                output_dir=effective_output_dir,
            )

            results = evaluator.run_evaluation()

            if results.get("success"):
                console.print(
                    "✅ Evaluation completed successfully!", style="bold green"
                )
            else:
                console.print(
                    f"❌ Evaluation failed: {results.get('error', 'Unknown error')}",
                    style="bold red",
                )
                raise typer.Exit(1) from None
        else:
            # Standard comparison mode
            console.print("📊 Running standard comparison mode...")

            # Import comparison logic
            from ..comparison.comparison_runner import ComparisonRunner

            # Run comparison using the existing comparison runner
            def run_comparison():
                config = ConfigManager.get_global_instance(config_path)

                # Apply CLI overrides to configuration
                if effective_datasets_path:
                    config.defaults.datasets_path = effective_datasets_path
                if effective_output_dir:
                    config.defaults.output_dir = effective_output_dir
                if effective_models:
                    config.defaults.models = effective_models
                if effective_max_tokens:
                    config.defaults.max_tokens = effective_max_tokens

                # Use comparison runner
                runner = ComparisonRunner(config)
                results = runner.run_comparison()

                return {
                    "success": True,
                    "results": results,
                    "config": config,
                    "execution_time": getattr(results, "total_execution_time", 0),
                }

            result = run_comparison()

            if result and result.get("success"):
                # Display comprehensive comparison summary like old model_comparison.py
                _display_comparison_summary(result["results"], result["config"])

                execution_time = result.get("execution_time", 0)
                if execution_time:
                    console.print(
                        f"⏱️  Total execution time: {execution_time:.1f}s",
                        style="bold green",
                    )
            else:
                error_msg = (
                    result.get("error", "Unknown error") if result else "Unknown error"
                )
                console.print(f"❌ Comparison failed: {error_msg}", style="bold red")
                raise typer.Exit(1) from None

    except Exception as e:
        if debug or verbose:
            import traceback

            console.print("🔍 Full traceback:", style="red")
            console.print(traceback.format_exc(), style="red")
        else:
            console.print(f"❌ Comparison failed: {e}", style="bold red")
        raise typer.Exit(1) from None


@app.command()
def visualize(
    input_file: Optional[str] = typer.Argument(
        None, help="Results JSON or ground truth CSV (auto-detects type)"
    ),
    ground_truth_csv: Optional[str] = typer.Option(
        None, help="Ground truth CSV for evaluation visualizations"
    ),
    output_dir: Optional[str] = typer.Option(
        None, help="Output directory (overrides YAML)"
    ),
    config_path: str = typer.Option(
        "model_comparison.yaml", help="Path to configuration file"
    ),
    open_browser: bool = typer.Option(
        False, "--browser/--no-browser", help="Auto-open visualizations"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    debug: bool = typer.Option(False, "--debug", help="Debug output"),
) -> None:
    """Generate visualizations with auto input type detection.

    Auto-detects input type:
    - No input: Uses latest results from output_dir
    - JSON file: Visualizes comparison results
    - CSV file: Generates ground truth analysis
    """
    console.print("📊 [bold blue]Visualization Generation[/bold blue]")

    try:
        # Load configuration
        config = ConfigManager.get_global_instance(config_path)

        # Apply CLI overrides
        if output_dir:
            config.defaults.output_dir = output_dir

        # Apply logging overrides
        if debug:
            config.defaults.debug_mode = True
            config.defaults.verbose_mode = True
        elif verbose:
            config.defaults.verbose_mode = True

        output_path = Path(config.defaults.output_dir)

        # Auto-detect input type and find file
        detected_file = None
        visualization_type = None

        if input_file:
            # User provided specific file
            input_path = Path(input_file)
            if not input_path.exists():
                console.print(
                    f"❌ Input file not found: {input_file}", style="bold red"
                )
                raise typer.Exit(1) from None

            detected_file = input_path

            # Determine type by extension
            if input_path.suffix.lower() == ".csv":
                visualization_type = "ground_truth"
                console.print(f"🎯 Detected ground truth CSV: {input_path.name}")
            elif input_path.suffix.lower() == ".json":
                visualization_type = "comparison_results"
                console.print(f"📊 Detected comparison results JSON: {input_path.name}")
            else:
                console.print(
                    f"❌ Unsupported file type: {input_path.suffix}", style="bold red"
                )
                console.print(
                    "💡 Supported: .json (comparison results), .csv (ground truth)",
                    style="yellow",
                )
                raise typer.Exit(1) from None

        elif ground_truth_csv:
            # User provided ground truth CSV via option
            gt_path = Path(ground_truth_csv)
            if not gt_path.exists():
                console.print(
                    f"❌ Ground truth CSV not found: {ground_truth_csv}",
                    style="bold red",
                )
                raise typer.Exit(1) from None

            detected_file = gt_path
            visualization_type = "ground_truth"
            console.print(f"🎯 Using ground truth CSV: {gt_path.name}")

        else:
            # Auto-detect latest results from output directory
            if not output_path.exists():
                console.print(
                    f"❌ Output directory not found: {output_path}", style="bold red"
                )
                console.print(
                    "💡 Run a comparison first or specify --output-dir", style="yellow"
                )
                raise typer.Exit(1) from None

            # Look for latest comparison results JSON
            json_files = list(output_path.glob("comparison_results_*.json"))
            if json_files:
                # Get most recent file
                detected_file = max(json_files, key=lambda p: p.stat().st_mtime)
                visualization_type = "comparison_results"
                console.print(f"📊 Auto-detected latest results: {detected_file.name}")
            else:
                console.print(
                    "❌ No comparison results found in output directory",
                    style="bold red",
                )
                console.print(f"💡 Searched in: {output_path}", style="yellow")
                console.print(
                    "💡 Run a comparison first or specify input file", style="yellow"
                )
                raise typer.Exit(1) from None

        # Generate visualizations based on detected type
        if visualization_type == "ground_truth":
            console.print("🎯 Generating ground truth analysis visualizations...")

            # Import and use evaluation visualization logic
            from ..analysis.dynamic_visualizations import DynamicModelVisualizer

            # Create visualization manager with ConfigManager
            viz_manager = DynamicModelVisualizer(
                config_manager=config, output_dir=str(output_path)
            )

            # For now, create dummy results structure since the correct method needs comparison_results
            # TODO: This needs to be implemented properly for ground truth analysis
            console.print(
                "⚠️  Ground truth visualization not fully implemented yet",
                style="yellow",
            )
            results = {
                "success": False,
                "error": "Ground truth analysis not implemented",
            }

            if results.get("success"):
                console.print(
                    "✅ Ground truth visualizations generated successfully!",
                    style="bold green",
                )

                # Show generated files
                if "files" in results:
                    console.print("\n📈 Generated visualizations:")
                    for file_path in results["files"]:
                        console.print(f"   📊 {Path(file_path).name}")

                    # Auto-open if requested
                    if open_browser and results["files"]:
                        import webbrowser

                        try:
                            webbrowser.open(
                                f"file://{Path(results['files'][0]).absolute()}"
                            )
                            console.print("🌐 Opened visualizations in browser")
                        except Exception as e:
                            console.print(
                                f"⚠️  Could not auto-open browser: {e}", style="yellow"
                            )
            else:
                console.print(
                    f"❌ Visualization generation failed: {results.get('error', 'Unknown error')}",
                    style="bold red",
                )
                raise typer.Exit(1) from None

        elif visualization_type == "comparison_results":
            console.print("📊 Generating comparison results visualizations...")

            # Import visualization logic
            import json

            from ..analysis.dynamic_visualizations import DynamicModelVisualizer

            # Load comparison results from JSON file
            with Path(detected_file).open("r") as f:
                comparison_results = json.load(f)

            # Create visualization manager with ConfigManager
            viz_manager = DynamicModelVisualizer(
                config_manager=config, output_dir=str(output_path)
            )

            # Generate all visualizations using the correct method
            viz_files = viz_manager.generate_all_visualizations(comparison_results)

            # Create results structure that matches expected format
            results = {"success": True if viz_files else False, "files": viz_files}

            if results.get("success"):
                console.print(
                    "✅ Comparison visualizations generated successfully!",
                    style="bold green",
                )

                # Show generated files
                if "files" in results:
                    console.print("\n📈 Generated visualizations:")
                    for file_path in results["files"]:
                        console.print(f"   📊 {Path(file_path).name}")

                    # Auto-open if requested
                    if open_browser and results["files"]:
                        import webbrowser

                        try:
                            webbrowser.open(
                                f"file://{Path(results['files'][0]).absolute()}"
                            )
                            console.print("🌐 Opened visualizations in browser")
                        except Exception as e:
                            console.print(
                                f"⚠️  Could not auto-open browser: {e}", style="yellow"
                            )
            else:
                console.print(
                    f"❌ Visualization generation failed: {results.get('error', 'Unknown error')}",
                    style="bold red",
                )
                raise typer.Exit(1) from None

    except ConfigurationError as e:
        console.print(f"❌ Configuration error: {e.message}", style="bold red")
        if verbose and hasattr(e, "details") and e.details:
            console.print(f"💡 Details: {e.details}", style="yellow")
        raise typer.Exit(1) from None
    except Exception as e:
        console.print(f"❌ Visualization generation failed: {e}", style="bold red")
        if debug or verbose:
            import traceback

            console.print("🔍 Full traceback:", style="red")
            console.print(traceback.format_exc(), style="red")
        raise typer.Exit(1) from None


@app.command()
def extract(
    image_path: str = typer.Argument(..., help="Path to document image"),
    model: Optional[str] = typer.Option(
        None, help="Model override: llama, internvl (overrides YAML)"
    ),
    output_format: Optional[str] = typer.Option(
        None, help="Output format: table, json, yaml (overrides YAML)"
    ),
    config_path: str = typer.Option(
        "model_comparison.yaml", help="Path to configuration file"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    debug: bool = typer.Option(False, "--debug", help="Debug output"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Minimal output"),
) -> None:
    """Extract data from single document image."""
    console.print("🔍 [bold blue]Single Image Extraction[/bold blue]")

    try:
        # Import required dependencies
        from ..config import ConfigManager
        from ..extraction.extraction_manager import SimpleExtractionManager
        from ..utils.path_resolver import PathResolver

        # Load configuration from YAML
        config = ConfigManager.get_global_instance(config_path)
        path_resolver = PathResolver(config)

        # Resolve image path using utility
        try:
            resolved_image_path = path_resolver.resolve_input_path(image_path)
        except ValueError as e:
            console.print(f"❌ {e}", style="bold red")
            raise typer.Exit(1) from None

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

        # Apply CLI overrides
        if model:
            config.set_model_type(model)
        if output_format:
            config.set_output_format(output_format)

        # Validate that a model has been selected
        config.validate_model_selected()

        # Process document
        console.print(f"🔍 Processing document: {Path(resolved_image_path).name}")

        with console.status(
            f"[bold green]Processing with {config.current_model_type}..."
        ):
            manager = SimpleExtractionManager(config)
            result = manager.process_document(resolved_image_path)

        # Display results based on configuration
        if config.output_format == "table":
            _display_table(result.extracted_fields)
        elif config.output_format == "json":
            print(json.dumps(result.extracted_fields, indent=2))
        elif config.output_format == "yaml":
            _display_clean_format(result.extracted_fields)
        else:
            # Default to clean format
            _display_clean_format(result.extracted_fields)

        # Show processing info
        console.print(f"⏱️  Processing Time: {result.processing_time:.2f}s")
        console.print(f"🎯 Model Confidence: {result.model_confidence:.3f}")
        console.print(f"🔧 Extraction Method: {result.extraction_method}")

    except ConfigurationError as e:
        console.print(f"❌ Configuration error: {e.message}", style="bold red")
        if verbose and hasattr(e, "details") and e.details:
            console.print(f"💡 Details: {e.details}", style="yellow")
        raise typer.Exit(1) from None
    except ModelLoadError as e:
        console.print(f"❌ Model loading error: {e.message}", style="bold red")
        if verbose and hasattr(e, "details") and e.details:
            console.print(f"💡 Details: {e.details}", style="yellow")
        raise typer.Exit(1) from None
    except ImageProcessingError as e:
        console.print(f"❌ Image processing error: {e.message}", style="bold red")
        if verbose and hasattr(e, "details") and e.details:
            console.print(f"💡 Details: {e.details}", style="yellow")
        raise typer.Exit(1) from None
    except VisionProcessorError as e:
        console.print(f"❌ {e.message}", style="bold red")
        if verbose and hasattr(e, "details") and e.details:
            console.print(f"💡 Details: {e.details}", style="yellow")
        raise typer.Exit(1) from None
    except ImportError as e:
        console.print(f"❌ Import error: {e}", style="bold red")
        console.print("💡 Please ensure all dependencies are installed", style="yellow")
        raise typer.Exit(1) from None
    except Exception as e:
        console.print(f"❌ Unexpected error: {e}", style="bold red")
        if debug or verbose:
            import traceback

            console.print("🔍 Full traceback:", style="red")
            console.print(traceback.format_exc(), style="red")
        raise typer.Exit(1) from None


@app.command()
def batch(
    input_dir: str = typer.Argument(..., help="Directory containing images"),
    output_dir: Optional[str] = typer.Option(
        None, help="Output directory (overrides YAML)"
    ),
    model: Optional[str] = typer.Option(
        None, help="Model override: llama, internvl (overrides YAML)"
    ),
    config_path: str = typer.Option(
        "model_comparison.yaml", help="Path to configuration file"
    ),
    max_images: Optional[int] = typer.Option(None, help="Maximum images to process"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    debug: bool = typer.Option(False, "--debug", help="Debug output"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Minimal output"),
) -> None:
    """Batch process directory of images."""
    console.print("📁 [bold blue]Batch Processing[/bold blue]")

    try:
        # Import required dependencies
        from ..config import ConfigManager
        from ..extraction.extraction_manager import SimpleExtractionManager
        from ..utils.path_resolver import PathResolver

        # Load configuration from YAML
        config = ConfigManager.get_global_instance(config_path)
        path_resolver = PathResolver(config)

        # Validate input directory
        input_path = Path(input_dir)
        if not input_path.exists():
            console.print(
                f"❌ Input directory does not exist: {input_dir}", style="bold red"
            )
            raise typer.Exit(1) from None

        if not input_path.is_dir():
            console.print(
                f"❌ Input path is not a directory: {input_dir}", style="bold red"
            )
            raise typer.Exit(1) from None

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

        # Apply CLI overrides
        if model:
            config.set_model_type(model)

        if output_dir:
            config.defaults.output_dir = output_dir

        # Validate that a model has been selected
        config.validate_model_selected()

        # Find image files
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        image_files = [
            f
            for f in input_path.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ]

        if not image_files:
            console.print(f"❌ No image files found in: {input_dir}", style="bold red")
            console.print(
                f"💡 Supported formats: {', '.join(sorted(image_extensions))}",
                style="yellow",
            )
            raise typer.Exit(1) from None

        # Apply max_images limit
        if max_images and max_images < len(image_files):
            image_files = image_files[:max_images]
            console.print(
                f"📊 Processing first {max_images} images out of {len(image_files)} found"
            )

        console.print(f"📊 Found {len(image_files)} images to process")
        console.print(f"🤖 Using model: {config.current_model_type}")

        # Setup output directory
        output_path = Path(config.defaults.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Initialize extraction manager
        manager = SimpleExtractionManager(config)

        # Process images
        results = []
        failed_images = []

        with console.status("[bold green]Processing images...") as status:
            for i, image_file in enumerate(image_files, 1):
                try:
                    status.update(
                        f"[bold green]Processing {i}/{len(image_files)}: {image_file.name}"
                    )

                    result = manager.process_document(str(image_file))

                    # Store result
                    image_result = {
                        "image_path": str(image_file),
                        "image_name": image_file.name,
                        "extracted_fields": result.extracted_fields,
                        "processing_time": result.processing_time,
                        "model_confidence": result.model_confidence,
                        "extraction_method": result.extraction_method,
                        "success": True,
                    }
                    results.append(image_result)

                    if not quiet:
                        console.print(
                            f"✅ {image_file.name} ({result.processing_time:.2f}s)"
                        )

                except Exception as e:
                    failed_images.append({"image": str(image_file), "error": str(e)})
                    if not quiet:
                        console.print(f"❌ {image_file.name}: {e}")

                    if debug:
                        import traceback

                        console.print(traceback.format_exc(), style="red")

        # Save batch results
        batch_results = {
            "model": config.current_model_type,
            "total_images": len(image_files),
            "successful_extractions": len(results),
            "failed_extractions": len(failed_images),
            "processing_time": sum(r["processing_time"] for r in results),
            "results": results,
            "failed_images": failed_images,
            "config": {
                "input_dir": str(input_path),
                "output_dir": str(output_path),
                "max_images": max_images,
            },
        }

        # Save results to JSON file
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = (
            output_path / f"batch_results_{config.current_model_type}_{timestamp}.json"
        )

        with output_file.open("w") as f:
            json.dump(batch_results, f, indent=2, default=str)

        # Summary
        console.print("\n📊 [bold blue]Batch Processing Complete[/bold blue]")
        console.print(f"✅ Successful: {len(results)}/{len(image_files)} images")
        console.print(f"❌ Failed: {len(failed_images)} images")
        console.print(f"⏱️  Total time: {batch_results['processing_time']:.2f}s")
        console.print(f"💾 Results saved: {output_file}")

        if failed_images and not quiet:
            console.print("\n❌ [bold red]Failed Images:[/bold red]")
            for failed in failed_images[:5]:  # Show first 5 failures
                console.print(f"   {Path(failed['image']).name}: {failed['error']}")
            if len(failed_images) > 5:
                console.print(f"   ... and {len(failed_images) - 5} more")

    except ConfigurationError as e:
        console.print(f"❌ Configuration error: {e.message}", style="bold red")
        if verbose and hasattr(e, "details") and e.details:
            console.print(f"💡 Details: {e.details}", style="yellow")
        raise typer.Exit(1) from None
    except Exception as e:
        console.print(f"❌ Batch processing failed: {e}", style="bold red")
        if debug or verbose:
            import traceback

            console.print("🔍 Full traceback:", style="red")
            console.print(traceback.format_exc(), style="red")
        raise typer.Exit(1) from None


# =============================================================================
# EVALUATION WORKFLOW COMMANDS
# =============================================================================


@app.command()
def evaluate(
    ground_truth_csv: str = typer.Argument(..., help="Path to ground truth CSV"),
    images_dir: Optional[str] = typer.Option(
        None, help="Images directory (overrides YAML)"
    ),
    models: Optional[str] = typer.Option(
        None, help="Models to evaluate: llama,internvl (overrides YAML)"
    ),
    output_dir: Optional[str] = typer.Option(
        None, help="Output directory (overrides YAML)"
    ),
    config_path: str = typer.Option(
        "model_comparison.yaml", help="Path to configuration file"
    ),
    visualizations: bool = typer.Option(
        True, "--visualizations/--no-visualizations", help="Generate visualizations"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    debug: bool = typer.Option(False, "--debug", help="Debug output"),
) -> None:
    """Evaluate models against ground truth data."""
    console.print("🎯 [bold blue]Ground Truth Evaluation[/bold blue]")

    try:
        # Validate ground truth CSV exists
        gt_path = Path(ground_truth_csv)
        if not gt_path.exists():
            console.print(
                f"❌ Ground truth CSV not found: {ground_truth_csv}", style="bold red"
            )
            raise typer.Exit(1) from None

        # Load configuration
        config = ConfigManager.get_global_instance(config_path)

        # Apply CLI overrides
        effective_images_dir = images_dir or config.defaults.datasets_path
        effective_models = models or config.defaults.models
        effective_output_dir = output_dir or config.defaults.output_dir

        # Validate required paths
        if not effective_images_dir:
            console.print("❌ No images directory configured", style="bold red")
            console.print(
                "💡 Set datasets_path in model_comparison.yaml OR use --images-dir",
                style="yellow",
            )
            raise typer.Exit(1) from None

        if not effective_output_dir:
            console.print("❌ No output directory configured", style="bold red")
            console.print(
                "💡 Set output_dir in model_comparison.yaml OR use --output-dir",
                style="yellow",
            )
            raise typer.Exit(1) from None

        # Validate images directory exists
        images_path = Path(effective_images_dir)
        if not images_path.exists():
            console.print(
                f"❌ Images directory not found: {effective_images_dir}",
                style="bold red",
            )
            raise typer.Exit(1) from None

        # Parse models list
        models_list = [m.strip() for m in effective_models.split(",")]

        # Apply logging overrides
        if debug:
            config.defaults.debug_mode = True
            config.defaults.verbose_mode = True
        elif verbose:
            config.defaults.verbose_mode = True

        console.print(f"📊 Ground truth CSV: {gt_path.name}")
        console.print(f"📁 Images directory: {effective_images_dir}")
        console.print(f"🤖 Models to evaluate: {', '.join(models_list)}")
        console.print(f"📁 Output directory: {effective_output_dir}")

        # Import evaluation logic
        from ..evaluation import ExtractionEvaluator

        # Create evaluator
        evaluator = ExtractionEvaluator(
            config_path=config_path,
            ground_truth_csv=ground_truth_csv,
            images_dir=effective_images_dir,
            models=models_list,
            output_dir=effective_output_dir,
        )

        # Run evaluation
        console.print("\n🔄 Starting evaluation...")
        with console.status("[bold green]Running model evaluation..."):
            results = evaluator.run_evaluation()

        if results.get("success"):
            console.print("✅ Evaluation completed successfully!", style="bold green")

            # Show evaluation summary
            if "summary" in results:
                summary = results["summary"]
                console.print("\n📊 [bold blue]Evaluation Summary[/bold blue]")
                console.print(
                    f"📈 Total images processed: {summary.get('total_images', 'N/A')}"
                )
                console.print(
                    f"⏱️  Total processing time: {summary.get('total_time', 'N/A'):.2f}s"
                )

                # Show model performance
                if "model_performance" in summary:
                    console.print("\n🏆 [bold blue]Model Performance[/bold blue]")
                    for model, perf in summary["model_performance"].items():
                        f1_score = perf.get("f1_score", "N/A")
                        console.print(f"   🤖 {model}: F1 Score = {f1_score}")

            # Show output files
            if "output_files" in results:
                console.print("\n💾 [bold blue]Generated Files[/bold blue]")
                for file_path in results["output_files"]:
                    console.print(f"   📄 {Path(file_path).name}")

            # Generate visualizations if requested
            if visualizations:
                console.print("\n📊 Generating visualizations...")
                try:
                    from ..analysis.dynamic_visualizations import (
                        DynamicModelVisualizer,
                    )

                    viz_manager = DynamicModelVisualizer(
                        config_manager=config, output_dir=effective_output_dir
                    )

                    # Ground truth analysis not implemented yet
                    console.print(
                        "⚠️  Ground truth visualization not fully implemented yet",
                        style="yellow",
                    )
                    viz_results = {
                        "success": False,
                        "error": "Ground truth analysis not implemented",
                    }

                    if viz_results.get("success"):
                        console.print(
                            "✅ Visualizations generated successfully!",
                            style="bold green",
                        )
                        if "files" in viz_results:
                            console.print("📈 Visualization files:")
                            for viz_file in viz_results["files"]:
                                console.print(f"   📊 {Path(viz_file).name}")
                    else:
                        console.print(
                            f"⚠️  Visualization generation failed: {viz_results.get('error', 'Unknown error')}",
                            style="yellow",
                        )

                except Exception as e:
                    console.print(
                        f"⚠️  Visualization generation failed: {e}", style="yellow"
                    )
                    if debug:
                        import traceback

                        console.print(traceback.format_exc(), style="red")
        else:
            console.print(
                f"❌ Evaluation failed: {results.get('error', 'Unknown error')}",
                style="bold red",
            )
            raise typer.Exit(1) from None

    except ConfigurationError as e:
        console.print(f"❌ Configuration error: {e.message}", style="bold red")
        if verbose and hasattr(e, "details") and e.details:
            console.print(f"💡 Details: {e.details}", style="yellow")
        raise typer.Exit(1) from None
    except Exception as e:
        console.print(f"❌ Evaluation failed: {e}", style="bold red")
        if debug or verbose:
            import traceback

            console.print("🔍 Full traceback:", style="red")
            console.print(traceback.format_exc(), style="red")
        raise typer.Exit(1) from None


@app.command()
def benchmark(
    images_dir: str = typer.Argument(..., help="Directory with benchmark images"),
    models: Optional[str] = typer.Option(
        None, help="Models to benchmark: llama,internvl (overrides YAML)"
    ),
    output_dir: Optional[str] = typer.Option(
        None, help="Output directory (overrides YAML)"
    ),
    config_path: str = typer.Option(
        "model_comparison.yaml", help="Path to configuration file"
    ),
    iterations: int = typer.Option(1, help="Number of benchmark iterations"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    debug: bool = typer.Option(False, "--debug", help="Debug output"),
) -> None:
    """Performance benchmark models on image set."""
    console.print("⚡ [bold blue]Performance Benchmark[/bold blue]")

    try:
        # Validate images directory
        images_path = Path(images_dir)
        if not images_path.exists():
            console.print(
                f"❌ Images directory not found: {images_dir}", style="bold red"
            )
            raise typer.Exit(1) from None

        if not images_path.is_dir():
            console.print(f"❌ Path is not a directory: {images_dir}", style="bold red")
            raise typer.Exit(1) from None

        # Load configuration
        config = ConfigManager.get_global_instance(config_path)

        # Apply CLI overrides
        effective_models = models or config.defaults.models
        effective_output_dir = output_dir or config.defaults.output_dir

        # Validate required settings
        if not effective_models:
            console.print("❌ No models configured", style="bold red")
            console.print(
                "💡 Set models in model_comparison.yaml OR use --models", style="yellow"
            )
            raise typer.Exit(1) from None

        if not effective_output_dir:
            console.print("❌ No output directory configured", style="bold red")
            console.print(
                "💡 Set output_dir in model_comparison.yaml OR use --output-dir",
                style="yellow",
            )
            raise typer.Exit(1) from None

        # Parse models list
        models_list = [m.strip() for m in effective_models.split(",")]

        # Apply logging overrides
        if debug:
            config.defaults.debug_mode = True
            config.defaults.verbose_mode = True
        elif verbose:
            config.defaults.verbose_mode = True

        # Find benchmark images
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        image_files = [
            f
            for f in images_path.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ]

        if not image_files:
            console.print(f"❌ No image files found in: {images_dir}", style="bold red")
            console.print(
                f"💡 Supported formats: {', '.join(sorted(image_extensions))}",
                style="yellow",
            )
            raise typer.Exit(1) from None

        console.print("📊 Benchmark configuration:")
        console.print(
            f"   📁 Images directory: {images_dir} ({len(image_files)} images)"
        )
        console.print(f"   🤖 Models: {', '.join(models_list)}")
        console.print(f"   🔄 Iterations: {iterations}")
        console.print(f"   📁 Output directory: {effective_output_dir}")

        # Import benchmark logic - we'll implement a simple benchmark
        import statistics

        from ..extraction.extraction_manager import SimpleExtractionManager

        # Setup output directory
        output_path = Path(effective_output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        benchmark_results = {}
        total_start_time = time.time()

        for model_name in models_list:
            console.print(f"\n🤖 [bold blue]Benchmarking {model_name}[/bold blue]")

            # Configure for this model
            config.set_model_type(model_name)

            model_results = []
            model_times = []

            for iteration in range(iterations):
                console.print(f"   🔄 Iteration {iteration + 1}/{iterations}")

                iteration_start = time.time()
                successful_extractions = 0
                failed_extractions = 0
                processing_times = []

                # Process each image
                with console.status(
                    f"[bold green]Processing {len(image_files)} images..."
                ) as status:
                    manager = SimpleExtractionManager(config)

                    for i, image_file in enumerate(image_files):
                        try:
                            status.update(
                                f"[bold green]Processing {i + 1}/{len(image_files)}: {image_file.name}"
                            )

                            result = manager.process_document(str(image_file))
                            processing_times.append(result.processing_time)
                            successful_extractions += 1

                        except Exception as e:
                            failed_extractions += 1
                            if debug:
                                console.print(f"❌ {image_file.name}: {e}")

                iteration_time = time.time() - iteration_start
                model_times.append(iteration_time)

                # Calculate iteration stats
                iteration_result = {
                    "iteration": iteration + 1,
                    "total_time": iteration_time,
                    "successful_extractions": successful_extractions,
                    "failed_extractions": failed_extractions,
                    "success_rate": successful_extractions / len(image_files)
                    if image_files
                    else 0,
                    "avg_processing_time": statistics.mean(processing_times)
                    if processing_times
                    else 0,
                    "min_processing_time": min(processing_times)
                    if processing_times
                    else 0,
                    "max_processing_time": max(processing_times)
                    if processing_times
                    else 0,
                }
                model_results.append(iteration_result)

                console.print(
                    f"   ✅ Success: {successful_extractions}/{len(image_files)} ({iteration_result['success_rate']:.1%})"
                )
                console.print(
                    f"   ⏱️  Time: {iteration_time:.2f}s (avg: {iteration_result['avg_processing_time']:.2f}s per image)"
                )

            # Calculate model summary
            model_summary = {
                "model": model_name,
                "iterations": iterations,
                "total_images_per_iteration": len(image_files),
                "avg_iteration_time": statistics.mean(model_times),
                "min_iteration_time": min(model_times),
                "max_iteration_time": max(model_times),
                "avg_success_rate": statistics.mean(
                    [r["success_rate"] for r in model_results]
                ),
                "avg_processing_time_per_image": statistics.mean(
                    [r["avg_processing_time"] for r in model_results]
                ),
                "iterations_detail": model_results,
            }

            benchmark_results[model_name] = model_summary

            console.print(f"\n📊 [bold cyan]{model_name} Summary[/bold cyan]:")
            console.print(
                f"   ⏱️  Average iteration time: {model_summary['avg_iteration_time']:.2f}s"
            )
            console.print(
                f"   📈 Average success rate: {model_summary['avg_success_rate']:.1%}"
            )
            console.print(
                f"   🎯 Average time per image: {model_summary['avg_processing_time_per_image']:.2f}s"
            )

        total_benchmark_time = time.time() - total_start_time

        # Create final benchmark report
        benchmark_report = {
            "benchmark_config": {
                "images_dir": str(images_path),
                "total_images": len(image_files),
                "models": models_list,
                "iterations": iterations,
                "total_benchmark_time": total_benchmark_time,
            },
            "results": benchmark_results,
            "summary": {
                "best_performance": None,
                "fastest_model": None,
                "most_accurate_model": None,
            },
        }

        # Determine best performers
        if benchmark_results:
            # Find fastest model (lowest avg processing time per image)
            fastest = min(
                benchmark_results.items(),
                key=lambda x: x[1]["avg_processing_time_per_image"],
            )
            benchmark_report["summary"]["fastest_model"] = {
                "model": fastest[0],
                "avg_time_per_image": fastest[1]["avg_processing_time_per_image"],
            }

            # Find most accurate model (highest avg success rate)
            most_accurate = max(
                benchmark_results.items(), key=lambda x: x[1]["avg_success_rate"]
            )
            benchmark_report["summary"]["most_accurate_model"] = {
                "model": most_accurate[0],
                "avg_success_rate": most_accurate[1]["avg_success_rate"],
            }

        # Save benchmark results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = output_path / f"benchmark_results_{timestamp}.json"

        with output_file.open("w") as f:
            json.dump(benchmark_report, f, indent=2, default=str)

        # Final summary
        console.print("\n🏁 [bold blue]Benchmark Complete[/bold blue]")
        console.print(f"⏱️  Total benchmark time: {total_benchmark_time:.2f}s")
        console.print(f"💾 Results saved: {output_file}")

        if benchmark_report["summary"]["fastest_model"]:
            fastest = benchmark_report["summary"]["fastest_model"]
            console.print(
                f"🏃 Fastest model: {fastest['model']} ({fastest['avg_time_per_image']:.2f}s per image)"
            )

        if benchmark_report["summary"]["most_accurate_model"]:
            accurate = benchmark_report["summary"]["most_accurate_model"]
            console.print(
                f"🎯 Most accurate model: {accurate['model']} ({accurate['avg_success_rate']:.1%} success rate)"
            )

    except ConfigurationError as e:
        console.print(f"❌ Configuration error: {e.message}", style="bold red")
        if verbose and hasattr(e, "details") and e.details:
            console.print(f"💡 Details: {e.details}", style="yellow")
        raise typer.Exit(1) from None
    except Exception as e:
        console.print(f"❌ Benchmark failed: {e}", style="bold red")
        if debug or verbose:
            import traceback

            console.print("🔍 Full traceback:", style="red")
            console.print(traceback.format_exc(), style="red")
        raise typer.Exit(1) from None


@app.command("validate-ground-truth")
def validate_ground_truth(
    ground_truth_csv: str = typer.Argument(..., help="Path to ground truth CSV"),
    images_dir: Optional[str] = typer.Option(
        None, help="Images directory (overrides YAML)"
    ),
    config_path: str = typer.Option(
        "model_comparison.yaml", help="Path to configuration file"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    debug: bool = typer.Option(False, "--debug", help="Debug output"),
) -> None:
    """Validate ground truth CSV against available images."""
    console.print("✅ [bold blue]Ground Truth Validation[/bold blue]")

    try:
        import csv

        # Validate ground truth CSV exists
        gt_path = Path(ground_truth_csv)
        if not gt_path.exists():
            console.print(
                f"❌ Ground truth CSV not found: {ground_truth_csv}", style="bold red"
            )
            raise typer.Exit(1) from None

        # Load configuration
        config = ConfigManager.get_global_instance(config_path)

        # Apply CLI overrides
        effective_images_dir = images_dir or config.defaults.datasets_path

        # Validate images directory
        if not effective_images_dir:
            console.print("❌ No images directory configured", style="bold red")
            console.print(
                "💡 Set datasets_path in model_comparison.yaml OR use --images-dir",
                style="yellow",
            )
            raise typer.Exit(1) from None

        images_path = Path(effective_images_dir)
        if not images_path.exists():
            console.print(
                f"❌ Images directory not found: {effective_images_dir}",
                style="bold red",
            )
            raise typer.Exit(1) from None

        if not images_path.is_dir():
            console.print(
                f"❌ Images path is not a directory: {effective_images_dir}",
                style="bold red",
            )
            raise typer.Exit(1) from None

        # Apply logging overrides
        if debug:
            config.defaults.debug_mode = True
            config.defaults.verbose_mode = True
        elif verbose:
            config.defaults.verbose_mode = True

        console.print(f"📊 Ground truth CSV: {gt_path.name}")
        console.print(f"📁 Images directory: {effective_images_dir}")

        # Read ground truth CSV
        console.print("\n🔍 Reading ground truth CSV...")
        gt_entries = []
        csv_headers = []

        try:
            with gt_path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                csv_headers = reader.fieldnames or []
                gt_entries = list(reader)
        except Exception as e:
            console.print(f"❌ Failed to read CSV: {e}", style="bold red")
            raise typer.Exit(1) from None

        console.print(f"📝 CSV contains {len(gt_entries)} entries")
        console.print(f"📋 Headers: {', '.join(csv_headers)}")

        # Check for required columns
        required_columns = ["image_name", "image_path"]  # Basic required columns
        missing_columns = [col for col in required_columns if col not in csv_headers]

        if missing_columns:
            console.print(
                f"❌ Missing required columns: {', '.join(missing_columns)}",
                style="bold red",
            )
            console.print(
                f"💡 Required columns: {', '.join(required_columns)}", style="yellow"
            )
            raise typer.Exit(1) from None

        # Find available image files
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        available_images = {}

        for img_file in images_path.iterdir():
            if img_file.is_file() and img_file.suffix.lower() in image_extensions:
                available_images[img_file.name] = img_file

        console.print(f"🖼️  Found {len(available_images)} image files in directory")

        # Validate each ground truth entry
        console.print("\n🔍 Validating ground truth entries...")

        valid_entries = []
        missing_images = []
        path_mismatches = []
        invalid_entries = []

        for i, entry in enumerate(gt_entries, 1):
            try:
                image_name = entry.get("image_name", "").strip()
                image_path = entry.get("image_path", "").strip()

                if not image_name:
                    invalid_entries.append({"row": i, "issue": "Empty image_name"})
                    continue

                # Check if image file exists
                if image_name not in available_images:
                    missing_images.append(
                        {
                            "row": i,
                            "image_name": image_name,
                            "expected_path": image_path,
                        }
                    )
                    continue

                # Check if path matches (if provided)
                if image_path:
                    expected_full_path = images_path / image_name
                    actual_path = Path(image_path)

                    # Check if paths match (allow for different absolute/relative formats)
                    if actual_path.name != image_name:
                        path_mismatches.append(
                            {
                                "row": i,
                                "image_name": image_name,
                                "csv_path": image_path,
                                "expected_name": image_name,
                            }
                        )
                        continue

                valid_entries.append(entry)

            except Exception as e:
                invalid_entries.append(
                    {"row": i, "issue": f"Error processing entry: {e}"}
                )

        # Validation summary
        console.print("\n📊 [bold blue]Validation Summary[/bold blue]")
        console.print(f"✅ Valid entries: {len(valid_entries)}/{len(gt_entries)}")
        console.print(f"❌ Invalid entries: {len(invalid_entries)}")
        console.print(f"🖼️  Missing images: {len(missing_images)}")
        console.print(f"📁 Path mismatches: {len(path_mismatches)}")

        # Show details for issues
        if invalid_entries:
            console.print("\n❌ [bold red]Invalid Entries[/bold red]")
            for inv in invalid_entries[:5]:  # Show first 5
                console.print(f"   Row {inv['row']}: {inv['issue']}")
            if len(invalid_entries) > 5:
                console.print(f"   ... and {len(invalid_entries) - 5} more")

        if missing_images:
            console.print("\n🖼️  [bold red]Missing Images[/bold red]")
            for missing in missing_images[:5]:  # Show first 5
                console.print(f"   Row {missing['row']}: {missing['image_name']}")
            if len(missing_images) > 5:
                console.print(f"   ... and {len(missing_images) - 5} more")

        if path_mismatches:
            console.print("\n📁 [bold yellow]Path Mismatches[/bold yellow]")
            for mismatch in path_mismatches[:5]:  # Show first 5
                console.print(
                    f"   Row {mismatch['row']}: CSV has '{mismatch['csv_path']}', expected '{mismatch['image_name']}'"
                )
            if len(path_mismatches) > 5:
                console.print(f"   ... and {len(path_mismatches) - 5} more")

        # Check for extra images (images not in CSV)
        csv_image_names = {entry.get("image_name", "").strip() for entry in gt_entries}
        extra_images = [
            img for img in available_images.keys() if img not in csv_image_names
        ]

        if extra_images:
            console.print("\n📸 [bold yellow]Extra Images (not in CSV)[/bold yellow]")
            console.print(
                f"Found {len(extra_images)} image files not referenced in CSV:"
            )
            for extra in extra_images[:5]:  # Show first 5
                console.print(f"   {extra}")
            if len(extra_images) > 5:
                console.print(f"   ... and {len(extra_images) - 5} more")

        # Field analysis
        if verbose and gt_entries:
            console.print("\n📋 [bold blue]Field Analysis[/bold blue]")
            field_stats = {}

            for header in csv_headers:
                if header in ["image_name", "image_path"]:
                    continue  # Skip system fields

                non_empty_count = sum(
                    1 for entry in gt_entries if entry.get(header, "").strip()
                )
                field_stats[header] = {
                    "total": len(gt_entries),
                    "non_empty": non_empty_count,
                    "fill_rate": non_empty_count / len(gt_entries) if gt_entries else 0,
                }

            for field, stats in field_stats.items():
                console.print(
                    f"   {field}: {stats['non_empty']}/{stats['total']} filled ({stats['fill_rate']:.1%})"
                )

        # Overall validation result
        total_issues = len(invalid_entries) + len(missing_images) + len(path_mismatches)

        if total_issues == 0:
            console.print(
                "\n🎉 [bold green]Ground Truth Validation PASSED[/bold green]"
            )
            console.print("✅ All entries are valid and all images are available")
        else:
            console.print(
                f"\n⚠️  [bold yellow]Ground Truth Validation completed with {total_issues} issues[/bold yellow]"
            )
            console.print(
                "💡 Review the issues above and fix the ground truth CSV or add missing images"
            )

            if total_issues > len(gt_entries) * 0.1:  # More than 10% issues
                console.print(
                    "❌ High number of issues detected - consider reviewing the entire dataset",
                    style="bold red",
                )

        # Recommendations
        console.print("\n💡 [bold blue]Recommendations[/bold blue]")

        if missing_images:
            console.print("   📁 Add missing images to the images directory")

        if path_mismatches:
            console.print(
                "   🔧 Update image_path column in CSV to match actual file names"
            )

        if invalid_entries:
            console.print("   📝 Fix invalid entries in the CSV file")

        if extra_images:
            console.print(
                "   📸 Consider adding extra images to ground truth CSV or remove unused files"
            )

        console.print(
            f"\n✅ Validation complete - {len(valid_entries)}/{len(gt_entries)} entries ready for evaluation"
        )

    except Exception as e:
        console.print(f"❌ Validation failed: {e}", style="bold red")
        if debug or verbose:
            import traceback

            console.print("🔍 Full traceback:", style="red")
            console.print(traceback.format_exc(), style="red")
        raise typer.Exit(1) from None


# =============================================================================
# SYSTEM MANAGEMENT COMMANDS
# =============================================================================


@app.command()
def check(
    datasets_path: Optional[str] = typer.Option(
        None, help="Datasets path to check (overrides YAML)"
    ),
    config_path: str = typer.Option(
        "model_comparison.yaml", help="Path to configuration file"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    debug: bool = typer.Option(False, "--debug", help="Debug output"),
) -> None:
    """Validate system environment and configuration."""
    console.print("🔍 [bold blue]Environment Check[/bold blue]")

    try:
        import importlib.util

        import torch

        # Load configuration
        config = ConfigManager.get_global_instance(config_path)

        # Apply CLI overrides
        effective_datasets_path = datasets_path or config.defaults.datasets_path

        # Apply logging overrides
        if debug:
            config.defaults.debug_mode = True
            config.defaults.verbose_mode = True
        elif verbose:
            config.defaults.verbose_mode = True

        # Check CUDA availability
        console.print("\n🖥️  [bold blue]Hardware Check[/bold blue]")
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_count = torch.cuda.device_count()
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3

            console.print(f"✅ CUDA available: {gpu_name}")
            console.print(f"✅ GPU count: {gpu_count}")
            console.print(f"✅ GPU memory: {memory_gb:.1f}GB")

            if memory_gb < 16:
                console.print(
                    "⚠️  GPU memory < 16GB - consider enabling quantization",
                    style="yellow",
                )

            if gpu_count > 1:
                console.print(
                    "💡 Multi-GPU detected - optimized for development", style="cyan"
                )
            else:
                console.print("💡 Single GPU - optimized for production", style="cyan")
        else:
            console.print("⚠️  CUDA not available - using CPU mode", style="yellow")
            console.print("💡 Performance will be significantly slower", style="yellow")

        # Check Python dependencies
        console.print("\n📦 [bold blue]Dependencies Check[/bold blue]")

        required_deps = {
            "torch": "PyTorch",
            "transformers": "Transformers",
            "typer": "Typer CLI",
            "yaml": "PyYAML",
            "pandas": "Pandas",
            "seaborn": "Seaborn",
            "sklearn": "Scikit-learn",
            "PIL": "Pillow",
            "rich": "Rich",
        }

        missing_deps = []

        for dep, name in required_deps.items():
            if importlib.util.find_spec(dep) is None:
                missing_deps.append(name)
                console.print(f"❌ {name}: Not installed", style="red")
            else:
                try:
                    if dep == "torch":
                        import torch

                        console.print(f"✅ {name}: {torch.__version__}")
                    elif dep == "transformers":
                        import transformers

                        console.print(f"✅ {name}: {transformers.__version__}")
                    elif dep == "typer":
                        import typer

                        console.print(f"✅ {name}: {typer.__version__}")
                    else:
                        console.print(f"✅ {name}: Available")
                except Exception:
                    console.print(f"✅ {name}: Available")

        if missing_deps:
            console.print(
                f"\n❌ Missing dependencies: {', '.join(missing_deps)}",
                style="bold red",
            )
            console.print(
                "💡 Install with: pip install -r requirements.txt", style="yellow"
            )
        else:
            console.print("✅ All required dependencies installed", style="bold green")

        # Check configuration file
        console.print("\n⚙️  [bold blue]Configuration Check[/bold blue]")

        config_file = Path(config_path)
        if config_file.exists():
            console.print(f"✅ Configuration file: {config_path}")

            # Show key configuration paths
            console.print(
                f"   📁 Datasets path: {config.defaults.datasets_path or 'Not set'}"
            )
            console.print(
                f"   📁 Output directory: {config.defaults.output_dir or 'Not set'}"
            )
            console.print(f"   🤖 Models: {config.defaults.models or 'Not set'}")

            # Validate paths exist
            if effective_datasets_path:
                datasets_path_obj = Path(effective_datasets_path)
                if datasets_path_obj.exists():
                    console.print(
                        f"✅ Datasets directory exists: {effective_datasets_path}"
                    )

                    # Count images
                    image_extensions = {
                        ".jpg",
                        ".jpeg",
                        ".png",
                        ".bmp",
                        ".tiff",
                        ".webp",
                    }
                    image_count = sum(
                        1
                        for f in datasets_path_obj.iterdir()
                        if f.is_file() and f.suffix.lower() in image_extensions
                    )
                    console.print(f"   📸 Found {image_count} image files")
                else:
                    console.print(
                        f"❌ Datasets directory not found: {effective_datasets_path}",
                        style="red",
                    )
            else:
                console.print("⚠️  No datasets path configured", style="yellow")

            if config.defaults.output_dir:
                output_path = Path(config.defaults.output_dir)
                if output_path.exists():
                    console.print(
                        f"✅ Output directory exists: {config.defaults.output_dir}"
                    )
                else:
                    console.print(
                        f"⚠️  Output directory will be created: {config.defaults.output_dir}",
                        style="yellow",
                    )
            else:
                console.print("⚠️  No output directory configured", style="yellow")

        else:
            console.print(
                f"❌ Configuration file not found: {config_path}", style="bold red"
            )
            console.print(
                "💡 Create model_comparison.yaml with required settings", style="yellow"
            )

        # Check model registry and paths
        console.print("\n🤖 [bold blue]Model Registry Check[/bold blue]")

        try:
            from ..config.model_registry import get_model_registry

            model_registry = get_model_registry()
            available_models = model_registry.list_available_models()

            console.print(
                f"✅ Model registry: {len(available_models)} models registered"
            )

            for model_name in available_models:
                info = model_registry.get_model_info(model_name)
                path_exists = info.get("path_exists", False)
                status = "✅" if path_exists else "❌"
                description = info.get("description", "No description")

                console.print(f"   {status} {model_name}: {description}")

                if verbose and not path_exists:
                    model_path = info.get("path", "Unknown")
                    console.print(f"      💡 Expected path: {model_path}")

        except Exception as e:
            console.print(f"❌ Model registry check failed: {e}", style="red")
            if debug:
                import traceback

                console.print(traceback.format_exc(), style="red")

        # Check field schema
        console.print("\n🏷️  [bold blue]Field Schema Check[/bold blue]")

        try:
            # Try to get expected fields from config
            if hasattr(config, "expected_fields") and config.expected_fields:
                console.print(
                    f"✅ Field schema: {len(config.expected_fields)} fields defined"
                )
                if verbose:
                    console.print("   📋 Fields:")
                    for field in config.expected_fields:
                        console.print(f"      • {field}")
            else:
                console.print(
                    "⚠️  No field schema found in configuration", style="yellow"
                )

        except Exception as e:
            console.print(f"⚠️  Field schema check failed: {e}", style="yellow")

        # Overall environment status
        console.print("\n📊 [bold blue]Environment Summary[/bold blue]")

        issues = []

        if missing_deps:
            issues.append(f"{len(missing_deps)} missing dependencies")

        if not config_file.exists():
            issues.append("Configuration file missing")

        # Check if any models are available
        try:
            model_registry = get_model_registry()
            available_models = model_registry.list_available_models()
            working_models = [
                m
                for m in available_models
                if model_registry.get_model_info(m).get("path_exists", False)
            ]

            if not working_models:
                issues.append("No working models found")

        except Exception:
            issues.append("Model registry issues")

        if not effective_datasets_path or not Path(effective_datasets_path).exists():
            issues.append("Datasets directory not accessible")

        if issues:
            console.print(f"⚠️  Environment has {len(issues)} issues:", style="yellow")
            for issue in issues:
                console.print(f"   • {issue}")
            console.print("\n💡 [bold cyan]Recommendations:[/bold cyan]")
            console.print(
                "   1. Install missing dependencies: pip install -r requirements.txt"
            )
            console.print("   2. Create/fix model_comparison.yaml configuration")
            console.print("   3. Verify model paths and datasets directory")
            console.print("   4. Run this check again after fixes")
        else:
            console.print(
                "🎉 [bold green]Environment is ready![/bold green]", style="bold green"
            )
            console.print("✅ All systems operational for vision processing")

        # Quick start suggestion
        if not issues:
            console.print("\n🚀 [bold blue]Quick Start Commands[/bold blue]")
            console.print("   # Single image extraction")
            console.print("   python -m vision_processor extract image.png")
            console.print("")
            console.print("   # Model comparison")
            console.print("   python -m vision_processor compare")
            console.print("")
            console.print("   # Batch processing")
            console.print("   python -m vision_processor batch ./images/")

    except Exception as e:
        console.print(f"❌ Environment check failed: {e}", style="bold red")
        if debug or verbose:
            import traceback

            console.print("🔍 Full traceback:", style="red")
            console.print(traceback.format_exc(), style="red")
        raise typer.Exit(1) from None


@app.command()
def models(
    list_models: bool = typer.Option(False, "--list", help="List available models"),
    validate_models: bool = typer.Option(
        False, "--validate", help="Validate model configurations"
    ),
    config_path: str = typer.Option(
        "model_comparison.yaml", help="Path to configuration file"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    debug: bool = typer.Option(False, "--debug", help="Debug output"),
) -> None:
    """List and validate available models."""
    console.print("🤖 [bold blue]Model Management[/bold blue]")

    # Default to list if no specific action
    if not list_models and not validate_models:
        list_models = True

    try:
        # Load configuration
        config = ConfigManager.get_global_instance(config_path)

        # Apply logging overrides
        if debug:
            config.defaults.debug_mode = True
            config.defaults.verbose_mode = True
        elif verbose:
            config.defaults.verbose_mode = True

        # Import model registry
        from ..config.model_registry import get_model_registry

        model_registry = get_model_registry()
        available_models = model_registry.list_available_models()

        if list_models:
            console.print(
                f"\n📋 [bold blue]Available Models ({len(available_models)} registered)[/bold blue]"
            )

            if not available_models:
                console.print(
                    "❌ No models registered in the model registry", style="red"
                )
                console.print(
                    "💡 Check your model configuration in model_comparison.yaml",
                    style="yellow",
                )
                return

            for model_name in available_models:
                info = model_registry.get_model_info(model_name)

                # Model header
                console.print(f"\n🤖 [bold cyan]{model_name}[/bold cyan]")

                # Description
                description = info.get("description", "No description available")
                console.print(f"   📄 Description: {description}")

                # Model path and availability
                model_path = info.get("path", "Unknown")
                path_exists = info.get("path_exists", False)
                status = "✅ Available" if path_exists else "❌ Not found"
                console.print(f"   📁 Path: {model_path}")
                console.print(f"   🔍 Status: {status}")

                # Additional model info
                if verbose:
                    model_type = info.get("type", "Unknown")
                    console.print(f"   🏷️  Type: {model_type}")

                    # Try to get more detailed info
                    try:
                        if path_exists:
                            model_path_obj = Path(model_path)
                            if model_path_obj.exists():
                                if model_path_obj.is_file():
                                    size_mb = model_path_obj.stat().st_size / (
                                        1024 * 1024
                                    )
                                    console.print(f"   📏 Size: {size_mb:.1f} MB")
                                elif model_path_obj.is_dir():
                                    # Count files in directory
                                    file_count = sum(
                                        1
                                        for _ in model_path_obj.rglob("*")
                                        if _.is_file()
                                    )
                                    console.print(
                                        f"   📂 Directory with {file_count} files"
                                    )
                    except Exception as e:
                        if debug:
                            console.print(
                                f"   ⚠️  Could not get detailed info: {e}",
                                style="yellow",
                            )

                # Configuration specific to this model
                if hasattr(config, "model_paths") and hasattr(
                    config.model_paths, model_name
                ):
                    config_path = getattr(config.model_paths, model_name, None)
                    if config_path and config_path != model_path:
                        console.print(f"   ⚙️  Config override: {config_path}")

        if validate_models:
            console.print("\n🔍 [bold blue]Model Validation[/bold blue]")

            if not available_models:
                console.print("❌ No models to validate", style="red")
                return

            validation_results = {}

            for model_name in available_models:
                console.print(f"\n🔍 Validating {model_name}...")

                info = model_registry.get_model_info(model_name)
                model_path = info.get("path", "")

                validation = {
                    "model_exists": False,
                    "path_accessible": False,
                    "loadable": False,
                    "issues": [],
                }

                # Check if model path exists
                if model_path:
                    model_path_obj = Path(model_path)
                    if model_path_obj.exists():
                        validation["model_exists"] = True
                        validation["path_accessible"] = True
                        console.print(f"   ✅ Model path exists: {model_path}")

                        # Try to validate model can be loaded (basic check)
                        try:
                            # For now, just check if it's a valid path structure
                            # More sophisticated loading tests could be added here
                            if model_path_obj.is_file():
                                # Single file model
                                if model_path_obj.suffix in [
                                    ".bin",
                                    ".safetensors",
                                    ".pt",
                                    ".pth",
                                ]:
                                    validation["loadable"] = True
                                    console.print("   ✅ Valid model file format")
                                else:
                                    validation["issues"].append(
                                        f"Unexpected file format: {model_path_obj.suffix}"
                                    )

                            elif model_path_obj.is_dir():
                                # Directory-based model (like HuggingFace)
                                required_files = ["config.json"]
                                missing_files = []

                                for req_file in required_files:
                                    if not (model_path_obj / req_file).exists():
                                        missing_files.append(req_file)

                                if not missing_files:
                                    validation["loadable"] = True
                                    console.print(
                                        "   ✅ Model directory structure valid"
                                    )
                                else:
                                    validation["issues"].append(
                                        f"Missing files: {', '.join(missing_files)}"
                                    )

                        except Exception as e:
                            validation["issues"].append(f"Validation error: {e}")

                    else:
                        validation["issues"].append(
                            f"Model path does not exist: {model_path}"
                        )
                        console.print(
                            f"   ❌ Model path not found: {model_path}", style="red"
                        )
                else:
                    validation["issues"].append("No model path configured")
                    console.print("   ❌ No model path configured", style="red")

                # Configuration validation
                try:
                    # Check if model can be configured
                    config.set_model_type(model_name)
                    console.print("   ✅ Model configuration valid")
                except Exception as e:
                    validation["issues"].append(f"Configuration error: {e}")
                    console.print(f"   ❌ Configuration error: {e}", style="red")

                validation_results[model_name] = validation

                # Show validation summary for this model
                if validation["issues"]:
                    console.print(
                        f"   ⚠️  Issues found: {len(validation['issues'])}",
                        style="yellow",
                    )
                    if verbose:
                        for issue in validation["issues"]:
                            console.print(f"      • {issue}")
                else:
                    console.print("   🎉 Model validation passed!", style="green")

            # Overall validation summary
            console.print("\n📊 [bold blue]Validation Summary[/bold blue]")

            total_models = len(validation_results)
            valid_models = sum(
                1
                for v in validation_results.values()
                if not v["issues"] and v["loadable"]
            )
            accessible_models = sum(
                1 for v in validation_results.values() if v["path_accessible"]
            )

            console.print(f"📈 Total models: {total_models}")
            console.print(f"✅ Fully valid: {valid_models}")
            console.print(f"📁 Path accessible: {accessible_models}")
            console.print(f"❌ With issues: {total_models - valid_models}")

            if valid_models == total_models:
                console.print(
                    "\n🎉 [bold green]All models validated successfully![/bold green]"
                )
            elif valid_models > 0:
                console.print(
                    f"\n⚠️  [bold yellow]{total_models - valid_models} models have issues[/bold yellow]"
                )
                console.print(
                    "💡 Review the issues above and fix model configurations",
                    style="yellow",
                )
            else:
                console.print("\n❌ [bold red]No models are currently valid[/bold red]")
                console.print(
                    "💡 Check model paths in model_comparison.yaml", style="yellow"
                )
                console.print(
                    "💡 Ensure models are downloaded and accessible", style="yellow"
                )

        # Show configured models from YAML
        if verbose:
            console.print("\n⚙️  [bold blue]Configuration Settings[/bold blue]")
            console.print(f"📁 Config file: {config_path}")
            console.print(f"🤖 Default models: {config.defaults.models or 'Not set'}")

            if hasattr(config, "model_paths"):
                console.print("\n📍 [bold blue]Model Path Configuration[/bold blue]")
                for attr_name in dir(config.model_paths):
                    if not attr_name.startswith("_"):
                        path_value = getattr(config.model_paths, attr_name, None)
                        if path_value:
                            console.print(f"   {attr_name}: {path_value}")

    except Exception as e:
        console.print(f"❌ Model management failed: {e}", style="bold red")
        if debug or verbose:
            import traceback

            console.print("🔍 Full traceback:", style="red")
            console.print(traceback.format_exc(), style="red")
        raise typer.Exit(1) from None


@app.command()
def config(
    config_path: str = typer.Option(
        "model_comparison.yaml", help="Path to configuration file"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show all configuration details"
    ),
    debug: bool = typer.Option(False, "--debug", help="Debug output"),
) -> None:
    """Show current configuration settings."""
    console.print("⚙️ [bold blue]Configuration Display[/bold blue]")

    try:
        import yaml

        # Check if config file exists
        config_file = Path(config_path)
        if not config_file.exists():
            console.print(
                f"❌ Configuration file not found: {config_path}", style="bold red"
            )
            console.print(
                f"💡 Expected location: {config_file.absolute()}", style="yellow"
            )
            console.print(
                "💡 Create model_comparison.yaml with required settings", style="yellow"
            )
            raise typer.Exit(1) from None

        console.print(f"📁 Configuration file: {config_path}")
        console.print(f"📍 Absolute path: {config_file.absolute()}")

        # Load and display raw YAML content
        try:
            with config_file.open("r", encoding="utf-8") as f:
                yaml_content = yaml.safe_load(f)
        except Exception as e:
            console.print(f"❌ Failed to parse YAML: {e}", style="bold red")
            raise typer.Exit(1) from None

        # Load configuration through ConfigManager
        try:
            config = ConfigManager.get_global_instance(config_path)
            console.print("✅ Configuration loaded successfully")
        except Exception as e:
            console.print(f"❌ Configuration loading failed: {e}", style="bold red")
            if debug:
                import traceback

                console.print(traceback.format_exc(), style="red")
            raise typer.Exit(1) from None

        # Display main configuration sections
        console.print("\n📊 [bold blue]Configuration Overview[/bold blue]")

        # Defaults section
        if "defaults" in yaml_content:
            defaults = yaml_content["defaults"]
            console.print("\n⚙️  [bold cyan]Default Settings[/bold cyan]")

            key_settings = [
                ("datasets_path", "📁 Datasets Path"),
                ("output_dir", "📁 Output Directory"),
                ("models", "🤖 Default Models"),
                ("max_tokens", "📝 Max Tokens"),
                ("quantization", "⚡ Quantization"),
                ("trust_remote_code", "🔒 Trust Remote Code"),
                ("console_output", "🖥️  Console Output"),
                ("verbose_mode", "🔍 Verbose Mode"),
                ("debug_mode", "🐛 Debug Mode"),
            ]

            for key, label in key_settings:
                value = defaults.get(key, "Not set")
                console.print(f"   {label}: {value}")

            # Show additional defaults in verbose mode
            if verbose:
                console.print("\n📋 [bold cyan]All Default Settings[/bold cyan]")
                for key, value in defaults.items():
                    if key not in [k for k, _ in key_settings]:
                        console.print(f"   {key}: {value}")

        # Model paths section
        if "model_paths" in yaml_content:
            model_paths = yaml_content["model_paths"]
            console.print("\n🤖 [bold cyan]Model Paths[/bold cyan]")

            for model_name, path in model_paths.items():
                # Check if path exists
                path_obj = Path(path) if path else None
                status = "✅" if path_obj and path_obj.exists() else "❌"
                console.print(f"   {status} {model_name}: {path}")

        # Device configuration section
        if "device_config" in yaml_content:
            device_config = yaml_content["device_config"]
            console.print("\n🖥️  [bold cyan]Device Configuration[/bold cyan]")

            for key, value in device_config.items():
                console.print(f"   {key}: {value}")

        # Expected fields section
        if "expected_fields" in yaml_content:
            expected_fields = yaml_content["expected_fields"]
            console.print(
                f"\n🏷️  [bold cyan]Expected Fields ({len(expected_fields)} fields)[/bold cyan]"
            )

            if verbose:
                # Show all fields in columns
                from rich.columns import Columns

                field_items = [f"• {field}" for field in expected_fields]
                console.print(Columns(field_items, equal=True, expand=True))
            else:
                # Show just the count and first few
                console.print(f"   Total fields: {len(expected_fields)}")
                console.print(f"   First 5: {', '.join(expected_fields[:5])}")
                if len(expected_fields) > 5:
                    console.print(
                        f"   ... and {len(expected_fields) - 5} more (use --verbose to see all)"
                    )

        # Additional sections in verbose mode
        if verbose:
            console.print(
                "\n📋 [bold blue]Complete Configuration Structure[/bold blue]"
            )

            for section_name, section_data in yaml_content.items():
                if section_name not in [
                    "defaults",
                    "model_paths",
                    "device_config",
                    "expected_fields",
                ]:
                    console.print(f"\n📂 [bold cyan]{section_name}[/bold cyan]")

                    if isinstance(section_data, dict):
                        for key, value in section_data.items():
                            console.print(f"   {key}: {value}")
                    elif isinstance(section_data, list):
                        console.print(f"   List with {len(section_data)} items")
                        for i, item in enumerate(
                            section_data[:3]
                        ):  # Show first 3 items
                            console.print(f"   [{i}]: {item}")
                        if len(section_data) > 3:
                            console.print(
                                f"   ... and {len(section_data) - 3} more items"
                            )
                    else:
                        console.print(f"   Value: {section_data}")

        # Configuration validation
        console.print("\n🔍 [bold blue]Configuration Validation[/bold blue]")

        validation_issues = []

        # Check required settings
        required_settings = [
            ("defaults.datasets_path", config.defaults.datasets_path),
            ("defaults.output_dir", config.defaults.output_dir),
            ("defaults.models", config.defaults.models),
        ]

        for setting_name, setting_value in required_settings:
            if not setting_value:
                validation_issues.append(f"Missing required setting: {setting_name}")
            else:
                console.print(f"   ✅ {setting_name}: Configured")

        # Check path accessibility
        path_checks = []
        if config.defaults.datasets_path:
            path_checks.append(("datasets_path", config.defaults.datasets_path))

        for path_name, path_value in path_checks:
            if path_value:
                path_obj = Path(path_value)
                if path_obj.exists():
                    console.print(f"   ✅ {path_name}: Path exists")
                else:
                    validation_issues.append(
                        f"Path does not exist: {path_name} = {path_value}"
                    )

        # Check model paths
        if hasattr(config, "model_paths"):
            model_issues = []
            try:
                from ..config.model_registry import get_model_registry

                model_registry = get_model_registry()
                available_models = model_registry.list_available_models()

                working_models = []
                for model_name in available_models:
                    info = model_registry.get_model_info(model_name)
                    if info.get("path_exists", False):
                        working_models.append(model_name)
                    else:
                        model_issues.append(f"Model path not accessible: {model_name}")

                if working_models:
                    console.print(f"   ✅ Working models: {', '.join(working_models)}")

                validation_issues.extend(model_issues)

            except Exception as e:
                validation_issues.append(f"Model registry check failed: {e}")

        # Show validation results
        if validation_issues:
            console.print(
                f"\n⚠️  [bold yellow]Configuration Issues ({len(validation_issues)} found)[/bold yellow]"
            )
            for issue in validation_issues:
                console.print(f"   • {issue}")
            console.print("\n💡 [bold cyan]Recommendations:[/bold cyan]")
            console.print("   1. Fix missing required settings")
            console.print("   2. Verify all paths exist and are accessible")
            console.print("   3. Check model configurations")
            console.print(
                "   4. Run 'python -m vision_processor check' for detailed validation"
            )
        else:
            console.print("🎉 [bold green]Configuration is valid![/bold green]")

        # Configuration file info
        console.print("\n📄 [bold blue]File Information[/bold blue]")

        try:
            stat = config_file.stat()
            import datetime

            modified_time = datetime.datetime.fromtimestamp(stat.st_mtime)
            file_size = stat.st_size

            console.print(f"   📏 File size: {file_size} bytes")
            console.print(
                f"   📅 Last modified: {modified_time.strftime('%Y-%m-%d %H:%M:%S')}"
            )
            console.print(f"   🔍 Readable: {'Yes' if config_file.is_file() else 'No'}")

        except Exception as e:
            console.print(f"   ⚠️  Could not get file info: {e}", style="yellow")

    except Exception as e:
        console.print(f"❌ Configuration display failed: {e}", style="bold red")
        if debug:
            import traceback

            console.print("🔍 Full traceback:", style="red")
            console.print(traceback.format_exc(), style="red")
        raise typer.Exit(1) from None


@app.command()
def schema(
    config_path: str = typer.Option(
        "model_comparison.yaml", help="Path to configuration file"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show detailed field information"
    ),
    format_output: str = typer.Option(
        "table", "--format", help="Output format: table, list, json"
    ),
) -> None:
    """Show field extraction schema."""
    console.print("🏷️ [bold blue]Field Schema[/bold blue]")

    try:
        # Import required dependencies
        from ..config import ConfigManager

        # Use unified config loading - no more raw YAML!
        config = ConfigManager.get_global_instance(config_path)

        # Parse expected fields from extraction prompt (single source of truth)
        expected_fields = config.get_expected_fields()

        if not expected_fields:
            console.print("⚠️  No fields found in extraction prompt", style="yellow")
            console.print(
                "💡 Check extraction_prompt section in model_comparison.yaml",
                style="yellow",
            )

        console.print(f"📋 Configuration source: {config_path}")
        console.print(f"🏷️  Total fields: {len(expected_fields)}")

        # Display fields based on format
        if format_output == "json":
            console.print("\n📄 [bold blue]Field Schema (JSON Format)[/bold blue]")
            field_schema = {
                "schema_version": "1.0",
                "total_fields": len(expected_fields),
                "fields": expected_fields,
                "field_details": {
                    field: {"required": True, "type": "string"}
                    for field in expected_fields
                },
            }
            print(json.dumps(field_schema, indent=2))

        elif format_output == "list":
            console.print("\n📋 [bold blue]Field Schema (List Format)[/bold blue]")
            for i, field in enumerate(expected_fields, 1):
                console.print(f"{i:2d}. {field}")

        else:  # table format (default)
            console.print("\n📊 [bold blue]Field Schema (Table Format)[/bold blue]")

            from rich.table import Table

            # Create table with appropriate columns
            table = Table(
                show_header=True, header_style="bold magenta", show_lines=True
            )
            table.add_column("#", style="cyan", width=4, justify="right")
            table.add_column("Field Name", style="green", min_width=20)

            if verbose:
                table.add_column("Type", style="yellow", width=10)
                table.add_column("Required", style="blue", width=8, justify="center")
                table.add_column("Description", style="white", min_width=30)

            # Field descriptions for verbose mode
            field_descriptions = {
                "DOCUMENT_TYPE": "Type of document (invoice, receipt, etc.)",
                "SUPPLIER": "Name of the supplier/vendor",
                "ABN": "Australian Business Number",
                "INVOICE_NUMBER": "Invoice or receipt number",
                "DATE": "Document date",
                "TOTAL_AMOUNT": "Total amount including GST",
                "GST_AMOUNT": "GST/tax amount",
                "DESCRIPTION": "Description of goods/services",
                "QUANTITY": "Quantity of items",
                "UNIT_PRICE": "Price per unit",
                "CUSTOMER_NAME": "Customer name",
                "CUSTOMER_ADDRESS": "Customer address",
                "SUPPLIER_ADDRESS": "Supplier address",
                "PAYMENT_METHOD": "Method of payment",
                "DUE_DATE": "Payment due date",
                "SUBTOTAL": "Subtotal before tax",
                "DISCOUNT": "Discount amount",
                "LINE_ITEMS": "Individual line items",
                "CURRENCY": "Currency code",
                "REFERENCE_NUMBER": "Reference or PO number",
            }

            # Add rows to table
            for i, field in enumerate(expected_fields, 1):
                if verbose:
                    description = field_descriptions.get(
                        field, "Standard extraction field"
                    )
                    table.add_row(str(i), field, "string", "✅", description)
                else:
                    table.add_row(str(i), field)

            console.print(table)

        # Field categories analysis
        if verbose:
            console.print("\n📂 [bold blue]Field Categories[/bold blue]")

            # Categorize fields
            categories = {
                "Document Info": [
                    "DOCUMENT_TYPE",
                    "INVOICE_NUMBER",
                    "REFERENCE_NUMBER",
                    "DATE",
                    "DUE_DATE",
                ],
                "Financial": [
                    "TOTAL_AMOUNT",
                    "SUBTOTAL",
                    "GST_AMOUNT",
                    "DISCOUNT",
                    "CURRENCY",
                ],
                "Supplier Info": ["SUPPLIER", "ABN", "SUPPLIER_ADDRESS"],
                "Customer Info": ["CUSTOMER_NAME", "CUSTOMER_ADDRESS"],
                "Item Details": ["DESCRIPTION", "QUANTITY", "UNIT_PRICE", "LINE_ITEMS"],
                "Payment": ["PAYMENT_METHOD"],
            }

            for category, category_fields in categories.items():
                fields_in_schema = [f for f in category_fields if f in expected_fields]
                if fields_in_schema:
                    console.print(
                        f"\n📁 [bold cyan]{category}[/bold cyan] ({len(fields_in_schema)} fields)"
                    )
                    for field in fields_in_schema:
                        console.print(f"   • {field}")

        # Schema statistics
        console.print("\n📊 [bold blue]Schema Statistics[/bold blue]")

        # Field name analysis
        avg_length = (
            sum(len(field) for field in expected_fields) / len(expected_fields)
            if expected_fields
            else 0
        )
        longest_field = max(expected_fields, key=len) if expected_fields else ""
        shortest_field = min(expected_fields, key=len) if expected_fields else ""

        console.print(f"   📏 Average field name length: {avg_length:.1f} characters")
        console.print(
            f"   📏 Longest field: {longest_field} ({len(longest_field)} chars)"
        )
        console.print(
            f"   📏 Shortest field: {shortest_field} ({len(shortest_field)} chars)"
        )

        # Field naming patterns
        underscore_fields = [f for f in expected_fields if "_" in f]
        console.print(
            f"   🔗 Fields with underscores: {len(underscore_fields)}/{len(expected_fields)}"
        )

        # Field prefixes
        prefixes = {}
        for field in expected_fields:
            prefix = field.split("_")[0] if "_" in field else field
            prefixes[prefix] = prefixes.get(prefix, 0) + 1

        if verbose and len(prefixes) > 1:
            console.print("\n🏷️  [bold blue]Field Prefixes[/bold blue]")
            sorted_prefixes = sorted(prefixes.items(), key=lambda x: x[1], reverse=True)
            for prefix, count in sorted_prefixes[:5]:  # Show top 5 prefixes
                console.print(f"   {prefix}: {count} fields")

        # Usage recommendations
        console.print("\n💡 [bold blue]Usage Recommendations[/bold blue]")
        console.print("   📋 Use these field names consistently across all extractions")
        console.print("   🎯 Models will be prompted to extract these specific fields")
        console.print(
            "   ⚙️  Modify expected_fields in model_comparison.yaml to customize"
        )
        console.print(
            "   🔍 Fields not in this schema may be ignored during extraction"
        )

        # Quick start examples
        console.print("\n🚀 [bold blue]Quick Start Examples[/bold blue]")
        console.print("   # Extract using current schema")
        console.print("   python -m vision_processor extract document.png")
        console.print("")
        console.print("   # View configuration")
        console.print("   python -m vision_processor config --verbose")
        console.print("")
        console.print("   # Check environment")
        console.print("   python -m vision_processor check")

        # Export options
        if format_output == "table" and not verbose:
            console.print("\n📤 [bold blue]Export Options[/bold blue]")
            console.print("   # Export as JSON")
            console.print(
                "   python -m vision_processor schema --format json > schema.json"
            )
            console.print("")
            console.print("   # Export as list")
            console.print("   python -m vision_processor schema --format list")
            console.print("")
            console.print("   # Detailed view")
            console.print("   python -m vision_processor schema --verbose")

    except ConfigurationError as e:
        console.print(f"❌ Configuration error: {e.message}", style="bold red")
        if verbose and hasattr(e, "details") and e.details:
            console.print(f"💡 Details: {e.details}", style="yellow")
        raise typer.Exit(1) from None
    except Exception as e:
        console.print(f"❌ Schema display failed: {e}", style="bold red")
        if verbose:
            import traceback

            console.print("🔍 Full traceback:", style="red")
            console.print(traceback.format_exc(), style="red")
        raise typer.Exit(1) from None


# =============================================================================
# UTILITY COMMANDS
# =============================================================================


@app.command()
def convert(
    batch_file: str = typer.Argument(..., help="Path to batch_results.json"),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output CSV path"
    ),
    config_path: str = typer.Option(
        "model_comparison.yaml", help="Path to configuration file"
    ),
    keep_na: bool = typer.Option(
        False, "--keep-na", help="Keep 'N/A' strings instead of NaN"
    ),
    info_only: bool = typer.Option(
        False, "--info", help="Show info without saving CSV"
    ),
    model_name: Optional[str] = typer.Option(
        None, "--model", help="Model name for filename (auto-detected if not provided)"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    debug: bool = typer.Option(False, "--debug", help="Debug output"),
) -> None:
    """Convert batch results JSON to CSV format."""
    console.print("🔄 [bold blue]Batch Results Conversion[/bold blue]")

    try:
        # Validate batch file exists
        batch_path = Path(batch_file)
        if not batch_path.exists():
            console.print(f"❌ Batch file not found: {batch_file}", style="bold red")
            raise typer.Exit(1) from None

        if not batch_path.is_file():
            console.print(f"❌ Path is not a file: {batch_file}", style="bold red")
            raise typer.Exit(1) from None

        console.print(f"📁 Input file: {batch_file}")

        # Load configuration (used for default model info if needed)
        try:
            config = ConfigManager.get_global_instance(config_path)
        except Exception as e:
            console.print(f"⚠️  Could not load configuration: {e}", style="yellow")
            config = None

        # Import conversion utilities
        from ..utils.batch_to_dataframe import (
            batch_results_to_dataframe,
            print_dataframe_info,
            save_dataframe_to_csv,
        )

        # Load batch results
        console.print("\n🔍 Loading batch results...")
        try:
            with batch_path.open("r", encoding="utf-8") as f:
                batch_data = json.load(f)
        except json.JSONDecodeError as e:
            console.print(f"❌ Invalid JSON format: {e}", style="bold red")
            raise typer.Exit(1) from None
        except Exception as e:
            console.print(f"❌ Failed to read batch file: {e}", style="bold red")
            raise typer.Exit(1) from None

        console.print("✅ Batch file loaded successfully")

        # Display batch file information
        console.print("\n📊 [bold blue]Batch File Information[/bold blue]")

        total_images = batch_data.get("total_images", "Unknown")
        successful = batch_data.get("successful_extractions", "Unknown")
        failed = batch_data.get("failed_extractions", "Unknown")
        model_used = batch_data.get("model", "Unknown")
        processing_time = batch_data.get("processing_time", 0)

        console.print(f"   🤖 Model: {model_used}")
        console.print(f"   📸 Total images: {total_images}")
        console.print(f"   ✅ Successful extractions: {successful}")
        console.print(f"   ❌ Failed extractions: {failed}")
        console.print(f"   ⏱️  Processing time: {processing_time:.2f}s")

        if (
            successful
            and total_images
            and str(successful).isdigit()
            and str(total_images).isdigit()
        ):
            success_rate = int(successful) / int(total_images)
            console.print(f"   📈 Success rate: {success_rate:.1%}")

        # Auto-detect model name if not provided
        if not model_name:
            model_name = batch_data.get("model", "unknown")
            console.print(f"💡 Auto-detected model: {model_name}")

        # Convert to DataFrame
        console.print("\n🔄 Converting to DataFrame...")
        try:
            dataframe = batch_results_to_dataframe(batch_data, keep_na=keep_na)
            console.print(f"✅ DataFrame created with {len(dataframe)} rows")
        except Exception as e:
            console.print(f"❌ Conversion failed: {e}", style="bold red")
            if debug:
                import traceback

                console.print(traceback.format_exc(), style="red")
            raise typer.Exit(1) from None

        # Show DataFrame information
        console.print("\n📋 [bold blue]DataFrame Information[/bold blue]")
        print_dataframe_info(dataframe, verbose=verbose)

        # Show sample data if verbose
        if verbose and not dataframe.empty:
            console.print("\n📊 [bold blue]Sample Data (First 3 Rows)[/bold blue]")

            from rich.table import Table

            # Create table with first few columns that fit
            table = Table(
                show_header=True, header_style="bold magenta", show_lines=False
            )

            # Get column names - limit to fit in terminal
            columns = list(dataframe.columns)
            display_columns = columns[:6]  # Show first 6 columns

            for col in display_columns:
                table.add_column(col, style="green", max_width=15, overflow="ellipsis")

            # Add sample rows
            for i in range(min(3, len(dataframe))):
                row_data = []
                for col in display_columns:
                    value = str(dataframe.iloc[i][col])
                    # Truncate long values
                    if len(value) > 12:
                        value = value[:12] + "..."
                    row_data.append(value)
                table.add_row(*row_data)

            console.print(table)

            if len(columns) > 6:
                console.print(f"   ... and {len(columns) - 6} more columns")

        # Info-only mode - just show information and exit
        if info_only:
            console.print("\n✅ [bold green]Information display complete[/bold green]")
            console.print("💡 Use without --info flag to save CSV file")
            return

        # Determine output filename
        if output:
            output_path = Path(output)
        else:
            # Auto-generate filename
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"batch_results_{model_name}_{timestamp}.csv"
            output_path = batch_path.parent / filename

        console.print(f"\n💾 Output file: {output_path}")

        # Check if output file already exists
        if output_path.exists():
            console.print(
                f"⚠️  Output file already exists: {output_path}", style="yellow"
            )
            if not typer.confirm("Overwrite existing file?"):
                console.print("❌ Conversion cancelled")
                raise typer.Exit(1) from None

        # Save DataFrame to CSV
        console.print("\n💾 Saving CSV file...")
        try:
            save_dataframe_to_csv(dataframe, str(output_path))
            console.print(f"✅ CSV saved successfully: {output_path}")
        except Exception as e:
            console.print(f"❌ Failed to save CSV: {e}", style="bold red")
            if debug:
                import traceback

                console.print(traceback.format_exc(), style="red")
            raise typer.Exit(1) from None

        # Verify saved file
        try:
            saved_df = __import__("pandas").read_csv(output_path)
            console.print(
                f"✅ Verification: {len(saved_df)} rows, {len(saved_df.columns)} columns"
            )
        except Exception as e:
            console.print(f"⚠️  Could not verify saved file: {e}", style="yellow")

        # Final summary
        console.print("\n📊 [bold blue]Conversion Summary[/bold blue]")
        console.print(f"   📁 Input: {batch_file}")
        console.print(f"   📄 Output: {output_path}")
        console.print(f"   📊 Records: {len(dataframe)}")
        console.print(f"   📋 Fields: {len(dataframe.columns)}")
        console.print(f"   🤖 Model: {model_name}")

        if not keep_na:
            console.print("   🔄 N/A values converted to NaN")
        else:
            console.print("   📝 N/A values preserved as strings")

        # Usage suggestions
        console.print("\n💡 [bold blue]Next Steps[/bold blue]")
        console.print(f"   📊 Analyze data: pandas.read_csv('{output_path}')")
        console.print("   📈 Generate reports: Use the CSV for further analysis")
        console.print("   🔍 Inspect results: Open in spreadsheet application")

        if verbose:
            console.print("\n🔧 [bold blue]Advanced Options[/bold blue]")
            console.print("   # Convert with custom model name")
            console.print(
                f"   python -m vision_processor convert {batch_file} --model custom_model"
            )
            console.print("")
            console.print("   # Keep N/A strings instead of converting to NaN")
            console.print(
                f"   python -m vision_processor convert {batch_file} --keep-na"
            )
            console.print("")
            console.print("   # Just show info without saving")
            console.print(f"   python -m vision_processor convert {batch_file} --info")

    except Exception as e:
        console.print(f"❌ Conversion failed: {e}", style="bold red")
        if debug or verbose:
            import traceback

            console.print("🔍 Full traceback:", style="red")
            console.print(traceback.format_exc(), style="red")
        raise typer.Exit(1) from None


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    app()
