"""Simplified CLI with YAML configuration support."""

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from ..exceptions import (
    ConfigurationError,
    ImageProcessingError,
    ModelLoadError,
    ValidationError,
    VisionProcessorError,
)

app = typer.Typer(
    name="simple-vision-processor",
    help="Simplified Vision Document Processing - Single-Step Extraction",
    rich_markup_mode="rich",
)
console = Console()


@app.command()
def extract(
    image_path: str = typer.Argument(..., help="Path to document image"),
    model: Optional[str] = typer.Option(
        None, help="Override model type (internvl or llama)"
    ),
    output_format: Optional[str] = typer.Option(
        None, help="Override output format (table, json, yaml)"
    ),
    yaml_file: str = typer.Option("model_comparison.yaml", help="Path to YAML configuration file"),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
) -> None:
    """Extract data using single-step processing with YAML configuration."""

    # Check image file exists
    image_file = Path(image_path)
    if not image_file.exists():
        console.print(f"[red]❌ Image file not found: {image_path}[/red]")
        raise typer.Exit(1) from None

    try:
        # Import dependencies
        if verbose:
            console.print("[yellow]Loading dependencies...[/yellow]")

        from ..config.simple_config import SimpleConfig
        from ..extraction.simple_extraction_manager import SimpleExtractionManager

        # Load configuration from YAML
        config = SimpleConfig(yaml_file)

        # Apply simple CLI overrides
        if model:
            config.set_model_type(model)
        if output_format:
            config.set_output_format(output_format)

        # Validate configuration
        try:
            config.validate()
        except ValidationError as e:
            console.print(f"[red]❌ Configuration validation failed: {e.message}[/red]")
            if verbose and e.details:
                console.print(f"[yellow]Details: {e.details}[/yellow]")
            raise typer.Exit(1) from None

        # Process document
        console.print(f"\n🔍 Processing document: {image_file.name}")

        with console.status(f"[bold green]Processing with {config.model_type}..."):
            manager = SimpleExtractionManager(config)
            result = manager.process_document(image_path)

        # Display results based on configuration
        if config.output_format == "table":
            display_table(result.extracted_fields)
        elif config.output_format == "json":
            print(json.dumps(result.extracted_fields, indent=2))
        elif config.output_format == "yaml":
            # Use clean aligned format for YAML (default)
            display_clean_format(result.extracted_fields)
        else:
            # Default to clean format
            display_clean_format(result.extracted_fields)

        # Show processing info
        console.print(f"\n⏱️  Processing Time: {result.processing_time:.2f}s")
        console.print(f"🎯 Model Confidence: {result.model_confidence:.3f}")
        console.print(f"🔧 Extraction Method: {result.extraction_method}")

    except ConfigurationError as e:
        console.print(f"[red]❌ Configuration error: {e.message}[/red]")
        if verbose and e.details:
            console.print(f"[yellow]Details: {e.details}[/yellow]")
        raise typer.Exit(1) from None
    except ModelLoadError as e:
        console.print(f"[red]❌ Model loading error: {e.message}[/red]")
        if verbose and e.details:
            console.print(f"[yellow]Details: {e.details}[/yellow]")
        raise typer.Exit(1) from None
    except ImageProcessingError as e:
        console.print(f"[red]❌ Image processing error: {e.message}[/red]")
        if verbose and e.details:
            console.print(f"[yellow]Details: {e.details}[/yellow]")
        raise typer.Exit(1) from None
    except VisionProcessorError as e:
        console.print(f"[red]❌ {e.message}[/red]")
        if verbose and e.details:
            console.print(f"[yellow]Details: {e.details}[/yellow]")
        raise typer.Exit(1) from None
    except ImportError as e:
        console.print(f"[red]❌ Import error: {e}[/red]")
        console.print("[yellow]Please ensure all dependencies are installed[/yellow]")
        raise typer.Exit(1) from None
    except Exception as e:
        console.print(f"[red]❌ Unexpected error: {e}[/red]")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        raise typer.Exit(1) from None


@app.command()
def compare(
    image_path: str = typer.Argument(..., help="Path to document image"),
    models: str = typer.Option(
        "internvl,llama", help="Models to compare (comma-separated)"
    ),
    yaml_file: str = typer.Option("model_comparison.yaml", help="Path to YAML configuration file"),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
) -> None:
    """Compare extraction results between models using YAML configuration."""

    # Check image file exists
    image_file = Path(image_path)
    if not image_file.exists():
        console.print(f"[red]❌ Image file not found: {image_path}[/red]")
        raise typer.Exit(1) from None

    try:
        # YAML configuration is passed directly to SimpleConfig

        # Import dependencies
        if verbose:
            console.print("[yellow]Loading dependencies...[/yellow]")

        from ..config.simple_config import SimpleConfig
        from ..extraction.simple_extraction_manager import SimpleExtractionManager

        results = {}
        model_list = [m.strip() for m in models.split(",")]

        for model_name in model_list:
            console.print(f"\n{'=' * 50}")
            console.print(f"🔄 Testing with {model_name}")
            console.print(f"{'=' * 50}")

            # Create config with model override
            config = SimpleConfig("model_comparison.yaml")
            config.set_model_type(model_name)

            # Validate configuration
            if not config.validate():
                console.print(
                    f"[red]❌ Configuration validation failed for {model_name}[/red]"
                )
                continue

            # Process document
            with console.status(f"[bold green]Processing with {model_name}..."):
                manager = SimpleExtractionManager(config)
                results[model_name] = manager.process_document(image_path)

            console.print(f"✅ {model_name} processing complete")

        # Display comparison
        if results:
            display_comparison(results)
        else:
            console.print("[red]❌ No valid results to compare[/red]")
            raise typer.Exit(1) from None

    except ImportError as e:
        console.print(f"[red]❌ Import error: {e}[/red]")
        console.print("[yellow]Please ensure all dependencies are installed[/yellow]")
        raise typer.Exit(1) from None
    except Exception as e:
        console.print(f"[red]❌ Comparison failed: {e}[/red]")
        if verbose:
            import traceback

            console.print(traceback.format_exc())
        raise typer.Exit(1) from None


@app.command()
def config_info(
    yaml_file: str = typer.Option("model_comparison.yaml", help="Path to YAML configuration file"),
) -> None:
    """Display current configuration from YAML file."""

    try:
        # YAML configuration is passed directly to SimpleConfig

        from ..config.simple_config import SimpleConfig

        config = SimpleConfig(yaml_file)
        config.print_configuration()

        # Also show YAML file location
        yaml_path = Path(yaml_file).absolute()
        if yaml_path.exists():
            console.print(f"\n📄 Configuration file: {yaml_path}")
            console.print(f"📝 File size: {yaml_path.stat().st_size} bytes")

            # Validate configuration
            console.print("\n🔍 Validating configuration...")
            try:
                config.validate()
                console.print("[green]✅ Configuration is valid[/green]")
            except ValidationError as e:
                console.print(f"[red]❌ Configuration has issues: {e.message}[/red]")
        else:
            console.print(f"\n⚠️  Configuration file not found: {yaml_path}")
            console.print("💡 Check that model_comparison.yaml exists")

    except ImportError as e:
        console.print(f"[red]❌ Import error: {e}[/red]")
        raise typer.Exit(1) from None
    except Exception as e:
        console.print(f"[red]❌ Error reading configuration: {e}[/red]")
        raise typer.Exit(1) from None


@app.command()
def batch(
    input_dir: str = typer.Argument(..., help="Directory containing images to process"),
    output_dir: str = typer.Option("output", help="Directory to save results"),
    model: Optional[str] = typer.Option(None, help="Override model type"),
    yaml_file: str = typer.Option("model_comparison.yaml", help="Path to YAML configuration file"),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
) -> None:
    """Process multiple documents in batch mode."""

    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists() or not input_path.is_dir():
        console.print(f"[red]❌ Input directory not found: {input_dir}[/red]")
        raise typer.Exit(1) from None

    try:
        # YAML configuration is passed directly to SimpleConfig

        # Import dependencies
        from ..config.simple_config import SimpleConfig
        from ..extraction.simple_extraction_manager import SimpleExtractionManager

        # Load configuration
        config = SimpleConfig(yaml_file)

        if model:
            config.set_model_type(model)

        # Find image files
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        image_files = [
            f
            for f in input_path.iterdir()
            if f.suffix.lower() in image_extensions and f.is_file()
        ]

        if not image_files:
            console.print(f"[red]❌ No image files found in: {input_dir}[/red]")
            raise typer.Exit(1) from None

        console.print(f"📦 Found {len(image_files)} image files")

        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)
        console.print(f"📁 Output directory: {output_path}")

        # Process batch
        manager = SimpleExtractionManager(config)
        results = manager.process_batch(image_files)

        # Save results
        batch_results = []
        for image_file, result in zip(image_files, results, strict=False):
            result_data = {
                "filename": image_file.name,
                "extracted_fields": result.extracted_fields,
                "processing_time": result.processing_time,
                "model_confidence": result.model_confidence,
                "extraction_method": result.extraction_method,
            }
            batch_results.append(result_data)

        # Save as JSON
        output_file = output_path / "batch_results.json"
        with output_file.open("w") as f:
            json.dump(batch_results, f, indent=2)

        console.print(f"💾 Results saved to: {output_file}")

        # Display summary
        successful = sum(1 for r in results if r.extraction_method != "error")
        console.print("\n📊 Batch Summary:")
        console.print(f"  Total files: {len(image_files)}")
        console.print(f"  Successful: {successful}")
        console.print(f"  Failed: {len(image_files) - successful}")

    except ImportError as e:
        console.print(f"[red]❌ Import error: {e}[/red]")
        raise typer.Exit(1) from None
    except Exception as e:
        console.print(f"[red]❌ Batch processing failed: {e}[/red]")
        if verbose:
            import traceback

            console.print(traceback.format_exc())
        raise typer.Exit(1) from None


def display_table(extracted_fields: dict) -> None:
    """Display extracted fields in a rich table."""
    table = Table(title="Extracted Data")
    table.add_column("Field", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")

    for key, value in extracted_fields.items():
        if isinstance(value, (list, dict)):
            value_str = str(value)[:80] + "..." if len(str(value)) > 80 else str(value)
        else:
            value_str = str(value)
        table.add_row(key, value_str)

    console.print(table)


def display_clean_format(extracted_fields: dict) -> None:
    """Display extracted fields in clean aligned format (like comparison output)."""
    console.print("\n📋 Extracted Fields:")

    # Sort fields alphabetically for consistent display
    sorted_pairs = sorted(extracted_fields.items())

    for key, value in sorted_pairs:
        # Clean up values - remove trailing asterisks and whitespace
        if isinstance(value, str):
            value = value.rstrip("*").strip()
        else:
            value = str(value)

        # Display with aligned formatting (20 char padding for field names)
        console.print(f"   {key:20}: {value}", style="dim green")


def display_comparison(results: dict) -> None:
    """Display side-by-side comparison of model results."""
    console.print(f"\n{'=' * 80}")
    console.print("📊 MODEL COMPARISON RESULTS")
    console.print(f"{'=' * 80}")

    # Extract common keys
    all_keys = set()
    for result in results.values():
        all_keys.update(result.extracted_fields.keys())

    # Display comparison table
    table = Table(title="Model Comparison")
    table.add_column("Field", style="cyan")

    for model_name in results.keys():
        table.add_column(model_name, style="green")

    for key in sorted(all_keys):
        row = [key]
        for model_name in results.keys():
            value = results[model_name].extracted_fields.get(key, "❌ Missing")
            if isinstance(value, (list, dict)):
                value = str(value)[:50] + "..." if len(str(value)) > 50 else str(value)
            row.append(str(value))
        table.add_row(*row)

    console.print(table)

    # Performance comparison
    console.print("\n⏱️  PERFORMANCE COMPARISON:")
    for model_name, result in results.items():
        console.print(f"  {model_name}: {result.processing_time:.2f}s")


if __name__ == "__main__":
    app()
