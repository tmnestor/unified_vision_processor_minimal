"""CLI for converting batch_results.json to CSV format.

Provides easy command-line access to batch results conversion functionality.
"""

from typing import Optional

import typer
from rich.console import Console

from ..config.config_manager import ConfigManager
from ..utils.batch_to_dataframe import (
    batch_results_to_dataframe,
    print_dataframe_info,
    save_dataframe_to_csv,
)
from ..utils.path_resolver import PathResolver

app = typer.Typer(help="Convert batch processing results to CSV format")
console = Console()


@app.command("convert")
def convert_to_csv(
    batch_file: str = typer.Argument(..., help="Path to batch_results.json file"),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output CSV path"
    ),
    config: Optional[str] = typer.Option(
        None, "--config", "-c", help="Config YAML path"
    ),
    keep_na: bool = typer.Option(
        False, "--keep-na", help="Keep 'N/A' strings instead of NaN"
    ),
    info_only: bool = typer.Option(
        False, "--info", help="Show info without saving CSV"
    ),
    yaml_file: Optional[str] = typer.Option(
        None, "--yaml-file", help="YAML configuration file"
    ),
) -> None:
    """Convert batch_results.json to CSV format with one row per image."""

    try:
        # Initialize configuration and path resolver
        config_manager = ConfigManager(yaml_file)
        path_resolver = PathResolver(config_manager)

        # Resolve batch file path (look in output directory if relative)
        resolved_batch_path = path_resolver.resolve_output_file_path(batch_file)

        console.print(f"📂 Loading batch results from: {resolved_batch_path}")

        # Convert to DataFrame
        batch_dataframe = batch_results_to_dataframe(
            batch_results_path=resolved_batch_path,
            config_path=config,
            use_na_strings=keep_na,
        )

        # Show DataFrame info
        print_dataframe_info(batch_dataframe)

        # Save to CSV unless info-only mode
        if not info_only:
            console.print("\n💾 Saving to CSV...")
            # Resolve output path if provided
            resolved_output_path = None
            if output:
                resolved_output_path = path_resolver.resolve_output_file_path(output)

            csv_path = save_dataframe_to_csv(
                batch_results_path=resolved_batch_path,
                output_csv_path=resolved_output_path,
                config_path=config,
                use_na_strings=keep_na,
            )
            console.print(f"✅ [green]Successfully saved to: {csv_path}[/green]")
        else:
            console.print("\n📋 [yellow]Info-only mode: CSV not saved[/yellow]")

    except FileNotFoundError as e:
        console.print(f"❌ [red]File not found: {e}[/red]")
        raise typer.Exit(1) from None
    except ValueError as e:
        console.print(f"❌ [red]Invalid data: {e}[/red]")
        raise typer.Exit(1) from None
    except Exception as e:
        console.print(f"❌ [red]Unexpected error: {e}[/red]")
        raise typer.Exit(1) from None


@app.command("analyze")
def analyze_batch_results(
    batch_file: str = typer.Argument(..., help="Path to batch_results.json file"),
    config: Optional[str] = typer.Option(
        None, "--config", "-c", help="Config YAML path"
    ),
    yaml_file: Optional[str] = typer.Option(
        None, "--yaml-file", help="YAML configuration file"
    ),
) -> None:
    """Analyze batch results without saving to CSV."""

    try:
        # Initialize configuration and path resolver
        config_manager = ConfigManager(yaml_file)
        path_resolver = PathResolver(config_manager)

        # Resolve batch file path (look in output directory if relative)
        resolved_batch_path = path_resolver.resolve_output_file_path(batch_file)

        console.print(f"📊 Analyzing batch results from: {resolved_batch_path}")

        batch_dataframe = batch_results_to_dataframe(
            batch_results_path=resolved_batch_path,
            config_path=config,
            use_na_strings=False,  # Use NaN for analysis
        )

        # Show detailed analysis
        print_dataframe_info(batch_dataframe)

        # Show sample of data
        console.print("\n📋 Sample Data (first 3 rows):")
        if len(batch_dataframe) > 0:
            # Show first few columns to avoid overwhelming output
            sample_cols = ["image"] + list(
                batch_dataframe.columns[1:6]
            )  # image + first 5 fields
            sample_dataframe = batch_dataframe[sample_cols].head(3)
            console.print(sample_dataframe.to_string(index=False))

            if len(batch_dataframe.columns) > 6:
                console.print(
                    f"... and {len(batch_dataframe.columns) - 6} more field columns"
                )
        else:
            console.print("No data found in batch results")

    except Exception as e:
        console.print(f"❌ [red]Error analyzing batch results: {e}[/red]")
        raise typer.Exit(1) from None


if __name__ == "__main__":
    app()
