#!/usr/bin/env python3
"""Simple Model Comparison CLI
=============================

Simplified model comparison using refactored components.
"""


import typer

from vision_processor.comparison.simple_runner import SimpleComparisonRunner

app = typer.Typer()


@app.command()
def compare(
    datasets_path: str = typer.Option(..., help="Path to dataset directory"),
    output_dir: str = typer.Option("./results", help="Output directory for results"),
    models: str = typer.Option("llama,internvl", help="Comma-separated list of models"),
    max_tokens: int = typer.Option(256, help="Maximum tokens for generation")
):
    """Run simplified model comparison."""
    # Parse models
    model_list = [m.strip() for m in models.split(',')]

    # Create runner
    runner = SimpleComparisonRunner(
        datasets_path=datasets_path,
        output_dir=output_dir,
        max_tokens=max_tokens
    )

    # Run comparison
    try:
        results = runner.run_comparison(model_list)
        typer.echo(f"\n✅ Comparison complete! Results saved to {output_dir}")
    except Exception as e:
        typer.echo(f"❌ Error: {e}", err=True)
        raise typer.Exit(1) from None


if __name__ == "__main__":
    app()
