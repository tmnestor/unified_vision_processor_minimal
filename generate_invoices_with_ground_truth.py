#!/usr/bin/env python3
"""
Generate Ground Truth for Existing Invoice Images

Since the invoices were already generated, this script helps create
ground truth annotations by processing existing images using both models
and manual verification.

Usage:
    python generate_invoices_with_ground_truth.py --images-dir ./datasets --output-dir ./ground_truth
"""

import csv
from pathlib import Path
from typing import Dict, Optional

import typer
from rich.console import Console
from rich.progress import track
from rich.table import Table

from vision_processor.config import ConfigManager
from vision_processor.extraction.extraction_manager import SimpleExtractionManager

# Import our ground truth generator and extraction tools
from vision_processor.utils.ground_truth_generator import GroundTruthGenerator

console = Console()
app = typer.Typer()


def extract_with_both_models(image_path: str, config: ConfigManager) -> Dict[str, Dict]:
    """Extract fields from image using both Llama and InternVL models.

    Returns:
        Dictionary with 'llama' and 'internvl' extraction results
    """
    results = {}

    # Extract with Llama
    console.print("[blue]Extracting with Llama-3.2-Vision...[/blue]")
    config.set_model_type("llama32_vision")
    llama_manager = SimpleExtractionManager(config)
    llama_result = llama_manager.process_document(image_path)
    results["llama"] = llama_result.extracted_fields

    # Extract with InternVL
    console.print("[blue]Extracting with InternVL3...[/blue]")
    config.set_model_type("internvl3")
    internvl_manager = SimpleExtractionManager(config)
    internvl_result = internvl_manager.process_document(image_path)
    results["internvl"] = internvl_result.extracted_fields

    return results


def create_ground_truth_from_extractions(
    image_filename: str,
    llama_fields: Dict,
    internvl_fields: Dict,
    manual_override: Optional[Dict] = None,
) -> Dict:
    """Create ground truth by comparing model extractions.

    Args:
        image_filename: Name of the image file
        llama_fields: Fields extracted by Llama
        internvl_fields: Fields extracted by InternVL
        manual_override: Manual corrections for specific fields

    Returns:
        Ground truth dictionary
    """
    ground_truth = {"image_filename": image_filename}

    # Expected fields from model_comparison.yaml
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

    for field in expected_fields:
        llama_val = llama_fields.get(field, "N/A")
        internvl_val = internvl_fields.get(field, "N/A")

        # Apply manual override if available
        if manual_override and field in manual_override:
            ground_truth[field] = manual_override[field]
        # If both models agree and it's not N/A, use that value
        elif llama_val == internvl_val and llama_val != "N/A":
            ground_truth[field] = llama_val
        # For critical fields, prefer non-N/A values
        elif field in ["SUPPLIER", "ABN", "TOTAL", "GST", "INVOICE_DATE"]:
            if llama_val != "N/A":
                ground_truth[field] = llama_val
            elif internvl_val != "N/A":
                ground_truth[field] = internvl_val
            else:
                ground_truth[field] = "N/A"
        else:
            # Default to N/A for disagreements
            ground_truth[field] = "N/A"

    return ground_truth


@app.command()
def reconstruct(
    images_dir: str = typer.Argument(
        ..., help="Directory containing existing invoice images"
    ),
    output_dir: str = typer.Option(
        "./ground_truth", "--output-dir", "-o", help="Output directory for ground truth"
    ),
    shell_example: bool = typer.Option(
        False, "--shell-example", help="Include hardcoded Shell invoice example"
    ),
    auto_mode: bool = typer.Option(
        False, "--auto", help="Automatic mode - no manual verification"
    ),
) -> None:
    """Reconstruct ground truth from existing invoice images."""

    images_path = Path(images_dir)
    if not images_path.exists():
        console.print(f"[red]❌ Images directory not found: {images_dir}[/red]")
        raise typer.Exit(1)

    # Find all invoice images
    image_extensions = ["*.png", "*.jpg", "*.jpeg"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(images_path.glob(ext))

    if not image_files:
        console.print(f"[red]❌ No images found in {images_dir}[/red]")
        raise typer.Exit(1)

    console.print(f"[green]Found {len(image_files)} images to process[/green]")

    # Initialize ground truth generator
    gt_generator = GroundTruthGenerator(output_dir=output_dir)
    config = ConfigManager()

    # Add hardcoded Shell example if requested
    if shell_example:
        shell_gt = {
            "image_filename": "shell_invoice.png",
            "DOCUMENT_TYPE": "INVOICE",
            "SUPPLIER": "Shell Australia",
            "ABN": "36 643 730 685",
            "PAYER_NAME": "Emily Thompson",
            "PAYER_ADDRESS": "821 King Street, Hobart TAS 7000",
            "PAYER_PHONE": "(71) 5765 9791",
            "PAYER_EMAIL": "N/A",
            "INVOICE_DATE": "28/06/2025",
            "DUE_DATE": "21/08/2025",
            "GST": "$6.52",
            "TOTAL": "$71.67",
            "SUBTOTAL": "$65.15",
            "SUPPLIER_WEBSITE": "N/A",
            "QUANTITIES": "2, 2, 2, 3, 3, 1",
            "PRICES": "$4.20, $15.00, $1.75, $4.50, $1.75, $4.50",
            "BUSINESS_ADDRESS": "724 Pitt Street, Darwin NT 0800",
            "BUSINESS_PHONE": "(66) 6115 5508",
            "BANK_NAME": "N/A",
            "BSB_NUMBER": "N/A",
            "BANK_ACCOUNT_NUMBER": "083-004-330319-1",
            "ACCOUNT_HOLDER": "N/A",
            "STATEMENT_PERIOD": "N/A",
            "OPENING_BALANCE": "N/A",
            "CLOSING_BALANCE": "N/A",
            "DESCRIPTIONS": "Energy Drink, Car Wash, Premium Unleaded, Coffee Large, Premium Unleaded, Coffee Large",
        }

        # Save Shell example to CSV
        with gt_generator.ground_truth_file.open(
            "a", newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(
                f, fieldnames=["image_filename"] + gt_generator.expected_fields
            )
            writer.writerow(shell_gt)

        console.print("[green]✓ Added Shell invoice example to ground truth[/green]")

    # Process each image
    ground_truth_data = []
    for image_file in track(image_files, description="Processing images"):
        console.print(f"\n[bold]Processing: {image_file.name}[/bold]")

        try:
            # Extract with both models
            extractions = extract_with_both_models(str(image_file), config)

            # Create ground truth from extractions
            ground_truth = create_ground_truth_from_extractions(
                image_file.name, extractions["llama"], extractions["internvl"]
            )

            if not auto_mode:
                # Show comparison and ask for verification
                table = Table(show_header=True, header_style="bold magenta")
                table.add_column("Field", style="cyan", width=20)
                table.add_column("Llama", style="green")
                table.add_column("InternVL", style="blue")
                table.add_column("Ground Truth", style="yellow")

                # Show key fields for verification
                key_fields = ["SUPPLIER", "ABN", "TOTAL", "GST", "INVOICE_DATE"]
                for field in key_fields:
                    llama_val = extractions["llama"].get(field, "N/A")
                    internvl_val = extractions["internvl"].get(field, "N/A")
                    gt_val = ground_truth.get(field, "N/A")
                    table.add_row(field, llama_val, internvl_val, gt_val)

                console.print(table)

                # Ask for confirmation
                if not typer.confirm(f"Accept ground truth for {image_file.name}?"):
                    console.print("[yellow]Skipping this image...[/yellow]")
                    continue

            ground_truth_data.append(ground_truth)

        except Exception as e:
            console.print(f"[red]❌ Error processing {image_file.name}: {e}[/red]")
            continue

    # Save all ground truth data
    if ground_truth_data:
        # Append to existing CSV or create new one
        mode = "a" if gt_generator.ground_truth_file.exists() else "w"
        with gt_generator.ground_truth_file.open(
            mode, newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(
                f, fieldnames=["image_filename"] + gt_generator.expected_fields
            )
            if mode == "w":
                writer.writeheader()
            writer.writerows(ground_truth_data)

        console.print(
            f"\n[green]✓ Generated ground truth for {len(ground_truth_data)} images[/green]"
        )
        console.print(
            f"[yellow]Ground truth file: {gt_generator.ground_truth_file}[/yellow]"
        )

        # Show evaluation command
        console.print("\n[bold]To evaluate models using this ground truth:[/bold]")
        console.print(
            f"python -m vision_processor.cli.evaluation_cli compare {gt_generator.ground_truth_file} --images-dir {images_dir}"
        )
    else:
        console.print("[red]❌ No ground truth data was generated[/red]")


@app.command()
def generate_with_synthetic(
    count: int = typer.Option(
        5, "--count", "-c", help="Number of invoices to generate"
    ),
    output_dir: str = typer.Option(
        "./evaluation_dataset", "--output-dir", "-o", help="Output directory"
    ),
    business_type: str = typer.Option(
        "mixed",
        "--business-type",
        help="Business type (retail, fuel, accommodation, professional, mixed)",
    ),
    min_items: int = typer.Option(2, "--min-items", help="Minimum items per invoice"),
    max_items: int = typer.Option(6, "--max-items", help="Maximum items per invoice"),
) -> None:
    """Generate new synthetic invoices WITH ground truth annotations."""

    # Import the synthetic invoice generator from current project
    import random

    from synthetic_invoice_generator import SyntheticInvoiceGenerator

    # Setup directories
    output_path = Path(output_dir)
    images_dir = output_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Initialize generators
    invoice_generator = SyntheticInvoiceGenerator()
    ground_truth_generator = GroundTruthGenerator(output_dir=output_path)

    # Business types
    business_types = ["retail", "fuel", "accommodation", "professional"]

    console.print(f"[green]Generating {count} invoices with ground truth...[/green]")
    console.print(f"[blue]Output directory: {output_path}[/blue]")
    console.print(f"[blue]Business type: {business_type}[/blue]")

    # Generate invoices
    for i in track(range(count), description="Generating invoices"):
        # Select business type
        if business_type == "mixed":
            selected_type = random.choice(business_types)
        else:
            selected_type = business_type

        # Random number of items
        num_items = random.randint(min_items, max_items)

        # Generate invoice data
        invoice_data = invoice_generator.create_invoice_data(selected_type, num_items)

        # Generate filename
        filename = f"synthetic_invoice_{i + 1:03d}_{selected_type}.png"
        image_path = images_dir / filename

        # Create invoice image
        invoice_generator.create_invoice_image(invoice_data, str(image_path))

        # Save ground truth using our mapping
        ground_truth_generator.save_ground_truth(filename, invoice_data)

        console.print(f"  Generated: {filename} ({selected_type}, {num_items} items)")

    console.print(f"\n[green]✓ Generated {count} invoices with ground truth[/green]")
    console.print(
        f"[yellow]Ground truth CSV: {ground_truth_generator.ground_truth_file}[/yellow]"
    )
    console.print(f"[yellow]Images directory: {images_dir}[/yellow]")

    # Print sample evaluation command
    console.print("\n[bold]To evaluate models on this dataset:[/bold]")
    console.print(
        f"python -m vision_processor.cli.evaluation_cli compare {ground_truth_generator.ground_truth_file} --images-dir {images_dir}"
    )


@app.command()
def verify(
    ground_truth_file: str = typer.Argument(..., help="Path to ground truth CSV file"),
) -> None:
    """Verify ground truth file and show sample data."""

    ground_truth_path = Path(ground_truth_file)
    if not ground_truth_path.exists():
        console.print(
            f"[red]Error: Ground truth file not found: {ground_truth_file}[/red]"
        )
        raise typer.Exit(1)

    # Load and display sample
    with ground_truth_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    console.print(f"[green]Ground truth file: {ground_truth_file}[/green]")
    console.print(f"[blue]Total records: {len(rows)}[/blue]")

    if rows:
        # Show first record
        console.print("\n[bold]Sample record:[/bold]")
        for key, value in rows[0].items():
            if value and value != "N/A":
                console.print(f"  {key}: {value}")

        # Show field coverage
        console.print("\n[bold]Field coverage:[/bold]")
        field_counts = {}
        for row in rows:
            for key, value in row.items():
                if key != "image_filename" and value != "N/A":
                    field_counts[key] = field_counts.get(key, 0) + 1

        for field, count in sorted(
            field_counts.items(), key=lambda x: x[1], reverse=True
        ):
            percentage = (count / len(rows)) * 100
            console.print(f"  {field}: {count}/{len(rows)} ({percentage:.1f}%)")


@app.command("quick-test")
def quick_test() -> None:
    """Generate a small test dataset (3 invoices) for quick testing."""
    import random

    from synthetic_invoice_generator import SyntheticInvoiceGenerator

    output_path = Path("./quick_test_dataset")
    images_dir = output_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    invoice_generator = SyntheticInvoiceGenerator()
    ground_truth_generator = GroundTruthGenerator(output_dir=output_path)

    console.print("[green]Generating quick test dataset (3 invoices)...[/green]")

    for i in range(3):
        # Generate different business types
        business_type = ["retail", "fuel", "professional"][i]
        num_items = random.randint(2, 4)

        invoice_data = invoice_generator.create_invoice_data(business_type, num_items)
        filename = f"test_invoice_{i + 1}_{business_type}.png"
        image_path = images_dir / filename

        invoice_generator.create_invoice_image(invoice_data, str(image_path))
        ground_truth_generator.save_ground_truth(filename, invoice_data)

        console.print(f"  Generated: {filename}")

    console.print("\n[green]✓ Quick test dataset ready![/green]")
    console.print(f"[yellow]Location: {output_path}[/yellow]")
    console.print(
        f"[yellow]Ground truth: {ground_truth_generator.ground_truth_file}[/yellow]"
    )


if __name__ == "__main__":
    app()
