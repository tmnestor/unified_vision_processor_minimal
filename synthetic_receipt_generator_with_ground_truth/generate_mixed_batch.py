#!/usr/bin/env python3
"""Generate a mixed batch of documents including bank statements."""

import random
from pathlib import Path

import typer
from rich.console import Console
from synthetic_invoice_generator import SyntheticInvoiceGenerator

console = Console()


def generate_mixed_batch_with_bank_statements(
    output_dir: str = typer.Option(
        "test_synthetic", help="Output directory for generated documents"
    ),
    count: int = typer.Option(10, help="Number of documents to generate"),
    bank_statements: int = typer.Option(2, help="Number of bank statements to include"),
):
    """Generate documents including bank statements with configurable output directory."""
    generator = SyntheticInvoiceGenerator()
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    console.print(
        f"🏭 Generating {count} mixed documents including {bank_statements} bank statements",
        style="bold blue",
    )
    console.print(f"📁 Output directory: {output_path.absolute()}", style="cyan")
    console.print("📊 Ground truth CSV will be generated for evaluation", style="blue")

    # Weight retail and fuel types more heavily for receipt-style documents
    business_types = [
        "retail",
        "retail",
        "retail",
        "fuel",
        "fuel",
        "accommodation",
        "professional",
    ]
    invoices_data = []

    # Determine which documents should be bank statements
    bank_statement_indices = random.sample(range(count), min(bank_statements, count))

    for i in range(count):
        # Include bank data for selected indices
        include_bank_data = i in bank_statement_indices

        if include_bank_data:
            # For bank statements, use retail business type but add bank data
            selected_type = "retail"
            console.print(f"  {i + 1:2d}. Will be BANK STATEMENT")
        else:
            # Regular invoice/receipt
            selected_type = random.choice(business_types)

        # Generate with varying number of items
        num_items = random.randint(2, 6)

        # Generate invoice data
        invoice_data = generator.create_invoice_data(
            selected_type, num_items, include_bank_data
        )

        # Override document type for bank statements
        if include_bank_data:
            invoice_data["document_type"] = "BANK STATEMENT"

        # Create filename
        filename = f"synthetic_invoice_{i + 1:03d}.png"
        file_path = output_path / filename

        # Store filename in invoice data for ground truth CSV
        invoice_data["image_filename"] = filename

        # Create image
        generator.create_invoice_image(invoice_data, str(file_path))

        # Add to collection for ground truth CSV
        invoices_data.append(invoice_data)

        doc_type = "BANK STATEMENT" if include_bank_data else selected_type.upper()
        bank_indicator = " + bank data" if include_bank_data else ""
        console.print(
            f"  {i + 1:2d}. {filename} ({doc_type}, {num_items} items{bank_indicator})"
        )

    # Generate ground truth CSV
    csv_path = output_path / "evaluation_ground_truth.csv"
    generator.generate_ground_truth_csv(invoices_data, str(csv_path))
    console.print(f"📄 Ground truth CSV: {csv_path}", style="cyan")

    console.print(
        f"✅ Generated {count} documents in {output_path}", style="bold green"
    )
    console.print(
        f"📋 Includes {bank_statements} bank statements with banking field data",
        style="green",
    )


if __name__ == "__main__":
    typer.run(generate_mixed_batch_with_bank_statements)
