#!/usr/bin/env python3
"""
Reconstruct Ground Truth from Existing Synthetic Invoices

This script attempts to reconstruct ground truth by extracting data from
existing synthetic invoice images using both models and manual verification.

Usage:
    python reconstruct_ground_truth.py --images-dir ./datasets --output ground_truth.csv
"""

import csv
import json
from pathlib import Path
from typing import Dict, List

import typer
from rich.console import Console
from rich.prompt import Confirm, Prompt
from rich.table import Table

console = Console()
app = typer.Typer()


class GroundTruthReconstructor:
    """Reconstruct ground truth from existing synthetic invoices."""

    def __init__(self, output_file: str = "ground_truth.csv"):
        self.output_file = Path(output_file)
        self.expected_fields = [
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

    def create_shell_invoice_ground_truth(self) -> Dict:
        """Create ground truth for the Shell invoice example."""
        return {
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

    def create_ground_truth_template(self) -> List[Dict]:
        """Create ground truth templates for common synthetic invoices."""
        templates = []

        # Add Shell invoice
        templates.append(self.create_shell_invoice_ground_truth())

        # Add more templates for common patterns
        # These are based on the synthetic invoice generator patterns

        # Woolworths template
        templates.append(
            {
                "image_filename": "woolworths_invoice.png",
                "DOCUMENT_TYPE": "TAX INVOICE",
                "SUPPLIER": "Woolworths Limited",
                "ABN": "88 000 014 675",  # Real Woolworths ABN
                "PAYER_NAME": "John Smith",
                "PAYER_ADDRESS": "123 Main Street, Sydney NSW 2000",
                "PAYER_PHONE": "(02) 1234 5678",
                "PAYER_EMAIL": "john.smith@email.com",
                "INVOICE_DATE": "01/01/2025",
                "DUE_DATE": "15/01/2025",
                "GST": "$10.00",
                "TOTAL": "$110.00",
                "SUBTOTAL": "$100.00",
                "SUPPLIER_WEBSITE": "www.woolworths.com.au",
                "QUANTITIES": "1, 2, 3",
                "PRICES": "$20.00, $30.00, $50.00",
                "BUSINESS_ADDRESS": "1 Woolworths Way, Bella Vista NSW 2153",
                "BUSINESS_PHONE": "1300 908 631",
                "BANK_NAME": "N/A",
                "BSB_NUMBER": "N/A",
                "BANK_ACCOUNT_NUMBER": "N/A",
                "ACCOUNT_HOLDER": "N/A",
                "STATEMENT_PERIOD": "N/A",
                "OPENING_BALANCE": "N/A",
                "CLOSING_BALANCE": "N/A",
                "DESCRIPTIONS": "Groceries, Household items, Fresh produce",
            }
        )

        return templates

    def save_ground_truth(self, ground_truth_data: List[Dict]) -> None:
        """Save ground truth data to CSV file."""
        with self.output_file.open("w", newline="", encoding="utf-8") as f:
            fieldnames = ["image_filename"] + self.expected_fields
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(ground_truth_data)

        console.print(f"[green]✓ Saved ground truth to: {self.output_file}[/green]")

    def interactive_ground_truth_builder(
        self,
        image_file: str,
        llama_extraction: Dict = None,
        internvl_extraction: Dict = None,
    ) -> Dict:
        """Build ground truth interactively using model extractions as hints."""
        console.print(f"\n[bold]Creating ground truth for: {image_file}[/bold]")

        ground_truth = {"image_filename": image_file}

        # Create comparison table
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Field", style="cyan", width=20)
        table.add_column("Llama", style="green")
        table.add_column("InternVL", style="blue")
        table.add_column("Ground Truth", style="yellow")

        for field in self.expected_fields:
            llama_val = (
                llama_extraction.get(field, "N/A") if llama_extraction else "N/A"
            )
            internvl_val = (
                internvl_extraction.get(field, "N/A") if internvl_extraction else "N/A"
            )

            # If both models agree, use that as default
            if llama_val == internvl_val and llama_val != "N/A":
                default_val = llama_val
            else:
                default_val = "N/A"

            # For critical fields, prompt for verification
            if field in ["SUPPLIER", "ABN", "TOTAL", "GST", "INVOICE_DATE"]:
                table.add_row(field, llama_val, internvl_val, "?")
                console.print(table)

                prompt = f"Enter value for {field}"
                if default_val != "N/A":
                    prompt += f" (default: {default_val})"
                prompt += ": "

                value = Prompt.ask(prompt, default=default_val)
                ground_truth[field] = value

                # Update table
                table.rows[-1] = (field, llama_val, internvl_val, value)
            else:
                # For less critical fields, auto-accept if models agree
                ground_truth[field] = default_val
                table.add_row(field, llama_val, internvl_val, default_val)

        console.print(table)

        if Confirm.ask("Save this ground truth?"):
            return ground_truth
        else:
            return None


@app.command()
def create_basic(
    output: str = typer.Option(
        "ground_truth.csv", "--output", "-o", help="Output CSV file"
    ),
) -> None:
    """Create basic ground truth file with known examples."""
    reconstructor = GroundTruthReconstructor(output)

    # Create basic templates
    templates = reconstructor.create_ground_truth_template()

    # Save to CSV
    reconstructor.save_ground_truth(templates)

    console.print(f"[green]Created ground truth with {len(templates)} examples[/green]")


@app.command()
def add_shell_invoice(
    ground_truth_file: str = typer.Option(
        "ground_truth.csv", "--file", "-f", help="Ground truth CSV file"
    ),
) -> None:
    """Add the Shell invoice example to ground truth."""
    ground_truth_path = Path(ground_truth_file)

    # Load existing ground truth if it exists
    existing_data = []
    if ground_truth_path.exists():
        with ground_truth_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_data = list(reader)

    # Create Shell invoice ground truth
    reconstructor = GroundTruthReconstructor()
    shell_gt = reconstructor.create_shell_invoice_ground_truth()

    # Check if already exists
    for row in existing_data:
        if row.get("image_filename") == shell_gt["image_filename"]:
            console.print(
                "[yellow]Shell invoice already exists in ground truth[/yellow]"
            )
            if Confirm.ask("Overwrite existing entry?"):
                existing_data.remove(row)
            else:
                return

    # Add to data
    existing_data.append(shell_gt)

    # Save
    reconstructor.output_file = ground_truth_path
    reconstructor.save_ground_truth(existing_data)


@app.command()
def from_extractions(
    llama_results: str = typer.Argument(..., help="Llama extraction results JSON"),
    internvl_results: str = typer.Argument(
        ..., help="InternVL extraction results JSON"
    ),
    output: str = typer.Option(
        "ground_truth.csv", "--output", "-o", help="Output CSV file"
    ),
    interactive: bool = typer.Option(
        True, "--interactive/--auto", help="Interactive mode"
    ),
) -> None:
    """Create ground truth by comparing model extractions."""

    # Load extraction results
    with Path(llama_results).open("r") as f:
        llama_data = json.load(f)

    with Path(internvl_results).open("r") as f:
        internvl_data = json.load(f)

    reconstructor = GroundTruthReconstructor(output)
    ground_truth_data = []

    # Process each image
    images = set(llama_data.keys()) | set(internvl_data.keys())

    for image in sorted(images):
        llama_ext = llama_data.get(image, {})
        internvl_ext = internvl_data.get(image, {})

        if interactive:
            gt = reconstructor.interactive_ground_truth_builder(
                image, llama_ext, internvl_ext
            )
            if gt:
                ground_truth_data.append(gt)
        else:
            # Auto mode - use agreed values
            gt = {"image_filename": image}
            for field in reconstructor.expected_fields:
                llama_val = llama_ext.get(field, "N/A")
                internvl_val = internvl_ext.get(field, "N/A")

                if llama_val == internvl_val:
                    gt[field] = llama_val
                else:
                    gt[field] = "N/A"  # Disagreement = unknown

            ground_truth_data.append(gt)

    # Save ground truth
    reconstructor.save_ground_truth(ground_truth_data)


if __name__ == "__main__":
    app()
