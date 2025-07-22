#!/usr/bin/env python3
"""Investigate Extraction Results
===============================

Analyzes why core fields aren't being recognized.
"""

import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from vision_processor.config.production_schema import PRODUCTION_SCHEMA

console = Console()


def load_results():
    """Load the extraction results."""
    results_path = Path("results/comparison_results.json")
    if not results_path.exists():
        console.print("❌ No results file found", style="red")
        return None

    with results_path.open() as f:
        return json.load(f)


def analyze_field_matching():
    """Analyze why fields aren't matching schema."""
    console.print("\n🔍 Analyzing Field Matching Issues", style="bold blue")
    console.print("=" * 50)

    # Load results
    results = load_results()
    if not results:
        return

    # Get schema fields
    schema_fields = set(PRODUCTION_SCHEMA.get_all_fields())
    core_fields = set(PRODUCTION_SCHEMA.get_core_fields())

    console.print("\n📋 Production Schema:")
    console.print(f"   Total fields: {len(schema_fields)}")
    console.print(f"   Core fields: {len(core_fields)}")
    console.print(f"   Sample core fields: {list(core_fields)[:5]}")

    # Check extracted fields from both models
    for model_name in ["llama", "internvl"]:
        if model_name in results and "extraction_results" in results[model_name]:
            console.print(f"\n🤖 {model_name.upper()} Extraction Analysis:")

            # Get all unique extracted fields
            all_extracted_fields = set()
            sample_extraction = None

            for img_result in results[model_name]["extraction_results"]:
                if "fields" in img_result:
                    all_extracted_fields.update(img_result["fields"].keys())
                    if not sample_extraction and img_result["fields"]:
                        sample_extraction = img_result

            console.print(f"   Unique fields extracted: {len(all_extracted_fields)}")
            console.print(f"   Sample fields: {list(all_extracted_fields)[:10]}")

            # Check overlap with schema
            extracted_upper = {f.upper() for f in all_extracted_fields}
            schema_upper = {f.upper() for f in schema_fields}

            overlap = extracted_upper & schema_upper
            console.print(f"   Fields matching schema: {len(overlap)}")
            if overlap:
                console.print(f"   Matching fields: {list(overlap)[:5]}")

            # Show a sample extraction
            if sample_extraction:
                console.print(f"\n   📄 Sample extraction from {sample_extraction.get('image_name', 'unknown')}:")
                for i, (key, value) in enumerate(sample_extraction["fields"].items()):
                    if i < 5:
                        console.print(f"      {key}: {value}")

            # Check why core fields aren't found
            missing_core = core_fields - extracted_upper
            if missing_core:
                console.print(f"\n   ⚠️  Missing core fields: {list(missing_core)[:5]}")


def check_field_definitions():
    """Check production schema field definitions."""
    console.print("\n📋 Production Schema Field Definitions", style="bold blue")
    console.print("=" * 50)

    # Show some field definitions
    sample_fields = ["DATE", "TOTAL", "GST", "ABN", "INVOICE_NUMBER"]

    table = Table(title="Sample Field Definitions")
    table.add_column("Field", style="cyan")
    table.add_column("Category", style="green")
    table.add_column("Required", style="yellow")
    table.add_column("Core", style="magenta")

    for field_name in sample_fields:
        field_def = PRODUCTION_SCHEMA.get_field_definition(field_name)
        if field_def:
            table.add_row(
                field_name,
                field_def.category.value,
                "Yes" if field_def.is_required else "No",
                "Yes" if field_def.is_core else "No"
            )
        else:
            table.add_row(field_name, "NOT FOUND", "-", "-")

    console.print(table)


def suggest_fixes():
    """Suggest fixes for the field matching issue."""
    console.print("\n💡 Suggested Fixes", style="bold yellow")
    console.print("=" * 50)

    console.print("\n1. **Field Name Normalization**")
    console.print("   The extracted field names may not match schema exactly.")
    console.print("   Example: 'Invoice_Date' vs 'DATE' vs 'INVOICE_DATE'")

    console.print("\n2. **Check DynamicFieldExtractor**")
    console.print("   Look at: vision_processor/extraction/dynamic_extractor.py")
    console.print("   The _map_to_production_fields() method may need adjustment")

    console.print("\n3. **Schema Field Names**")
    console.print("   Verify production_schema.py has the expected field names")

    console.print("\n4. **Field Mapping Logic**")
    console.print("   The comparison may be case-sensitive or have other issues")


def main():
    """Run the investigation."""
    console.print("🔍 Investigating Extraction Results", style="bold")

    analyze_field_matching()
    check_field_definitions()
    suggest_fixes()

    console.print("\n✅ Investigation complete!", style="bold green")


if __name__ == "__main__":
    main()
