#!/usr/bin/env python3
"""Helper script to add ground truth data for new images interactively."""

import csv
from pathlib import Path
from typing import Dict, List

from rich.console import Console
from rich.prompt import Prompt


def load_existing_data(csv_file: str) -> List[Dict[str, str]]:
    """Load existing ground truth data."""
    if not Path(csv_file).exists():
        return []

    data = []
    with Path(csv_file).open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        data = list(reader)
    return data


def get_csv_headers() -> List[str]:
    """Get the standard CSV headers."""
    return [
        "image_file",
        "DATE",
        "STORE",
        "ABN",
        "GST",
        "TOTAL",
        "SUBTOTAL",
        "ITEMS",
        "QUANTITIES",
        "PRICES",
        "RECEIPT_NUMBER",
        "PAYMENT_METHOD",
        "DOCUMENT_TYPE",
        "ADDRESS",
        "PHONE",
        "TIME",
        "CARD_NUMBER",
        "AUTH_CODE",
        "STATUS",
    ]


def add_ground_truth_entry(console: Console) -> Dict[str, str]:
    """Interactively collect ground truth data for one image."""
    console.print("\n📝 Adding new ground truth entry")
    console.print("💡 Press Enter to skip optional fields")

    # Required fields
    image_file = Prompt.ask("📁 Image filename (e.g., image14.png)")

    # Core extraction fields
    date = Prompt.ask("📅 DATE (DD/MM/YYYY)", default="")
    store = Prompt.ask("🏪 STORE name", default="")
    abn = Prompt.ask("🔢 ABN (XX XXX XXX XXX)", default="")
    gst = Prompt.ask("💰 GST amount (numeric only)", default="")
    total = Prompt.ask("💵 TOTAL amount (numeric only)", default="")
    subtotal = Prompt.ask("📊 SUBTOTAL amount (numeric only)", default="")

    # Item details
    console.print("\n📦 Item Information (use | to separate multiple items)")
    items = Prompt.ask("📝 ITEMS (item1|item2|item3)", default="")
    quantities = Prompt.ask("📈 QUANTITIES (1|2|1)", default="")
    prices = Prompt.ask("💲 PRICES (1.50|2.30|5.00)", default="")

    # Document details
    receipt_number = Prompt.ask("🧾 RECEIPT_NUMBER", default="")
    payment_method = Prompt.ask("💳 PAYMENT_METHOD", default="")
    document_type = Prompt.ask("📄 DOCUMENT_TYPE", default="")

    # Optional details
    console.print("\n🔍 Optional Details")
    address = Prompt.ask("🏠 ADDRESS", default="")
    phone = Prompt.ask("📞 PHONE", default="")
    time = Prompt.ask("⏰ TIME", default="")
    card_number = Prompt.ask("💳 CARD_NUMBER (masked)", default="")
    auth_code = Prompt.ask("🔐 AUTH_CODE", default="")
    status = Prompt.ask("✅ STATUS", default="")

    return {
        "image_file": image_file,
        "DATE": date,
        "STORE": store,
        "ABN": abn,
        "GST": gst,
        "TOTAL": total,
        "SUBTOTAL": subtotal,
        "ITEMS": items,
        "QUANTITIES": quantities,
        "PRICES": prices,
        "RECEIPT_NUMBER": receipt_number,
        "PAYMENT_METHOD": payment_method,
        "DOCUMENT_TYPE": document_type,
        "ADDRESS": address,
        "PHONE": phone,
        "TIME": time,
        "CARD_NUMBER": card_number,
        "AUTH_CODE": auth_code,
        "STATUS": status,
    }


def save_ground_truth(data: List[Dict[str, str]], csv_file: str) -> None:
    """Save ground truth data to CSV."""
    headers = get_csv_headers()

    with Path(csv_file).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(data)


def main():
    """Main interactive ground truth addition."""
    console = Console()
    csv_file = "evaluation_ground_truth.csv"

    console.print("🎯 Ground Truth Data Entry Tool")
    console.print("=" * 50)

    # Load existing data
    existing_data = load_existing_data(csv_file)

    if existing_data:
        console.print(f"📊 Found {len(existing_data)} existing entries")
        for entry in existing_data[-3:]:  # Show last 3
            console.print(f"  - {entry['image_file']}: {entry['STORE']} ({entry['DATE']})")
    else:
        console.print("📝 Starting with empty ground truth file")

    while True:
        console.print(f"\n📋 Current entries: {len(existing_data)}")

        action = Prompt.ask(
            "What would you like to do?", choices=["add", "list", "save", "quit"], default="add"
        )

        if action == "add":
            try:
                new_entry = add_ground_truth_entry(console)
                existing_data.append(new_entry)
                console.print(f"✅ Added ground truth for {new_entry['image_file']}")
            except KeyboardInterrupt:
                console.print("\n⚠️ Entry cancelled")

        elif action == "list":
            if existing_data:
                console.print(f"\n📋 All {len(existing_data)} entries:")
                for i, entry in enumerate(existing_data, 1):
                    console.print(
                        f"  {i:2d}. {entry['image_file']:15} | {entry['STORE']:20} | {entry['DATE']}"
                    )
            else:
                console.print("📝 No entries yet")

        elif action == "save":
            save_ground_truth(existing_data, csv_file)
            console.print(f"💾 Saved {len(existing_data)} entries to {csv_file}")

        elif action == "quit":
            if existing_data:
                save = Prompt.ask("Save changes before quitting?", choices=["yes", "no"], default="yes")
                if save == "yes":
                    save_ground_truth(existing_data, csv_file)
                    console.print(f"💾 Saved {len(existing_data)} entries to {csv_file}")
            console.print("👋 Goodbye!")
            break


if __name__ == "__main__":
    main()
