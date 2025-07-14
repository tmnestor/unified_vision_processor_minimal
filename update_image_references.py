#!/usr/bin/env python3
"""Update the ground truth CSV to reference actual image files in datasets."""

import csv
from pathlib import Path


def update_image_references():
    """Update ground truth to reference existing image files."""

    csv_file = "evaluation_ground_truth.csv"

    # Load current ground truth
    with Path(csv_file).open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        data = list(reader)

    print("Current ground truth entries:")
    for entry in data:
        print(f"  - {entry['image_file']}: {entry['STORE']} (${entry['TOTAL']})")

    print(f"\nYou have {len(data)} ground truth entries for:")
    print("1. image14.png - SPOTLIGHT")
    print("2. the_good_guys.png - THE GOOD GUYS")
    print("3. dan_murphys.png - DAN MURPHY'S")
    print("4. david_jones.png - DAVID JONES")
    print("5. bcfaustralia.png - BCFAUSTRALIA")

    print("\nTo run the evaluation with existing images, you can either:")
    print("1. Rename your 4 new receipt images to match the ground truth filenames:")
    print("   - the_good_guys.png")
    print("   - dan_murphys.png")
    print("   - david_jones.png")
    print("   - bcfaustralia.png")

    print("\n2. OR manually map them to existing image files if they match:")

    # Show some existing files for reference
    datasets_dir = Path("datasets")
    existing_images = list(datasets_dir.glob("*.png"))[:10]
    print("\nFirst 10 images in datasets/:")
    for img in existing_images:
        print(f"  - {img.name}")

    print(f"\nTotal images available: {len(list(datasets_dir.glob('*.png')))}")

    print("\n📋 To run evaluation with current ground truth:")
    print("1. Make sure your receipt images are named correctly in datasets/")
    print("2. Run: python evaluate_extraction_performance.py")


if __name__ == "__main__":
    update_image_references()
