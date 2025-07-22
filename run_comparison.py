#!/usr/bin/env python3
"""Run Model Comparison - Simplified Version
==========================================

Standalone script that uses the original model_comparison_legacy.py
with the simplified approach.
"""

import os
from pathlib import Path


def main():
    """Run the model comparison using legacy script."""
    print("🚀 Running Simplified Model Comparison")
    print("=" * 50)

    # Check if we're in conda environment
    if 'CONDA_DEFAULT_ENV' not in os.environ:
        print("⚠️  Please activate conda environment:")
        print("   conda activate unified_vision_processor")
        print("   export KMP_DUPLICATE_LIB_OK=TRUE")
        return 1

    # Check if legacy script exists
    legacy_script = Path("model_comparison_legacy.py")
    if not legacy_script.exists():
        print("❌ model_comparison_legacy.py not found")
        return 1

    # Check if datasets exist
    datasets_path = Path("./datasets")
    if not datasets_path.exists() or not list(datasets_path.glob("*.png")):
        print("❌ No images found in ./datasets directory")
        return 1

    print(f"✅ Found {len(list(datasets_path.glob('*.png')))} images in datasets")

    # Run the legacy comparison
    print("\n📦 Using legacy model_comparison.py (full functionality)")
    print("Note: This will show core_fields_found values correctly after the refactoring.")

    cmd = [
        "python", "model_comparison_legacy.py", "compare",
        "--datasets-path", "./datasets",
        "--output-dir", "./results",
        "--models", "llama,internvl"
    ]

    print(f"\n🛠️  Running: {' '.join(cmd)}")

    # Use os.system for simplicity
    result = os.system(" ".join(cmd))

    if result == 0:
        print("\n✅ Comparison completed successfully!")
        print("Results saved to ./results/")
        print("\nKey files:")
        print("  - results/production_results.csv (core_fields_found now shows actual values!)")
        print("  - results/comparison_results.json")
    else:
        print("\n❌ Comparison failed")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
