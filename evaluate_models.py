#!/usr/bin/env python3
"""Simple evaluation script using the vision_processor.evaluation module."""

import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from vision_processor.evaluation import ExtractionEvaluator


def main():
    """Run model comparison evaluation."""
    # Setup
    ground_truth_csv = "evaluation_ground_truth.csv"
    images_dir = "datasets"

    if not Path(ground_truth_csv).exists():
        print(f"❌ Ground truth file not found: {ground_truth_csv}")
        return 1

    if not Path(images_dir).exists():
        print(f"❌ Images directory not found: {images_dir}")
        return 1

    # Create evaluator
    evaluator = ExtractionEvaluator(ground_truth_csv, images_dir)

    # Run comparison
    results = evaluator.compare_models()

    # Generate report
    evaluator.generate_report(results)

    print("🎉 Evaluation completed successfully!")
    print("📁 Results saved in: evaluation_results/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
