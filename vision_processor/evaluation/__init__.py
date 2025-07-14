"""Vision Processor Evaluation Module.

This module provides comprehensive evaluation tools for comparing vision model
performance on document extraction tasks.

Components:
- ExtractionEvaluator: Main evaluation engine
- FieldAccuracyCalculator: Accuracy metrics
- GroundTruthManager: Test data management
- ReportGenerator: Results visualization

Example:
    from vision_processor.evaluation import ExtractionEvaluator

    evaluator = ExtractionEvaluator("ground_truth.csv", "datasets/")
    results = evaluator.compare_models(["internvl3", "llama32_vision"])
    evaluator.generate_report(results)
"""

from .evaluator import ExtractionEvaluator

__all__ = ["ExtractionEvaluator"]
__version__ = "1.0.0"
