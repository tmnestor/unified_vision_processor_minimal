"""Model Comparison Orchestration
=============================

Orchestrates the complete model comparison pipeline including model loading,
image processing, field extraction, analysis, and reporting.
"""

from .comparison_runner import ComparisonRunner
from .model_validator import ModelValidator

__all__ = ["ComparisonRunner", "ModelValidator"]
