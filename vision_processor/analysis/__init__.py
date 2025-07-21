"""Analysis Modules for Model Comparison
====================================

Focused analysis modules for processing extraction results and generating
comprehensive performance metrics, field analysis, and comparison reports.
"""

from .comparison_metrics import ComparisonMetrics
from .field_analyzer import FieldAnalyzer
from .performance_analyzer import PerformanceAnalyzer

__all__ = ["PerformanceAnalyzer", "FieldAnalyzer", "ComparisonMetrics"]
