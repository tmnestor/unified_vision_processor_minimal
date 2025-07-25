"""Extraction Package

Simplified single-step extraction with YAML configuration and dynamic parsing.
"""

from .extraction_manager import ExtractionResult, SimpleExtractionManager

__all__ = [
    "ExtractionResult",
    "SimpleExtractionManager",
]
