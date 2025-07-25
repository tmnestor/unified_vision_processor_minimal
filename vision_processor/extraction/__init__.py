"""Extraction Package

Simplified single-step extraction with .env configuration and YAML-driven parsing.
"""

from .simple_extraction_manager import ExtractionResult, SimpleExtractionManager

__all__ = [
    "ExtractionResult",
    "SimpleExtractionManager",
]
