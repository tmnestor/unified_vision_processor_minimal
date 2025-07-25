"""Unified Vision Document Processing System

A production-ready system for processing documents using InternVL3 and Llama-3.2-Vision models.
Clean architecture with unified configuration management and maintainable patterns.

Key Features:
- Clean, maintainable architecture
- Unified ConfigManager system
- Universal KEY-VALUE extraction
- Model-agnostic interface (InternVL3, Llama-3.2-Vision)
- Fail-fast configuration validation
- Rich CLI with typer framework
"""

__version__ = "0.1.0"
__author__ = "Tod"

# Core imports for easy access
from .config import ConfigManager
from .extraction.simple_extraction_manager import SimpleExtractionManager

__all__ = [
    "ConfigManager",
    "SimpleExtractionManager",
]
