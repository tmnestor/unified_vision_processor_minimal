"""Simplified Vision Document Processing System

A streamlined system for processing documents using InternVL3 and Llama-3.2-Vision models.
Single-step processing with .env configuration and YAML-driven key extraction.

Key Features:
- Single-step processing pipeline
- .env configuration management
- Universal KEY-VALUE extraction
- Model-agnostic interface (InternVL3, Llama-3.2-Vision)
- YAML-driven key schema
- Rich CLI with typer framework
"""

__version__ = "0.1.0"
__author__ = "Developer"

# Core imports for easy access
from .config.simple_config import SimpleConfig
from .extraction.simple_extraction_manager import SimpleExtractionManager

__all__ = [
    "SimpleConfig",
    "SimpleExtractionManager",
]
