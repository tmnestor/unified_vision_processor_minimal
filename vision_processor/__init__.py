"""Unified Vision Document Processing Architecture

A comprehensive system for processing Australian tax documents using both InternVL3 and Llama-3.2-Vision models.
Integrates advanced computer vision capabilities with robust document processing pipelines.

Key Features:
- Model-agnostic processing with Llama 7-step pipeline foundation
- Australian Tax Office (ATO) compliance validation
- Multi-GPU optimization and single V100 production deployment
- 11 specialized document type handlers
- 4-component confidence scoring system
- Graceful degradation with intelligent fallbacks
"""

__version__ = "0.1.0"
__author__ = "Developer"

# Core imports for easy access
from .config.unified_config import UnifiedConfig
from .extraction.hybrid_extraction_manager import UnifiedExtractionManager

__all__ = [
    "UnifiedConfig",
    "UnifiedExtractionManager",
]
