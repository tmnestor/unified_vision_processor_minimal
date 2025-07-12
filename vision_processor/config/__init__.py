"""Configuration Package

Unified configuration management combining Llama and InternVL capabilities.
Supports environment-driven configuration with intelligent hardware detection.
"""

from .model_factory import ModelFactory
from .unified_config import (
    ExtractionMethod,
    ProcessingPipeline,
    ProductionAssessment,
    UnifiedConfig,
)

__all__ = [
    "ExtractionMethod",
    "ModelFactory",
    "ProcessingPipeline",
    "ProductionAssessment",
    "UnifiedConfig",
]
