"""Vision Models Package

Provides unified interface for vision models with model-agnostic processing capabilities.
Supports both InternVL3 and Llama-3.2-Vision models with optimizations for different hardware configurations.
"""

from .base_model import (
    BaseVisionModel,
    DeviceConfig,
    ModelCapabilities,
    ModelResponse,
    ModelType,
)
from .model_utils import (
    DeviceManager,
    ModelProfiler,
    QuantizationHelper,
)

# Import actual models if available
try:
    from .internvl_model import InternVLModel  # noqa: F401

    _has_internvl = True
except ImportError:
    _has_internvl = False

try:
    from .llama_model import LlamaVisionModel  # noqa: F401

    _has_llama = True
except ImportError:
    _has_llama = False

__all__ = [
    # Base model components
    "BaseVisionModel",
    "ModelType",
    "DeviceConfig",
    "ModelResponse",
    "ModelCapabilities",
    # Utility classes
    "DeviceManager",
    "QuantizationHelper",
    "ModelProfiler",
]

# Add models to exports if available
if _has_internvl:
    __all__.append("InternVLModel")
if _has_llama:
    __all__.append("LlamaVisionModel")
