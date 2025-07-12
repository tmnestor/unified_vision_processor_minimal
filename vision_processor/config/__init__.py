"""Configuration Package

Simplified configuration management with .env loading.
Supports model factory pattern for InternVL3 and Llama-3.2-Vision.
"""

from .model_factory import ModelFactory
from .simple_config import SimpleConfig

__all__ = [
    "ModelFactory",
    "SimpleConfig",
]
