"""Configuration Package

Unified configuration management with fail-fast validation.
Supports model factory pattern for InternVL3 and Llama-3.2-Vision.
"""

from .simple_config import SimpleConfig

# Note: ModelFactory removed - use model_registry directly
# Production config and unified config moved to legacy files
# Use SimpleConfig for new simplified system

__all__ = [
    "SimpleConfig",
]
