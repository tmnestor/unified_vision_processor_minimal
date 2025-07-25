"""Configuration Package

Unified configuration management with fail-fast validation.
Supports model factory pattern for InternVL3 and Llama-3.2-Vision.
"""

from .config_manager import ConfigManager
from .simple_config import SimpleConfig

# ConfigManager is the new unified configuration system
# SimpleConfig maintained for backward compatibility

__all__ = [
    "ConfigManager",
    "SimpleConfig",
]
