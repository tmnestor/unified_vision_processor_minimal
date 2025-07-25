"""Configuration Package

Unified configuration management with fail-fast validation.
Supports model factory pattern for InternVL3 and Llama-3.2-Vision.
"""

from .config_manager import ConfigManager

# ConfigManager is the unified configuration system
# Legacy SimpleConfig moved to backup/legacy_config/

__all__ = [
    "ConfigManager",
]
