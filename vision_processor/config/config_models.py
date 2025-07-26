"""Configuration dataclass models for Vision Processor.

Provides structured configuration objects to replace dynamic type() generation,
improving IDE support, type safety, and maintainability.
"""

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class ModelPaths:
    """Model file paths configuration."""

    llama: str = ""
    internvl: str = ""


@dataclass
class MemoryConfig:
    """Memory and hardware configuration."""

    v100_limit_gb: float = 16.0
    safety_margin: float = 0.85


@dataclass
class ImageProcessingConfig:
    """Image processing configuration."""

    max_image_size: int = 1024
    timeout_seconds: int = 10


@dataclass
class DefaultsConfig:
    """CLI default settings."""

    datasets_path: str = ""  # Will be set from YAML, no hardcoded default
    max_tokens: int = 700
    quantization: bool = True
    output_dir: str = ""  # Will be set from YAML, no hardcoded default
    models: str = "llama,internvl"
    trust_remote_code: bool = True
    debug_mode: bool = False
    verbose_mode: bool = False  # NEW - Controls detailed status messages
    console_output: bool = True  # NEW - Enable rich console formatting
    log_level: str = "INFO"  # NEW - ERROR, WARNING, INFO, DEBUG


@dataclass
class ModelSpecificConfig:
    """Model-specific settings."""

    max_new_tokens_limit: int = 2024
    confidence_score: float = 0.85


@dataclass
class RepetitionControlConfig:
    """Repetition control configuration."""

    enabled: bool = True
    word_threshold: float = 0.15
    phrase_threshold: int = 2
    fallback_max_tokens: int = 1000


@dataclass
class DeviceMapConfig:
    """Device mapping for a single model."""

    strategy: str = "single_gpu"
    device_map: Dict[str, Any] = field(default_factory=lambda: {"": 0})
    quantization_compatible: bool = True


@dataclass
class DeviceConfig:
    """Device configuration for deployment."""

    gpu_strategy: str = "single_gpu"
    target_gpu: int = 0
    v100_mode: bool = True
    memory_limit_gb: int = 16
    device_maps: Dict[str, DeviceMapConfig] = field(default_factory=dict)
    # For backward compatibility
    original_device_config: str = "auto"

    def get_device_map_for_model(self, model_name: str) -> Dict[str, Any]:
        """Get device map for a specific model."""
        if model_name in self.device_maps:
            return self.device_maps[model_name].device_map
        return {"": 0}  # Default to single GPU


@dataclass
class ProcessingConfig:
    """Processing configuration for model loading."""

    memory_limit_mb: int
    enable_gradient_checkpointing: bool = True
    use_flash_attention: bool = True
    quantization: bool = True
    batch_size: int = 1
    max_tokens: int = 800


@dataclass
class QualityThresholds:
    """Quality rating thresholds for field extraction."""

    excellent: int = 12
    good: int = 8
    fair: int = 5
    poor: int = 0


@dataclass
class SpeedThresholds:
    """Processing speed thresholds."""

    very_fast: float = 15.0
    fast: float = 25.0
    moderate: float = 40.0
