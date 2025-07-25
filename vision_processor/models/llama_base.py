"""Llama-3.2-Vision Base Model Implementation

Core model class and initialization logic separated from implementation details.
"""

import logging

from .base_model import BaseVisionModel, ModelCapabilities

logger = logging.getLogger(__name__)


class LlamaVisionModelBase(BaseVisionModel):
    """Base Llama-3.2-Vision implementation focusing on core functionality.

    This class contains the essential model structure and initialization,
    with implementation details delegated to specialized modules.
    """

    def __init__(self, *args, **kwargs):
        """Initialize LlamaVisionModel with configuration extraction."""
        super().__init__(*args, **kwargs)

        # Extract config from kwargs and store as direct attribute
        self.config = kwargs.get("config")

        # Extract repetition control configuration
        yaml_repetition_config = getattr(self.config, "yaml_config", {}).get(
            "repetition_control", {}
        )
        repetition_config = kwargs.get("repetition_control", yaml_repetition_config)
        
        # Store repetition control settings
        self.repetition_enabled = repetition_config.get("enabled", True)

        # Read max_new_tokens_limit from YAML config
        yaml_limit = None
        if hasattr(self, "config") and self.config:
            model_config = getattr(self.config, "yaml_config", {}).get(
                "model_config", {}
            )
            yaml_limit = model_config.get("llama", {}).get("max_new_tokens_limit")

        # Use YAML config as single source of truth
        self.max_new_tokens_limit = yaml_limit or 1024

        # Special tokens to clean from responses
        self.cleanup_tokens = [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|image|>",
            "[INST]",
            "[/INST]",
            "<s>",
            "</s>",
        ]

        logger.info(
            f"Llama repetition control - enabled={self.repetition_enabled}, "
            f"max_new_tokens_limit={self.max_new_tokens_limit}"
        )

    def _get_capabilities(self) -> ModelCapabilities:
        """Return Llama-3.2-Vision capabilities."""
        return ModelCapabilities(
            supports_multi_gpu=True,
            supports_quantization=True,
            supports_highlight_detection=False,  # Not available in Llama
            supports_batch_processing=True,
            max_image_size=(1024, 1024),  # Llama has lower limits
            memory_efficient=True,
            cross_platform=True,
        )

    @property
    def cross_platform_compatible(self) -> bool:
        """Check if model is cross-platform compatible."""
        return self.capabilities.cross_platform

    def _validate_config(self) -> None:
        """Validate that required configuration is available.
        
        Raises:
            RuntimeError: If required configuration is missing
        """
        if not hasattr(self, "config"):
            raise RuntimeError(
                "❌ FATAL: No production config found for Llama model\n"
                "💡 Expected: config parameter passed during model creation\n"
                "💡 Fix: Ensure config is passed via model_registry.create_model()\n"
                "💡 YAML file: model_comparison.yaml with device_config section"
            )

        if not hasattr(self.config, "device_config"):
            raise RuntimeError(
                "❌ FATAL: No device_config found in production config\n"
                "💡 Expected: device_config section in YAML configuration\n"
                "💡 Fix: Add device_config section to model_comparison.yaml\n"
                "💡 Example:\n"
                "   device_config:\n"
                "     gpu_strategy: 'single_gpu'\n"
                "     device_maps:\n"
                "       llama:\n"
                "         device_map: {'': 0}"
            )