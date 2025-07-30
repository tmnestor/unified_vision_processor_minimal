"""Unified Configuration Manager

Consolidates all configuration logic into a single, maintainable class.
Eliminates dynamic fallbacks in favor of explicit validation and clear error messages.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from ..exceptions import ConfigurationError, ValidationError
from ..utils.logging_config import VisionProcessorLogger
from .config_models import (
    DefaultsConfig,
    DeviceConfig,
    DeviceMapConfig,
    ImageProcessingConfig,
    MemoryConfig,
    ModelPaths,
    ModelSpecificConfig,
    ProcessingConfig,
    QualityThresholds,
    RepetitionControlConfig,
    SpeedThresholds,
)


class DeviceMapManager:
    """Manages device mapping configurations."""

    def __init__(self, device_config: DeviceConfig):
        self.device_config = device_config

    def get_device_map_for_model(self, model_name: str) -> Dict[str, int]:
        """Get device map for a specific model with explicit error handling."""
        if model_name not in self.device_config.device_maps:
            available_models = list(self.device_config.device_maps.keys())
            raise ConfigurationError(
                f"❌ FATAL: No device mapping found for model '{model_name}'\n"
                f"💡 Available models: {available_models}\n"
                f"💡 Fix: Add device mapping to model_comparison.yaml:\n"
                f"   device_config:\n"
                f"     device_maps:\n"
                f"       {model_name}:\n"
                f"         strategy: 'single_gpu'\n"
                f"         device_map: {{'': 0}}\n"
                f"         quantization_compatible: true"
            )

        device_map_config = self.device_config.device_maps[model_name]
        return device_map_config.device_map

    def validate_device_configuration(self) -> None:
        """Validate device configuration with explicit error messages."""
        if not self.device_config.device_maps:
            raise ConfigurationError(
                "❌ FATAL: No device mappings found in configuration\n"
                "💡 Expected: device_config.device_maps section in YAML\n"
                "💡 Fix: Add device_config section to model_comparison.yaml"
            )

        # Validate each device map
        for model_name, device_map_config in self.device_config.device_maps.items():
            if not device_map_config.device_map:
                raise ConfigurationError(
                    f"❌ FATAL: Empty device_map for model '{model_name}'\n"
                    f"💡 Fix: Specify device_map in configuration:\n"
                    f"   device_config:\n"
                    f"     device_maps:\n"
                    f"       {model_name}:\n"
                    f"         device_map: {{'': 0}}"
                )


class ConfigManager:
    """Unified configuration manager that eliminates complexity and dynamic fallbacks."""

    def __init__(self, yaml_file: str = "model_comparison.yaml"):
        """Initialize configuration manager with fail-fast validation.

        Args:
            yaml_file: Path to YAML configuration file

        Raises:
            ConfigurationError: If configuration cannot be loaded or is invalid
        """
        self.yaml_file = Path(yaml_file)
        # Store original YAML for backward compatibility
        self._yaml_config_data = self._load_yaml_config()

        # Parse configuration sections into structured objects
        self._parse_configuration()

        # Set up device mapping manager
        self.device_manager = DeviceMapManager(self.device_config)

        # Validate configuration
        self._validate_configuration()

        # Initialize logger last, after all configuration is complete
        self.logger = VisionProcessorLogger(self)

        # Show delayed warnings now that logger is available
        self._show_model_path_warnings()

        # Set environment variables for offline mode
        if self.defaults.trust_remote_code:
            self._setup_offline_mode()

    def _load_yaml_config(self) -> Dict:
        """Load YAML configuration with explicit error handling."""
        if not self.yaml_file.exists():
            raise ConfigurationError(
                f"❌ FATAL: Configuration file not found: {self.yaml_file}\n"
                f"💡 Expected location: {self.yaml_file.absolute()}\n"
                f"💡 Fix: Create configuration file using example template",
                config_path=self.yaml_file,
            )

        try:
            with self.yaml_file.open("r") as f:
                config = yaml.safe_load(f) or {}
                if not config:
                    raise ConfigurationError(
                        f"❌ FATAL: Empty configuration file: {self.yaml_file}\n"
                        f"💡 Fix: Add configuration sections to YAML file",
                        config_path=self.yaml_file,
                    )
                return config
        except yaml.YAMLError as e:
            raise ConfigurationError(
                f"❌ FATAL: Invalid YAML syntax in {self.yaml_file}\n"
                f"💡 Parse error: {e}\n"
                f"💡 Fix: Check YAML syntax and structure",
                config_path=self.yaml_file,
                parse_error=str(e),
            ) from e
        except IOError as e:
            raise ConfigurationError(
                f"❌ FATAL: Cannot read configuration file: {e}\n"
                f"💡 Fix: Check file permissions",
                config_path=self.yaml_file,
                io_error=str(e),
            ) from e

    def _parse_configuration(self) -> None:
        """Parse YAML configuration into structured objects."""
        # Parse defaults
        defaults_data = self._yaml_config_data.get("defaults", {})
        self.defaults = DefaultsConfig(**defaults_data)

        # Parse model paths
        model_paths_data = self._yaml_config_data.get("model_paths", {})
        self.model_paths = ModelPaths(**model_paths_data)

        # Parse memory configuration
        memory_data = self._yaml_config_data.get("memory_config", {})
        self.memory_config = MemoryConfig(**memory_data)

        # Parse image processing configuration
        image_data = self._yaml_config_data.get("image_processing", {})
        self.image_processing = ImageProcessingConfig(**image_data)

        # Parse repetition control
        repetition_data = self._yaml_config_data.get("repetition_control", {})
        self.repetition_control = RepetitionControlConfig(**repetition_data)

        # Parse quality and speed thresholds
        quality_data = self._yaml_config_data.get("quality_thresholds", {})
        self.quality_thresholds = QualityThresholds(**quality_data)

        speed_data = self._yaml_config_data.get("speed_thresholds", {})
        self.speed_thresholds = SpeedThresholds(**speed_data)

        # Parse device configuration
        device_data = self._yaml_config_data.get("device_config", {})
        if not device_data:
            raise ConfigurationError(
                "❌ FATAL: No device_config section found in configuration\n"
                "💡 Fix: Add device_config section to model_comparison.yaml"
            )

        # Parse device maps
        device_maps_data = device_data.get("device_maps", {})
        device_maps: Dict[str, DeviceMapConfig] = {}
        for model_name, map_config in device_maps_data.items():
            device_maps[model_name] = DeviceMapConfig(
                strategy=map_config.get("strategy", "single_gpu"),
                device_map=map_config.get("device_map", {"": 0}),
                quantization_compatible=map_config.get("quantization_compatible", True),
            )

        self.device_config = DeviceConfig(
            gpu_strategy=device_data.get("gpu_strategy", "single_gpu"),
            target_gpu=device_data.get("target_gpu", 0),
            v100_mode=device_data.get("v100_mode", True),
            memory_limit_gb=device_data.get("memory_limit_gb", 16),
            device_maps=device_maps,
            original_device_config="auto",
        )

        # Parse model-specific configurations
        model_config_data = self._yaml_config_data.get("model_config", {})
        self.llama_config = ModelSpecificConfig(**model_config_data.get("llama", {}))
        self.internvl_config = ModelSpecificConfig(
            **model_config_data.get("internvl", {})
        )

        # Create processing configuration
        self.processing = ProcessingConfig(
            memory_limit_mb=int(
                self.memory_config.v100_limit_gb
                * 1024
                * self.memory_config.safety_margin
            ),
            enable_gradient_checkpointing=True,
            use_flash_attention=True,
            quantization=self.defaults.quantization,
            batch_size=1,
            max_tokens=self.defaults.max_tokens,
        )

        # Store system prompts and extraction prompt
        self.system_prompts = self._yaml_config_data.get("system_prompts", {})
        self.extraction_prompt = self._yaml_config_data.get("extraction_prompt", "")

        # Runtime settings (can be overridden by CLI)
        self.current_model_type = None  # Must be explicitly set
        self.current_output_format = "yaml"  # Default
        self.log_level = "INFO"

    def _validate_configuration(self) -> None:
        """Validate configuration with explicit error messages."""
        # Validate model paths exist (warnings will be shown by logger when initialized)
        for _model_name, path in [
            ("llama", self.model_paths.llama),
            ("internvl", self.model_paths.internvl),
        ]:
            if path and not Path(path).exists():
                # Store warnings to be shown later when logger is available
                pass  # Skip model path warnings during validation

        # Validate device configuration
        self.device_manager.validate_device_configuration()

        # Validate memory settings
        if self.memory_config.v100_limit_gb < 1.0:
            raise ValidationError(
                field="memory_config.v100_limit_gb",
                value=self.memory_config.v100_limit_gb,
                reason="Must be at least 1.0 GB",
            )

        if not 0.1 <= self.memory_config.safety_margin <= 1.0:
            raise ValidationError(
                field="memory_config.safety_margin",
                value=self.memory_config.safety_margin,
                reason="Must be between 0.1 and 1.0",
            )

        # Validate extraction prompt exists
        if not self.extraction_prompt.strip():
            raise ConfigurationError(
                "❌ FATAL: No extraction_prompt found in configuration\n"
                "💡 Fix: Add extraction_prompt section to model_comparison.yaml"
            )

    def _show_model_path_warnings(self) -> None:
        """Show model path warnings now that logger is available."""
        for model_name, path in [
            ("llama", self.model_paths.llama),
            ("internvl", self.model_paths.internvl),
        ]:
            if path and not Path(path).exists():
                self.logger.warning(f"Model path does not exist: {path}")
                self.logger.warning(f"Model: {model_name}")
                self.logger.warning(
                    f"Fix: Update model_paths.{model_name} in {self.yaml_file}"
                )

    def _setup_offline_mode(self) -> None:
        """Set up environment variables for offline operation."""
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"

    # === PUBLIC API ===

    def get_model_path(self, model_type: str) -> str:
        """Get model path for specified model type."""
        if model_type == "llama":
            return self.model_paths.llama
        elif model_type == "internvl":
            return self.model_paths.internvl
        else:
            raise ValidationError(
                field="model_type",
                value=model_type,
                reason="Must be 'llama' or 'internvl'",
            )

    def get_device_map_for_model(self, model_name: str) -> Dict[str, int]:
        """Get device map for specified model."""
        return self.device_manager.get_device_map_for_model(model_name)

    def get_model_config(self, model_type: str) -> ModelSpecificConfig:
        """Get model-specific configuration."""
        if model_type == "llama":
            return self.llama_config
        elif model_type == "internvl":
            return self.internvl_config
        else:
            raise ValidationError(
                field="model_type",
                value=model_type,
                reason="Must be 'llama' or 'internvl'",
            )

    def get_system_prompt(self, model_type: str) -> str:
        """Get system prompt for specified model."""
        prompt = self.system_prompts.get(model_type, "You are a helpful assistant.")
        return str(prompt)

    def get_expected_fields(self) -> List[str]:
        """Parse expected fields from extraction prompt."""
        import re

        fields = []
        for line in self.extraction_prompt.split("\n"):
            match = re.match(r"^\s*([A-Z_]+):\s*\[.*\]", line.strip())
            if match:
                field_name = match.group(1)
                if field_name not in ["CORRECT", "WRONG", "EXAMPLE"]:
                    fields.append(field_name)
        return fields

    def get_prompts(self) -> Dict[str, str]:
        """Get prompts for all models (shared extraction prompt)."""
        return {
            "llama": self.extraction_prompt,
            "internvl": self.extraction_prompt,
        }

    def get_field_weights(self) -> Dict[str, float]:
        """Get field importance weights from configuration.

        Returns:
            Dictionary mapping field names to importance weights.
            Fields not specified in config get default weight of 1.0.
        """
        # Get field weights from YAML config
        yaml_weights = self._yaml_config_data.get("field_weights", {})

        # Get all expected fields
        expected_fields = self.get_expected_fields()

        # Build complete weights dictionary with defaults
        field_weights = {}
        for field in expected_fields:
            field_weights[field] = yaml_weights.get(field, 1.0)

        return field_weights

    def set_model_type(self, model_type: str) -> None:
        """Set current model type with validation."""
        if model_type not in ["llama", "internvl"]:
            raise ValidationError(
                field="model_type",
                value=model_type,
                reason="Must be 'llama' or 'internvl'",
            )
        self.current_model_type = model_type

    def validate_model_selected(self) -> None:
        """Ensure a model has been explicitly selected."""
        if self.current_model_type is None:
            available = list(self.get_available_models())
            raise ConfigurationError(
                f"❌ FATAL: No model specified\n"
                f"💡 Available models: {available}\n"
                f"💡 Fix: Add --model flag to CLI command\n"
                f"💡 Example: --model llama or --model internvl"
            )

    def set_output_format(self, output_format: str) -> None:
        """Set output format with validation."""
        if output_format not in ["yaml", "json", "table"]:
            raise ValidationError(
                field="output_format",
                value=output_format,
                reason="Must be 'yaml', 'json', or 'table'",
            )
        self.current_output_format = output_format

    @property
    def output_format(self) -> str:
        """Get current output format for compatibility."""
        return self.current_output_format

    @property
    def datasets_path(self) -> str:
        """Get datasets path for compatibility."""
        return self.defaults.datasets_path

    @property
    def output_dir(self) -> str:
        """Get output directory for compatibility."""
        return self.defaults.output_dir

    @property
    def models_list(self) -> List[str]:
        """Get list of models for comparison."""
        return [m.strip() for m in self.defaults.models.split(",")]

    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        return self.models_list

    def get_model_display_name(self, model_type: str) -> str:
        """Get the display name for a model from its path.

        Args:
            model_type: Model type key ('llama' or 'internvl')

        Returns:
            Display name extracted from model path (e.g., 'Llama-3.2-11B-Vision-Instruct')
        """
        model_path = self.get_model_path(model_type)
        if model_path:
            # Extract the model name from the path (last directory component)
            from pathlib import Path

            return Path(model_path).name
        else:
            # Fallback to uppercase model type if no path available
            return model_type.upper()

    def is_multi_gpu_enabled(self, model_type: Optional[str] = None) -> bool:
        """Check if multi-GPU is enabled for specified model."""
        if model_type is None:
            model_type = self.current_model_type

        if model_type in self.device_config.device_maps:
            model_strategy = self.device_config.device_maps[model_type].strategy
            return model_strategy == "multi_gpu"

        return self.device_config.gpu_strategy == "multi_gpu"

    def get_legacy_config_dict(self) -> Dict:
        """Get configuration in legacy format for backward compatibility."""
        return {
            "model_type": self.current_model_type,
            "model_path": self.get_model_path(self.current_model_type),
            "device_config": self.device_config.original_device_config,
            "enable_multi_gpu": self.is_multi_gpu_enabled(),
            "gpu_memory_fraction": 0.9,  # Legacy default
            "memory_limit_mb": self.processing.memory_limit_mb,
            "enable_quantization": self.processing.quantization,
            "enable_gradient_checkpointing": self.processing.enable_gradient_checkpointing,
            "use_flash_attention": self.processing.use_flash_attention,
            "trust_remote_code": self.defaults.trust_remote_code,
            "offline_mode": True,
            "repetition_control": {
                "enabled": self.repetition_control.enabled,
                "word_threshold": self.repetition_control.word_threshold,
                "phrase_threshold": self.repetition_control.phrase_threshold,
                "max_new_tokens_limit": self.get_model_config(
                    self.current_model_type
                ).max_new_tokens_limit,
            },
        }

    def print_configuration(self) -> None:
        """Print current configuration for debugging."""
        # Only print detailed config in verbose mode
        if self.defaults.verbose_mode:
            self.logger.info("Vision Processor Configuration:")
            self.logger.info(f"Model Type: {self.current_model_type}")
            self.logger.info(
                f"Model Path: {self.get_model_path(self.current_model_type)}"
            )
            self.logger.info(f"Device Strategy: {self.device_config.gpu_strategy}")
            self.logger.info(f"Multi-GPU: {self.is_multi_gpu_enabled()}")
            self.logger.info(f"Memory Limit: {self.processing.memory_limit_mb}MB")
            self.logger.info(f"Quantization: {self.processing.quantization}")
            self.logger.info(f"V100 Mode: {self.device_config.v100_mode}")
            self.logger.info(f"Output Format: {self.current_output_format}")
            self.logger.status("Repetition Control:")
            self.logger.info(f"Enabled: {self.repetition_control.enabled}")
            self.logger.info(
                f"Word Threshold: {self.repetition_control.word_threshold}"
            )
            self.logger.info(
                f"Phrase Threshold: {self.repetition_control.phrase_threshold}"
            )
            model_config = self.get_model_config(self.current_model_type)
            self.logger.info(f"Max Tokens Limit: {model_config.max_new_tokens_limit}")
