"""Simple configuration loader from YAML config only."""

from pathlib import Path

import yaml

from ..exceptions import ConfigurationError, ValidationError


class SimpleConfig:
    """Simplified configuration loader from YAML only."""

    def __init__(self, yaml_file: str = "model_comparison.yaml"):
        """Initialize configuration from YAML file only.

        Args:
            yaml_file: Path to YAML config file (default: model_comparison.yaml).
            
        Raises:
            ConfigurationError: If YAML file cannot be loaded or parsed
        """
        # Load YAML configuration
        self.yaml_config = {}
        yaml_path = Path(yaml_file)
        
        if not yaml_path.exists():
            raise ConfigurationError(
                f"Configuration file not found: {yaml_file}",
                config_path=yaml_path
            )
            
        try:
            with yaml_path.open("r") as f:
                self.yaml_config = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ConfigurationError(
                f"Failed to parse YAML configuration: {e}",
                config_path=yaml_path,
                parse_error=str(e)
            ) from e
        except IOError as e:
            raise ConfigurationError(
                f"Failed to read configuration file: {e}",
                config_path=yaml_path,
                io_error=str(e)
            ) from e

        # Get defaults from YAML
        defaults = self.yaml_config.get("defaults", {})

        # Model settings from YAML
        self.model_type = "internvl"  # Default, can be overridden by CLI

        # GPU and memory settings from YAML
        memory_config = self.yaml_config.get("memory_config", {})
        self.gpu_memory_fraction = 0.9
        self.memory_limit_mb = int(
            memory_config.get("v100_limit_gb", 16)
            * 1024
            * memory_config.get("safety_margin", 0.85)
        )
        self.enable_quantization = defaults.get("quantization", True)

        # Processing settings from YAML
        self.enable_gradient_checkpointing = True
        self.use_flash_attention = True
        self.trust_remote_code = defaults.get("trust_remote_code", True)
        self.offline_mode = True

        # Output settings
        self.output_format = "yaml"
        self.log_level = "INFO"

        # Repetition control settings from YAML only
        yaml_repetition = self.yaml_config.get("repetition_control", {})
        self.repetition_control_enabled = yaml_repetition.get("enabled", True)
        self.repetition_word_threshold = float(
            yaml_repetition.get("word_threshold", 0.15)
        )
        self.repetition_phrase_threshold = int(
            yaml_repetition.get("phrase_threshold", 2)
        )

        # Max tokens from YAML model config
        yaml_token_limit = (
            self.yaml_config.get("model_config", {})
            .get("llama", {})
            .get("max_new_tokens_limit", 2024)
        )
        self.repetition_max_tokens_limit = int(yaml_token_limit)

        # Set offline mode for transformers
        if self.offline_mode:
            import os

            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["HF_DATASETS_OFFLINE"] = "1"

        # Load comparison-specific settings from YAML defaults
        self.datasets_path = defaults.get("datasets_path", "datasets")
        self.output_dir = defaults.get("output_dir", "results")
        models_str = defaults.get("models", "llama,internvl")
        self.models_list = [m.strip() for m in models_str.split(",")]

        # Load model paths from YAML
        yaml_model_paths = self.yaml_config.get("model_paths", {})
        self.model_paths = type("ModelPaths", (), yaml_model_paths)()

        # Create processing object with all required attributes for model loading
        yaml_max_tokens = defaults.get("max_tokens", 800)

        self.processing = type(
            "Processing",
            (),
            {
                "memory_limit_mb": self.memory_limit_mb,
                "enable_gradient_checkpointing": self.enable_gradient_checkpointing,
                "use_flash_attention": self.use_flash_attention,
                "quantization": self.enable_quantization,
                "batch_size": 1,
                "max_tokens": yaml_max_tokens,
            },
        )()

        # Create device config object with device map method
        yaml_device_config = self.yaml_config.get("device_config", {})
        device_maps = yaml_device_config.get("device_maps", {})

        class SimpleDeviceConfig:
            def __init__(self, device_maps_dict):
                self.device_maps = device_maps_dict
                self.original_device_config = "auto"  # Simplified - no env vars

            def get_device_map_for_model(self, model_name: str):
                # Check if we have specific device map for this model in YAML
                if model_name in self.device_maps:
                    return self.device_maps[model_name].get("device_map", {"": 0})
                # Default to single GPU
                return {"": 0}

        self.device_config = SimpleDeviceConfig(device_maps)

    def print_configuration(self):
        """Print current configuration for debugging."""
        print("🔧 Vision Processor Configuration:")
        print(f"  Model Type: {self.model_type}")
        print(f"  Model Path: {self.model_path}")
        print(f"  Device Config: {self.device_config.original_device_config}")
        print(f"  Multi-GPU: {self.enable_multi_gpu}")
        print(f"  GPU Memory Fraction: {self.gpu_memory_fraction}")
        print(f"  Memory Limit: {self.memory_limit_mb}MB")
        print(f"  Quantization: {self.enable_quantization}")
        print(f"  Gradient Checkpointing: {self.enable_gradient_checkpointing}")
        print(f"  Flash Attention: {self.use_flash_attention}")
        print(f"  Offline Mode: {self.offline_mode}")
        print(f"  Output Format: {self.output_format}")
        print(f"  Log Level: {self.log_level}")
        print("🧹 Repetition Control:")
        print(f"  Enabled: {self.repetition_control_enabled}")
        print(f"  Word Threshold: {self.repetition_word_threshold}")
        print(f"  Phrase Threshold: {self.repetition_phrase_threshold}")
        print(f"  Max Tokens Limit: {self.repetition_max_tokens_limit}")

    def validate_prompt_fields(self) -> tuple[bool, str]:
        """Validate that all model prompts have the same fields.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        prompts = self.get_prompts()
        if not prompts:
            return False, "No prompts found in configuration"
            
        # Parse fields from each prompt
        all_fields = {}
        for model_type, prompt in prompts.items():
            fields = self._parse_fields_from_prompt(prompt)
            all_fields[model_type] = set(fields)
            
        if len(all_fields) < 2:
            return True, ""  # Only one prompt, nothing to compare
            
        # Compare all field sets
        field_sets = list(all_fields.values())
        first_set = field_sets[0]
        
        for i, other_set in enumerate(field_sets[1:], 1):
            if first_set != other_set:
                model_names = list(all_fields.keys())
                missing_in_second = first_set - other_set
                extra_in_second = other_set - first_set
                
                error_msg = f"Field mismatch between {model_names[0]} and {model_names[i]}:\n"
                if missing_in_second:
                    error_msg += f"  Missing in {model_names[i]}: {missing_in_second}\n"
                if extra_in_second:
                    error_msg += f"  Extra in {model_names[i]}: {extra_in_second}"
                    
                return False, error_msg
                
        return True, ""
    
    def _parse_fields_from_prompt(self, prompt: str) -> list[str]:
        """Parse field names from a prompt text."""
        import re
        
        fields = []
        for line in prompt.split("\n"):
            match = re.match(r"^\s*([A-Z_]+):\s*\[.*\]", line.strip())
            if match:
                field_name = match.group(1)
                if field_name not in ["CORRECT", "WRONG", "EXAMPLE"]:
                    fields.append(field_name)
        return fields

    def validate(self) -> None:
        """Validate configuration settings.
        
        Raises:
            ValidationError: If any configuration setting is invalid
        """
        # Check model type
        if self.model_type not in ["internvl", "llama"]:
            raise ValidationError(
                field="model_type",
                value=self.model_type,
                reason="Must be 'internvl' or 'llama'"
            )

        # Check model path exists
        model_path = Path(self.model_path)
        if not model_path.exists():
            # This is a warning, not an error - model might be downloaded later
            print(f"⚠️  Model path does not exist: {self.model_path}")
            print("  Please update model_paths in model_comparison.yaml")

        # Check device config
        valid_devices = ["auto", "cpu", "cuda", "cuda:0", "cuda:1", "mps"]
        device_str = self.device_config.original_device_config
        if device_str not in valid_devices and not device_str.startswith("cuda:"):
            raise ValidationError(
                field="device_config",
                value=device_str,
                reason=f"Must be one of {valid_devices} or cuda:N"
            )

        # Check memory settings
        if self.memory_limit_mb < 1024:
            print(f"⚠️  Memory limit seems low: {self.memory_limit_mb}MB")

        # Check GPU memory fraction
        if not 0.1 <= self.gpu_memory_fraction <= 1.0:
            raise ValidationError(
                field="gpu_memory_fraction",
                value=self.gpu_memory_fraction,
                reason="Must be between 0.1 and 1.0"
            )

        # Check output format
        if self.output_format not in ["table", "json", "yaml"]:
            raise ValidationError(
                field="output_format", 
                value=self.output_format,
                reason="Must be 'table', 'json', or 'yaml'"
            )

    def get_model_config(self) -> dict:
        """Get model-specific configuration as a dictionary.

        Returns:
            Dictionary with model configuration settings.
        """
        return {
            "model_type": self.model_type,
            "model_path": self.model_path,
            "device_config": self.device_config.original_device_config,
            "enable_multi_gpu": self.enable_multi_gpu,
            "gpu_memory_fraction": self.gpu_memory_fraction,
            "memory_limit_mb": self.memory_limit_mb,
            "enable_quantization": self.enable_quantization,
            "enable_gradient_checkpointing": self.enable_gradient_checkpointing,
            "use_flash_attention": self.use_flash_attention,
            "trust_remote_code": self.trust_remote_code,
            "offline_mode": self.offline_mode,
            "repetition_control": {
                "enabled": self.repetition_control_enabled,
                "word_threshold": self.repetition_word_threshold,
                "phrase_threshold": self.repetition_phrase_threshold,
                "max_new_tokens_limit": self.repetition_max_tokens_limit,
            },
        }

    def set_model_type(self, model_type: str):
        """Set the model type (simple override for CLI)."""
        if model_type in ["llama", "internvl"]:
            self.model_type = model_type
            print(f"🔄 Using model type: {self.model_type}")
        else:
            print(f"❌ Invalid model type: {model_type}. Must be 'llama' or 'internvl'")

    def set_output_format(self, output_format: str):
        """Set the output format (simple override for CLI)."""
        if output_format in ["yaml", "json", "table"]:
            self.output_format = output_format
            print(f"🔄 Using output format: {self.output_format}")
        else:
            print(f"❌ Invalid output format: {output_format}")

    def get_expected_fields(self) -> list[str]:
        """Get expected fields by parsing from the prompt text.
        
        This ensures a single source of truth - the fields are defined
        only in the prompt, not in a separate list.
        """
        # First try to get the shared extraction_prompt
        extraction_prompt = self.yaml_config.get("extraction_prompt", "")
        
        if extraction_prompt:
            return self._parse_fields_from_prompt(extraction_prompt)
        
        # Otherwise get the prompt for the current model type
        prompts = self.get_prompts()
        model_prompt = prompts.get(self.model_type, "")
        
        if not model_prompt:
            # Fallback to llama prompt if model-specific prompt not found
            model_prompt = prompts.get("llama", "")
            
        if not model_prompt:
            return []

        return self._parse_fields_from_prompt(model_prompt)

    def get_prompts(self) -> dict[str, str]:
        """Get prompts from YAML configuration.
        
        Returns the shared extraction_prompt for all models.
        """
        # First check if there's a shared extraction_prompt
        extraction_prompt = self.yaml_config.get("extraction_prompt", "")
        
        if extraction_prompt:
            # Return the same prompt for all models
            return {
                "llama": extraction_prompt,
                "internvl": extraction_prompt,
            }
        
        # Fallback to legacy model-specific prompts if they exist
        return self.yaml_config.get(
            "prompts",
            {
                "llama": "Extract data from this image in KEY-VALUE format.",
                "internvl": "Extract data from this image in KEY-VALUE format.",
            },
        )

    @property
    def model_path(self) -> str:
        """Get the model path for the current model type."""
        paths = self.yaml_config.get("model_paths", {})
        return paths.get(self.model_type, "")

    @property
    def enable_multi_gpu(self) -> bool:
        """Check if multi-GPU is enabled based on device configuration."""
        # Check device_config in YAML
        device_config = self.yaml_config.get("device_config", {})
        gpu_strategy = device_config.get("gpu_strategy", "single_gpu")

        # Also check model-specific device maps
        device_maps = device_config.get("device_maps", {})
        if self.model_type in device_maps:
            model_strategy = device_maps[self.model_type].get("strategy", "single_gpu")
            return model_strategy == "multi_gpu"

        return gpu_strategy == "multi_gpu"
