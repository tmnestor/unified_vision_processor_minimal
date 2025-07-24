"""Simple configuration loader from .env file and YAML config."""

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv


class SimpleConfig:
    """Simplified configuration loader from .env file."""

    def __init__(self, env_file: str | None = None, yaml_file: str | None = None):
        """Initialize configuration from .env file and optional YAML file.

        Args:
            env_file: Path to .env file. If None, uses default .env in project root.
            yaml_file: Path to YAML config file (e.g., model_comparison.yaml).
        """
        # Load environment variables first
        if env_file:
            load_dotenv(env_file)
        else:
            load_dotenv()  # Load default .env file

        # Load YAML configuration if provided
        self.yaml_config = {}
        if yaml_file and Path(yaml_file).exists():
            with Path(yaml_file).open("r") as f:
                self.yaml_config = yaml.safe_load(f) or {}

        # Model settings
        self.model_type = os.getenv("VISION_MODEL_TYPE", "internvl")

        # Try model-specific path first, then fall back to generic path
        match self.model_type:
            case "internvl":
                self.model_path = os.getenv("VISION_INTERNVL_MODEL_PATH") or os.getenv(
                    "VISION_MODEL_PATH", "/path/to/models"
                )
            case "llama":
                self.model_path = os.getenv("VISION_LLAMA_MODEL_PATH") or os.getenv(
                    "VISION_MODEL_PATH", "/path/to/models"
                )
            case _:
                self.model_path = os.getenv("VISION_MODEL_PATH", "/path/to/models")

        # GPU and memory settings
        self.device_config = os.getenv("VISION_DEVICE_CONFIG", "auto")
        self.enable_multi_gpu = (
            os.getenv("VISION_ENABLE_MULTI_GPU", "false").lower() == "true"
        )
        self.gpu_memory_fraction = float(os.getenv("VISION_GPU_MEMORY_FRACTION", "0.9"))
        self.memory_limit_mb = int(os.getenv("VISION_MEMORY_LIMIT_MB", "15360"))
        self.enable_quantization = (
            os.getenv("VISION_ENABLE_QUANTIZATION", "false").lower() == "true"
        )

        # Processing settings
        self.enable_gradient_checkpointing = (
            os.getenv("VISION_ENABLE_GRADIENT_CHECKPOINTING", "true").lower() == "true"
        )
        self.use_flash_attention = (
            os.getenv("VISION_USE_FLASH_ATTENTION", "true").lower() == "true"
        )
        self.trust_remote_code = (
            os.getenv("VISION_TRUST_REMOTE_CODE", "true").lower() == "true"
        )
        self.offline_mode = os.getenv("VISION_OFFLINE_MODE", "true").lower() == "true"

        # Output settings
        self.output_format = os.getenv("VISION_OUTPUT_FORMAT", "table")
        self.log_level = os.getenv("VISION_LOG_LEVEL", "INFO")

        # Repetition control settings - read from YAML first (single source of truth)
        yaml_repetition = self.yaml_config.get("repetition_control", {})
        self.repetition_control_enabled = (
            os.getenv("VISION_REPETITION_CONTROL_ENABLED", 
                     str(yaml_repetition.get("enabled", "true"))).lower() == "true"
        )
        self.repetition_word_threshold = float(
            os.getenv("VISION_REPETITION_WORD_THRESHOLD", 
                     str(yaml_repetition.get("word_threshold", "0.15")))
        )
        self.repetition_phrase_threshold = int(
            os.getenv("VISION_REPETITION_PHRASE_THRESHOLD", 
                     str(yaml_repetition.get("phrase_threshold", "2")))
        )
        # Read max_tokens_limit from YAML config (single source of truth)
        yaml_token_limit = (
            self.yaml_config.get("model_config", {})
            .get("llama", {})
            .get("max_new_tokens_limit")
        )
        env_token_limit = os.getenv("VISION_REPETITION_MAX_TOKENS_LIMIT")

        # Priority: YAML config > Environment variable > Default
        if yaml_token_limit:
            self.repetition_max_tokens_limit = int(yaml_token_limit)
        elif env_token_limit:
            self.repetition_max_tokens_limit = int(env_token_limit)
        else:
            self.repetition_max_tokens_limit = 384  # Final fallback

        # Set offline mode for transformers
        if self.offline_mode:
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["HF_DATASETS_OFFLINE"] = "1"

        # Load comparison-specific settings from YAML defaults
        defaults = self.yaml_config.get("defaults", {})
        self.datasets_path = defaults.get("datasets_path", "datasets")
        self.output_dir = defaults.get("output_dir", "results")
        models_str = defaults.get("models", "llama,internvl")
        self.models_list = [m.strip() for m in models_str.split(",")]

        # Load model paths from YAML
        yaml_model_paths = self.yaml_config.get("model_paths", {})
        self.model_paths = type("ModelPaths", (), yaml_model_paths)()

        # Create processing object with all required attributes for model loading
        # Read max_tokens from YAML defaults (single source of truth)
        yaml_max_tokens = defaults.get(
            "max_tokens", 800
        )  # Use YAML default or fallback to 800

        self.processing = type(
            "Processing",
            (),
            {
                "memory_limit_mb": self.memory_limit_mb,
                "enable_gradient_checkpointing": self.enable_gradient_checkpointing,
                "use_flash_attention": self.use_flash_attention,
                "quantization": self.enable_quantization,
                "batch_size": 1,
                "max_tokens": yaml_max_tokens,  # Use YAML config value
            },
        )()

        # Create device config object with device map method
        yaml_device_config = self.yaml_config.get("device_config", {})
        device_maps = yaml_device_config.get("device_maps", {})

        class SimpleDeviceConfig:
            def __init__(self, device_maps_dict):
                self.device_maps = device_maps_dict
                self.original_device_config = os.getenv("VISION_DEVICE_CONFIG", "auto")

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

    def validate(self) -> bool:
        """Validate configuration settings.

        Returns:
            True if configuration is valid, False otherwise.
        """
        # Check model type
        if self.model_type not in ["internvl", "llama"]:
            print(f"❌ Invalid model type: {self.model_type}")
            return False

        # Check model path exists
        model_path = Path(self.model_path)
        if not model_path.exists():
            print(f"⚠️  Model path does not exist: {self.model_path}")
            print("  Please update VISION_MODEL_PATH in .env file")

        # Check device config
        valid_devices = ["auto", "cpu", "cuda", "cuda:0", "cuda:1", "mps"]
        device_str = self.device_config.original_device_config
        if device_str not in valid_devices and not device_str.startswith("cuda:"):
            print(f"❌ Invalid device config: {device_str}")
            return False

        # Check memory settings
        if self.memory_limit_mb < 1024:
            print(f"⚠️  Memory limit seems low: {self.memory_limit_mb}MB")

        # Check GPU memory fraction
        if not 0.1 <= self.gpu_memory_fraction <= 1.0:
            print(f"❌ Invalid GPU memory fraction: {self.gpu_memory_fraction}")
            return False

        # Check output format
        if self.output_format not in ["table", "json", "yaml"]:
            print(f"❌ Invalid output format: {self.output_format}")
            return False

        return True

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

    def update_from_cli(self, **kwargs):
        """Update configuration from CLI arguments.

        Args:
            **kwargs: Configuration overrides from CLI.
        """
        if "model" in kwargs and kwargs["model"]:
            self.model_type = kwargs["model"]
            # Reload .env file to ensure we get the latest values
            from dotenv import load_dotenv

            load_dotenv(override=True)

            # Update model path based on new model type
            if self.model_type == "internvl":
                self.model_path = os.getenv("VISION_INTERNVL_MODEL_PATH") or os.getenv(
                    "VISION_MODEL_PATH", "/path/to/models"
                )
            elif self.model_type == "llama":
                self.model_path = os.getenv("VISION_LLAMA_MODEL_PATH") or os.getenv(
                    "VISION_MODEL_PATH", "/path/to/models"
                )
            else:
                self.model_path = os.getenv("VISION_MODEL_PATH", "/path/to/models")
            print(f"🔄 Overriding model type to: {self.model_type}")
            print(f"🔄 Using model path: {self.model_path}")

        if "output_format" in kwargs and kwargs["output_format"]:
            self.output_format = kwargs["output_format"]
            print(f"🔄 Overriding output format to: {self.output_format}")

        if "device" in kwargs and kwargs["device"]:
            # Update the original device config and recreate the device config object
            device_str = kwargs["device"]
            self.device_config.original_device_config = device_str
            print(f"🔄 Overriding device config to: {device_str}")

        if "quantization" in kwargs and kwargs["quantization"] is not None:
            self.enable_quantization = kwargs["quantization"]
            print(f"🔄 Overriding quantization to: {self.enable_quantization}")

    def get_expected_fields(self) -> list[str]:
        """Get expected fields by parsing from llama prompt or using YAML configuration.
        
        Priority:
        1. If expected_fields is defined in YAML, use that
        2. Otherwise, parse fields from the llama prompt
        """
        # First check if expected_fields is explicitly defined
        expected_fields = self.yaml_config.get("expected_fields", [])
        if expected_fields:
            return expected_fields
            
        # Otherwise, parse from llama prompt
        prompts = self.yaml_config.get("prompts", {})
        llama_prompt = prompts.get("llama", "")
        
        if not llama_prompt:
            return []
            
        # Parse field names from lines that match "FIELD_NAME: [value or N/A]"
        import re
        fields = []
        for line in llama_prompt.split('\n'):
            match = re.match(r'^\s*([A-Z_]+):\s*\[.*\]', line.strip())
            if match:
                fields.append(match.group(1))
                
        return fields

    def get_prompts(self) -> dict[str, str]:
        """Get model-specific prompts from YAML configuration."""
        return self.yaml_config.get(
            "prompts",
            {
                "llama": "Extract data from this image in KEY-VALUE format.",
                "internvl": "Extract data from this image in KEY-VALUE format.",
            },
        )
