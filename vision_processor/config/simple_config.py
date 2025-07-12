"""Simple configuration loader from .env file."""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


class SimpleConfig:
    """Simplified configuration loader from .env file."""

    def __init__(self, env_file: Optional[str] = None):
        """Initialize configuration from .env file.

        Args:
            env_file: Path to .env file. If None, uses default .env in project root.
        """
        if env_file:
            load_dotenv(env_file)
        else:
            load_dotenv()  # Load default .env file

        # Model settings
        self.model_type = os.getenv("VISION_MODEL_TYPE", "internvl3")

        # Try model-specific path first, then fall back to generic path
        if self.model_type == "internvl3":
            self.model_path = os.getenv("VISION_INTERNVL_MODEL_PATH") or os.getenv("VISION_MODEL_PATH", "/path/to/models")
        elif self.model_type == "llama32_vision":
            self.model_path = os.getenv("VISION_LLAMA_MODEL_PATH") or os.getenv("VISION_MODEL_PATH", "/path/to/models")
        else:
            self.model_path = os.getenv("VISION_MODEL_PATH", "/path/to/models")

        # GPU and memory settings
        self.device_config = os.getenv("VISION_DEVICE_CONFIG", "auto")
        self.enable_multi_gpu = os.getenv("VISION_ENABLE_MULTI_GPU", "false").lower() == "true"
        self.gpu_memory_fraction = float(os.getenv("VISION_GPU_MEMORY_FRACTION", "0.9"))
        self.memory_limit_mb = int(os.getenv("VISION_MEMORY_LIMIT_MB", "15360"))
        self.enable_quantization = os.getenv("VISION_ENABLE_QUANTIZATION", "false").lower() == "true"

        # Processing settings
        self.enable_gradient_checkpointing = (
            os.getenv("VISION_ENABLE_GRADIENT_CHECKPOINTING", "true").lower() == "true"
        )
        self.use_flash_attention = os.getenv("VISION_USE_FLASH_ATTENTION", "true").lower() == "true"
        self.trust_remote_code = os.getenv("VISION_TRUST_REMOTE_CODE", "true").lower() == "true"
        self.offline_mode = os.getenv("VISION_OFFLINE_MODE", "true").lower() == "true"

        # Output settings
        self.output_format = os.getenv("VISION_OUTPUT_FORMAT", "table")
        self.log_level = os.getenv("VISION_LOG_LEVEL", "INFO")

        # Set offline mode for transformers
        if self.offline_mode:
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["HF_DATASETS_OFFLINE"] = "1"

    def print_configuration(self):
        """Print current configuration for debugging."""
        print("🔧 Vision Processor Configuration:")
        print(f"  Model Type: {self.model_type}")
        print(f"  Model Path: {self.model_path}")
        print(f"  Device Config: {self.device_config}")
        print(f"  Multi-GPU: {self.enable_multi_gpu}")
        print(f"  GPU Memory Fraction: {self.gpu_memory_fraction}")
        print(f"  Memory Limit: {self.memory_limit_mb}MB")
        print(f"  Quantization: {self.enable_quantization}")
        print(f"  Gradient Checkpointing: {self.enable_gradient_checkpointing}")
        print(f"  Flash Attention: {self.use_flash_attention}")
        print(f"  Offline Mode: {self.offline_mode}")
        print(f"  Output Format: {self.output_format}")
        print(f"  Log Level: {self.log_level}")

    def validate(self) -> bool:
        """Validate configuration settings.

        Returns:
            True if configuration is valid, False otherwise.
        """
        # Check model type
        if self.model_type not in ["internvl3", "llama32_vision"]:
            print(f"❌ Invalid model type: {self.model_type}")
            return False

        # Check model path exists
        model_path = Path(self.model_path)
        if not model_path.exists():
            print(f"⚠️  Model path does not exist: {self.model_path}")
            print("  Please update VISION_MODEL_PATH in .env file")

        # Check device config
        valid_devices = ["auto", "cpu", "cuda", "cuda:0", "cuda:1", "mps"]
        if self.device_config not in valid_devices and not self.device_config.startswith("cuda:"):
            print(f"❌ Invalid device config: {self.device_config}")
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
            "device_config": self.device_config,
            "enable_multi_gpu": self.enable_multi_gpu,
            "gpu_memory_fraction": self.gpu_memory_fraction,
            "memory_limit_mb": self.memory_limit_mb,
            "enable_quantization": self.enable_quantization,
            "enable_gradient_checkpointing": self.enable_gradient_checkpointing,
            "use_flash_attention": self.use_flash_attention,
            "trust_remote_code": self.trust_remote_code,
            "offline_mode": self.offline_mode,
        }

    def update_from_cli(self, **kwargs):
        """Update configuration from CLI arguments.

        Args:
            **kwargs: Configuration overrides from CLI.
        """
        if "model" in kwargs and kwargs["model"]:
            self.model_type = kwargs["model"]
            # Update model path based on new model type
            if self.model_type == "internvl3":
                self.model_path = os.getenv("VISION_INTERNVL_MODEL_PATH") or os.getenv("VISION_MODEL_PATH", "/path/to/models")
            elif self.model_type == "llama32_vision":
                self.model_path = os.getenv("VISION_LLAMA_MODEL_PATH") or os.getenv("VISION_MODEL_PATH", "/path/to/models")
            else:
                self.model_path = os.getenv("VISION_MODEL_PATH", "/path/to/models")
            print(f"🔄 Overriding model type to: {self.model_type}")
            print(f"🔄 Using model path: {self.model_path}")

        if "output_format" in kwargs and kwargs["output_format"]:
            self.output_format = kwargs["output_format"]
            print(f"🔄 Overriding output format to: {self.output_format}")

        if "device" in kwargs and kwargs["device"]:
            self.device_config = kwargs["device"]
            print(f"🔄 Overriding device config to: {self.device_config}")

        if "quantization" in kwargs and kwargs["quantization"] is not None:
            self.enable_quantization = kwargs["quantization"]
            print(f"🔄 Overriding quantization to: {self.enable_quantization}")
