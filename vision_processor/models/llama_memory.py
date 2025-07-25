"""Llama-3.2-Vision Memory and Device Management

Handles device setup, device mapping, and memory management.
"""

import gc
import logging
from typing import Dict, Union

import torch

from .base_model import DeviceConfig
from .model_utils import DeviceManager

logger = logging.getLogger(__name__)


class LlamaMemoryManager:
    """Manages memory and device configuration for Llama-3.2-Vision model."""

    def __init__(self, config, device_config: DeviceConfig, memory_limit_mb: int):
        """Initialize memory manager.
        
        Args:
            config: Model configuration object
            device_config: Device configuration
            memory_limit_mb: Memory limit in MB
        """
        self.config = config
        self.device_config = device_config
        self.memory_limit_mb = memory_limit_mb
        self.device_manager = DeviceManager(memory_limit_mb)
        self.num_gpus = 0

    def setup_device(self) -> torch.device:
        """Setup device configuration for Llama-3.2-Vision.
        
        Returns:
            Configured torch device
        """
        # Use device manager to select optimal device
        if self.device_config == DeviceConfig.AUTO:
            device = self.device_manager.select_device(DeviceConfig.AUTO)
        else:
            device = self.device_manager.select_device(self.device_config)

        # Force single GPU for fair V100 production comparison
        if device.type == "cuda":
            # Force single GPU mode to match V100 production constraints
            # This ensures fair comparison with InternVL model that's V100-optimized
            available_gpus = torch.cuda.device_count()
            self.num_gpus = 1  # Force single GPU regardless of available GPUs
            logger.info(
                f"🔧 V100 Production Mode: Using 1 GPU (detected {available_gpus} available)"
            )

            # Enable TF32 for GPU optimization on compatible hardware
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("TF32 enabled for GPU optimization")
        else:
            self.num_gpus = 0

        return device

    def get_device_map_from_config(self):
        """Get device map configuration from production config.

        FAILS EXPLICITLY if YAML config is not available - no silent fallbacks.
        
        Returns:
            Device mapping configuration
            
        Raises:
            RuntimeError: If configuration is missing or invalid
        """
        # Validate configuration exists
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

        device_config = self.config.device_config
        device_map = device_config.get_device_map_for_model("llama")

        if device_map is None:
            raise RuntimeError(
                f"❌ FATAL: No device mapping found for llama model\n"
                f"💡 Expected: llama entry in device_config.device_maps\n"
                f"💡 Current device_maps: {list(device_config.device_maps.keys())}\n"
                f"💡 Fix: Add llama device mapping to model_comparison.yaml:\n"
                f"   device_config:\n"
                f"     device_maps:\n"
                f"       llama:\n"
                f"         strategy: 'single_gpu'\n"
                f"         device_map: {{'': 0}}\n"
                f"         quantization_compatible: true"
            )

        return device_map

    def detect_device_mapping(self, device: torch.device) -> Union[str, Dict[str, int]]:
        """Detect optimal device mapping based on hardware.
        
        Args:
            device: The torch device being used
            
        Returns:
            Device mapping configuration
        """
        if device.type == "cuda":
            if torch.cuda.device_count() > 1:
                return "balanced"  # Distribute across multiple GPUs
            return "cuda:0"  # Single GPU
        if device.type == "mps":
            return "mps"
        return "cpu"

    def cleanup_memory(self) -> None:
        """Clean up GPU and system memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()

    def get_memory_usage(self, device: torch.device) -> float:
        """Get current GPU memory usage in MB.
        
        Args:
            device: The torch device to check
            
        Returns:
            Memory usage in MB
        """
        try:
            if torch.cuda.is_available() and device.type == "cuda":
                return torch.cuda.memory_allocated(device) / 1024 / 1024
        except Exception as e:
            logger.warning(f"Failed to get memory usage: {e}")
        return 0.0

    def configure_model_loading_args(
        self, device: torch.device, device_map, quantization_config, offline_mode: bool = True
    ) -> Dict[str, any]:
        """Configure loading parameters based on device and settings.
        
        Args:
            device: Target device
            device_map: Device mapping configuration
            quantization_config: Quantization configuration
            offline_mode: Whether to load in offline mode
            
        Returns:
            Dictionary of model loading arguments
        """
        model_loading_args = {
            "low_cpu_mem_usage": True,
            "torch_dtype": torch.float16,
        }

        # Add offline mode configuration
        if offline_mode:
            model_loading_args["local_files_only"] = True
            logger.info("Loading model in offline mode")

        # Configure based on device
        if device.type == "cpu":
            model_loading_args["device_map"] = None
            logger.info("Loading model on CPU (will be slow)...")
        elif self.num_gpus == 1:
            model_loading_args["device_map"] = device_map

            # Add quantization configuration if enabled
            if quantization_config:
                model_loading_args["quantization_config"] = quantization_config
                # Determine quantization type for logging
                use_4bit = (
                    hasattr(quantization_config, "load_in_4bit")
                    and quantization_config.load_in_4bit
                )
                quant_type = "4-bit" if use_4bit else "8-bit"
                logger.info(
                    f"🔧 V100 Mode: Loading 11B model on single GPU with {quant_type} quantization..."
                )
            else:
                logger.info(
                    "🔧 V100 Mode: Loading model on single GPU without quantization..."
                )
                logger.warning(
                    "⚠️  No quantization - 11B model may exceed V100 16GB limit"
                )
        else:  # Multi-GPU (should not happen in V100 production mode)
            logger.warning(
                f"Multi-GPU mode detected ({self.num_gpus} GPUs) - this may exceed V100 limits"
            )
            model_loading_args["device_map"] = device_map
            logger.info(f"Loading model across {self.num_gpus} GPUs...")

        return model_loading_args