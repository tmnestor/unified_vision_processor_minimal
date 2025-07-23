"""Model Utilities for Device Optimization and Memory Management

Provides shared utilities for both InternVL and Llama models including:
- Multi-GPU auto-configuration
- Memory optimization
- Device selection
- Quantization helpers
"""

import logging
import platform
from typing import TYPE_CHECKING, Union

import torch

if TYPE_CHECKING:
    from torch import nn
    from transformers import BitsAndBytesConfig

from .base_model import DeviceConfig

logger = logging.getLogger(__name__)


class DeviceManager:
    """Manages device selection and configuration for vision models."""

    def __init__(self, memory_limit_mb: Union[int, None] = None):
        self.memory_limit_mb = memory_limit_mb
        self.system_info = self._get_system_info()

    def _get_system_info(self) -> dict[str, any]:
        """Get comprehensive system information."""
        import torch

        info = {
            "platform": platform.system(),
            "architecture": platform.machine(),
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
        }

        # CUDA information
        if torch.cuda.is_available():
            info.update(
                {
                    "cuda_available": True,
                    "cuda_version": torch.version.cuda,
                    "gpu_count": torch.cuda.device_count(),
                    "gpus": [
                        {
                            "id": i,
                            "name": torch.cuda.get_device_name(i),
                            "memory_total": torch.cuda.get_device_properties(
                                i,
                            ).total_memory,
                            "memory_reserved": torch.cuda.memory_reserved(i),
                            "memory_allocated": torch.cuda.memory_allocated(i),
                        }
                        for i in range(torch.cuda.device_count())
                    ],
                },
            )
        else:
            info["cuda_available"] = False

        # MPS information (Apple Silicon)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            info["mps_available"] = True
        else:
            info["mps_available"] = False

        return info

    def select_device(self, device_config: DeviceConfig) -> "torch.device":
        """Select optimal device based on configuration and system capabilities.

        Args:
            device_config: Requested device configuration

        Returns:
            Selected torch device

        """
        if device_config == DeviceConfig.CPU:
            return torch.device("cpu")

        if device_config == DeviceConfig.SINGLE_GPU:
            if torch.cuda.is_available():
                return torch.device("cuda:0")
            logger.warning("CUDA not available, falling back to CPU")
            return torch.device("cpu")

        if device_config == DeviceConfig.MULTI_GPU:
            if torch.cuda.device_count() > 1:
                return torch.device("cuda:0")  # Primary device for multi-GPU
            if torch.cuda.is_available():
                logger.warning("Only one GPU available, using single GPU mode")
                return torch.device("cuda:0")
            logger.warning("CUDA not available, falling back to CPU")
            return torch.device("cpu")

        # AUTO configuration - intelligent device selection
        if device_config == DeviceConfig.AUTO:
            return self._auto_select_device()

        raise ValueError(f"Unsupported device configuration: {device_config}")

    def _auto_select_device(self) -> "torch.device":
        """Automatically select the best available device."""
        import torch

        # Priority 1: Multiple high-memory GPUs (H200 development)
        if torch.cuda.device_count() > 1:
            gpu_memory = [
                torch.cuda.get_device_properties(i).total_memory
                for i in range(torch.cuda.device_count())
            ]
            if all(mem >= 70 * 1024**3 for mem in gpu_memory):  # 70GB+ for H200
                logger.info(
                    "Multiple high-memory GPUs detected, using multi-GPU configuration",
                )
                return torch.device("cuda:0")

        # Priority 2: Single GPU with sufficient memory (V100 production)
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            if gpu_memory >= 15 * 1024**3:  # 15GB+ for V100
                logger.info(
                    f"Single GPU with {gpu_memory // 1024**3}GB memory detected",
                )
                return torch.device("cuda:0")

        # Priority 3: Apple Silicon MPS (Mac M1 development)
        if (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
            and platform.system() == "Darwin"
        ):
            logger.info("Apple Silicon MPS detected, using MPS acceleration")
            return torch.device("mps")

        # Fallback: CPU
        logger.info("Using CPU for processing")
        return torch.device("cpu")

    def get_optimal_quantization_config(
        self,
        _model_type: str,
        device: "torch.device",
    ) -> Union["BitsAndBytesConfig", None]:
        """Get optimal quantization configuration based on model and device.

        Args:
            model_type: Type of model (internvl3, llama32_vision)
            device: Target device

        Returns:
            BitsAndBytesConfig or None if quantization not recommended

        """
        import torch
        from transformers import BitsAndBytesConfig

        if device.type == "cpu":
            return None  # No quantization on CPU

        if device.type == "mps":
            return None  # MPS doesn't support quantization yet

        if device.type != "cuda":
            return None

        # Get GPU memory
        gpu_memory = torch.cuda.get_device_properties(device).total_memory
        available_memory = gpu_memory - torch.cuda.memory_allocated(device)

        # Determine quantization strategy based on available memory
        if available_memory < 20 * 1024**3:  # Less than 20GB (V100 production)
            logger.info("Low GPU memory detected, using 8-bit quantization")
            return BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=True,
            )
        if available_memory < 40 * 1024**3:  # 20-40GB (moderate memory)
            logger.info("Moderate GPU memory detected, using 4-bit quantization")
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        # High memory (H200 development)
        logger.info("High GPU memory detected, no quantization needed")
        return None

    def setup_multi_gpu(self, model: "nn.Module") -> "nn.Module":
        """Setup model for multi-GPU processing if available.

        Args:
            model: PyTorch model to parallelize

        Returns:
            Model wrapped for multi-GPU or original model

        """
        import torch
        from torch import nn

        if torch.cuda.device_count() > 1:
            logger.info(f"Setting up model for {torch.cuda.device_count()} GPUs")
            model = nn.DataParallel(model)

        return model

    def optimize_memory_usage(self, device: "torch.device") -> None:
        """Optimize memory usage for the given device."""
        import torch

        if device.type == "cuda":
            # Clear cache
            torch.cuda.empty_cache()

            # Set memory fraction if limit specified
            if self.memory_limit_mb:
                total_memory = torch.cuda.get_device_properties(device).total_memory
                memory_fraction = (self.memory_limit_mb * 1024 * 1024) / total_memory
                memory_fraction = min(memory_fraction, 0.95)  # Cap at 95%
                torch.cuda.set_per_process_memory_fraction(memory_fraction, device)
                logger.info(f"Set GPU memory fraction to {memory_fraction:.2f}")

    def get_memory_stats(self, device: "torch.device") -> dict[str, float]:
        """Get detailed memory statistics for the device."""
        import torch

        if device.type == "cuda":
            return {
                "allocated_mb": torch.cuda.memory_allocated(device) / 1024 / 1024,
                "reserved_mb": torch.cuda.memory_reserved(device) / 1024 / 1024,
                "total_mb": torch.cuda.get_device_properties(device).total_memory
                / 1024
                / 1024,
            }
        return {"allocated_mb": 0.0, "reserved_mb": 0.0, "total_mb": 0.0}


class QuantizationHelper:
    """Helper utilities for model quantization."""

    @staticmethod
    def apply_dynamic_quantization(model: "nn.Module") -> "nn.Module":
        """Apply dynamic quantization to the model."""
        import torch
        from torch import nn

        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear},
            dtype=torch.qint8,
        )
        logger.info("Applied dynamic quantization")
        return quantized_model

    @staticmethod
    def check_quantization_support(device: "torch.device") -> bool:
        """Check if quantization is supported on the device."""
        if device.type == "cpu":
            return True  # CPU supports quantization
        if device.type == "cuda":
            return True  # CUDA supports quantization
        if device.type == "mps":
            return False  # MPS doesn't support quantization yet
        return False


class ModelProfiler:
    """Profile model performance and memory usage."""

    def __init__(self):
        self.profile_data = []

    def profile_inference(
        self,
        model_fn: callable,
        *args,
        **kwargs,
    ) -> tuple[any, dict[str, float]]:
        """Profile a model inference call.

        Args:
            model_fn: Model function to profile
            *args: Arguments to pass to model function
            **kwargs: Keyword arguments to pass to model function

        Returns:
            Tuple of (result, profile_stats)

        """
        import time

        import torch

        # Get initial memory
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            start_memory = torch.cuda.memory_allocated()
        else:
            start_memory = 0

        # Profile inference
        start_time = time.perf_counter()
        result = model_fn(*args, **kwargs)
        end_time = time.perf_counter()

        # Get final memory
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            end_memory = torch.cuda.memory_allocated()
        else:
            end_memory = 0

        stats = {
            "inference_time": end_time - start_time,
            "memory_used_mb": (end_memory - start_memory) / 1024 / 1024,
            "peak_memory_mb": end_memory / 1024 / 1024,
        }

        self.profile_data.append(stats)
        return result, stats

    def get_average_stats(self) -> dict[str, float]:
        """Get average statistics across all profiled calls."""
        if not self.profile_data:
            return {}

        return {
            "avg_inference_time": sum(d["inference_time"] for d in self.profile_data)
            / len(self.profile_data),
            "avg_memory_used_mb": sum(d["memory_used_mb"] for d in self.profile_data)
            / len(self.profile_data),
            "max_peak_memory_mb": max(d["peak_memory_mb"] for d in self.profile_data),
        }
