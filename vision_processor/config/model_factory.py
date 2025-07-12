"""Model Factory for Unified Vision Processing

Creates and configures vision models with optimal settings for different hardware configurations.
Integrates InternVL multi-GPU optimization with Llama processing capabilities.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from ..models.base_model import BaseVisionModel, DeviceConfig

if TYPE_CHECKING:
    from .simple_config import SimpleConfig
    from .unified_config import ModelType, UnifiedConfig
else:
    from .unified_config import ModelType

logger = logging.getLogger(__name__)


class ModelCreationError(Exception):
    """Exception raised when model creation fails."""

    def __init__(self, message: str, model_type: ModelType | None = None):
        self.model_type = model_type
        super().__init__(message)


class ModelFactory:
    """Factory for creating and configuring vision models.

    Handles:
    - Model selection and instantiation
    - Device configuration and optimization
    - Multi-GPU setup for InternVL
    - Memory optimization for V100 production
    - Cross-platform compatibility
    """

    # Registry of available models
    _model_registry: dict[ModelType, type[BaseVisionModel]] = {}

    @classmethod
    def register_model(
        cls,
        model_type: ModelType,
        model_class: type[BaseVisionModel],
    ) -> None:
        """Register a model class for the given type."""
        cls._model_registry[model_type] = model_class
        logger.info(f"Registered model: {model_type.value} -> {model_class.__name__}")

    @classmethod
    def create_model(
        cls,
        config: "SimpleConfig" = None,
        model_type: ModelType | str = None,
        model_path: str | Path | None = None,
        **kwargs,
    ) -> BaseVisionModel:
        """Create a vision model instance with optimal configuration.

        Args:
            config: SimpleConfig object with settings
            model_type: Type of model to create (overrides config if provided)
            model_path: Path to model files (overrides config if provided)
            **kwargs: Additional model parameters

        Returns:
            Configured model instance

        Raises:
            ValueError: If model type is not registered or invalid parameters
            ModelCreationError: If model creation fails

        """
        # Get values from SimpleConfig if provided
        if config:
            model_type = model_type or config.model_type
            model_path = model_path or config.model_path

        # Validate path first
        if model_path is None or (isinstance(model_path, str) and not model_path.strip()):
            raise ValueError("Model path cannot be None or empty")

        # Convert string model_type to enum if needed
        if isinstance(model_type, str):
            try:
                model_type = ModelType(model_type)
            except ValueError as e:
                raise ValueError(f"Unsupported model type: {model_type}") from e

        if model_type not in cls._model_registry:
            available_models = list(cls._model_registry.keys())
            raise ValueError(
                f"Unsupported model type: {model_type.value}. "
                f"Available models: {[m.value for m in available_models]}",
            )

        try:
            # Get model class
            model_class = cls._model_registry[model_type]

            # Prepare configuration
            model_config = cls._prepare_model_config(model_type, config, **kwargs)

            # Handle offline mode
            if config and config.offline_mode:
                model_config["offline_mode"] = True
                logger.info(f"Offline mode enabled - using local model at {model_path}")

            # Create model instance
            logger.info(
                f"Creating {model_type.value} model with config: {model_config}",
            )
            model = model_class(model_path, **model_config)

            # Apply optimizations
            cls._apply_optimizations(model, config)

            return model

        except Exception as e:
            logger.error(f"Failed to create {model_type.value} model: {e}")
            raise ModelCreationError(f"Failed to create model: {e}", model_type) from e

    @classmethod
    def _prepare_model_config(
        cls,
        model_type: ModelType,
        config: Optional["SimpleConfig"],
        **kwargs,
    ) -> dict:
        """Prepare model-specific configuration."""
        # Base configuration
        model_config = {
            "device_config": DeviceConfig.AUTO,
            "enable_quantization": True,
            "memory_limit_mb": None,
        }

        # Apply simple config if provided
        if config:
            # Convert device_config to enum if it's a string
            device_config = config.device_config
            if isinstance(device_config, str):
                try:
                    device_config = DeviceConfig(device_config)
                except ValueError:
                    # Handle common string mappings
                    device_mapping = {
                        "cpu": DeviceConfig.CPU,
                        "cuda": DeviceConfig.SINGLE_GPU,
                        "cuda:0": DeviceConfig.SINGLE_GPU,
                        "auto": DeviceConfig.AUTO,
                        "multi_gpu": DeviceConfig.MULTI_GPU,
                        "mps": DeviceConfig.AUTO,  # MPS will be detected by AUTO
                    }
                    device_config = device_mapping.get(device_config.lower(), DeviceConfig.AUTO)
                    logger.warning(
                        f"Converted device config string '{config.device_config}' to {device_config}"
                    )

            model_config.update(
                {
                    "device_config": device_config,
                    "enable_quantization": config.enable_quantization,
                    "memory_limit_mb": config.memory_limit_mb,
                    "gpu_memory_fraction": config.gpu_memory_fraction,
                    "enable_gradient_checkpointing": config.enable_gradient_checkpointing,
                    "use_flash_attention": config.use_flash_attention,
                    "trust_remote_code": config.trust_remote_code,
                },
            )

        # Apply model-specific optimizations
        if model_type == ModelType.INTERNVL3:
            model_config.update(cls._get_internvl_config(config))
        elif model_type == ModelType.LLAMA32_VISION:
            model_config.update(cls._get_llama_config(config))

        # Override with explicit kwargs
        model_config.update(kwargs)

        return model_config

    @classmethod
    def _get_internvl_config(cls, config: Optional["SimpleConfig"]) -> dict:
        """Get InternVL-specific configuration optimizations."""
        internvl_config = {
            # Multi-GPU optimization settings
            "enable_multi_gpu": config.enable_multi_gpu if config else False,
            "gpu_memory_fraction": config.gpu_memory_fraction if config else 0.9,
            "enable_gradient_checkpointing": config.enable_gradient_checkpointing if config else True,
            # InternVL-specific features
            "enable_highlight_detection": True,
            "enable_enhanced_parsing": True,
            "enable_computer_vision": True,
            # Performance optimizations
            "use_flash_attention": config.use_flash_attention if config else True,
            "enable_compilation": False,  # Disable for compatibility
            "trust_remote_code": config.trust_remote_code if config else True,
        }

        return internvl_config

    @classmethod
    def _get_llama_config(cls, config: Optional["SimpleConfig"]) -> dict:
        """Get Llama-3.2-Vision specific configuration."""
        llama_config = {
            # Llama-specific optimizations - simplified for single-step
            "enable_graceful_degradation": False,  # Not needed for single-step
            "confidence_threshold": 0.8,
            "enable_awk_fallback": False,  # Not needed for single-step
            # Processing pipeline settings - simplified
            "processing_pipeline": "single_step",
            "extraction_method": "key_value",
            # Memory optimization for single GPU
            "enable_compilation": False,  # Disable for CUDA graph compatibility
            "use_cache": True,
            "low_cpu_mem_usage": True,
        }

        return llama_config

    @classmethod
    def _apply_optimizations(
        cls,
        model: BaseVisionModel,
        config: Optional["UnifiedConfig"],
    ) -> None:
        """Apply post-creation optimizations to the model."""
        try:
            # Device-specific optimizations
            if model.device.type == "cuda":
                cls._apply_cuda_optimizations(model, config)
            elif model.device.type == "mps":
                cls._apply_mps_optimizations(model, config)
            elif model.device.type == "cpu":
                cls._apply_cpu_optimizations(model, config)

            logger.info(f"Applied optimizations for {model.device} device")

        except Exception as e:
            logger.warning(f"Failed to apply some optimizations: {e}")

    @classmethod
    def _apply_cuda_optimizations(
        cls,
        model: BaseVisionModel,
        _config: Optional["UnifiedConfig"],
    ) -> None:
        """Apply CUDA-specific optimizations."""
        import torch

        # Enable optimized attention if available
        try:
            torch.backends.cuda.enable_flash_sdp(True)
        except Exception:
            pass

        # Set memory format for better performance
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass

        # Multi-GPU setup for InternVL
        if (
            hasattr(model, "capabilities")
            and getattr(model.capabilities, "supports_multi_gpu", False)
            and torch.cuda.device_count() > 1
        ):
            try:
                from ..models.model_utils import DeviceManager

                device_manager = DeviceManager()
                if hasattr(model, "model") and model.model is not None:
                    model.model = device_manager.setup_multi_gpu(model.model)
                    logger.info("Applied multi-GPU optimization")
            except ImportError as e:
                logger.warning(f"DeviceManager not available for multi-GPU setup: {e}")

    @classmethod
    def _apply_mps_optimizations(
        cls,
        model: BaseVisionModel,
        _config: Optional["UnifiedConfig"],
    ) -> None:
        """Apply MPS (Apple Silicon) optimizations."""
        # MPS optimizations for Mac M1 development
        try:
            # Reduce memory pressure on unified memory
            if hasattr(model, "model") and model.model is not None:
                model.model.half()  # Use FP16 to reduce memory usage
                logger.info("Applied MPS FP16 optimization")
        except Exception as e:
            logger.warning(f"MPS optimization failed: {e}")

    @classmethod
    def _apply_cpu_optimizations(
        cls,
        model: BaseVisionModel,
        _config: Optional["UnifiedConfig"],
    ) -> None:
        """Apply CPU-specific optimizations."""
        try:
            import torch

            # Enable CPU optimizations
            torch.set_num_threads(torch.get_num_threads())

            # Apply dynamic quantization if supported
            if hasattr(model, "capabilities") and getattr(
                model.capabilities, "supports_quantization", False
            ):
                try:
                    from ..models.model_utils import QuantizationHelper

                    if hasattr(model, "model") and model.model is not None:
                        model.model = QuantizationHelper.apply_dynamic_quantization(
                            model.model,
                        )
                        logger.info("Applied CPU quantization optimization")
                except ImportError as e:
                    logger.warning(f"QuantizationHelper not available: {e}")

        except Exception as e:
            logger.warning(f"CPU optimization failed: {e}")

    @classmethod
    def get_available_models(cls) -> dict[str, str]:
        """Get list of available model types and their descriptions."""
        descriptions = {
            ModelType.INTERNVL3: "Advanced multi-modal model with highlight detection and multi-GPU support",
            ModelType.LLAMA32_VISION: "Robust vision-language model with graceful degradation and production optimizations",
        }

        return {
            model_type.value: descriptions.get(model_type, "Vision model")
            for model_type in cls._model_registry.keys()
        }

    @classmethod
    def get_recommended_config(
        cls,
        model_type: ModelType,
        hardware_profile: str = "auto",
    ) -> dict:
        """Get recommended configuration for a model type and hardware profile.

        Args:
            model_type: Type of model
            hardware_profile: Hardware profile (auto, mac_m1, h200_dev, v100_prod)

        Returns:
            Recommended configuration dictionary

        """
        base_config = {
            "device_config": DeviceConfig.AUTO,
            "enable_quantization": True,
            "memory_limit_mb": None,
        }

        # Hardware-specific recommendations
        if hardware_profile == "mac_m1":
            base_config.update(
                {
                    "device_config": DeviceConfig.AUTO,  # Will select MPS
                    "enable_quantization": False,  # MPS doesn't support quantization
                    "memory_limit_mb": 16384,  # Conservative for unified memory
                },
            )
        elif hardware_profile == "h200_dev":
            base_config.update(
                {
                    "device_config": DeviceConfig.MULTI_GPU,
                    "enable_quantization": True,  # Required for CUDA scatter operation compatibility
                    "memory_limit_mb": None,
                },
            )
        elif hardware_profile == "v100_prod":
            base_config.update(
                {
                    "device_config": DeviceConfig.SINGLE_GPU,
                    "enable_quantization": True,  # Required for 16GB memory
                    "memory_limit_mb": 15360,  # V100 16GB with buffer
                },
            )

        # Model-specific adjustments
        if model_type == ModelType.INTERNVL3:
            if hardware_profile in ["h200_dev", "auto"]:
                base_config["enable_multi_gpu"] = True
                base_config["enable_highlight_detection"] = True

        elif model_type == ModelType.LLAMA32_VISION:
            base_config["enable_graceful_degradation"] = True
            base_config["processing_pipeline"] = "7step"

        return base_config

    @classmethod
    def get_supported_models(cls) -> list[str]:
        """Get list of supported model types."""
        return [model_type.value for model_type in cls._model_registry.keys()]

    @classmethod
    def is_model_supported(cls, model_type: ModelType | str) -> bool:
        """Check if a model type is supported."""
        # Handle string input
        if isinstance(model_type, str):
            try:
                model_type = ModelType(model_type)
            except ValueError:
                return False
        return model_type in cls._model_registry


# Auto-register models when available
def _auto_register_models():
    """Automatically register available model implementations."""
    # Try to register actual models first
    try:
        from ..models.internvl_model import InternVLModel

        ModelFactory.register_model(ModelType.INTERNVL3, InternVLModel)
        logger.debug("Registered InternVL model")
    except ImportError as e:
        logger.debug(f"InternVL model import failed: {e}")
        # Fall back to placeholder for Phase 1
        try:
            from ..models.placeholder_models import PlaceholderInternVLModel

            ModelFactory.register_model(ModelType.INTERNVL3, PlaceholderInternVLModel)
            logger.debug("Using placeholder InternVL model")
        except ImportError as e2:
            logger.warning(f"InternVL placeholder model not available: {e2}")

    try:
        from ..models.llama_model import LlamaVisionModel

        ModelFactory.register_model(ModelType.LLAMA32_VISION, LlamaVisionModel)
        logger.debug("Registered Llama Vision model")
    except ImportError as e:
        logger.debug(f"Llama Vision model import failed: {e}")
        # Fall back to placeholder for Phase 1
        try:
            from ..models.placeholder_models import PlaceholderLlamaVisionModel

            ModelFactory.register_model(
                ModelType.LLAMA32_VISION,
                PlaceholderLlamaVisionModel,
            )
            logger.debug("Using placeholder Llama Vision model")
        except ImportError as e2:
            logger.warning(f"Llama Vision placeholder model not available: {e2}")


# Register models on import
_auto_register_models()
