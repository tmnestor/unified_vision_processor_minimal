"""Model Registry for Extensible Model Management
===========================================

Factory system for managing vision models in the comparison framework.
Provides extensible registration and instantiation of models using the
BaseVisionModel interface.
"""

import importlib
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from ..constants import ModelNames
from ..exceptions import ModelLoadError
from ..models.base_model import (
    BaseVisionModel,
    DeviceConfig,
    ModelCapabilities,
    ModelType,
)
from ..utils.logging_config import VisionProcessorLogger
from .config_manager import ConfigManager


class ModelStatus(Enum):
    """Model registration and validation status."""

    REGISTERED = "registered"
    VALIDATED = "validated"
    LOADED = "loaded"
    ERROR = "error"
    NOT_FOUND = "not_found"


@dataclass
class ModelRegistration:
    """Model registration information."""

    name: str
    model_type: ModelType
    model_class: Type[BaseVisionModel]
    default_path: str
    description: str
    status: ModelStatus = ModelStatus.REGISTERED
    error_message: Optional[str] = None
    capabilities: Optional[ModelCapabilities] = None

    def __post_init__(self):
        """Validate registration after initialization."""
        if not issubclass(self.model_class, BaseVisionModel):
            raise ValueError(
                f"Model class {self.model_class} must inherit from BaseVisionModel"
            )


class ModelFactory:
    """Factory for creating model instances with standardized configuration."""

    def __init__(self, config_manager: ConfigManager):
        """Initialize factory with configuration manager.

        Args:
            config_manager: Unified configuration manager
        """
        self.config_manager = config_manager

    def create_model(
        self,
        registration: ModelRegistration,
        model_path: Optional[str] = None,
        device_config: Optional[DeviceConfig] = None,
        **kwargs,
    ) -> BaseVisionModel:
        """Create model instance from registration.

        Args:
            registration: Model registration information
            model_path: Override model path (uses registration default if None)
            device_config: Device configuration (uses auto-detection if None)
            **kwargs: Additional model-specific arguments

        Returns:
            Configured model instance
        """
        effective_path = model_path or registration.default_path
        effective_device = device_config or DeviceConfig.AUTO

        # Merge configuration with any overrides
        model_kwargs = {
            "model_path": effective_path,
            "device_config": effective_device,
            "enable_quantization": self.config_manager.processing.quantization,
            "memory_limit_mb": self.config_manager.processing.memory_limit_mb,
            "config": self.config_manager,  # Pass the entire config manager
            **kwargs,
        }

        try:
            return registration.model_class(**model_kwargs)
        except ImportError as e:
            raise ModelLoadError(
                model_name=registration.name,
                original_error=e,
                missing_dependency=str(e),
            ) from e
        except Exception as e:
            raise ModelLoadError(
                model_name=registration.name,
                original_error=e,
                model_path=effective_path,
            ) from e


class ModelRegistry:
    """Registry for managing available vision models."""

    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """Initialize registry with configuration manager.

        Args:
            config_manager: Unified configuration manager (creates default if None)
        """
        self._models: Dict[str, ModelRegistration] = {}
        self._factory: Optional[ModelFactory] = None
        self.config_manager = config_manager or ConfigManager()
        self.logger = VisionProcessorLogger(self.config_manager)
        self._register_builtin_models()

    def _register_builtin_models(self):
        """Register built-in model implementations."""
        try:
            # Import and register Llama model
            from ..models.llama_model import LlamaVisionModel

            self.register_model(
                name=ModelNames.LLAMA,
                model_type=ModelType.LLAMA32_VISION,
                model_class=LlamaVisionModel,
                default_path=self.config_manager.model_paths.llama,
                description="Llama-3.2-11B-Vision model with 7-step processing pipeline",
            )
        except ImportError as e:
            self.logger.warning(f"Failed to register Llama model: {e}")

        try:
            # Import and register InternVL model
            from ..models.internvl_model import InternVLModel

            self.register_model(
                name=ModelNames.INTERNVL,
                model_type=ModelType.INTERNVL3,
                model_class=InternVLModel,
                default_path=self.config_manager.model_paths.internvl,
                description="InternVL3-8B model with multi-GPU optimization and highlight detection",
            )
        except ImportError as e:
            self.logger.warning(f"Failed to register InternVL model: {e}")

    def register_model(
        self,
        name: str,
        model_type: ModelType,
        model_class: Type[BaseVisionModel],
        default_path: str,
        description: str,
    ):
        """Register a new model in the registry.

        Args:
            name: Unique model name (e.g., "llama", "internvl")
            model_type: Model type classification
            model_class: Model implementation class
            default_path: Default model path
            description: Human-readable description
        """
        registration = ModelRegistration(
            name=name,
            model_type=model_type,
            model_class=model_class,
            default_path=default_path,
            description=description,
        )

        self._models[name] = registration
        self.logger.status(f"Registered model: {name} ({model_type.value})")

    def register_external_model(
        self,
        name: str,
        module_path: str,
        class_name: str,
        model_type: ModelType,
        default_path: str,
        description: str,
    ):
        """Register an external model by importing it dynamically.

        Args:
            name: Unique model name
            module_path: Python module path (e.g., "my_models.custom_model")
            class_name: Class name within the module
            model_type: Model type classification
            default_path: Default model path
            description: Description
        """
        try:
            module = importlib.import_module(module_path)
            model_class = getattr(module, class_name)

            self.register_model(
                name, model_type, model_class, default_path, description
            )

        except (ImportError, AttributeError) as e:
            error_msg = f"Failed to import external model {name}: {e}"
            self.logger.error(error_msg)

            # Register with error status for debugging
            registration = ModelRegistration(
                name=name,
                model_type=model_type,
                model_class=BaseVisionModel,  # Placeholder
                default_path=default_path,
                description=description,
                status=ModelStatus.ERROR,
                error_message=error_msg,
            )
            self._models[name] = registration

    def get_model(self, name: str) -> Optional[ModelRegistration]:
        """Get model registration by name."""
        return self._models.get(name)

    def list_models(self) -> List[str]:
        """Get list of all registered model names."""
        return list(self._models.keys())

    def list_available_models(self) -> List[str]:
        """Get list of models that can be instantiated (no errors)."""
        return [
            name
            for name, reg in self._models.items()
            if reg.status != ModelStatus.ERROR
        ]

    def validate_model(self, name: str, model_path: Optional[str] = None) -> bool:
        """Validate that a model can be loaded.

        Args:
            name: Model name
            model_path: Override model path for validation

        Returns:
            True if model can be loaded, False otherwise
        """
        registration = self.get_model(name)
        if not registration:
            self.logger.error(f"Model not found: {name}")
            return False

        if registration.status == ModelStatus.ERROR:
            self.logger.error(
                f"Model has registration error: {registration.error_message}"
            )
            return False

        # Validate model path
        effective_path = model_path or registration.default_path
        if not Path(effective_path).exists():
            self.logger.error(f"Model path does not exist: {effective_path}")
            registration.status = ModelStatus.NOT_FOUND
            return False

        # Try to get model capabilities (lightweight validation)
        try:
            # Create a dummy instance just to check class instantiation
            temp_model = registration.model_class(
                model_path=effective_path,
                device_config=DeviceConfig.CPU,  # Use CPU for validation
            )
            registration.capabilities = temp_model.capabilities
            registration.status = ModelStatus.VALIDATED
            del temp_model  # Clean up
            self.logger.success(f"Model validation passed: {name}")
            return True

        except Exception as e:
            error_msg = f"Model validation failed: {e}"
            self.logger.error(f"{name}: {error_msg}")
            registration.status = ModelStatus.ERROR
            registration.error_message = error_msg
            return False

    def create_model(
        self,
        name: str,
        model_path: Optional[str] = None,
        device_config: Optional[DeviceConfig] = None,
        **kwargs,
    ) -> BaseVisionModel:
        """Create and return a model instance.

        Args:
            name: Model name
            model_path: Override model path
            device_config: Device configuration
            **kwargs: Additional model arguments

        Returns:
            Configured model instance

        Raises:
            ValueError: If model is not registered
            RuntimeError: If model cannot be created
        """
        registration = self.get_model(name)
        if not registration:
            raise ValueError(f"Model not registered: {name}")

        if registration.status == ModelStatus.ERROR:
            raise RuntimeError(f"Model has error: {registration.error_message}")

        # Create factory if needed
        if not self._factory:
            self._factory = ModelFactory(self.config_manager)

        return self._factory.create_model(
            registration, model_path, device_config, **kwargs
        )

    def get_model_info(self, name: str) -> Dict[str, Any]:
        """Get comprehensive model information.

        Args:
            name: Model name

        Returns:
            Dictionary with model information
        """
        registration = self.get_model(name)
        if not registration:
            return {"error": f"Model not found: {name}"}

        info = {
            "name": registration.name,
            "type": registration.model_type.value,
            "description": registration.description,
            "default_path": registration.default_path,
            "status": registration.status.value,
            "path_exists": Path(registration.default_path).exists(),
        }

        if registration.error_message:
            info["error_message"] = registration.error_message

        if registration.capabilities:
            info["capabilities"] = {
                "supports_multi_gpu": registration.capabilities.supports_multi_gpu,
                "supports_quantization": registration.capabilities.supports_quantization,
                "supports_highlight_detection": registration.capabilities.supports_highlight_detection,
                "supports_batch_processing": registration.capabilities.supports_batch_processing,
                "max_image_size": registration.capabilities.max_image_size,
                "memory_efficient": registration.capabilities.memory_efficient,
                "cross_platform": registration.capabilities.cross_platform,
            }

        return info

    def print_registry_status(self):
        """Print comprehensive registry status."""
        # Only print if console output enabled
        if self.logger._should_use_console():
            self.logger.info("Model Registry Status:")
            self.logger.info(f"Total registered: {len(self._models)}")

            available_count = len(self.list_available_models())
            self.logger.info(f"Available: {available_count}")

            # Detailed results only in verbose mode
            if self.logger._is_verbose():
                self.logger.status("Registered Models:")
                for name, registration in self._models.items():
                    status_emoji = {
                        ModelStatus.REGISTERED: "📝",
                        ModelStatus.VALIDATED: "✅",
                        ModelStatus.LOADED: "🚀",
                        ModelStatus.ERROR: "❌",
                        ModelStatus.NOT_FOUND: "🔍",
                    }.get(registration.status, "❓")

                    path_status = (
                        "✅" if Path(registration.default_path).exists() else "❌"
                    )

                    if registration.status == ModelStatus.ERROR:
                        self.logger.error(
                            f"{status_emoji} {name}: {registration.model_type.value}"
                        )
                    else:
                        self.logger.info(
                            f"{status_emoji} {name}: {registration.model_type.value}"
                        )

                    self.logger.debug(f"   Description: {registration.description}")
                    self.logger.debug(
                        f"   Path: {path_status} {registration.default_path}"
                    )

                    if registration.error_message:
                        self.logger.error(f"   Error: {registration.error_message}")

                    if registration.capabilities:
                        caps = registration.capabilities
                        self.logger.debug(
                            f"   Capabilities: Multi-GPU:{caps.supports_multi_gpu}, "
                            f"Quantization:{caps.supports_quantization}, "
                            f"Highlights:{caps.supports_highlight_detection}"
                        )

    def validate_all_models(
        self, model_paths: Optional[Dict[str, str]] = None
    ) -> Dict[str, bool]:
        """Validate all registered models.

        Args:
            model_paths: Override paths for specific models

        Returns:
            Dictionary mapping model names to validation status
        """
        model_paths = model_paths or {}
        results = {}

        self.logger.info("Validating all registered models...")
        for name in self.list_models():
            override_path = model_paths.get(name)
            results[name] = self.validate_model(name, override_path)

        successful = sum(results.values())
        total = len(results)
        self.logger.info(
            f"Validation Summary: {successful}/{total} models validated successfully"
        )

        return results


# Global registry instance
GLOBAL_MODEL_REGISTRY = None


def get_model_registry(config_manager: Optional[ConfigManager] = None) -> ModelRegistry:
    """Get the global model registry instance."""
    global GLOBAL_MODEL_REGISTRY
    if GLOBAL_MODEL_REGISTRY is None:
        GLOBAL_MODEL_REGISTRY = ModelRegistry(config_manager)
    return GLOBAL_MODEL_REGISTRY


def register_model(
    name: str,
    model_type: ModelType,
    model_class: Type[BaseVisionModel],
    default_path: str,
    description: str,
):
    """Convenience function to register a model globally."""
    registry = get_model_registry()
    registry.register_model(name, model_type, model_class, default_path, description)


def create_model(name: str, **kwargs) -> BaseVisionModel:
    """Convenience function to create a model from global registry."""
    registry = get_model_registry()
    return registry.create_model(name, **kwargs)
