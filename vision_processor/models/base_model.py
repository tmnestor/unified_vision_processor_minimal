"""Base Vision Model Abstraction

Provides a unified interface for both InternVL3 and Llama-3.2-Vision models,
preserving their unique capabilities while enabling model-agnostic processing.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

if TYPE_CHECKING:
    import torch
    from PIL import Image

from ..exceptions import InvalidImageError


class ModelType(Enum):
    """Supported vision model types."""

    INTERNVL3 = "internvl3"
    LLAMA32_VISION = "llama32_vision"


class DeviceConfig(Enum):
    """Device configuration options."""

    AUTO = "auto"
    CPU = "cpu"
    SINGLE_GPU = "cuda:0"
    MULTI_GPU = "multi_gpu"


@dataclass
class ModelResponse:
    """Standardized model response format."""

    raw_text: str
    confidence: float
    processing_time: float
    device_used: str
    memory_usage: Union[float, None] = None
    model_type: Union[str, None] = None
    quantized: bool = False

    # Additional metadata for analysis
    metadata: Union[dict[str, Any], None] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ModelCapabilities:
    """Model-specific capabilities and features."""

    supports_multi_gpu: bool
    supports_quantization: bool
    supports_highlight_detection: bool
    supports_batch_processing: bool
    max_image_size: tuple[int, int]
    memory_efficient: bool
    cross_platform: bool


class BaseVisionModel(ABC):
    """Abstract base class for vision models.

    Provides unified interface while preserving model-specific optimizations:
    - InternVL3: Multi-GPU optimization, highlight detection, enhanced parsing
    - Llama-3.2-Vision: Robust processing, graceful degradation, production-ready
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        device_config: DeviceConfig = DeviceConfig.AUTO,
        enable_quantization: bool = True,
        memory_limit_mb: Union[int, None] = None,
        **kwargs,
    ):
        self.model_path = Path(model_path)
        self.device_config = device_config
        self.enable_quantization = enable_quantization
        self.memory_limit_mb = memory_limit_mb
        self.kwargs = kwargs

        # Initialize device management
        self.device = self._setup_device()
        self.model = None
        self.tokenizer = None
        self.processor = None

        # Model state
        self.is_loaded = False
        self.capabilities = self._get_capabilities()

    @abstractmethod
    def _get_capabilities(self) -> ModelCapabilities:
        """Return model-specific capabilities."""

    @abstractmethod
    def _setup_device(self) -> "torch.device":
        """Setup device configuration for the model."""

    @abstractmethod
    def load_model(self) -> None:
        """Load the model into memory."""

    @abstractmethod
    def unload_model(self) -> None:
        """Unload the model from memory."""

    @abstractmethod
    def process_image(
        self,
        image_path: Union[str, Path, "Image.Image"],
        prompt: str,
        **kwargs,
    ) -> ModelResponse:
        """Process a single image with the given prompt.

        Args:
            image_path: Path to image file or PIL Image
            prompt: Text prompt for processing
            **kwargs: Model-specific parameters

        Returns:
            ModelResponse with standardized format

        """

    @abstractmethod
    def process_batch(
        self,
        image_paths: list[Union[str, Path, "Image.Image"]],
        prompts: list[str],
        **kwargs,
    ) -> list[ModelResponse]:
        """Process multiple images in batch.

        Args:
            image_paths: List of image paths or PIL Images
            prompts: List of text prompts
            **kwargs: Model-specific parameters

        Returns:
            List of ModelResponse objects

        """

    def get_memory_usage(self) -> float:
        """Get current GPU memory usage in MB."""
        try:
            import torch

            if torch.cuda.is_available() and self.device.type == "cuda":
                return torch.cuda.memory_allocated(self.device) / 1024 / 1024
        except ImportError:
            pass
        return 0.0

    def optimize_for_inference(self) -> None:
        """Apply model-specific optimizations for inference."""
        if self.model is not None:
            self.model.eval()

            # Apply quantization if enabled and supported
            if self.enable_quantization and self.capabilities.supports_quantization:
                self._apply_quantization()

    @abstractmethod
    def _apply_quantization(self) -> None:
        """Apply model-specific quantization."""

    def validate_image(self, image_path: Union[str, Path, "Image.Image"]) -> bool:
        """Validate image format and size constraints.

        Args:
            image_path: Path to image or PIL Image

        Returns:
            True if image is valid for processing

        """
        try:
            if isinstance(image_path, (str, Path)):
                from PIL import Image

                image = Image.open(image_path)
            else:
                image = image_path

            # Check image size constraints
            max_width, max_height = self.capabilities.max_image_size
            if image.size[0] > max_width or image.size[1] > max_height:
                return False

            # Check image format
            if image.format not in ["JPEG", "PNG", "RGB"]:
                return False

            return True

        except Exception as e:
            raise InvalidImageError(
                image_path=Path(str(image_path)) if isinstance(image_path, (str, Path)) else Path("unknown"),
                reason=f"Failed to validate image: {str(e)}"
            ) from e

    def get_device_info(self) -> dict[str, Any]:
        """Get detailed device information."""
        info = {
            "device": str(self.device),
            "device_config": self.device_config.value,
            "memory_limit_mb": self.memory_limit_mb,
            "quantization_enabled": self.enable_quantization,
        }

        try:
            import torch

            if torch.cuda.is_available():
                info.update(
                    {
                        "cuda_version": torch.version.cuda,
                        "gpu_count": torch.cuda.device_count(),
                        "current_gpu": torch.cuda.current_device()
                        if self.device.type == "cuda"
                        else None,
                        "gpu_memory_total": torch.cuda.get_device_properties(
                            0
                        ).total_memory
                        / 1024
                        / 1024
                        if self.device.type == "cuda"
                        else None,
                        "gpu_memory_allocated": self.get_memory_usage(),
                    },
                )
        except ImportError:
            pass

        return info

    def __enter__(self):
        """Context manager entry."""
        if not self.is_loaded:
            self.load_model()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.is_loaded:
            self.unload_model()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_path={self.model_path}, device={self.device}, loaded={self.is_loaded})"
