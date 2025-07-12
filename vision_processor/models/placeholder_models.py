"""Placeholder Model Implementations for Phase 1 Testing

These are temporary implementations to verify the pipeline framework.
They will be replaced with actual model implementations in Phase 2.
"""

import time
from pathlib import Path

import torch
from PIL import Image

from .base_model import BaseVisionModel, ModelCapabilities, ModelResponse


class PlaceholderInternVLModel(BaseVisionModel):
    """Placeholder InternVL model for testing."""

    def _get_capabilities(self) -> ModelCapabilities:
        """Return InternVL-like capabilities."""
        return ModelCapabilities(
            supports_multi_gpu=True,
            supports_quantization=True,
            supports_highlight_detection=True,
            supports_batch_processing=True,
            max_image_size=(3840, 2160),
            memory_efficient=True,
            cross_platform=True,
        )

    def _setup_device(self) -> torch.device:
        """Setup device for testing."""
        # Simple device selection for testing
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def load_model(self) -> None:
        """Placeholder model loading."""
        self.is_loaded = True
        self.model = "placeholder_internvl_model"
        self.tokenizer = "placeholder_tokenizer"
        self.processor = "placeholder_processor"

    def unload_model(self) -> None:
        """Placeholder model unloading."""
        self.is_loaded = False
        self.model = None
        self.tokenizer = None
        self.processor = None

    def process_image(
        self,
        _image_path: str | Path | Image.Image,
        prompt: str,
        **_kwargs,
    ) -> ModelResponse:
        """Process image with placeholder logic."""
        if not self.is_loaded:
            self.load_model()

        # Simulate processing
        start_time = time.time()
        time.sleep(0.1)  # Simulate processing time

        # Generate placeholder response
        raw_text = f"[InternVL Placeholder] Processing {prompt}\n"
        raw_text += "Extracted fields:\n"
        raw_text += "- date_value: 2024-01-15\n"
        raw_text += "- store_name_value: Placeholder Store\n"
        raw_text += "- total_value: $123.45\n"
        raw_text += "- tax_value: $12.35\n"

        processing_time = time.time() - start_time

        return ModelResponse(
            raw_text=raw_text,
            confidence=0.85,
            processing_time=processing_time,
            device_used=str(self.device),
            memory_usage=100.0,
            model_type="internvl3_placeholder",
            quantized=False,
        )

    def process_batch(
        self,
        image_paths: list[str | Path | Image.Image],
        prompts: list[str],
        **_kwargs,
    ) -> list[ModelResponse]:
        """Process batch of images."""
        return [self.process_image(img, prompt) for img, prompt in zip(image_paths, prompts, strict=False)]

    def _apply_quantization(self) -> None:
        """Placeholder quantization."""


class PlaceholderLlamaVisionModel(BaseVisionModel):
    """Placeholder Llama Vision model for testing."""

    def _get_capabilities(self) -> ModelCapabilities:
        """Return Llama-like capabilities."""
        return ModelCapabilities(
            supports_multi_gpu=False,
            supports_quantization=True,
            supports_highlight_detection=False,
            supports_batch_processing=True,
            max_image_size=(2048, 2048),
            memory_efficient=True,
            cross_platform=True,
        )

    def _setup_device(self) -> torch.device:
        """Setup device for testing."""
        # Simple device selection for testing
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def load_model(self) -> None:
        """Placeholder model loading."""
        self.is_loaded = True
        self.model = "placeholder_llama_model"
        self.tokenizer = "placeholder_tokenizer"
        self.processor = "placeholder_processor"

    def unload_model(self) -> None:
        """Placeholder model unloading."""
        self.is_loaded = False
        self.model = None
        self.tokenizer = None
        self.processor = None

    def process_image(
        self,
        _image_path: str | Path | Image.Image,
        prompt: str,
        **_kwargs,
    ) -> ModelResponse:
        """Process image with placeholder logic."""
        if not self.is_loaded:
            self.load_model()

        # Simulate processing
        start_time = time.time()
        time.sleep(0.1)  # Simulate processing time

        # Generate placeholder response
        raw_text = f"[Llama Vision Placeholder] Processing {prompt}\n"
        raw_text += "Extracted information:\n"
        raw_text += "- Date: January 15, 2024\n"
        raw_text += "- Vendor: Placeholder Business\n"
        raw_text += "- Amount: 123.45\n"
        raw_text += "- GST: 12.35\n"

        processing_time = time.time() - start_time

        return ModelResponse(
            raw_text=raw_text,
            confidence=0.90,
            processing_time=processing_time,
            device_used=str(self.device),
            memory_usage=80.0,
            model_type="llama32_vision_placeholder",
            quantized=False,
        )

    def process_batch(
        self,
        image_paths: list[str | Path | Image.Image],
        prompts: list[str],
        **_kwargs,
    ) -> list[ModelResponse]:
        """Process batch of images."""
        return [self.process_image(img, prompt) for img, prompt in zip(image_paths, prompts, strict=False)]

    def _apply_quantization(self) -> None:
        """Placeholder quantization."""
