"""Llama-3.2-Vision Model Implementation - Refactored

Integrates specialized components for better maintainability and separation of concerns.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Union

import torch
from PIL import Image

from .base_model import ModelResponse
from .llama_base import LlamaVisionModelBase
from .llama_classification import LlamaDocumentClassifier
from .llama_inference import LlamaInferenceManager
from .llama_memory import LlamaMemoryManager
from .llama_quantization import LlamaQuantizationManager

logger = logging.getLogger(__name__)


class LlamaVisionModel(LlamaVisionModelBase):
    """Llama-3.2-Vision implementation with modular architecture.

    Features:
    - Single GPU optimization with 8-bit quantization
    - Graceful degradation capabilities
    - 7-step pipeline integration
    - Australian tax document processing
    - Production-ready inference
    - Ultra-aggressive repetition control for Llama-3.2-Vision bugs
    
    This refactored version delegates specialized functionality to focused modules.
    """

    def __init__(self, *args, **kwargs):
        """Initialize LlamaVisionModel with modular components."""
        super().__init__(*args, **kwargs)
        
        # Validate configuration before proceeding
        self._validate_config()
        
        # Initialize specialized managers
        self.quantization_manager = LlamaQuantizationManager(
            self.config, 
            self.enable_quantization, 
            **kwargs
        )
        
        self.memory_manager = LlamaMemoryManager(
            self.config,
            self.device_config,
            self.memory_limit_mb
        )
        
        self.inference_manager = LlamaInferenceManager(
            self.config,
            self.memory_manager,
            self.quantization_manager
        )
        
        self.document_classifier = LlamaDocumentClassifier(
            self.inference_manager
        )
        
        # Setup device through memory manager
        self.device = self.memory_manager.setup_device()

    def _setup_device(self) -> torch.device:
        """Setup device configuration - delegated to memory manager."""
        return self.memory_manager.setup_device()

    def load_model(self) -> None:
        """Load Llama-3.2-Vision model - delegated to inference manager."""
        model_path = Path(self.model_path)
        self.inference_manager.load_model(model_path, self.device)
        self.is_loaded = self.inference_manager.is_loaded

    def unload_model(self) -> None:
        """Unload model from memory - delegated to inference manager."""
        self.inference_manager.unload_model()
        self.is_loaded = False

    def process_image(
        self,
        image_path: Union[str, Path, Image.Image],
        prompt: str,
        **kwargs,
    ) -> ModelResponse:
        """Process image with Llama-3.2-Vision model.

        Args:
            image_path: Path to image file or PIL Image
            prompt: Text prompt for processing
            **kwargs: Additional parameters (max_new_tokens, temperature, etc.)

        Returns:
            ModelResponse with standardized format
        """
        return self.inference_manager.process_image(image_path, prompt, **kwargs)

    def process_batch(
        self,
        image_paths: List[Union[str, Path, Image.Image]],
        prompts: List[str],
        **kwargs,
    ) -> List[ModelResponse]:
        """Process multiple images in batch."""
        return self.inference_manager.process_batch(image_paths, prompts, **kwargs)

    def classify_document(
        self,
        image_path: Union[str, Path, Image.Image],
    ) -> Dict[str, Any]:
        """Classify document type using Llama-3.2-Vision.

        Args:
            image_path: Path to image file or PIL Image

        Returns:
            Classification result dictionary
        """
        return self.document_classifier.classify_document(image_path)

    def predict(self, image_path: Union[str, Path], prompt: str) -> str:
        """Generate prediction with CUDA-safe parameters.

        Args:
            image_path: Path to image file or HTTP URL
            prompt: Text prompt for extraction

        Returns:
            Generated response text
        """
        try:
            response = self.inference_manager.process_image(image_path, prompt)
            return response.raw_text
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return f"Error: {str(e)}"

    def _apply_quantization(self) -> None:
        """Apply quantization to model - delegated to quantization manager."""
        if hasattr(self.inference_manager, 'model') and self.inference_manager.model:
            self.quantization_manager.apply_quantization(self.inference_manager.model)

    def get_memory_usage(self) -> float:
        """Get current GPU memory usage in MB."""
        return self.memory_manager.get_memory_usage(self.device)

    # Additional utility methods for backward compatibility
    def _cleanup_memory(self) -> None:
        """Clean up GPU and system memory."""
        self.memory_manager.cleanup_memory()

    def _preprocess_image(self, image_path: Union[str, Path, Image.Image]) -> Image.Image:
        """Preprocess image for Llama-3.2-Vision compatibility."""
        return self.inference_manager.preprocess_image(image_path)

    def _prepare_inputs(self, image: Image.Image, prompt: str) -> Dict[str, Any]:
        """Prepare inputs for model inference."""
        return self.inference_manager.prepare_inputs(image, prompt)

    def _clean_response(self, response: str, image_name: str = "") -> str:
        """Clean response by removing repetition and special tokens."""
        return self.inference_manager.clean_response(response, image_name)

    def __repr__(self) -> str:
        """String representation of the model."""
        return (f"{self.__class__.__name__}("
                f"model_path={self.model_path}, "
                f"device={self.device}, "
                f"loaded={self.is_loaded}, "
                f"quantization={self.quantization_manager.enable_quantization})")