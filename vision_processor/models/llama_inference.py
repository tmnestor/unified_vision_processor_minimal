"""Llama-3.2-Vision Inference Pipeline

Handles model loading, image processing, and inference operations.
"""

import logging
import re
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Union

import torch
from PIL import Image
from transformers import AutoProcessor, MllamaForConditionalGeneration

from .base_model import ModelResponse

logger = logging.getLogger(__name__)


class LlamaInferenceManager:
    """Manages inference pipeline for Llama-3.2-Vision model."""

    def __init__(self, config, memory_manager, quantization_manager):
        """Initialize inference manager.
        
        Args:
            config: Model configuration object
            memory_manager: Memory management instance
            quantization_manager: Quantization management instance
        """
        self.config = config
        self.memory_manager = memory_manager
        self.quantization_manager = quantization_manager
        
        # Model components
        self.model = None
        self.processor = None
        self.device = None
        self.is_loaded = False
        
        # Configuration extracted from config
        self.model_path = None
        self.repetition_enabled = True
        self.max_new_tokens_limit = 1024
        self.cleanup_tokens = [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|image|>",
            "[INST]",
            "[/INST]",
            "<s>",
            "</s>",
        ]

    def load_model(self, model_path: Path, device: torch.device) -> None:
        """Load Llama-3.2-Vision model with auto-configuration.
        
        Args:
            model_path: Path to the model files
            device: Target device for the model
        """
        if self.is_loaded:
            logger.warning("Model already loaded")
            return

        self.model_path = model_path
        self.device = device
        
        logger.info(f"Loading Llama-3.2-Vision model from {model_path}")

        # Suppress warnings
        warnings.filterwarnings("ignore", message=".*pad_token_id.*")

        # Clean memory before loading
        self.memory_manager.cleanup_memory()

        # Get configurations
        device_map = self.memory_manager.get_device_map_from_config()
        quantization_config = self.quantization_manager.get_quantization_config()

        logger.info(f"🔧 V100 Mode: Using device configuration: {device_map}")

        # V100 compliance validation
        estimated_memory = self.quantization_manager.estimate_memory_usage(quantization_config)
        self.quantization_manager.validate_v100_compliance(estimated_memory)

        # Configure loading parameters
        model_loading_args = self.memory_manager.configure_model_loading_args(
            device, device_map, quantization_config
        )

        try:
            # Load processor first
            processor_config = {}
            if model_loading_args.get("local_files_only"):
                processor_config["local_files_only"] = True

            self.processor = AutoProcessor.from_pretrained(
                str(model_path),
                trust_remote_code=True,
                **processor_config,
            )
            logger.info("Processor loaded successfully")

            # Load model with suppressed generation config warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                self.model = MllamaForConditionalGeneration.from_pretrained(
                    str(model_path),
                    **model_loading_args,
                ).eval()

            # Move to device if needed (CPU only)
            if device.type == "cpu":
                self.model = self.model.to("cpu")

            self.is_loaded = True
            logger.info(f"Model loaded successfully on {device}")

            # Configure generation settings
            self._configure_generation_settings()

            # Store metadata for inference
            self.model._llama_config = {
                "device_map": device_map,
                "quantization_enabled": quantization_config is not None,
                "offline_mode": True,
            }

            # Test basic functionality
            self._test_model_functionality()

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.memory_manager.cleanup_memory()
            raise RuntimeError(f"Failed to load Llama-3.2-Vision model: {e}") from e

    def _configure_generation_settings(self) -> None:
        """Configure generation settings for stable inference."""
        # Set sampling parameters to None to suppress warnings when do_sample=False
        self.model.generation_config.max_new_tokens = self.max_new_tokens_limit
        self.model.generation_config.do_sample = False  # Deterministic for consistency
        self.model.generation_config.temperature = None  # Set to None to suppress warnings
        self.model.generation_config.top_p = None  # Set to None to suppress warnings
        self.model.generation_config.top_k = None  # Set to None to suppress warnings
        self.model.config.use_cache = True  # Enable KV cache
        logger.info("Configured generation settings for deterministic inference")

    def unload_model(self) -> None:
        """Unload model from memory."""
        if not self.is_loaded:
            return

        self.model = None
        self.processor = None
        self.memory_manager.cleanup_memory()
        self.is_loaded = False
        logger.info("Model unloaded")

    def _test_model_functionality(self) -> None:
        """Test basic model functionality."""
        try:
            logger.info("Testing model functionality...")

            # Simple text-only test
            test_prompt = "Hello, how are you?"
            test_inputs = self.processor.tokenizer(test_prompt, return_tensors="pt")

            # Move to appropriate device
            if hasattr(self.model, "device") and self.model.device.type != "cpu":
                test_inputs = {
                    k: v.to(self.model.device) for k, v in test_inputs.items()
                }

            with torch.no_grad():
                test_outputs = self.model.generate(
                    **test_inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                )

            test_response = self.processor.decode(
                test_outputs[0][test_inputs["input_ids"].shape[-1] :],
                skip_special_tokens=True,
            )

            logger.info(f"Model test successful: '{test_response[:50]}...'")

        except Exception as e:
            logger.warning(f"Model functionality test failed: {e}")

    def preprocess_image(self, image_path: Union[str, Path, Image.Image]) -> Image.Image:
        """Preprocess image for Llama-3.2-Vision compatibility.
        
        Args:
            image_path: Path to image file or PIL Image
            
        Returns:
            Preprocessed PIL Image
        """
        # Load image
        if isinstance(image_path, (str, Path)):
            image = Image.open(image_path)
        else:
            image = image_path

        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Resize if too large using YAML config
        image_config = getattr(self.config, "yaml_config", {}).get(
            "image_processing", {}
        )
        max_size = image_config.get("max_image_size", 1024)

        if max(image.size) > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            logger.info(f"Image resized to {image.size} (max: {max_size})")

        return image

    def prepare_inputs(self, image: Image.Image, prompt: str) -> Dict[str, Any]:
        """Prepare inputs for model inference using official chat template.

        Args:
            image: Preprocessed PIL Image
            prompt: Text prompt (without <|image|> token - will be added by chat template)

        Returns:
            Processed inputs dictionary
        """
        # Clean prompt of any existing image tokens - chat template will handle this
        clean_prompt = prompt.replace("<|image|>", "").strip()

        # Get system prompt from YAML configuration
        system_prompts = getattr(self.config, "yaml_config", {}).get(
            "system_prompts", {}
        )
        system_prompt = system_prompts.get("llama", "You are a helpful assistant.")

        # Use official HuggingFace chat template format with configurable system prompt
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": clean_prompt}],
            },
        ]

        # Apply chat template with generation prompt
        try:
            input_text = self.processor.apply_chat_template(
                messages, add_generation_prompt=True
            )

            # Process inputs using the formatted text
            inputs = self.processor(text=input_text, images=image, return_tensors="pt")

            logger.info("✅ Chat template applied successfully - using official format")

        except Exception as e:
            logger.warning(
                f"⚠️ Chat template failed, falling back to manual format: {e}"
            )
            # Fallback to original manual approach if chat template fails
            if not prompt.startswith("<|image|>"):
                prompt_with_image = f"<|image|>{prompt}"
            else:
                prompt_with_image = prompt

            inputs = self.processor(
                text=prompt_with_image, images=image, return_tensors="pt"
            )

        logger.debug(
            f"Input shapes - IDs: {inputs['input_ids'].shape}, Pixels: {inputs['pixel_values'].shape}"
        )

        # Move to correct device
        if self.device.type != "cpu":
            device_target = (
                str(self.device).split(":")[0]
                if ":" in str(self.device)
                else str(self.device)
            )
            inputs = {
                k: v.to(device_target) if hasattr(v, "to") else v
                for k, v in inputs.items()
            }

        return inputs

    def clean_response(self, response: str, image_name: str = "") -> str:
        """Clean response by removing repetition and special tokens.
        
        Args:
            response: Raw model response
            image_name: Name of the processed image (for logging)
            
        Returns:
            Cleaned response text
        """
        if not response:
            return ""
            
        cleaned = response
        
        # Remove special tokens
        if self.repetition_enabled:
            for token in self.cleanup_tokens:
                cleaned = cleaned.replace(token, "")
        
            # Remove consecutive duplicate lines
            lines = cleaned.split("\n")
            unique_lines = []
            prev_line = None
            
            for line in lines:
                line_cleaned = line.strip()
                if line_cleaned and line_cleaned != prev_line:
                    unique_lines.append(line)
                    prev_line = line_cleaned
                    
            cleaned = "\n".join(unique_lines)
            
            # Remove excessive whitespace
            cleaned = re.sub(r"\s+", " ", cleaned)
            
            # Truncate if too long
            if len(cleaned) > self.max_new_tokens_limit * 5:  # Rough char estimate
                cleaned = cleaned[:self.max_new_tokens_limit * 5] + "..."
        else:
            # Basic cleaning only
            cleaned = re.sub(r"\s+", " ", cleaned)
            if len(cleaned) > 1000:
                cleaned = cleaned[:1000] + "..."
                
        return cleaned.strip()

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
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        start_time = time.time()

        try:
            # Extract image name for logging
            if isinstance(image_path, (str, Path)):
                image_name = Path(image_path).name
            else:
                image_name = "unknown_image"

            # Preprocess image
            image = self.preprocess_image(image_path)

            # Prepare inputs
            inputs = self.prepare_inputs(image, prompt)

            # Configure generation parameters
            max_tokens = kwargs.get("max_new_tokens", 1024)
            if self.repetition_enabled:
                max_tokens = min(max_tokens, self.max_new_tokens_limit)

            generation_kwargs = {
                **inputs,
                "max_new_tokens": max_tokens,
                "do_sample": False,  # Deterministic generation
                "temperature": None,  # Explicitly disable to suppress warnings
                "top_p": None,  # Explicitly disable to suppress warnings
                "top_k": None,  # Explicitly disable to suppress warnings
                "pad_token_id": self.processor.tokenizer.eos_token_id,
                "eos_token_id": self.processor.tokenizer.eos_token_id,
                "use_cache": True,
            }

            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(**generation_kwargs)

            # Decode response - extract only the new tokens
            response = self.processor.decode(
                outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
            )

            # Clean response
            response = self.clean_response(response, image_name)

            processing_time = time.time() - start_time
            logger.info(f"Inference completed in {processing_time:.2f}s")

            # Get confidence score from YAML config
            model_config = getattr(self.config, "yaml_config", {}).get(
                "model_config", {}
            )
            confidence_score = model_config.get("llama", {}).get(
                "confidence_score", 0.85
            )

            return ModelResponse(
                raw_text=response.strip(),
                confidence=confidence_score,
                processing_time=processing_time,
                device_used=str(self.device),
                memory_usage=self.memory_manager.get_memory_usage(self.device),
                model_type="llama32_vision",
                quantized=self.quantization_manager.enable_quantization,
                metadata={
                    "graceful_degradation": True,
                    "processing_pipeline": "7step",
                    "max_image_size": "1024x1024",
                },
            )

        except Exception as e:
            logger.error(f"Error during Llama inference: {e}")
            return ModelResponse(
                raw_text=f"Error: {str(e)}",
                confidence=0.0,
                processing_time=time.time() - start_time,
                device_used=str(self.device),
                memory_usage=0.0,
                model_type="llama32_vision",
                quantized=False,
                metadata={"error": str(e)},
            )

    def process_batch(
        self,
        image_paths: List[Union[str, Path, Image.Image]],
        prompts: List[str],
        **kwargs,
    ) -> List[ModelResponse]:
        """Process multiple images in batch.
        
        Args:
            image_paths: List of image paths or PIL Images
            prompts: List of text prompts
            **kwargs: Additional parameters
            
        Returns:
            List of ModelResponse objects
        """
        results = []

        for image_path, prompt in zip(image_paths, prompts, strict=False):
            try:
                result = self.process_image(image_path, prompt, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process image: {e}")
                # Create error response
                error_response = ModelResponse(
                    raw_text=f"Error: {e!s}",
                    confidence=0.0,
                    processing_time=0.0,
                    device_used=str(self.device) if self.device else "unknown",
                    memory_usage=0.0,
                    model_type="llama32_vision",
                    quantized=False,
                    metadata={"error": str(e)},
                )
                results.append(error_response)

        return results