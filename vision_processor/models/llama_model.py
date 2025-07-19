"""Llama-3.2-Vision Model Implementation

Implements Llama-3.2-Vision with graceful degradation, 7-step pipeline integration,
and production optimizations for single GPU deployment.
"""

import gc
import logging
import time
import warnings
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    MllamaForConditionalGeneration,
)

from ..utils.repetition_control import UltraAggressiveRepetitionController
from .base_model import BaseVisionModel, DeviceConfig, ModelCapabilities, ModelResponse
from .model_utils import DeviceManager

logger = logging.getLogger(__name__)


class LlamaVisionModel(BaseVisionModel):
    """Llama-3.2-Vision implementation with production optimizations.

    Features:
    - Single GPU optimization with 8-bit quantization
    - Graceful degradation capabilities
    - 7-step pipeline integration
    - Australian tax document processing
    - Production-ready inference
    - Ultra-aggressive repetition control for Llama-3.2-Vision bugs
    """

    def __init__(self, *args, **kwargs):
        """Initialize LlamaVisionModel with repetition control."""
        super().__init__(*args, **kwargs)

        # Initialize ultra-aggressive repetition controller
        # Extract configuration from kwargs
        repetition_config = kwargs.get("repetition_control", {})
        word_threshold = repetition_config.get("word_threshold", 0.15)
        phrase_threshold = repetition_config.get("phrase_threshold", 2)

        self.repetition_controller = UltraAggressiveRepetitionController(
            word_threshold=word_threshold, phrase_threshold=phrase_threshold
        )

        # Store repetition control settings
        self.repetition_enabled = repetition_config.get("enabled", True)
        self.max_new_tokens_limit = repetition_config.get("max_new_tokens_limit", 384)

        logger.info(
            f"UltraAggressiveRepetitionController initialized - "
            f"word_threshold={word_threshold}, phrase_threshold={phrase_threshold}"
        )

    def _get_capabilities(self) -> ModelCapabilities:
        """Return Llama-3.2-Vision capabilities."""
        return ModelCapabilities(
            supports_multi_gpu=True,
            supports_quantization=True,
            supports_highlight_detection=False,  # Not available in Llama
            supports_batch_processing=True,
            max_image_size=(1024, 1024),  # Llama has lower limits
            memory_efficient=True,
            cross_platform=True,
        )

    @property
    def cross_platform_compatible(self) -> bool:
        """Check if model is cross-platform compatible."""
        return self.capabilities.cross_platform

    def _setup_device(self) -> torch.device:
        """Setup device configuration for Llama-3.2-Vision."""
        device_manager = DeviceManager(self.memory_limit_mb)

        # Use device manager to select optimal device
        if self.device_config == DeviceConfig.AUTO:
            device = device_manager.select_device(DeviceConfig.AUTO)
        else:
            device = device_manager.select_device(self.device_config)

        # Store device manager for later use
        self.device_manager = device_manager

        # Enable TF32 for GPU optimization on compatible hardware
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("TF32 enabled for GPU optimization")

        return device

    def _get_quantization_config(self) -> BitsAndBytesConfig | None:
        """Get quantization configuration if enabled."""
        if not self.enable_quantization:
            return None

        try:
            # Use int8 quantization for production V100 deployment
            return BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True,
                llm_int8_skip_modules=["vision_tower", "multi_modal_projector"],
                llm_int8_threshold=6.0,
            )
        except ImportError:
            logger.warning("BitsAndBytesConfig not available - falling back to FP16")
            return None

    def _detect_device_mapping(self) -> str | dict[str, int]:
        """Detect optimal device mapping based on hardware."""
        if self.device.type == "cuda":
            if torch.cuda.device_count() > 1:
                return "balanced"  # Distribute across multiple GPUs
            return "cuda:0"  # Single GPU
        if self.device.type == "mps":
            return "mps"
        return "cpu"

    def load_model(self) -> None:
        """Load Llama-3.2-Vision model with auto-configuration."""
        if self.is_loaded:
            logger.warning("Model already loaded")
            return

        logger.info(f"Loading Llama-3.2-Vision model from {self.model_path}")

        # Suppress warnings
        warnings.filterwarnings("ignore", message=".*pad_token_id.*")

        # Clean memory before loading
        self._cleanup_memory()

        # Get device configuration
        device_map = self._detect_device_mapping()
        quantization_config = self._get_quantization_config()

        # Configure loading parameters
        model_loading_args = {
            "low_cpu_mem_usage": True,
            "torch_dtype": torch.float16,
        }

        # Add offline mode configuration
        if hasattr(self, "kwargs") and self.kwargs.get("offline_mode", True):
            model_loading_args["local_files_only"] = True
            logger.info("Loading model in offline mode")

        # Configure based on device
        if self.device.type == "cpu":
            model_loading_args["device_map"] = None
            logger.info("Loading model on CPU (will be slow)...")
        elif torch.cuda.device_count() == 1:
            model_loading_args["device_map"] = device_map
            if quantization_config:
                model_loading_args["quantization_config"] = quantization_config
                logger.info("Loading model on single GPU with quantization...")
            else:
                logger.info("Loading model on single GPU...")
        else:  # Multi-GPU
            model_loading_args["device_map"] = device_map
            logger.info(f"Loading model across {torch.cuda.device_count()} GPUs...")

        try:
            # Load processor first
            processor_config = {}
            if model_loading_args.get("local_files_only"):
                processor_config["local_files_only"] = True

            self.processor = AutoProcessor.from_pretrained(
                str(self.model_path),
                trust_remote_code=True,
                **processor_config,
            )
            logger.info("Processor loaded successfully")

            # Load model
            self.model = MllamaForConditionalGeneration.from_pretrained(
                str(self.model_path),
                **model_loading_args,
            ).eval()

            # Move to device if needed (CPU only)
            if self.device.type == "cpu":
                self.model = self.model.to("cpu")

            self.is_loaded = True
            logger.info(f"Model loaded successfully on {self.device}")

            # Configure generation settings for stable inference (based on working implementation)
            self.model.generation_config.max_new_tokens = 1024  # Use max_new_tokens instead of max_length
            self.model.generation_config.do_sample = False  # Deterministic for consistency
            
            # Completely remove sampling parameters to avoid warnings when do_sample=False
            # These parameters are not needed for deterministic generation
            try:
                self.model.generation_config.temperature = 1.0  # Reset to default, will be ignored
                self.model.generation_config.top_p = 1.0  # Reset to default, will be ignored
                self.model.generation_config.top_k = 50  # Reset to default, will be ignored
                
                # The key is to ensure do_sample=False overrides these in generation calls
                logger.info("Generation config updated for deterministic inference")
            except Exception as e:
                logger.warning(f"Could not update generation config: {e}")
            self.model.config.use_cache = True  # Enable KV cache

            # Store metadata for inference
            self.model._llama_config = {
                "device_map": device_map,
                "quantization_enabled": quantization_config is not None,
                "offline_mode": self.kwargs.get("offline_mode", True),
            }

            # Test basic functionality
            self._test_model_functionality()

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self._cleanup_memory()
            raise RuntimeError(f"Failed to load Llama-3.2-Vision model: {e}") from e

    def unload_model(self) -> None:
        """Unload model from memory."""
        if not self.is_loaded:
            return

        self.model = None
        self.processor = None

        # Clean memory
        self._cleanup_memory()

        self.is_loaded = False
        logger.info("Model unloaded")

    def _cleanup_memory(self) -> None:
        """Clean up GPU and system memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()

    def _test_model_functionality(self) -> None:
        """Test basic model functionality."""
        try:
            logger.info("Testing model functionality...")

            # Simple text-only test
            test_prompt = "Hello, how are you?"
            test_inputs = self.processor.tokenizer(test_prompt, return_tensors="pt")

            # Move to appropriate device
            if hasattr(self.model, "device") and self.model.device.type != "cpu":
                test_inputs = {k: v.to(self.model.device) for k, v in test_inputs.items()}

            with torch.no_grad():
                test_outputs = self.model.generate(
                    **test_inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    temperature=None,  # Explicitly disable to avoid warnings
                    top_p=None,  # Explicitly disable to avoid warnings
                    top_k=None,  # Explicitly disable to avoid warnings
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                )

            test_response = self.processor.decode(
                test_outputs[0][test_inputs["input_ids"].shape[-1] :],
                skip_special_tokens=True,
            )

            logger.info(f"Model test successful: '{test_response[:50]}...'")

        except Exception as e:
            logger.warning(f"Model functionality test failed: {e}")

    def _preprocess_image(
        self,
        image_path: str | Path | Image.Image,
    ) -> Image.Image:
        """Preprocess image for Llama-3.2-Vision compatibility."""
        # Load image
        if isinstance(image_path, (str, Path)):
            image = Image.open(image_path)
        else:
            image = image_path

        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Resize if too large (Llama has size limits)
        max_size = 1024
        if max(image.size) > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            logger.info(f"Image resized to {image.size} (max: {max_size})")

        return image

    def _prepare_inputs(self, image: Image.Image, prompt: str) -> dict[str, Any]:
        """Prepare inputs for model inference.

        Args:
            image: Preprocessed PIL Image
            prompt: Text prompt (should include <|image|> token)

        Returns:
            Processed inputs dictionary
        """
        # Ensure prompt includes image token
        if not prompt.startswith("<|image|>"):
            prompt_with_image = f"<|image|>{prompt}"
        else:
            prompt_with_image = prompt

        # Process inputs
        inputs = self.processor(text=prompt_with_image, images=image, return_tensors="pt")

        logger.debug(
            f"Input shapes - IDs: {inputs['input_ids'].shape}, Pixels: {inputs['pixel_values'].shape}"
        )

        # Move to correct device - use the exact device detection from working code
        if self.device.type != "cpu":
            device_target = str(self.device).split(":")[0] if ":" in str(self.device) else str(self.device)
            inputs = {k: v.to(device_target) if hasattr(v, "to") else v for k, v in inputs.items()}

        return inputs

    def _clean_response(self, response: str) -> str:
        """Clean response using ultra-aggressive repetition control."""
        if not self.repetition_enabled:
            # Fallback to basic cleaning if repetition control is disabled
            import re

            response = re.sub(r"\s+", " ", response)
            if len(response) > 1000:
                response = response[:1000] + "..."
            return response.strip()

        # Use the ultra-aggressive repetition controller
        return self.repetition_controller.clean_response(response)

    def process_image(
        self,
        image_path: str | Path | Image.Image,
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
            self.load_model()

        start_time = time.time()

        try:
            # Preprocess image
            image = self._preprocess_image(image_path)

            # Prepare inputs
            inputs = self._prepare_inputs(image, prompt)

            # Generate with deterministic parameters to bypass safety mode
            # Use working implementation settings for OCR extraction
            # Apply ultra-aggressive token limit to prevent repetition
            max_tokens = kwargs.get("max_new_tokens", 1024)
            if self.repetition_enabled:
                max_tokens = min(max_tokens, self.max_new_tokens_limit)

            generation_kwargs = {
                **inputs,
                "max_new_tokens": max_tokens,
                "do_sample": False,  # Deterministic generation bypasses safety checks
                "temperature": None,  # Explicitly disable to avoid warnings
                "top_p": None,  # Explicitly disable to avoid warnings
                "top_k": None,  # Explicitly disable to avoid warnings
                "pad_token_id": self.processor.tokenizer.eos_token_id,
                "eos_token_id": self.processor.tokenizer.eos_token_id,
                "use_cache": True,
            }

            with torch.no_grad():
                outputs = self.model.generate(**generation_kwargs)

            # Decode response - extract only the new tokens
            response = self.processor.decode(
                outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
            )

            # Clean response from repetitive text and common artifacts
            response = self._clean_response(response)

            processing_time = time.time() - start_time
            logger.info(f"Inference completed in {processing_time:.2f}s")

            return ModelResponse(
                raw_text=response.strip(),
                confidence=0.85,  # Llama doesn't provide confidence scores
                processing_time=processing_time,
                device_used=str(self.device),
                memory_usage=self.get_memory_usage(),
                model_type="llama32_vision",
                quantized=self.enable_quantization,
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
        image_paths: list[str | Path | Image.Image],
        prompts: list[str],
        **kwargs,
    ) -> list[ModelResponse]:
        """Process multiple images in batch."""
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
                    device_used=str(self.device),
                    memory_usage=0.0,
                    model_type="llama32_vision",
                    quantized=False,
                    metadata={"error": str(e)},
                )
                results.append(error_response)

        return results

    def classify_document(
        self,
        image_path: str | Path | Image.Image,
    ) -> dict[str, Any]:
        """Classify document type using Llama-3.2-Vision.

        Args:
            image_path: Path to image file or PIL Image

        Returns:
            Classification result dictionary

        """
        # Use the improved classification prompt from working implementation
        classification_prompt = """<|image|>Analyze document structure and format. Classify based on layout patterns:

- fuel_receipt: Contains fuel quantities (L, litres), price per unit
- tax_invoice: Formal invoice layout, tax calculations
- receipt: Product lists, subtotals, retail format
- bank_statement: Account numbers, transaction records
- unknown: Cannot determine format

Output document type only."""

        try:
            # Use direct prediction method like working implementation
            response = self.process_image(image_path, classification_prompt, do_sample=False)
            response_lower = response.raw_text.lower()

            # Parse classification response with improved fuel detection
            # First check OCR content for fuel indicators (override classification if needed)
            response_text = response.raw_text.lower()

            # Look for fuel indicators in the actual OCR text
            fuel_indicators = [
                "13ulp",
                "ulp",
                "unleaded",
                "diesel",
                "litre",
                " l ",
                ".l ",
                "price/l",
                "per litre",
                "fuel",
            ]
            has_fuel_content = any(indicator in response_text for indicator in fuel_indicators)

            # Look for quantity patterns that indicate fuel
            import re

            fuel_quantity_pattern = r"\d+\.\d{2,3}\s*l\b|\d+\s*litre"
            has_fuel_quantity = bool(re.search(fuel_quantity_pattern, response_text))

            # Look for bank statement indicators in the actual OCR text
            bank_indicators = [
                "account",
                "balance",
                "transaction",
                "deposit",
                "withdrawal",
                "bsb",
                "opening balance",
                "closing balance",
                "statement period",
                "account number",
                "sort code",
                "debit",
                "credit",
                "available balance",
                "current balance",
            ]
            has_bank_content = any(indicator in response_text for indicator in bank_indicators)

            # Look for account number patterns (Australian BSB + Account format)
            bank_account_pattern = r"\d{3}-\d{3}\s+\d{4,10}|\bBSB\b|\baccount\s+number\b"
            has_bank_account = bool(re.search(bank_account_pattern, response_text, re.IGNORECASE))

            if "fuel_receipt" in response_lower or "fuel receipt" in response_lower:
                doc_type = "fuel_receipt"
                confidence = 0.90
            elif has_fuel_content or has_fuel_quantity:
                # Override other classifications if we see clear fuel indicators
                doc_type = "fuel_receipt"
                confidence = 0.95
                logger.info("Overriding classification to fuel_receipt based on content indicators")
            elif "fuel" in response_lower or "petrol" in response_lower:
                doc_type = "fuel_receipt"
                confidence = 0.85
            elif "tax_invoice" in response_lower or "tax invoice" in response_lower:
                doc_type = "tax_invoice"
                confidence = 0.85
            elif "tax" in response_lower and "invoice" in response_lower:
                doc_type = "tax_invoice"
                confidence = 0.80
            elif "bank_statement" in response_lower or "bank statement" in response_lower:
                doc_type = "bank_statement"
                confidence = 0.90
            elif has_bank_content or has_bank_account:
                # Override other classifications if we see clear bank indicators
                doc_type = "bank_statement"
                confidence = 0.95
                logger.info("Overriding classification to bank_statement based on content indicators")
            elif "bank" in response_lower:
                doc_type = "bank_statement"
                confidence = 0.75
            elif "receipt" in response_lower:
                doc_type = "receipt"
                confidence = 0.75
            elif "invoice" in response_lower:
                doc_type = "tax_invoice"  # Default invoices to tax_invoice
                confidence = 0.70
            else:
                doc_type = "unknown"
                confidence = 0.50

            return {
                "document_type": doc_type,
                "confidence": confidence,
                "classification_response": response.raw_text,
                "is_business_document": doc_type
                in [
                    "receipt",
                    "tax_invoice",
                    "fuel_receipt",
                    "bank_statement",
                    "invoice",
                ]
                and confidence > 0.7,
            }

        except Exception as e:
            logger.error(f"Document classification failed: {e}")
            return {
                "document_type": "unknown",
                "confidence": 0.0,
                "classification_response": f"Error: {str(e)}",
                "is_business_document": False,
            }

    def predict(self, image_path: str | Path, prompt: str) -> str:
        """Generate prediction with CUDA-safe parameters.

        Args:
            image_path: Path to image file or HTTP URL
            prompt: Text prompt for extraction

        Returns:
            Generated response text
        """
        try:
            start_time = time.time()

            # Preprocess image
            image = self._preprocess_image(image_path)

            # Prepare inputs
            inputs = self._prepare_inputs(image, prompt)

            # Generate with CUDA-safe parameters (NO repetition_penalty)
            # Use optimized settings for receipt extraction with ultra-aggressive repetition control
            max_tokens = 1024
            if self.repetition_enabled:
                max_tokens = min(max_tokens, self.max_new_tokens_limit)

            generation_kwargs = {
                **inputs,
                "max_new_tokens": max_tokens,
                "do_sample": False,
                "temperature": None,  # Explicitly disable to avoid warnings
                "top_p": None,  # Explicitly disable to avoid warnings
                "top_k": None,  # Explicitly disable to avoid warnings
                "pad_token_id": self.processor.tokenizer.eos_token_id,
                "eos_token_id": self.processor.tokenizer.eos_token_id,
                "use_cache": True,
            }

            with torch.no_grad():
                outputs = self.model.generate(**generation_kwargs)

            # Decode response - extract only the new tokens
            response = self.processor.decode(
                outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
            )

            # Clean response from repetitive text and common artifacts
            response = self._clean_response(response)

            inference_time = time.time() - start_time
            logger.info(f"Inference completed in {inference_time:.2f}s")

            return response.strip()

        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return f"Error: {str(e)}"

    def _apply_quantization(self) -> None:
        """Apply quantization to model (handled during loading)."""
        if not self.capabilities.supports_quantization:
            return

        logger.info("Quantization applied during model loading")
