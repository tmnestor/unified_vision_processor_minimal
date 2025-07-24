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

# We'll suppress warnings during model loading using context managers
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

        # Extract config from kwargs and store as direct attribute
        self.config = kwargs.get("config")

        # Initialize ultra-aggressive repetition controller
        # Extract configuration from YAML config first, then kwargs
        yaml_repetition_config = getattr(self.config, "yaml_config", {}).get("repetition_control", {})
        repetition_config = kwargs.get("repetition_control", yaml_repetition_config)
        word_threshold = repetition_config.get("word_threshold", 0.15)
        phrase_threshold = repetition_config.get("phrase_threshold", 2)

        # Store repetition control settings
        self.repetition_enabled = repetition_config.get("enabled", True)

        # Read max_new_tokens_limit from YAML config (single source of truth) FIRST
        yaml_limit = None
        if hasattr(self, "config") and self.config:
            model_config = getattr(self.config, "yaml_config", {}).get(
                "model_config", {}
            )
            yaml_limit = model_config.get("llama", {}).get("max_new_tokens_limit")

        # Use YAML config as single source of truth, fallback to repetition_config, then default
        fallback_tokens = yaml_repetition_config.get("fallback_max_tokens", 1000)
        self.max_new_tokens_limit = (
            yaml_limit
            or repetition_config.get("max_new_tokens_limit")
            or fallback_tokens
        )

        # Now create repetition controller with the correct token limit
        self.repetition_controller = UltraAggressiveRepetitionController(
            word_threshold=word_threshold,
            phrase_threshold=phrase_threshold,
            max_tokens_limit=self.max_new_tokens_limit,  # Pass the YAML config limit
        )

        logger.info(
            f"UltraAggressiveRepetitionController initialized - "
            f"word_threshold={word_threshold}, phrase_threshold={phrase_threshold}, "
            f"max_new_tokens_limit={self.max_new_tokens_limit}"
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

    def _get_quantization_config(self) -> BitsAndBytesConfig | None:
        """Get quantization configuration if enabled."""
        if not self.enable_quantization:
            return None

        try:
            # Check if aggressive quantization is requested for V100 safety
            use_4bit = getattr(self, "kwargs", {}).get("aggressive_quantization", False)

            if use_4bit:
                # Use 4-bit quantization for 11B model V100 deployment
                logger.info(
                    "🔧 Using 4-bit quantization for aggressive memory reduction"
                )
                return BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            else:
                # Use int8 quantization for standard V100 deployment
                logger.info("🔧 Using 8-bit quantization for V100 deployment")
                return BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=True,
                    llm_int8_skip_modules=["vision_tower", "multi_modal_projector"],
                    llm_int8_threshold=6.0,
                )
        except ImportError:
            logger.warning("BitsAndBytesConfig not available - falling back to FP16")
            return None

    def _get_device_map_from_config(self):
        """Get device map configuration from production config.

        FAILS EXPLICITLY if YAML config is not available - no silent fallbacks.
        """
        # FAIL FAST: Production config must be available
        if not hasattr(self, "config"):
            raise RuntimeError(
                "❌ FATAL: No production config found for Llama model\n"
                "💡 Expected: config parameter passed during model creation\n"
                "💡 Fix: Ensure config is passed via model_registry.create_model()\n"
                "💡 YAML file: model_comparison.yaml with device_config section"
            )

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

    def _detect_device_mapping(self) -> str | dict[str, int]:
        """Detect optimal device mapping based on hardware."""
        if self.device.type == "cuda":
            if torch.cuda.device_count() > 1:
                return "balanced"  # Distribute across multiple GPUs
            return "cuda:0"  # Single GPU
        if self.device.type == "mps":
            return "mps"
        return "cpu"

    def _estimate_memory_usage(
        self, quantization_config: BitsAndBytesConfig | None
    ) -> float:
        """Estimate model memory usage in GB for V100 validation."""
        # Base model size: Llama-3.2-11B ≈ 11B parameters
        base_params = 11_000_000_000  # 11B parameters

        if quantization_config:
            if (
                hasattr(quantization_config, "load_in_4bit")
                and quantization_config.load_in_4bit
            ):
                # 4-bit quantization: ~0.5 bytes per parameter
                memory_gb = (base_params * 0.5) / (1024**3)
                overhead = 1.5  # Quantization overhead
                return memory_gb * overhead
            elif (
                hasattr(quantization_config, "load_in_8bit")
                and quantization_config.load_in_8bit
            ):
                # 8-bit quantization: ~1 byte per parameter
                memory_gb = (base_params * 1.0) / (1024**3)
                overhead = 1.3  # Quantization overhead
                return memory_gb * overhead

        # FP16: 2 bytes per parameter
        memory_gb = (base_params * 2.0) / (1024**3)
        overhead = 1.2  # Model overhead
        return memory_gb * overhead

    def _validate_v100_compliance(self, estimated_memory_gb: float) -> None:
        """Validate that estimated memory usage complies with V100 limits."""
        # Get memory config from YAML (single source of truth)
        memory_config = getattr(self.config, "yaml_config", {}).get("memory_config", {})
        v100_limit_gb = memory_config.get("v100_limit_gb", 16.0)
        safety_margin = memory_config.get("safety_margin", 0.85)
        effective_limit = v100_limit_gb * safety_margin

        if estimated_memory_gb > effective_limit:
            logger.error("❌ V100 COMPLIANCE FAILURE:")
            logger.error(f"   Estimated memory: {estimated_memory_gb:.1f}GB")
            logger.error(f"   V100 safe limit: {effective_limit:.1f}GB (85% of 16GB)")
            logger.error(f"   Excess: {estimated_memory_gb - effective_limit:.1f}GB")
            logger.error("")
            logger.error("💡 SOLUTIONS:")
            if not self.enable_quantization:
                logger.error("   1. Enable quantization (set quantization=true)")
            else:
                use_4bit = getattr(self, "kwargs", {}).get(
                    "aggressive_quantization", False
                )
                if not use_4bit:
                    logger.error("   1. Enable aggressive 4-bit quantization")
                    logger.error(
                        "   2. Use smaller model variant (e.g., 7B instead of 11B)"
                    )
                else:
                    logger.error(
                        "   1. Use smaller model variant (e.g., 7B instead of 11B)"
                    )

            raise RuntimeError(
                f"Model memory ({estimated_memory_gb:.1f}GB) exceeds V100 safe limit ({effective_limit:.1f}GB)"
            )
        else:
            safety_percent = (
                (effective_limit - estimated_memory_gb) / effective_limit * 100
            )
            logger.info(
                f"✅ V100 Compliance: {estimated_memory_gb:.1f}GB / {v100_limit_gb:.1f}GB ({safety_percent:.1f}% safety margin)"
            )

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

        # Get device configuration from YAML config (FAIL FAST - no fallbacks)
        device_map = self._get_device_map_from_config()
        quantization_config = self._get_quantization_config()

        logger.info(f"🔧 V100 Mode: Using device configuration: {device_map}")

        # V100 compliance validation
        estimated_memory = self._estimate_memory_usage(quantization_config)
        self._validate_v100_compliance(estimated_memory)

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
        elif self.num_gpus == 1:
            model_loading_args["device_map"] = device_map

            # Add quantization configuration if enabled
            if quantization_config:
                model_loading_args["quantization_config"] = quantization_config
                use_4bit = getattr(self, "kwargs", {}).get(
                    "aggressive_quantization", False
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

            # Load model with suppressed generation config warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
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
            # Note: Set sampling parameters to None to suppress warnings when do_sample=False
            self.model.generation_config.max_new_tokens = (
                self.max_new_tokens_limit  # Use YAML config limit instead of hardcoded
            )
            self.model.generation_config.do_sample = (
                False  # Deterministic for consistency
            )

            # Set sampling parameters to None to suppress warnings (working script approach)
            self.model.generation_config.temperature = (
                None  # Set to None to suppress warnings
            )
            self.model.generation_config.top_p = (
                None  # Set to None to suppress warnings
            )
            self.model.generation_config.top_k = (
                None  # Set to None to suppress warnings
            )

            self.model.config.use_cache = True  # Enable KV cache
            logger.info("Configured generation settings for deterministic inference")

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

        # Resize if too large using YAML config
        image_config = getattr(self.config, "yaml_config", {}).get("image_processing", {})
        max_size = image_config.get("max_image_size", 1024)
        
        if max(image.size) > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            logger.info(f"Image resized to {image.size} (max: {max_size})")

        return image

    def _prepare_inputs(self, image: Image.Image, prompt: str) -> dict[str, Any]:
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
        system_prompts = getattr(self.config, "yaml_config", {}).get("system_prompts", {})
        system_prompt = system_prompts.get("llama", "You are a helpful assistant.")
        
        # Use official HuggingFace chat template format with configurable system prompt
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": clean_prompt}
                ]
            }
        ]
        
        # Apply chat template with generation prompt
        try:
            input_text = self.processor.apply_chat_template(
                messages, 
                add_generation_prompt=True
            )
            
            # Process inputs using the formatted text
            inputs = self.processor(
                text=input_text, 
                images=image, 
                return_tensors="pt"
            )
            
            logger.info("✅ Chat template applied successfully - using official format")
            
        except Exception as e:
            logger.warning(f"⚠️ Chat template failed, falling back to manual format: {e}")
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

        # Move to correct device - use the exact device detection from working code
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

    def _clean_response(self, response: str, image_name: str = "") -> str:
        """Clean response using ultra-aggressive repetition control."""
        if not self.repetition_enabled:
            # Fallback to basic cleaning if repetition control is disabled
            import re

            response = re.sub(r"\s+", " ", response)
            if len(response) > 1000:
                response = response[:1000] + "..."
            return response.strip()

        # Use the ultra-aggressive repetition controller
        return self.repetition_controller.clean_response(response, image_name)

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
            # Extract image name for repetition controller
            if isinstance(image_path, (str, Path)):
                image_name = Path(image_path).name
            else:
                image_name = "unknown_image"

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

            # Only log generation parameters in debug mode
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"Generation parameters: max_new_tokens={kwargs.get('max_new_tokens', 1024)}, max_new_tokens_limit={self.max_new_tokens_limit}, final_max_tokens={max_tokens}"
                )

            generation_kwargs = {
                **inputs,
                "max_new_tokens": max_tokens,
                "do_sample": False,  # Deterministic generation bypasses safety checks
                "temperature": None,  # Explicitly disable to suppress warnings
                "top_p": None,  # Explicitly disable to suppress warnings
                "top_k": None,  # Explicitly disable to suppress warnings
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
            response = self._clean_response(response, image_name)

            processing_time = time.time() - start_time
            logger.info(f"Inference completed in {processing_time:.2f}s")

            # Get confidence score from YAML config
            model_config = getattr(self.config, "yaml_config", {}).get("model_config", {})
            confidence_score = model_config.get("llama", {}).get("confidence_score", 0.85)
            
            return ModelResponse(
                raw_text=response.strip(),
                confidence=confidence_score,
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
            response = self.process_image(
                image_path, classification_prompt, do_sample=False
            )
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
            has_fuel_content = any(
                indicator in response_text for indicator in fuel_indicators
            )

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
            has_bank_content = any(
                indicator in response_text for indicator in bank_indicators
            )

            # Look for account number patterns (Australian BSB + Account format)
            bank_account_pattern = (
                r"\d{3}-\d{3}\s+\d{4,10}|\bBSB\b|\baccount\s+number\b"
            )
            has_bank_account = bool(
                re.search(bank_account_pattern, response_text, re.IGNORECASE)
            )

            if "fuel_receipt" in response_lower or "fuel receipt" in response_lower:
                doc_type = "fuel_receipt"
                confidence = 0.90
            elif has_fuel_content or has_fuel_quantity:
                # Override other classifications if we see clear fuel indicators
                doc_type = "fuel_receipt"
                confidence = 0.95
                logger.info(
                    "Overriding classification to fuel_receipt based on content indicators"
                )
            elif "fuel" in response_lower or "petrol" in response_lower:
                doc_type = "fuel_receipt"
                confidence = 0.85
            elif "tax_invoice" in response_lower or "tax invoice" in response_lower:
                doc_type = "tax_invoice"
                confidence = 0.85
            elif "tax" in response_lower and "invoice" in response_lower:
                doc_type = "tax_invoice"
                confidence = 0.80
            elif (
                "bank_statement" in response_lower or "bank statement" in response_lower
            ):
                doc_type = "bank_statement"
                confidence = 0.90
            elif has_bank_content or has_bank_account:
                # Override other classifications if we see clear bank indicators
                doc_type = "bank_statement"
                confidence = 0.95
                logger.info(
                    "Overriding classification to bank_statement based on content indicators"
                )
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

            # Extract image name for repetition controller
            if isinstance(image_path, (str, Path)):
                image_name = Path(image_path).name
            else:
                image_name = "unknown_image"

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
                "temperature": None,  # Explicitly disable to suppress warnings
                "top_p": None,  # Explicitly disable to suppress warnings
                "top_k": None,  # Explicitly disable to suppress warnings
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
            response = self._clean_response(response, image_name)

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
