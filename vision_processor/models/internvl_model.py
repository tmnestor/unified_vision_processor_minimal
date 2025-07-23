"""InternVL Model Implementation

Implements InternVL3 with multi-GPU auto-configuration, quantization support,
and advanced features like highlight detection integration.
"""

import logging
import math
import time
import warnings
from pathlib import Path
from typing import Any

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

from .base_model import BaseVisionModel, DeviceConfig, ModelCapabilities, ModelResponse
from .model_utils import DeviceManager

logger = logging.getLogger(__name__)

# Constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class InternVLModel(BaseVisionModel):
    """InternVL3 implementation with multi-GPU support and advanced features.

    Supports InternVL Version 3 models only from https://huggingface.co/OpenGVLab/InternVL
    - InternVL3-1B (0.9B parameters)
    - InternVL3-2B (2B parameters)
    - InternVL3-8B (8B parameters - primary model)
    - InternVL3-9B (9B parameters)
    - InternVL3-14B (15B parameters)
    - InternVL3-38B (38B parameters)
    - InternVL3-78B (78B parameters)

    Features:
    - Multi-GPU auto-configuration
    - 8-bit quantization for single GPU
    - Dynamic image tiling for better detail preservation
    - Highlight detection support
    - Cross-platform compatibility
    """

    def __init__(self, *args, **kwargs):
        """Initialize InternVLModel with production config support."""
        super().__init__(*args, **kwargs)

        # Extract config from kwargs and store as direct attribute
        self.config = kwargs.get("config")

    def _get_capabilities(self) -> ModelCapabilities:
        """Return InternVL capabilities."""
        return ModelCapabilities(
            supports_multi_gpu=True,
            supports_quantization=True,
            supports_highlight_detection=True,
            supports_batch_processing=True,
            max_image_size=(3840, 2160),  # 4K support
            memory_efficient=True,
            cross_platform=True,
        )

    def _setup_device(self) -> torch.device:
        """Setup device configuration for InternVL."""
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
            # This ensures fair comparison with Llama model that's V100-optimized
            available_gpus = torch.cuda.device_count()
            self.num_gpus = 1  # Force single GPU regardless of available GPUs
            logger.info(
                f"🔧 V100 Production Mode: Using 1 GPU (detected {available_gpus} available)"
            )
        else:
            self.num_gpus = 0

        return device

    def _get_device_map_from_config(self):
        """Get device map configuration from production config.

        FAILS EXPLICITLY if YAML config is not available - no silent fallbacks.
        """
        # FAIL FAST: Production config must be available
        if not hasattr(self, "config"):
            raise RuntimeError(
                "❌ FATAL: No production config found for InternVL model\n"
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
                "       internvl:\n"
                "         device_map: {'': 0}"
            )

        device_config = self.config.device_config
        device_map = device_config.get_device_map_for_model("internvl")

        if device_map is None:
            raise RuntimeError(
                f"❌ FATAL: No device mapping found for internvl model\n"
                f"💡 Expected: internvl entry in device_config.device_maps\n"
                f"💡 Current device_maps: {list(device_config.device_maps.keys())}\n"
                f"💡 Fix: Add internvl device mapping to model_comparison.yaml:\n"
                f"   device_config:\n"
                f"     device_maps:\n"
                f"       internvl:\n"
                f"         strategy: 'single_gpu'\n"
                f"         device_map: {{'': 0}}\n"
                f"         quantization_compatible: true"
            )

        return device_map

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

    def _split_model(self, model_name: str) -> dict[str, int]:
        """Create device mapping for multi-GPU configuration.
        Based on the model architecture, distributes layers across available GPUs.
        """
        device_map = {}
        world_size = self.num_gpus

        # Model layer counts for InternVL Version 3 models only
        # Reference: https://huggingface.co/collections/OpenGVLab/internvl3-67f7f690be79c2fe9d74fe9d
        num_layers_mapping = {
            "InternVL3-1B": 24,  # 0.9B parameters
            "InternVL3-2B": 24,  # 2B parameters
            "InternVL3-8B": 28,  # 8B parameters (primary)
            "InternVL3-9B": 32,  # 9B parameters
            "InternVL3-14B": 40,  # 15B parameters
            "InternVL3-38B": 60,  # 38B parameters
            "InternVL3-78B": 80,  # 78B parameters
        }

        # Extract model size from path
        model_size = None
        for size_key in num_layers_mapping:
            if (
                size_key in str(model_name)
                or size_key.replace("-", "_").lower() in str(model_name).lower()
            ):
                model_size = size_key
                break

        if model_size is None:
            # Default to InternVL3-8B (the primary InternVL3 model)
            logger.warning(
                f"Could not determine InternVL3 model size from {model_name}, defaulting to InternVL3-8B",
            )
            logger.info(
                "Supported InternVL3 models: InternVL3-1B, InternVL3-2B, InternVL3-8B, InternVL3-9B, InternVL3-14B, InternVL3-38B, InternVL3-78B"
            )
            model_size = "InternVL3-8B"

        num_layers = num_layers_mapping[model_size]

        # Since the first GPU will be used for ViT, treat it as half a GPU
        num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
        num_layers_per_gpu = [num_layers_per_gpu] * world_size
        num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)

        layer_cnt = 0
        for i, num_layer in enumerate(num_layers_per_gpu):
            for _j in range(num_layer):
                device_map[f"language_model.model.layers.{layer_cnt}"] = i
                layer_cnt += 1

        # Vision and core components on GPU 0
        device_map["vision_model"] = 0
        device_map["mlp1"] = 0
        device_map["language_model.model.tok_embeddings"] = 0
        device_map["language_model.model.embed_tokens"] = 0
        device_map["language_model.output"] = 0
        device_map["language_model.model.norm"] = 0
        device_map["language_model.lm_head"] = 0
        device_map[f"language_model.model.layers.{num_layers - 1}"] = 0

        return device_map

    def load_model(self) -> None:
        """Load InternVL model with auto-configuration."""
        if self.is_loaded:
            logger.warning("Model already loaded")
            return

        logger.info(f"Loading InternVL model from {self.model_path}")

        # Suppress tokenizer warnings
        warnings.filterwarnings("ignore", message=".*pad_token_id.*")
        warnings.filterwarnings("ignore", message=".*Setting `pad_token_id`.*")

        # Configure loading parameters
        model_loading_args = {
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        }

        # Add offline mode configuration
        if hasattr(self, "kwargs") and self.kwargs.get("offline_mode", True):
            model_loading_args["local_files_only"] = True
            logger.info("Loading model in offline mode")

        # Configure based on device
        if self.device.type == "cpu":
            model_loading_args["torch_dtype"] = torch.float32
            logger.info("Loading model on CPU (will be slow)...")
        elif self.num_gpus == 1:
            model_loading_args["torch_dtype"] = torch.float16

            # Use device configuration from YAML config (FAIL FAST - no fallbacks)
            device_map = self._get_device_map_from_config()
            model_loading_args["device_map"] = device_map
            logger.info(f"🔧 V100 Mode: Using YAML device configuration: {device_map}")

            # Add quantization configuration if enabled
            quantization_config = self._get_quantization_config()
            if quantization_config:
                model_loading_args["quantization_config"] = quantization_config
                logger.info(
                    "🔧 V100 Mode: Loading model on single GPU with 8-bit quantization..."
                )
            else:
                logger.info("🔧 V100 Mode: Loading model on single GPU...")
        elif hasattr(self, "kwargs") and self.kwargs.get("force_multi_gpu", False):
            logger.info(
                f"Multi-GPU mode requested, distributing across {self.num_gpus} GPUs",
            )
            model_loading_args["torch_dtype"] = torch.float16
            device_map = self._split_model(self.model_path)
            model_loading_args["device_map"] = device_map
            logger.info(f"Device mapping: {device_map}")
        else:
            logger.warning(
                f"Multi-GPU detected ({self.num_gpus} GPUs), but using single GPU mode (safer)",
            )
            model_loading_args["torch_dtype"] = torch.float16
            # Force single GPU mode instead of device mapping
            model_loading_args["device_map"] = {"": 0}  # Put everything on GPU 0
            logger.info(
                "Loading model on single GPU (GPU 0) with device_map override...",
            )

        # Add flash attention if requested and available
        if hasattr(self, "kwargs") and self.kwargs.get("use_flash_attention", False):
            try:
                # Only enable if flash_attn is available
                import flash_attn  # noqa: F401

                model_loading_args["use_flash_attn"] = True
                logger.info("🔧 Flash Attention enabled")
            except ImportError:
                logger.info(
                    "🔧 Flash Attention not available - using standard attention"
                )

        try:
            # Load tokenizer
            tokenizer_config = {"trust_remote_code": True}
            if model_loading_args.get("local_files_only"):
                tokenizer_config["local_files_only"] = True

            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path),
                **tokenizer_config,
            )
            logger.info("Tokenizer loaded successfully")

            # Load model
            self.model = AutoModel.from_pretrained(
                str(self.model_path),
                **model_loading_args,
            ).eval()

            # Move to device if needed (single GPU only) - but not for quantized models
            if (
                self.device.type == "cuda"
                and "device_map" not in model_loading_args
                and "quantization_config" not in model_loading_args
            ):
                self.model = self.model.cuda()
                logger.info("Model moved to CUDA device")
            elif "quantization_config" in model_loading_args:
                logger.info("Quantized model - device placement handled automatically")

            self.is_loaded = True
            logger.info(f"Model loaded successfully on {self.device}")

            # Optimize for inference
            self.optimize_for_inference()

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Failed to load InternVL model: {e}") from e

    def unload_model(self) -> None:
        """Unload model from memory."""
        if not self.is_loaded:
            return

        self.model = None
        self.tokenizer = None
        self.processor = None

        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.is_loaded = False
        logger.info("Model unloaded")

    def _build_transform(self, input_size: int) -> T.Compose:
        """Build image transformation pipeline."""
        transform = T.Compose(
            [
                T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                T.Resize(
                    (input_size, input_size),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ],
        )
        return transform

    def _dynamic_preprocess(
        self,
        image: Image.Image,
        min_num: int = 1,
        max_num: int = 12,
        image_size: int = 448,
        use_thumbnail: bool = False,
    ) -> list[Image.Image]:
        """Process images with dynamic tiling based on aspect ratio."""
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # Calculate target ratios
        target_ratios = {
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_num and i * j >= min_num
        }
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # Find best aspect ratio
        best_ratio = self._find_closest_aspect_ratio(
            aspect_ratio,
            target_ratios,
            orig_width,
            orig_height,
            image_size,
        )

        target_width = image_size * best_ratio[0]
        target_height = image_size * best_ratio[1]
        blocks = best_ratio[0] * best_ratio[1]

        # Resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []

        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size,
            )
            split_img = resized_img.crop(box)
            processed_images.append(split_img)

        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)

        return processed_images

    def _find_closest_aspect_ratio(
        self,
        aspect_ratio: float,
        target_ratios: list[tuple[int, int]],
        width: int,
        height: int,
        image_size: int,
    ) -> tuple[int, int]:
        """Find the closest valid aspect ratio for image tiling."""
        best_ratio_diff = float("inf")
        best_ratio = (1, 1)
        area = width * height

        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio

        return best_ratio

    def process_image(
        self,
        image_path: str | Path | Image.Image,
        prompt: str,
        **kwargs,
    ) -> ModelResponse:
        """Process image with InternVL model.

        Args:
            image_path: Path to image file or PIL Image
            prompt: Text prompt for processing
            **kwargs: Additional parameters (image_size, max_tiles, etc.)

        Returns:
            ModelResponse with standardized format

        """
        if not self.is_loaded:
            self.load_model()

        start_time = time.time()

        try:
            # Load image
            if isinstance(image_path, (str, Path)):
                image = Image.open(image_path)
            else:
                image = image_path

            # Get processing parameters
            image_size = kwargs.get("image_size", 448)

            # For InternVL3, try simpler single-image approach first
            if len(image.size) == 2 or min(image.size) < 224:
                # Resize small images
                image = image.resize((448, 448), Image.Resampling.LANCZOS)

            # Build transform
            transform = self._build_transform(image_size)

            # Process as single image first (simpler approach)
            pixel_values = transform(image).unsqueeze(0)  # Shape: [1, 3, 448, 448]

            # Critical: Move to appropriate device and dtype BEFORE inference
            # This ensures pixel_values matches the model's device and dtype
            if self.device.type == "cuda":
                pixel_values = pixel_values.cuda().to(torch.float16)
            else:
                pixel_values = pixel_values.to(torch.float32)

            logger.info(f"Single image pixel_values shape: {pixel_values.shape}")
            logger.info(f"Pixel values device: {pixel_values.device}")

            # Log model device info for debugging
            if hasattr(self.model, "device"):
                logger.info(f"Model device: {self.model.device}")
            elif hasattr(self.model, "vision_model"):
                logger.info(
                    f"Vision model device: {next(self.model.vision_model.parameters()).device}",
                )

            logger.info("Processed image as single tile")

            # Run inference - fix generation config for deterministic output
            generation_config = {
                "max_new_tokens": kwargs.get("max_new_tokens", 1024),
                "do_sample": False,
                "pad_token_id": self.tokenizer.eos_token_id,
            }

            try:
                # InternVL chat interface - ensure all inputs are on the same device
                response = self.model.chat(
                    tokenizer=self.tokenizer,
                    pixel_values=pixel_values,
                    question=prompt,
                    generation_config=generation_config,
                )
            except Exception as e:
                logger.error(f"Error during InternVL inference: {e}")
                logger.info("Attempting alternative inference method...")

                # Try alternative interface with explicit device handling
                input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
                if self.device.type == "cuda":
                    input_ids = (
                        input_ids.cuda()
                    )  # Use .cuda() to match pixel_values device

                try:
                    with torch.no_grad():
                        outputs = self.model.generate(
                            input_ids=input_ids,
                            pixel_values=pixel_values,
                            **generation_config,
                        )
                    response = self.tokenizer.decode(
                        outputs[0],
                        skip_special_tokens=True,
                    )
                    logger.info("Alternative inference successful")
                except Exception as e2:
                    logger.error(f"Alternative inference also failed: {e2}")
                    raise RuntimeError(f"InternVL inference failed: {e}") from e

            # Handle response format
            raw_text = response[0] if isinstance(response, tuple) else response

            processing_time = time.time() - start_time

            return ModelResponse(
                raw_text=raw_text,
                confidence=0.95,  # InternVL doesn't provide confidence scores
                processing_time=processing_time,
                device_used=str(self.device),
                memory_usage=self.get_memory_usage(),
                model_type="internvl3",
                quantized=self.enable_quantization and self.num_gpus == 1,
                metadata={
                    "tiles_created": 1,  # Single image processing
                    "image_size": image_size,
                    "multi_gpu": self.num_gpus > 1,
                },
            )

        except Exception as e:
            logger.error(f"Error during InternVL inference: {e}")
            raise RuntimeError(f"InternVL inference failed: {e}") from e

    def process_batch(
        self,
        image_paths: list[str | Path | Image.Image],
        prompts: list[str],
        **kwargs,
    ) -> list[ModelResponse]:
        """Process multiple images in batch."""
        # InternVL doesn't support true batch processing
        # Process sequentially but reuse loaded model
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
                    model_type="internvl3",
                    quantized=False,
                    metadata={"error": str(e)},
                )
                results.append(error_response)

        return results

    def classify_document(
        self,
        image_path: str | Path | Image.Image,
    ) -> dict[str, Any]:
        """Classify document type using InternVL model.

        Args:
            image_path: Path to image file or PIL Image

        Returns:
            Classification result dictionary

        """
        classification_prompt = """Analyze document structure and format. Classify based on layout patterns:

- fuel_receipt: Contains fuel quantities (L, litres), price per unit
- tax_invoice: Formal invoice layout, tax calculations
- receipt: Product lists, subtotals, retail format
- bank_statement: Account numbers, transaction records
- unknown: Cannot determine format

Output document type only."""

        try:
            response = self.process_image(image_path, classification_prompt)
            response_text = response.raw_text.lower()

            # Parse classification response
            confidence = 0.75  # Default confidence

            if "fuel_receipt" in response_text or "fuel receipt" in response_text:
                doc_type = "fuel_receipt"
                confidence = 0.90
            elif "tax_invoice" in response_text or "tax invoice" in response_text:
                doc_type = "tax_invoice"
                confidence = 0.85
            elif "bank_statement" in response_text or "bank statement" in response_text:
                doc_type = "bank_statement"
                confidence = 0.85
            elif "receipt" in response_text:
                doc_type = "receipt"
                confidence = 0.75
            elif "invoice" in response_text:
                doc_type = "tax_invoice"
                confidence = 0.70
            else:
                doc_type = "unknown"
                confidence = 0.50

            return {
                "document_type": doc_type,
                "confidence": confidence,
                "classification_response": response.raw_text,
                "is_business_document": doc_type
                in ["receipt", "tax_invoice", "fuel_receipt", "bank_statement"]
                and confidence > 0.7,
            }

        except Exception as e:
            logger.error(f"Document classification failed: {e}")
            return {
                "document_type": "unknown",
                "confidence": 0.0,
                "classification_response": f"Error: {e!s}",
                "is_business_document": False,
            }

    def _apply_quantization(self) -> None:
        """Apply 8-bit quantization to model."""
        if not self.capabilities.supports_quantization:
            return

        try:
            import importlib.util

            if importlib.util.find_spec("bitsandbytes"):
                logger.info("Quantization will be applied during model loading")
            else:
                logger.warning("bitsandbytes not available for quantization")
        except ImportError:
            logger.warning("bitsandbytes not available for quantization")
