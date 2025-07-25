"""Simplified single-step extraction manager."""

import time
from pathlib import Path
from typing import Any, Dict, Union

import torch
from rich.console import Console

from ..config.simple_config import SimpleConfig
from ..exceptions import (
    ImageProcessingError,
    ModelInferenceError,
    ModelLoadError,
)

console = Console()


class ExtractionResult:
    """Result from extraction process."""

    def __init__(
        self,
        extracted_fields: Dict[str, Any],
        processing_time: float,
        model_confidence: float = 0.0,
        extraction_method: str = "single_step",
    ):
        self.extracted_fields = extracted_fields
        self.processing_time = processing_time
        self.model_confidence = model_confidence
        self.extraction_method = extraction_method


class SimpleExtractionManager:
    """Simplified single-step extraction manager."""

    def __init__(self, config: SimpleConfig):
        """Initialize extraction manager.

        Args:
            config: SimpleConfig instance with settings.
        """
        self.config = config

        # Print configuration before model loading
        self.config.print_configuration()

        # Initialize model with detailed logging
        self.model = self._load_model_with_logging()
        # Use expected_fields from model_comparison.yaml only
        self.expected_fields = self.config.get_expected_fields()

    def _load_model_with_logging(self):
        """Load model with detailed configuration logging."""

        print(f"\n🚀 Loading {self.config.current_model_type} model...")
        print(f"📍 Model Path: {self.config.get_model_path(self.config.current_model_type)}")

        # Detect system capabilities
        gpu_count = torch.cuda.device_count()
        if gpu_count > 0:
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"🔌 GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            # Check for MPS (Mac M1/M2)
            if torch.backends.mps.is_available():
                print("💻 Using MPS (Apple Silicon GPU)")
            else:
                print("💻 Using CPU (no CUDA GPUs detected)")

        # Print memory settings
        print("🧠 Memory Configuration:")
        print(f"  - Memory Limit: {self.config.processing.memory_limit_mb}MB")
        print("  - GPU Memory Fraction: 0.9")  # Legacy value
        print(
            f"  - Quantization: {'Enabled' if self.config.processing.quantization else 'Disabled'}"
        )
        print(
            f"  - Multi-GPU: {'Enabled' if self.config.is_multi_gpu_enabled() else 'Disabled'}"
        )

        # Print processing optimizations
        print("⚡ Processing Optimizations:")
        print(
            f"  - Gradient Checkpointing: {'Enabled' if self.config.processing.enable_gradient_checkpointing else 'Disabled'}"
        )
        print(
            f"  - Flash Attention: {'Enabled' if self.config.processing.use_flash_attention else 'Disabled'}"
        )

        # Load the model
        start_time = time.time()
        try:
            # Use same model creation pattern as ComparisonRunner
            # Create model registry with our config manager if it's a ConfigManager
            from ..config import ConfigManager
            from ..config.model_registry import get_model_registry
            config_manager = self.config if isinstance(self.config, ConfigManager) else None
            model_registry = get_model_registry(config_manager)

            # Get model path from ConfigManager
            model_type = self.config.current_model_type
            model_path = self.config.get_model_path(model_type)

            model = model_registry.create_model(
                model_type,
                model_path=model_path,
                config=self.config,
            )
            load_time = time.time() - start_time

            # Print successful loading
            print(f"✅ Model loaded successfully in {load_time:.2f} seconds")

            # Print actual device usage
            if hasattr(model, "device"):
                print(f"🎯 Model running on: {model.device}")

            # Print memory usage after loading
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                cached = torch.cuda.memory_reserved() / 1024**3
                print(
                    f"📊 GPU Memory Usage: {allocated:.2f}GB allocated, {cached:.2f}GB cached"
                )

            return model

        except ImportError as e:
            raise ModelLoadError(
                model_name=self.config.current_model_type,
                original_error=e,
                suggestion="Check if all model dependencies are installed"
            ) from e
        except Exception as e:
            raise ModelLoadError(
                model_name=self.config.current_model_type,
                original_error=e,
                model_path=str(model_path)
            ) from e

    def process_document(self, image_path: Union[str, Path]) -> ExtractionResult:
        """Single-step document processing.

        Args:
            image_path: Path to the document image.

        Returns:
            ExtractionResult with extracted data.
        """
        start_time = time.time()

        # Convert to Path if string
        image_path = Path(image_path)

        print(f"\n🔍 Processing document: {image_path.name}")

        # Step 1: Get model-specific prompt
        prompt = self._get_model_prompt()

        # Step 2: Process with model
        print(f"📸 Sending to {self.config.current_model_type} model...")
        response = self.model.process_image(str(image_path), prompt)

        # Step 3: Parse KEY-VALUE response
        print("📝 Parsing response...")
        print("🔍 DEBUG - Raw model response (first 500 chars):")
        print(f"'{response.raw_text[:500]}...'")

        # Use the same parsing logic as comparison runner
        extracted_data = self._parse_clean_response(response.raw_text)

        # Step 4: Validate against schema
        print("✓ Validating extracted data...")
        validated_data = self._validate_against_schema(extracted_data)

        processing_time = time.time() - start_time
        print(f"⏱️  Processing completed in {processing_time:.2f} seconds")

        return ExtractionResult(
            extracted_fields=validated_data,
            processing_time=processing_time,
            model_confidence=response.confidence
            if hasattr(response, "confidence")
            else 0.9,
            extraction_method="single_step",
        )

    def _parse_clean_response(self, raw_text: str) -> dict:
        """Parse clean KEY: value format response (same logic as comparison runner).

        Args:
            raw_text: Raw response text from model

        Returns:
            Dictionary of extracted key-value pairs
        """
        extracted_fields = {}
        raw_text = raw_text.strip()

        # Get expected fields for validation
        expected_fields = self.config.get_expected_fields()

        # Split on spaces and look for KEY: value patterns
        parts = raw_text.split()
        i = 0
        while i < len(parts) - 1:
            part = parts[i]
            if part.endswith(":"):
                key = part[:-1]  # Remove the colon
                # Only process if it's an expected field
                if key in expected_fields:
                    # Collect value(s) until next key or end
                    value_parts = []
                    j = i + 1
                    while j < len(parts) and not parts[j].endswith(":"):
                        value_parts.append(parts[j])
                        j += 1
                    value = " ".join(value_parts) if value_parts else "N/A"
                    extracted_fields[key] = value
                    i = j
                else:
                    i += 1
            else:
                i += 1

        return extracted_fields

    def _get_model_prompt(self) -> str:
        """Get model-specific prompt from model_comparison.yaml."""
        prompts = self.config.get_prompts()
        model_prompt = prompts.get(self.config.current_model_type, "")

        # Generate field list from expected fields
        expected_fields = self.config.get_expected_fields()
        if expected_fields and model_prompt:
            # Add the field list to the prompt
            field_list = "\n".join(
                [f"{field}: [value or N/A]" for field in expected_fields]
            )
            # Replace any placeholder or append field list
            if "[FIELD_LIST]" in model_prompt:
                model_prompt = model_prompt.replace("[FIELD_LIST]", field_list)
            elif not any(field in model_prompt for field in expected_fields):
                # If prompt doesn't contain fields, append them
                model_prompt = model_prompt.rstrip() + "\n\n" + field_list

        return model_prompt

    def _validate_against_schema(self, extracted_data: dict) -> dict:
        """Validate extracted data against expected fields.

        Args:
            extracted_data: Raw extracted data.

        Returns:
            Validated data dictionary.
        """
        validated = {}

        # Only include fields that are in expected_fields
        for key in self.expected_fields:
            if key in extracted_data:
                validated[key] = extracted_data[key]

        return validated

    def process_batch(
        self, image_paths: list[Union[str, Path]]
    ) -> list[ExtractionResult]:
        """Process multiple documents.

        Args:
            image_paths: List of image paths to process.

        Returns:
            List of ExtractionResult objects.
        """
        results = []
        total = len(image_paths)

        print(f"\n📦 Processing batch of {total} documents...")

        for idx, image_path in enumerate(image_paths, 1):
            print(f"\n[{idx}/{total}] Processing {Path(image_path).name}")
            try:
                result = self.process_document(image_path)
                results.append(result)
            except (ModelInferenceError, ImageProcessingError) as e:
                print(f"❌ Error processing {image_path}: {e.message}")
                # Create error result with specific error info
                results.append(
                    ExtractionResult(
                        extracted_fields={"error": e.message, "error_type": type(e).__name__},
                        processing_time=0.0,
                        model_confidence=0.0,
                        extraction_method="error",
                    )
                )
            except Exception as e:
                print(f"❌ Unexpected error processing {image_path}: {str(e)}")
                # Create error result for unexpected errors
                results.append(
                    ExtractionResult(
                        extracted_fields={"error": str(e), "error_type": "UnexpectedError"},
                        processing_time=0.0,
                        model_confidence=0.0,
                        extraction_method="error",
                    )
                )

        print(f"\n✅ Batch processing complete: {len(results)} documents processed")
        return results
