"""Simplified single-step extraction manager."""

import time
from pathlib import Path
from typing import Any, Dict, Union

import torch
import yaml
from rich.console import Console

from ..config.model_factory import ModelFactory
from ..config.simple_config import SimpleConfig
from .universal_key_value_parser import UniversalKeyValueParser

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
        self.key_schema = self._load_key_schema()
        self.universal_prompt = self._load_universal_prompt()
        self.parser = UniversalKeyValueParser(self.key_schema)

    def _load_model_with_logging(self):
        """Load model with detailed configuration logging."""

        print(f"\n🚀 Loading {self.config.model_type} model...")
        print(f"📍 Model Path: {self.config.model_path}")

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
        print(f"  - Memory Limit: {self.config.memory_limit_mb}MB")
        print(f"  - GPU Memory Fraction: {self.config.gpu_memory_fraction}")
        print(f"  - Quantization: {'Enabled' if self.config.enable_quantization else 'Disabled'}")
        print(f"  - Multi-GPU: {'Enabled' if self.config.enable_multi_gpu else 'Disabled'}")

        # Print processing optimizations
        print("⚡ Processing Optimizations:")
        print(
            f"  - Gradient Checkpointing: {'Enabled' if self.config.enable_gradient_checkpointing else 'Disabled'}"
        )
        print(f"  - Flash Attention: {'Enabled' if self.config.use_flash_attention else 'Disabled'}")

        # Load the model
        start_time = time.time()
        try:
            model = ModelFactory.create_model(self.config)
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
                print(f"📊 GPU Memory Usage: {allocated:.2f}GB allocated, {cached:.2f}GB cached")

            return model

        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            raise

    def _load_key_schema(self) -> dict:
        """Load key schema from prompts.yaml."""
        prompts_path = Path(__file__).parent.parent / "config" / "prompts.yaml"
        with prompts_path.open("r") as f:
            prompts_data = yaml.safe_load(f)

        return prompts_data.get("key_schema", {})

    def _load_universal_prompt(self) -> str:
        """Load universal prompt from prompts.yaml."""
        prompts_path = Path(__file__).parent.parent / "config" / "prompts.yaml"
        with prompts_path.open("r") as f:
            prompts_data = yaml.safe_load(f)

        # Check for model-specific prompt first
        model_prompts = prompts_data.get("model_prompts", {})
        if self.config.model_type in model_prompts:
            return model_prompts[self.config.model_type]["prompt"]

        # Fall back to universal prompt
        return prompts_data.get("universal_extraction_prompt", "")

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
        print(f"📸 Sending to {self.config.model_type} model...")
        response = self.model.process_image(str(image_path), prompt)

        # Step 3: Parse KEY-VALUE response
        print("📝 Parsing response...")
        print(f"🔍 DEBUG - Raw model response (first 500 chars):")
        print(f"'{response.raw_text[:500]}...'")
        extracted_data = self.parser.parse(response.raw_text)

        # Step 4: Validate against schema
        print("✓ Validating extracted data...")
        validated_data = self._validate_against_schema(extracted_data)

        processing_time = time.time() - start_time
        print(f"⏱️  Processing completed in {processing_time:.2f} seconds")

        return ExtractionResult(
            extracted_fields=validated_data,
            processing_time=processing_time,
            model_confidence=response.confidence if hasattr(response, "confidence") else 0.9,
            extraction_method="single_step",
        )

    def _get_model_prompt(self) -> str:
        """Get model-specific prompt."""
        return self.universal_prompt

    def _validate_against_schema(self, extracted_data: dict) -> dict:
        """Validate extracted data against schema.

        Args:
            extracted_data: Raw extracted data.

        Returns:
            Validated data dictionary.
        """
        validated = {}

        # Check required keys
        missing_required = []
        for key in self.key_schema.get("required_keys", []):
            if key in extracted_data:
                validated[key] = extracted_data[key]
            else:
                missing_required.append(key)

        # Add optional keys if present
        for key in self.key_schema.get("optional_keys", []):
            if key in extracted_data:
                validated[key] = extracted_data[key]

        # Warn about missing required keys
        if missing_required:
            console.print(f"[yellow]⚠️  Missing required keys: {', '.join(missing_required)}[/yellow]")

        return validated

    def process_batch(self, image_paths: list[Union[str, Path]]) -> list[ExtractionResult]:
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
            except Exception as e:
                print(f"❌ Error processing {image_path}: {str(e)}")
                # Create error result
                results.append(
                    ExtractionResult(
                        extracted_fields={"error": str(e)},
                        processing_time=0.0,
                        model_confidence=0.0,
                        extraction_method="error",
                    )
                )

        print(f"\n✅ Batch processing complete: {len(results)} documents processed")
        return results
