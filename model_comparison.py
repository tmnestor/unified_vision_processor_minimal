#!/usr/bin/env python3
"""
Unified Vision Model Comparison Script
=====================================

Converts model_comparison.ipynb to standalone script for V100 production environment.
Compares Llama 3.2-11B-Vision vs InternVL3-8B for Australian business document extraction.

KFP (Kubeflow Pipelines) Usage:
    python model_comparison.py compare --datasets-path /mnt/datasets --output-dir /mnt/output
    python model_comparison.py compare --datasets-path /data/images --models llama --quantization
    python model_comparison.py check-environment --datasets-path /mnt/input

Local Usage:
    python model_comparison.py compare --datasets-path ./datasets --output-dir ./results
    python model_comparison.py compare --datasets-path ~/data --models internvl --max-tokens 128

Features:
- ✅ KFP-ready: Configurable input/output paths for ephemeral environments
- ✅ Sequential model loading (optimized for 16GB V100)
- ✅ Comprehensive analytics with sklearn, pandas, seaborn
- ✅ Memory monitoring and GPU management
- ✅ Saves visualizations as PNG files (no interactive dependencies)
- ✅ Exports detailed results as JSON/CSV
- ✅ Australian business document focus (ABN, DD/MM/YYYY dates)
"""

import gc
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import typer
import yaml
from PIL import Image
from rich.console import Console
from rich.progress import track
from rich.table import Table
from sklearn.metrics import f1_score, precision_score, recall_score

# Configure matplotlib for headless environment
plt.switch_backend("Agg")  # Non-interactive backend for V100
console = Console()

# Suppress pandas future warnings about downcasting
pd.set_option("future.no_silent_downcasting", True)

# =============================================================================
# EXTRACTION CONFIGURATION SYSTEM
# =============================================================================


class ExtractionConfigLoader:
    """Loads and manages simplified extraction configuration from YAML files"""

    def __init__(self, config_path: str = "model_comparison.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with self.config_path.open("r") as f:
            return yaml.safe_load(f)

    def get_field_names_from_response(self, response: str) -> List[str]:
        """Dynamically extract field names from response"""
        import re

        # Find all "FIELD:" patterns in the response
        field_pattern = r"([A-Z_]+):\s*"
        matches = re.findall(field_pattern, response)
        return list(set(matches))  # Remove duplicates

    def get_success_criteria(self) -> Dict[str, int]:
        """Get success criteria configuration"""
        return {
            "min_fields_for_success": self.config["min_fields_for_success"],
        }

    def get_expected_abn_images(self) -> List[str]:
        """Get list of images expected to have ABN (for evaluation)"""
        return []  # Simplified - no evaluation config

    def generate_extraction_prompt(self) -> str:
        """Generate extraction prompt for InternVL from simplified configuration"""
        return f"<|image|>{self.config['prompts']['internvl']}"

    def generate_llama_safe_prompt(self) -> str:
        """Generate ultra-safe Llama prompt that bypasses safety mode"""
        return f"<|image|>{self.config['prompts']['llama']}"

    def get_field_config(self, field_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific field"""
        for field in self.get_all_fields():
            if field["name"] == field_name:
                return field
        return None

    def is_core_field(self, field_name: str) -> bool:
        """Check if a field is a core field"""
        return field_name in self.get_core_field_names()

    def get_validation_type(self, field_name: str) -> Optional[str]:
        """Get validation type for a field"""
        field_config = self.get_field_config(field_name)
        return field_config.get("validation_type") if field_config else None

    def get_fallback_patterns(self, field_name: str) -> List[str]:
        """Get fallback regex patterns for a field"""
        field_config = self.get_field_config(field_name)
        return field_config.get("fallback_patterns", []) if field_config else []

    def get_invalid_values(self, field_name: str) -> List[str]:
        """Get list of invalid values for a field (like N/A, etc.)"""
        field_config = self.get_field_config(field_name)
        return field_config.get("invalid_values", []) if field_config else []

    def get_validation_rules(self, field_name: str) -> List[str]:
        """Get validation rules for a field"""
        field_config = self.get_field_config(field_name)
        return field_config.get("validation_rules", []) if field_config else []


class ConfigurableFieldValidator:
    """Validates extracted field values based on configuration"""

    def __init__(self, config_loader: ExtractionConfigLoader):
        self.config_loader = config_loader

    def validate_field(self, field_name: str, field_value: str) -> bool:
        """Validate a field value based on its configuration"""
        if not field_value:
            return False

        field_config = self.config_loader.get_field_config(field_name)
        if not field_config:
            return False

        validation_type = field_config.get("validation_type")

        if validation_type == "australian_abn":
            return self._validate_australian_abn(field_name, field_value)
        elif validation_type == "australian_date":
            return self._validate_australian_date(field_value)
        elif validation_type == "australian_currency":
            return self._validate_australian_currency(field_value)
        elif validation_type == "text":
            return self._validate_text(field_name, field_value)
        elif validation_type == "text_list":
            return self._validate_text_list(field_name, field_value)
        else:
            # Default validation - just check it's not empty or N/A
            return self._validate_not_empty_or_na(field_name, field_value)

    def _validate_australian_abn(self, field_name: str, abn_value: str) -> bool:
        """Validate Australian Business Number"""
        abn_clean = abn_value.strip().upper()

        # Check for invalid values
        invalid_values = self.config_loader.get_invalid_values(field_name)
        if abn_clean in invalid_values:
            return False

        # Check if it's actually a valid 11-digit ABN pattern
        digits_only = re.sub(r"[^\d]", "", abn_clean)
        if len(digits_only) != 11:
            return False

        # Additional validation rules
        validation_rules = self.config_loader.get_validation_rules(field_name)
        if "no_all_zeros" in validation_rules and digits_only == "00000000000":
            return False
        if "no_repeating_pattern" in validation_rules and len(set(digits_only)) == 1:
            return False

        return True

    def _validate_australian_date(self, date_value: str) -> bool:
        """Validate Australian date format (DD/MM/YYYY)"""
        date_clean = date_value.strip()
        # Basic pattern check for Australian date formats
        pattern = r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}\.\d{1,2}\.\d{2,4})\b"
        return bool(re.search(pattern, date_clean))

    def _validate_australian_currency(self, currency_value: str) -> bool:
        """Validate Australian currency format"""
        currency_clean = currency_value.strip()
        # Basic pattern check for Australian currency
        pattern = r"(\$\d+\.\d{2}|\$\d+|AUD\s*\d+\.\d{2})"
        return bool(re.search(pattern, currency_clean))

    def _validate_text(self, field_name: str, text_value: str) -> bool:
        """Validate text field"""
        return self._validate_not_empty_or_na(field_name, text_value)

    def _validate_text_list(self, field_name: str, list_value: str) -> bool:
        """Validate text list field (items separated by |)"""
        return self._validate_not_empty_or_na(field_name, list_value)

    def _validate_not_empty_or_na(self, field_name: str, value: str) -> bool:
        """Basic validation - not empty and not N/A"""
        value_clean = value.strip().upper()

        # Check for invalid values
        invalid_values = self.config_loader.get_invalid_values(field_name)
        if not invalid_values:
            # Default invalid values if none specified
            invalid_values = ["N/A", "NA", "NOT AVAILABLE", "NOT FOUND", "NONE", "-"]

        return value_clean not in invalid_values and len(value_clean) > 0


# =============================================================================
# CONFIGURATION
# =============================================================================


def load_extraction_config(config_path: str = "model_comparison.yaml") -> Dict[str, Any]:
    """Load extraction configuration from YAML file - FAIL FAST if config missing"""
    try:
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with config_file.open("r") as f:
            config = yaml.safe_load(f)

        # Validate required sections exist
        if "prompts" not in config:
            raise ValueError(f"Missing 'prompts' section in {config_path}")

        if "internvl" not in config["prompts"]:
            raise ValueError(f"Missing 'internvl' prompt in {config_path}")

        if "llama" not in config["prompts"]:
            raise ValueError(f"Missing 'llama' prompt in {config_path}")

        # Extract prompts directly with explicit model names
        internvl_prompt = f"<|image|>{config['prompts']['internvl']}"
        llama_prompt = f"<|image|>{config['prompts']['llama']}"

        # Create config_loader for dynamic field handling
        config_loader = ExtractionConfigLoader(config_path)

        model_paths_config = config.get(
            "model_paths",
            {
                "llama": "/home/jovyan/nfs_share/models/Llama-3.2-11B-Vision",
                "internvl": "/home/jovyan/nfs_share/models/InternVL3-8B",
            },
        )

        console.print(f"✅ Model comparison configuration loaded from: {config_path}", style="green")
        console.print(f"   InternVL prompt: {len(internvl_prompt)} characters", style="dim")
        console.print(f"   Llama prompt: {len(llama_prompt)} characters", style="dim")
        defaults = config.get("defaults", {})
        console.print(f"   Max tokens: {defaults.get('max_tokens', 256)}", style="dim")
        console.print(f"   Llama path: {model_paths_config.get('llama', 'default')}", style="dim")
        console.print(f"   InternVL path: {model_paths_config.get('internvl', 'default')}", style="dim")

        return {
            "model_paths": model_paths_config,
            "internvl_prompt": internvl_prompt,
            "llama_prompt": llama_prompt,
            "test_images": [],  # Will be populated dynamically from datasets directory
            "config": config,
            "config_loader": config_loader,
        }
    except FileNotFoundError:
        console.print(f"❌ FATAL: Configuration file not found: {config_path}", style="bold red")
        console.print(f"💡 Expected location: {Path(config_path).absolute()}", style="yellow")
        console.print("💡 Create the YAML file with 'prompts' section", style="yellow")
        raise typer.Exit(1) from None
    except Exception as e:
        console.print(f"❌ FATAL: Failed to load configuration: {e}", style="bold red")
        console.print(f"💡 Configuration file: {config_path}", style="yellow")
        console.print("💡 Check YAML syntax and required sections", style="yellow")
        raise typer.Exit(1) from None


# Load default configuration lazily (will be loaded in CLI)
DEFAULT_CONFIG = None

# =============================================================================
# UTILITY CLASSES (From Notebook Cells 1-2)
# =============================================================================


class MemoryManager:
    """Memory management and monitoring utilities for V100 production"""

    @staticmethod
    def cleanup_gpu_memory():
        """Aggressive memory cleanup for V100 16GB limit"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """Get current GPU memory usage in GB"""
        if torch.cuda.is_available():
            return {
                "allocated": torch.cuda.memory_allocated() / 1024**3,
                "reserved": torch.cuda.memory_reserved() / 1024**3,
                "free": (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved())
                / 1024**3,
            }
        return {"allocated": 0.0, "reserved": 0.0, "free": 0.0}

    @staticmethod
    def print_memory_usage(label: str = "Memory"):
        """Print formatted memory usage for V100 monitoring"""
        if torch.cuda.is_available():
            memory = MemoryManager.get_memory_usage()
            total_gpu = torch.cuda.get_device_properties(0).total_memory / 1024**3
            console.print(
                f"   💾 {label}: {memory['allocated']:.1f}GB allocated | {memory['reserved']:.1f}GB reserved | {memory['free']:.1f}GB free | {total_gpu:.1f}GB total"
            )
        else:
            console.print(f"   💾 {label}: No CUDA available")

    @staticmethod
    def get_memory_delta(before: Dict[str, float], after: Dict[str, float]) -> Dict[str, float]:
        """Calculate memory usage delta"""
        return {
            "allocated_delta": after["allocated"] - before["allocated"],
            "reserved_delta": after["reserved"] - before["reserved"],
        }


class UltraAggressiveRepetitionController:
    """Business document repetition detection and cleanup"""

    def __init__(self, word_threshold: float = 0.15, phrase_threshold: int = 2):
        self.word_threshold = word_threshold
        self.phrase_threshold = phrase_threshold

        self.toxic_patterns = [
            r"THANK YOU FOR SHOPPING WITH US[^.]*",
            r"All prices include GST where applicable[^.]*",
            r"applicable\.\s*applicable\.",
            r"GST where applicable[^.]*applicable",
            r"\\+[a-zA-Z]*\{[^}]*\}",
            r"\(\s*\)",
            r"[.-]\s*THANK YOU",
        ]

    def clean_response(self, response: str) -> str:
        """Clean business document extraction response"""
        if not response or len(response.strip()) == 0:
            return ""

        # First convert markdown to key-value format
        response = self._convert_markdown_to_keyvalue(response)

        response = self._remove_business_patterns(response)
        response = self._remove_word_repetition(response)
        response = self._remove_phrase_repetition(response)

        # Clean up multiple spaces but preserve newlines
        response = re.sub(r"[ \t]+", " ", response)  # Only replace spaces/tabs, not newlines
        response = re.sub(r"[.]{2,}", ".", response)
        response = re.sub(r"[!]{2,}", "!", response)

        return response.strip()

    def _convert_markdown_to_keyvalue(self, text: str) -> str:
        """Convert markdown formatting to clean KEY: VALUE pairs"""
        if not text or not text.strip():
            return text

        # Simple approach: convert all recognizable patterns to KEY: VALUE format
        converted_text = text

        # Remove markdown table pipes and separators
        converted_text = re.sub(r"\|", "", converted_text)
        converted_text = re.sub(r"^-+\s*$", "", converted_text, flags=re.MULTILINE)

        # Remove markdown bold formatting
        converted_text = re.sub(r"\*\*([^*]+)\*\*", r"\1", converted_text)

        # Convert bullet points to clean format
        converted_text = re.sub(
            r"^\*\s*([A-Za-z][A-Za-z\s]*?):\s*(.+)$", r"\1: \2", converted_text, flags=re.MULTILINE
        )

        # Normalize field names - convert to uppercase and replace spaces with underscores
        def normalize_field_name(match):
            field = match.group(1).strip().upper().replace(" ", "_")
            field = re.sub(r"[^A-Z_]", "", field)  # Remove non-alphanumeric chars except underscore
            value = match.group(2).strip()
            return f"{field}: {value}"

        converted_text = re.sub(
            r"^([A-Za-z][A-Za-z\s_]*?):\s*(.+)$", normalize_field_name, converted_text, flags=re.MULTILINE
        )

        # Clean up extra whitespace but preserve single line breaks
        converted_text = re.sub(r"\n\s*\n+", "\n", converted_text)
        converted_text = re.sub(r"  +", " ", converted_text)

        # Filter to only keep lines that look like KEY: VALUE pairs
        lines = converted_text.split("\n")
        keyvalue_lines = []

        for line in lines:
            line = line.strip()
            if (
                line
                and ":" in line
                and not line.startswith("Note:")
                and not line.startswith("NOTE:")
                and not line.startswith("#")
                and len(line.split(":", 1)) == 2
            ):
                field, value = line.split(":", 1)
                field = field.strip()
                value = value.strip()

                # Filter out explanatory text and non-data fields
                if (
                    field
                    and value
                    and field.isupper()
                    and field not in ["NOTE", "EXPLANATION", "OUTPUT", "FORMAT", "INSTRUCTION"]
                ):
                    keyvalue_lines.append(f"{field}: {value}")

        return "\n".join(keyvalue_lines)

    def _remove_business_patterns(self, text: str) -> str:
        """Remove business document specific repetitive patterns"""
        for pattern in self.toxic_patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)

        text = re.sub(r"(applicable\.\s*){2,}", "applicable. ", text, flags=re.IGNORECASE)
        return text

    def _remove_word_repetition(self, text: str) -> str:
        """Remove word repetition in business documents"""
        text = re.sub(r"\b(\w+)(\s+\1){1,}", r"\1", text, flags=re.IGNORECASE)
        return text

    def _remove_phrase_repetition(self, text: str) -> str:
        """Remove phrase repetition"""
        for phrase_length in range(2, 7):
            pattern = r"\b((?:\w+\s+){" + str(phrase_length - 1) + r"}\w+)(\s+\1){1,}"
            text = re.sub(pattern, r"\1", text, flags=re.IGNORECASE)

        return text


class ConfigurableKeyValueExtractionAnalyzer:
    """Analyzer for KEY-VALUE extraction results using configurable fields - FAIL FAST design"""

    def __init__(self, config_loader: ExtractionConfigLoader):
        if not config_loader:
            raise ValueError(
                "ConfigurableKeyValueExtractionAnalyzer requires a valid ExtractionConfigLoader"
            )

        self.config_loader = config_loader
        # Simplified - no complex field validator needed
        self.validator = None  # Explicitly set to None for simplified config

        # Validate configuration at initialization
        self._validate_config()

    def _validate_config(self):
        """Validate configuration at startup - FAIL FAST if invalid"""
        success_criteria = self.config_loader.get_success_criteria()
        if "min_fields_for_success" not in success_criteria:
            raise ValueError("Success criteria missing 'min_fields_for_success'")

    def analyze(self, response: str, img_name: str) -> Dict[str, Any]:
        """Analyze KEY-VALUE extraction results with configurable fields"""
        response_clean = response.strip()

        # Dynamically detect fields from response
        detected_fields = self.config_loader.get_field_names_from_response(response_clean)

        # Check if response is structured (contains any field patterns)
        is_structured = len(detected_fields) > 0

        # Extract and validate each detected field
        field_results = {}
        field_matches = {}

        for field_name in detected_fields:
            field_detected, field_match = self._extract_and_validate_field_simple(
                field_name, response_clean
            )

            field_results[f"has_{field_name.lower()}"] = field_detected
            field_matches[field_name.lower()] = field_match

        # Calculate scores - simplified
        all_scores = [field_results.get(key, False) for key in field_results.keys()]
        extraction_score = sum(all_scores)

        # Success criteria from config - simplified
        success_criteria = self.config_loader.get_success_criteria()
        min_fields = success_criteria.get("min_fields_for_success", 2)
        successful = extraction_score >= min_fields

        # Build result dictionary
        result = {
            "img_name": img_name,
            "response": response_clean,
            "is_structured": is_structured,
            "extraction_score": extraction_score,
            "successful": successful,
        }

        # Add individual field results
        result.update(field_results)

        return result

    def _extract_and_validate_field_simple(
        self, field_name: str, response: str
    ) -> Tuple[bool, Optional[str]]:
        """Extract and validate a specific field from the response - dynamic version"""
        # Try structured extraction first
        pattern = rf'(?:{field_name}|{field_name.lower()}):\s*"?([^"\n]+)"?'
        match = re.search(pattern, response, re.IGNORECASE)

        if match:
            field_value = match.group(1).strip()
            # Simplified validation - just check if value exists and isn't N/A
            if field_value and field_value.upper() not in [
                "N/A",
                "NA",
                "NOT AVAILABLE",
                "NOT FOUND",
                "NONE",
                "-",
                "UNKNOWN",
                "NULL",
                "EMPTY",
            ]:
                return True, field_value

        return False, None


# REMOVED: No backwards compatibility aliases - use explicit class names for clarity
# This follows the "Fail Fast with Diagnostics" principle


class DatasetManager:
    """Dataset verification and management"""

    def __init__(self, datasets_path: str = "datasets"):
        self.datasets_path = Path(datasets_path)

    def discover_all_images(self) -> List[Tuple[str, str]]:
        """Discover all PNG images in the datasets directory"""
        discovered_images = []

        if not self.datasets_path.exists():
            return discovered_images

        # Find all PNG files in the datasets directory
        png_files = list(self.datasets_path.glob("*.png"))

        for png_file in sorted(png_files):
            img_name = png_file.name
            # Classify document type based on filename patterns
            doc_type = self._classify_document_type(img_name)
            discovered_images.append((img_name, doc_type))

        return discovered_images

    def _classify_document_type(self, img_name: str) -> str:
        """Classify document type based on filename patterns"""
        img_lower = img_name.lower()

        if "invoice" in img_lower or "tax" in img_lower:
            return "TAX_INVOICE"
        elif "fuel" in img_lower or "petrol" in img_lower or "gas" in img_lower:
            return "FUEL_RECEIPT"
        elif "bank" in img_lower or "statement" in img_lower:
            return "BANK_STATEMENT"
        elif "receipt" in img_lower:
            return "RECEIPT"
        else:
            # Default classification for numbered images
            return "BUSINESS_DOCUMENT"

    def verify_images(self, test_images: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """Verify that test images exist and return verified list"""
        verified_images = []

        for img_name, doc_type in test_images:
            img_path = self.datasets_path / img_name
            if img_path.exists():
                verified_images.append((img_name, doc_type))

        return verified_images

    def print_verification_report(
        self, test_images: List[Tuple[str, str]], verified_images: List[Tuple[str, str]]
    ):
        """Print dataset verification report"""
        console.print("📊 DATASET VERIFICATION", style="bold blue")

        table = Table()
        table.add_column("Image", style="cyan")
        table.add_column("Type", style="magenta")
        table.add_column("Status", style="green")

        for img_name, doc_type in test_images:
            img_path = self.datasets_path / img_name
            status = "✅ Found" if img_path.exists() else "❌ Missing"
            table.add_row(img_name, doc_type, status)

        console.print(table)

        console.print("\n📋 Dataset Summary:")
        console.print(f"   - Expected: {len(test_images)} documents")
        console.print(f"   - Found: {len(verified_images)} documents")
        console.print(f"   - Missing: {len(test_images) - len(verified_images)} documents")

        if len(verified_images) == 0:
            console.print("❌ No test images found! Check datasets/ directory", style="bold red")
            raise FileNotFoundError("No test images found")
        elif len(verified_images) < len(test_images):
            console.print("⚠️ Some test images missing but proceeding with available images", style="yellow")
        else:
            console.print("✅ All test images found", style="bold green")


# =============================================================================
# MODEL LOADERS (From Notebook Cell 3)
# =============================================================================


class LlamaModelLoader:
    """Modular Llama model loader with V100 optimization"""

    @staticmethod
    def load_model(model_path: str, enable_quantization: bool = True):
        """Load Llama model with proper V100 configuration"""
        from transformers import AutoProcessor, BitsAndBytesConfig, MllamaForConditionalGeneration

        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)

        model_kwargs = {"torch_dtype": torch.float16, "local_files_only": True}

        if enable_quantization:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True,
                llm_int8_skip_modules=["vision_tower", "multi_modal_projector"],
            )
            model_kwargs["quantization_config"] = quantization_config

        model = MllamaForConditionalGeneration.from_pretrained(model_path, **model_kwargs)

        model.eval()
        return model, processor

    @staticmethod
    def run_inference(model, processor, prompt: str, image, max_new_tokens: int = 64):
        """Run inference with proper device handling for V100"""
        model.eval()

        inputs = processor(text=prompt, images=image, return_tensors="pt")
        device = next(model.parameters()).device
        if device.type != "cpu":
            device_target = str(device).split(":")[0]
            inputs = {k: v.to(device_target) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,  # Explicitly disable to suppress warnings
                top_p=None,  # Explicitly disable to suppress warnings
                top_k=None,  # Explicitly disable to suppress warnings
                pad_token_id=processor.tokenizer.eos_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                use_cache=True,
            )

        raw_response = processor.decode(
            outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
        )

        del inputs, outputs
        return raw_response


class InternVLModelLoader:
    """Modular InternVL model loader with V100 optimization"""

    @staticmethod
    def load_model(model_path: str, enable_quantization: bool = True):
        """Load InternVL model with proper V100 configuration"""
        import warnings

        from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            warnings.simplefilter("ignore", FutureWarning)

            tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True, local_files_only=True
            )

            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id

        model_kwargs = {"trust_remote_code": True, "torch_dtype": torch.bfloat16, "local_files_only": True}

        if enable_quantization:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True,
            )
            model_kwargs["quantization_config"] = quantization_config

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            warnings.simplefilter("ignore", FutureWarning)

            model = AutoModel.from_pretrained(model_path, **model_kwargs)

        model.eval()
        return model, tokenizer

    @staticmethod
    def run_inference(model, tokenizer, prompt: str, image, max_new_tokens: int = 64):
        """Run inference with comprehensive warning suppression for V100"""
        import io
        import sys
        import warnings

        model.eval()

        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode

        transform = T.Compose(
            [
                T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

        pixel_values = transform(image).unsqueeze(0)
        if torch.cuda.is_available():
            pixel_values = pixel_values.cuda().to(torch.bfloat16).contiguous()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            warnings.simplefilter("ignore", FutureWarning)
            warnings.filterwarnings("ignore", message="Setting `pad_token_id`")
            warnings.filterwarnings("ignore", message=".*pad_token_id.*")

            old_stderr = sys.stderr
            sys.stderr = buffer = io.StringIO()

            try:
                raw_response = model.chat(
                    tokenizer=tokenizer,
                    pixel_values=pixel_values,
                    question=prompt,
                    generation_config={"max_new_tokens": max_new_tokens, "do_sample": False},
                )
            finally:
                sys.stderr = old_stderr

        if isinstance(raw_response, tuple):
            raw_response = raw_response[0]

        del pixel_values
        return raw_response


# =============================================================================
# COMPREHENSIVE ANALYTICS (From Notebook Cell 6)
# =============================================================================


class ComprehensiveResultsAnalyzer:
    """Advanced results analysis with statistical metrics and visualizations for V100"""

    def __init__(self, output_dir: Path, config_loader: Any = None):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config_loader = config_loader
        plt.style.use("default")
        sns.set_palette("husl")

    def create_detailed_dataframe(self, extraction_results: Dict, verified_images: List) -> pd.DataFrame:
        """Create comprehensive DataFrame for analysis"""
        all_results = []

        for model_name, results in extraction_results.items():
            if not results["documents"]:
                continue

            for doc in results["documents"]:
                # Build row dynamically based on what fields are present in doc
                row = {
                    "model": model_name.upper(),
                    "image": doc["img_name"],
                    "doc_type": doc["doc_type"],
                    "inference_time": doc["inference_time"],
                    "is_structured": doc["is_structured"],
                    "extraction_score": doc["extraction_score"],
                    "successful": doc["successful"],
                }

                # Add all has_* fields dynamically
                for key, value in doc.items():
                    if key.startswith("has_"):
                        row[key] = value

                all_results.append(row)

        return pd.DataFrame(all_results)

    def calculate_field_f1_scores(self, df: pd.DataFrame) -> Dict:
        """Calculate F1 scores for each field and model"""
        # Get field names dynamically from DataFrame columns
        fields = [col for col in df.columns if col.startswith("has_")]
        f1_results = {}

        # Ground truth for realistic evaluation - build dynamically
        ground_truth = {}

        # Set ground truth based on field names
        for field in fields:
            if field == "has_abn":
                ground_truth[field] = [
                    1 if img in ["image39.png", "image76.png", "image71.png"] else 0 for img in df["image"]
                ]
            else:
                # Most fields should be present in all images
                ground_truth[field] = [1] * len(df)

        for model in df["model"].unique():
            model_df = df[df["model"] == model]
            f1_results[model] = {}

            for field in fields:
                if len(model_df) > 0:
                    # Fill NaN values with False (0) before converting to int
                    predictions = model_df[field].fillna(False).infer_objects(copy=False).values.astype(int)
                    gt_indices = model_df.index
                    gt = [ground_truth[field][i] for i in range(len(predictions))]

                    f1 = f1_score(gt, predictions, zero_division=0)
                    precision = precision_score(gt, predictions, zero_division=0)
                    recall = recall_score(gt, predictions, zero_division=0)

                    f1_results[model][field] = {"f1": f1, "precision": precision, "recall": recall}

        return f1_results

    def create_performance_visualizations(self, df: pd.DataFrame, f1_results: Dict):
        """Create comprehensive performance visualizations saved as PNG files"""

        fig = plt.figure(figsize=(20, 15))

        # 1. Field Detection Rates Comparison
        plt.subplot(2, 3, 1)

        # Get fields dynamically from actual data
        fields = [col for col in df.columns if col.startswith("has_")]
        field_names = [field[4:].upper() for field in fields]  # Remove "has_" prefix and uppercase

        detection_rates = []
        models = df["model"].unique()

        for model in models:
            model_df = df[df["model"] == model]
            # Fill NaN values with False (0) before calculating mean
            rates = [
                model_df[field].fillna(False).infer_objects(copy=False).mean() * 100 for field in fields
            ]
            detection_rates.append(rates)

        x = np.arange(len(field_names))
        width = 0.35

        for i, (model, rates) in enumerate(zip(models, detection_rates, strict=False)):
            plt.bar(x + i * width, rates, width, label=model, alpha=0.8)

        plt.xlabel("Fields")
        plt.ylabel("Detection Rate (%)")
        plt.title("Field Detection Rates by Model")
        plt.xticks(x + width / 2, field_names)
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 2. F1 Scores Heatmap
        plt.subplot(2, 3, 2)
        f1_matrix = []
        for model in models:
            if model in f1_results:
                f1_scores = [f1_results[model][field]["f1"] for field in fields]
                f1_matrix.append(f1_scores)

        if f1_matrix:
            sns.heatmap(
                f1_matrix,
                annot=True,
                fmt=".3f",
                xticklabels=field_names,
                yticklabels=models,
                cmap="RdYlGn",
                vmin=0,
                vmax=1,
                cbar_kws={"label": "F1 Score"},
            )
            plt.title("F1 Scores by Model and Field")

        # 3. Inference Time Distribution
        plt.subplot(2, 3, 3)
        for model in models:
            model_df = df[df["model"] == model]
            if len(model_df) > 0:
                plt.hist(model_df["inference_time"], alpha=0.7, label=model, bins=10, density=True)

        plt.xlabel("Inference Time (seconds)")
        plt.ylabel("Density")
        plt.title("Inference Time Distribution")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 4. Success Rate by Document Type
        plt.subplot(2, 3, 4)
        success_by_type = df.groupby(["model", "doc_type"])["successful"].mean().unstack(fill_value=0)
        success_by_type.plot(kind="bar", ax=plt.gca(), width=0.8)
        plt.xlabel("Model")
        plt.ylabel("Success Rate")
        plt.title("Success Rate by Document Type")
        plt.xticks(rotation=0)
        plt.legend(title="Document Type", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, alpha=0.3)

        # 5. Extraction Score Distribution
        plt.subplot(2, 3, 5)
        for model in models:
            model_df = df[df["model"] == model]
            if len(model_df) > 0:
                scores = model_df["extraction_score"].value_counts().sort_index()
                plt.plot(scores.index, scores.values, marker="o", label=model, linewidth=2)

        plt.xlabel("Extraction Score (fields extracted)")
        plt.ylabel("Number of Documents")
        plt.title("Extraction Score Distribution")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 6. Structured vs Unstructured Output
        plt.subplot(2, 3, 6)
        structured_rates = df.groupby("model")["is_structured"].mean() * 100
        colors = sns.color_palette("husl", len(structured_rates))
        bars = plt.bar(structured_rates.index, structured_rates.values, color=colors, alpha=0.8)
        plt.xlabel("Model")
        plt.ylabel("Structured Output Rate (%)")
        plt.title("Structured Output Rate by Model")
        plt.grid(True, alpha=0.3)

        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0, height + 1, f"{height:.1f}%", ha="center", va="bottom"
            )

        plt.tight_layout()

        # Save visualization
        viz_path = self.output_dir / "performance_analysis.png"
        plt.savefig(viz_path, dpi=300, bbox_inches="tight")
        console.print(f"📊 Visualizations saved: {viz_path}")
        plt.close()

    def export_results(self, df: pd.DataFrame, f1_results: Dict, extraction_results: Dict):
        """Export detailed results to JSON and CSV"""

        # Export DataFrame
        csv_path = self.output_dir / "detailed_results.csv"
        df.to_csv(csv_path, index=False)
        console.print(f"📁 Results CSV saved: {csv_path}")

        # Export F1 scores
        f1_path = self.output_dir / "f1_scores.json"
        with f1_path.open("w") as f:
            json.dump(f1_results, f, indent=2)
        console.print(f"📁 F1 scores saved: {f1_path}")

        # Export summary
        summary_path = self.output_dir / "summary_results.json"
        summary = {
            "models_tested": list(extraction_results.keys()),
            "total_documents": len(df) // len(df["model"].unique()) if len(df) > 0 else 0,
            "overall_performance": {},
        }

        for model in df["model"].unique():
            model_df = df[df["model"] == model]
            if len(model_df) > 0:
                summary["overall_performance"][model] = {
                    "success_rate": float(model_df["successful"].mean()),
                    "avg_inference_time": float(model_df["inference_time"].mean()),
                    "structured_output_rate": float(model_df["is_structured"].mean()),
                    "abn_detection_rate": float(
                        model_df["has_abn"].fillna(False).infer_objects(copy=False).mean()
                    ),
                }

        with summary_path.open("w") as f:
            json.dump(summary, f, indent=2)
        console.print(f"📁 Summary saved: {summary_path}")


# =============================================================================
# MAIN EXECUTION LOGIC
# =============================================================================


def validate_model(
    model_loader_class, model_path: str, config: Dict, model_name: str
) -> Tuple[bool, Optional[Any], Optional[Any], float]:
    """Validate model loading and basic inference"""

    memory_manager = MemoryManager()
    memory_manager.cleanup_gpu_memory()
    memory_before = memory_manager.get_memory_usage()

    model_start_time = time.time()

    try:
        console.print(f"🔄 Loading {model_name.upper()} model...")

        model, processor_or_tokenizer = model_loader_class.load_model(
            model_path, config["enable_quantization"]
        )

        model.eval()
        model_load_time = time.time() - model_start_time

        memory_after = memory_manager.get_memory_usage()
        memory_delta = memory_manager.get_memory_delta(memory_before, memory_after)

        console.print(f"✅ {model_name.upper()} loaded in {model_load_time:.1f}s")
        memory_manager.print_memory_usage(f"{model_name}-loaded")

        # Simple validation test
        test_img_path = Path("datasets/image14.png")
        if test_img_path.exists():
            image = Image.open(test_img_path).convert("RGB")
            simple_prompt = "<|image|>What do you see?"

            try:
                raw_response = model_loader_class.run_inference(
                    model, processor_or_tokenizer, simple_prompt, image, 32
                )

                if raw_response and len(raw_response.strip()) > 0:
                    console.print(f"✅ {model_name.upper()} validation passed")
                    return True, model, processor_or_tokenizer, model_load_time
                else:
                    console.print(f"❌ {model_name.upper()} validation failed - no response")
                    del model, processor_or_tokenizer
                    memory_manager.cleanup_gpu_memory()
                    return False, None, None, model_load_time

            except Exception as e:
                console.print(f"❌ {model_name.upper()} inference error: {str(e)[:100]}...")
                del model, processor_or_tokenizer
                memory_manager.cleanup_gpu_memory()
                return False, None, None, model_load_time
        else:
            console.print(f"⚠️ Test image not found, assuming {model_name.upper()} is valid")
            return True, model, processor_or_tokenizer, model_load_time

    except Exception as e:
        console.print(f"❌ {model_name.upper()} loading failed: {str(e)[:100]}...")
        memory_manager.cleanup_gpu_memory()
        return False, None, None, 0.0


def run_model_comparison(
    models: List[str],
    datasets_path: str,
    output_dir: str,
    max_tokens: int,
    quantization: bool,
    model_paths: Dict[str, str] = None,
    extraction_config: Dict[str, Any] = None,
):
    """Main model comparison execution"""

    # Use the already loaded extraction_config passed from CLI
    config = extraction_config.get("config")

    # Initialize components
    memory_manager = MemoryManager()
    repetition_controller = UltraAggressiveRepetitionController()
    # Create simple config_loader for compatibility with existing analyzer
    config_loader = extraction_config["config_loader"]
    extraction_analyzer = ConfigurableKeyValueExtractionAnalyzer(config_loader)
    dataset_manager = DatasetManager(datasets_path)

    # Use provided model paths or defaults
    if model_paths is None:
        model_paths = extraction_config["model_paths"]

    config = {
        "model_paths": model_paths,
        "internvl_prompt": extraction_config["internvl_prompt"],
        "llama_prompt": extraction_config["llama_prompt"],
        "max_new_tokens": max_tokens,  # Already effective value from CLI
        "enable_quantization": quantization,  # Already effective value from CLI
        "test_models": models,
        "test_images": extraction_config["test_images"],
        "config": config,
    }

    console.print("🏆 UNIFIED VISION MODEL COMPARISON", style="bold blue")
    console.print(f"📋 Models: {', '.join(models)}")
    console.print(f"📋 Max tokens: {max_tokens}")
    console.print(f"📋 Quantization: {quantization}")

    # CUDA diagnostics
    console.print("\n🔍 CUDA DIAGNOSTICS:")
    console.print(f"   CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        console.print(f"   CUDA Version: {torch.version.cuda}")
        console.print(f"   Device: {torch.cuda.get_device_name(0)}")
        memory_manager.print_memory_usage("Baseline")

    # Discover all images in datasets directory
    if not config["test_images"]:  # If test_images is empty, discover all images
        config["test_images"] = dataset_manager.discover_all_images()
        console.print(f"📁 Discovered {len(config['test_images'])} images in {datasets_path}")

    # Verify dataset
    verified_images = dataset_manager.verify_images(config["test_images"])
    dataset_manager.print_verification_report(config["test_images"], verified_images)

    # Initialize results
    extraction_results = {model: {"documents": [], "successful": 0, "total_time": 0} for model in models}
    model_loaders = {"llama": LlamaModelLoader, "internvl": InternVLModelLoader}

    # Sequential model testing
    for model_name in models:
        if model_name not in model_loaders:
            console.print(f"❌ Unknown model: {model_name}", style="red")
            continue

        console.print(f"\n{'=' * 70}")
        console.print(f"🔥 TESTING {model_name.upper()}", style="bold yellow")
        console.print(f"{'=' * 70}")

        # Load and validate model
        model_valid, model, processor, load_time = validate_model(
            model_loaders[model_name], config["model_paths"][model_name], config, model_name
        )

        if not model_valid:
            console.print(f"❌ Skipping {model_name} - model validation failed")
            continue

        # Run inference on all images
        total_inference_time = 0

        with console.status(f"Running {model_name} inference..."):
            for i, (img_name, doc_type) in enumerate(
                track(verified_images, description=f"{model_name.upper()} Processing")
            ):
                try:
                    img_path = Path(datasets_path) / img_name
                    image = Image.open(img_path).convert("RGB")

                    inference_start = time.time()

                    # Use model-specific prompts
                    if model_name == "llama":
                        prompt = config["llama_prompt"]
                    else:  # internvl
                        prompt = config["internvl_prompt"]

                    # Debug: Print prompt being used
                    # if i == 0:  # Only print for first image to avoid spam
                    #     console.print(f"[dim]🔧 {model_name.upper()} PROMPT:[/dim]")
                    #     truncated_prompt = prompt[:150] + ("..." if len(prompt) > 150 else "")
                    #     console.print(f"[dim]'{truncated_prompt}'[/dim]")

                    raw_response = model_loaders[model_name].run_inference(
                        model, processor, prompt, image, config["max_new_tokens"]
                    )
                    inference_time = time.time() - inference_start
                    total_inference_time += inference_time

                    # Debug: Print raw response to detect safety mode
                    # console.print(f"[dim]🔍 {img_name}[/dim]")
                    # console.print(f"[dim]{raw_response}[/dim]")

                    cleaned_response = repetition_controller.clean_response(raw_response)

                    # Debug: Print cleaned/processed response with proper formatting
                    console.print(f"[dim]🔍 {img_name}[/dim]")
                    if cleaned_response:
                        console.print(f"[dim]{cleaned_response}[/dim]")
                    else:
                        console.print("[dim]No cleaned response[/dim]")

                    analysis = extraction_analyzer.analyze(cleaned_response, img_name)
                    analysis["inference_time"] = inference_time
                    analysis["doc_type"] = doc_type

                    extraction_results[model_name]["documents"].append(analysis)

                    if analysis["successful"]:
                        extraction_results[model_name]["successful"] += 1

                    # Show progress
                    status = "✅" if analysis["successful"] else "❌"
                    fields_detected = []

                    # Dynamically check all has_* fields
                    for key, value in analysis.items():
                        if key.startswith("has_") and value:
                            # Convert has_supplier -> SUPPLIER, has_abn -> ABN, etc.
                            field_name = key[4:].upper()  # Remove "has_" prefix and uppercase
                            fields_detected.append(field_name)

                    fields_str = "|".join(fields_detected) if fields_detected else "none"

                    console.print(
                        f"   {i + 1:2d}. {img_name:<12} {status} {inference_time:.1f}s | {analysis['extraction_score']} fields | {fields_str}"
                    )

                    del image

                    if (i + 1) % 3 == 0:
                        memory_manager.cleanup_gpu_memory()

                except Exception as e:
                    console.print(f"   {i + 1:2d}. {img_name:<12} ❌ Error: {str(e)[:30]}...")

        # Calculate results
        extraction_results[model_name]["total_time"] = total_inference_time
        extraction_results[model_name]["avg_time"] = (
            total_inference_time / len(verified_images) if verified_images else 0
        )

        # Print model summary
        abn_count = sum(
            1 for doc in extraction_results[model_name]["documents"] if doc.get("has_abn", False)
        )

        # Calculate average extraction score
        avg_extraction_score = (
            sum(doc.get("extraction_score", 0) for doc in extraction_results[model_name]["documents"])
            / len(extraction_results[model_name]["documents"])
            if extraction_results[model_name]["documents"]
            else 0
        )

        console.print(f"\n📊 {model_name.upper()} Results:")
        console.print(
            f"   Success rate: {extraction_results[model_name]['successful']}/{len(verified_images)}"
        )
        console.print(
            f"   ABN detection: {abn_count}/{len(verified_images)} ({abn_count / len(verified_images) * 100:.1f}%)"
            if verified_images
            else "   ABN detection: 0/0"
        )
        console.print(f"   Average time: {extraction_results[model_name]['avg_time']:.1f}s per document")
        console.print(f"   Average fields extracted: {avg_extraction_score:.1f} key-value pairs")

        # Clean up model
        console.print(f"\n🧹 Cleaning up {model_name.upper()}")
        del model, processor
        memory_manager.cleanup_gpu_memory()

    # Generate comprehensive analytics
    console.print(f"\n{'=' * 70}")
    console.print("📊 GENERATING COMPREHENSIVE ANALYTICS", style="bold green")
    console.print(f"{'=' * 70}")

    output_path = Path(output_dir)
    analyzer = ComprehensiveResultsAnalyzer(output_path, config_loader)

    results_df = analyzer.create_detailed_dataframe(extraction_results, verified_images)

    if not results_df.empty:
        f1_scores = analyzer.calculate_field_f1_scores(results_df)
        analyzer.create_performance_visualizations(results_df, f1_scores)
        analyzer.export_results(results_df, f1_scores, extraction_results)

        # Print summary
        console.print("\n🏆 FINAL SUMMARY:")
        for model in results_df["model"].unique():
            model_df = results_df[results_df["model"] == model]
            success_rate = model_df["successful"].mean() * 100
            avg_time = model_df["inference_time"].mean()
            abn_rate = model_df["has_abn"].fillna(False).infer_objects(copy=False).mean() * 100
            avg_extraction_score = model_df["extraction_score"].mean()

            console.print(
                f"{model}: {success_rate:.1f}% success | {avg_time:.1f}s avg | {avg_extraction_score:.1f} avg fields | {abn_rate:.1f}% ABN detection"
            )

        console.print(f"\n✅ Analysis complete! Results saved to: {output_path}")

    else:
        console.print("❌ No results to analyze")


# =============================================================================
# CLI INTERFACE
# =============================================================================

app = typer.Typer(help="Unified Vision Model Comparison for V100 Production Environment")


@app.command()
def compare(
    datasets_path: str = typer.Option(None, help="Path to input datasets directory (default from config)"),
    output_dir: str = typer.Option(None, help="Output directory for results (default from config)"),
    models: str = typer.Option(None, help="Comma-separated list of models (default from config)"),
    max_tokens: int = typer.Option(None, help="Maximum new tokens for generation (default from config)"),
    quantization: bool = typer.Option(
        None, help="Enable 8-bit quantization for V100 (default from config)"
    ),
    llama_path: str = typer.Option(None, help="Custom path to Llama model"),
    internvl_path: str = typer.Option(None, help="Custom path to InternVL model"),
    config_path: str = typer.Option(
        "model_comparison.yaml", help="Path to model comparison configuration YAML file"
    ),
):
    """Run comprehensive model comparison with analytics"""

    # Load configuration first to get defaults
    extraction_config = load_extraction_config(config_path)
    config = extraction_config.get("config", {})
    defaults = config.get("defaults", {})

    # Apply effective values (CLI overrides config defaults)
    effective_datasets_path = (
        datasets_path if datasets_path is not None else defaults.get("datasets_path", "datasets")
    )
    effective_output_dir = output_dir if output_dir is not None else defaults.get("output_dir", "results")
    effective_models = models if models is not None else defaults.get("models", "llama,internvl")
    effective_max_tokens = max_tokens if max_tokens is not None else defaults.get("max_tokens", 256)
    effective_quantization = (
        quantization if quantization is not None else defaults.get("quantization", True)
    )

    models_list = [m.strip() for m in effective_models.split(",")]

    # Custom model paths if provided
    if DEFAULT_CONFIG is None:
        # Load default paths from config for the first time
        model_paths = extraction_config["model_paths"].copy()
    else:
        model_paths = DEFAULT_CONFIG["model_paths"].copy()

    if llama_path:
        model_paths["llama"] = llama_path
    if internvl_path:
        model_paths["internvl"] = internvl_path

    run_model_comparison(
        models=models_list,
        datasets_path=effective_datasets_path,
        output_dir=effective_output_dir,
        max_tokens=effective_max_tokens,
        quantization=effective_quantization,
        model_paths=model_paths,
        extraction_config=extraction_config,
    )


@app.command()
def check_environment(
    datasets_path: str = typer.Option("datasets", help="Path to datasets directory to check"),
):
    """Check V100 environment and dependencies"""

    console.print("🔍 V100 ENVIRONMENT CHECK", style="bold blue")

    # Check CUDA
    console.print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        console.print(f"CUDA Version: {torch.version.cuda}")
        console.print(f"Device: {torch.cuda.get_device_name(0)}")
        memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        console.print(f"Total Memory: {memory:.1f}GB")

    # Check dependencies
    try:
        import pandas

        console.print(f"✅ Pandas: {pandas.__version__}")
    except ImportError:
        console.print("❌ Pandas not available")

    try:
        import seaborn

        console.print(f"✅ Seaborn: {seaborn.__version__}")
    except ImportError:
        console.print("❌ Seaborn not available")

    try:
        import sklearn

        console.print(f"✅ Scikit-learn: {sklearn.__version__}")
    except ImportError:
        console.print("❌ Scikit-learn not available")

    # Check datasets
    datasets_dir = Path(datasets_path)
    if datasets_dir.exists():
        image_count = len(list(datasets_dir.glob("*.png")))
        console.print(f"✅ Datasets directory found: {datasets_dir} with {image_count} PNG files")
    else:
        console.print(f"❌ Datasets directory not found: {datasets_dir}")


if __name__ == "__main__":
    app()
