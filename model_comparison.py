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
import textwrap
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import typer
from PIL import Image
from rich.console import Console
from rich.progress import track
from rich.table import Table
from sklearn.metrics import f1_score, precision_score, recall_score

# Configure matplotlib for headless environment
plt.switch_backend("Agg")  # Non-interactive backend for V100
console = Console()

# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_CONFIG = {
    "model_paths": {
        "llama": "/home/jovyan/nfs_share/models/Llama-3.2-11B-Vision",
        "internvl": "/home/jovyan/nfs_share/models/InternVL3-8B",
    },
    "extraction_prompt": textwrap.dedent("""
        <|image|>Extract data from this Australian business document in KEY-VALUE format.

        Output format:
        STORE: [business name]
        ABN: [11-digit Australian Business Number]
        DATE: [date in DD/MM/YYYY format]
        TOTAL: [total amount in AUD]
        SUBTOTAL: [subtotal amount]
        GST: [GST amount]
        ITEMS: [item names separated by |]

        ABN is crucial - look for 11-digit numbers formatted as XX XXX XXX XXX or XXXXXXXXXXX. Use Australian date format (DD/MM/YYYY) and include currency symbols. Extract all visible text and format as KEY: VALUE pairs only. Stop after completion.
    """).strip(),
    "test_images": [
        ("image14.png", "TAX_INVOICE"),
        ("image65.png", "TAX_INVOICE"),
        ("image71.png", "TAX_INVOICE"),
        ("image74.png", "TAX_INVOICE"),
        ("image205.png", "FUEL_RECEIPT"),
        ("image23.png", "TAX_INVOICE"),
        ("image45.png", "TAX_INVOICE"),
        ("image1.png", "BANK_STATEMENT"),
        ("image39.png", "TAX_INVOICE"),
        ("image76.png", "TAX_INVOICE"),
    ],
}

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

        response = self._remove_business_patterns(response)
        response = self._remove_word_repetition(response)
        response = self._remove_phrase_repetition(response)

        response = re.sub(r"\s+", " ", response)
        response = re.sub(r"[.]{2,}", ".", response)
        response = re.sub(r"[!]{2,}", "!", response)

        return response.strip()

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


class KeyValueExtractionAnalyzer:
    """Analyzer for KEY-VALUE extraction results with realistic Australian business requirements"""

    @staticmethod
    def analyze(response: str, img_name: str) -> Dict[str, Any]:
        """Analyze KEY-VALUE extraction results with Australian format but realistic success criteria"""
        response_clean = response.strip()

        is_structured = bool(
            re.search(
                r"(STORE:|ABN:|DATE:|TOTAL:|store_name:|abn:|date:|total:)", response_clean, re.IGNORECASE
            )
        )

        # Extract data from KEY-VALUE format
        store_match = re.search(r'(?:STORE|store_name):\s*"?([^"\n]+)"?', response_clean, re.IGNORECASE)
        abn_match = re.search(r'(?:ABN|abn):\s*"?([^"\n]+)"?', response_clean, re.IGNORECASE)
        date_match = re.search(r'(?:DATE|date):\s*"?([^"\n]+)"?', response_clean, re.IGNORECASE)
        total_match = re.search(
            r'(?:TOTAL|total_amount|total):\s*"?([^"\n]+)"?', response_clean, re.IGNORECASE
        )

        # Fallback detection for non-structured responses
        if not store_match:
            store_match = re.search(
                r"(spotlight|woolworths|coles|bunnings|officeworks|kmart|target|harvey norman|jb hi-fi)",
                response_clean,
                re.IGNORECASE,
            )

        if not abn_match:
            abn_match = re.search(r"\b(\d{2}\s*\d{3}\s*\d{3}\s*\d{3}|\d{11})\b", response_clean)
            if not abn_match:
                abn_match = re.search(
                    r"(?:ABN|A\.B\.N\.?)\s*:?\s*(\d{2}\s*\d{3}\s*\d{3}\s*\d{3}|\d{11})",
                    response_clean,
                    re.IGNORECASE,
                )

        if not date_match:
            date_match = re.search(
                r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}\.\d{1,2}\.\d{2,4})\b", response_clean
            )

        if not total_match:
            total_match = re.search(r"(\$\d+\.\d{2}|\$\d+|AUD\s*\d+\.\d{2})", response_clean)

        has_store = bool(store_match)
        has_abn = bool(abn_match)
        has_date = bool(date_match)
        has_total = bool(total_match)

        core_fields = [has_store, has_date, has_total]
        all_fields = [has_store, has_abn, has_date, has_total]

        extraction_score = sum(all_fields)
        core_score = sum(core_fields)

        # Success = at least 2/3 core fields (STORE, DATE, TOTAL)
        successful = core_score >= 2

        return {
            "img_name": img_name,
            "response": response_clean,
            "is_structured": is_structured,
            "has_store": has_store,
            "has_abn": has_abn,
            "has_date": has_date,
            "has_total": has_total,
            "extraction_score": extraction_score,
            "core_score": core_score,
            "successful": successful,
        }


class DatasetManager:
    """Dataset verification and management"""

    def __init__(self, datasets_path: str = "datasets"):
        self.datasets_path = Path(datasets_path)

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

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        plt.style.use("default")
        sns.set_palette("husl")

    def create_detailed_dataframe(self, extraction_results: Dict, verified_images: List) -> pd.DataFrame:
        """Create comprehensive DataFrame for analysis"""
        all_results = []

        for model_name, results in extraction_results.items():
            if not results["documents"]:
                continue

            for doc in results["documents"]:
                all_results.append(
                    {
                        "model": model_name.upper(),
                        "image": doc["img_name"],
                        "doc_type": doc["doc_type"],
                        "inference_time": doc["inference_time"],
                        "is_structured": doc["is_structured"],
                        "has_store": doc["has_store"],
                        "has_abn": doc["has_abn"],
                        "has_date": doc["has_date"],
                        "has_total": doc["has_total"],
                        "extraction_score": doc["extraction_score"],
                        "core_score": doc["core_score"],
                        "successful": doc["successful"],
                    }
                )

        return pd.DataFrame(all_results)

    def calculate_field_f1_scores(self, df: pd.DataFrame) -> Dict:
        """Calculate F1 scores for each field and model"""
        fields = ["has_store", "has_abn", "has_date", "has_total"]
        f1_results = {}

        # Ground truth for realistic evaluation
        ground_truth = {
            "has_store": [1] * len(df),
            "has_abn": [
                1 if img in ["image39.png", "image76.png", "image71.png"] else 0 for img in df["image"]
            ],
            "has_date": [1] * len(df),
            "has_total": [1] * len(df),
        }

        for model in df["model"].unique():
            model_df = df[df["model"] == model]
            f1_results[model] = {}

            for field in fields:
                if len(model_df) > 0:
                    predictions = model_df[field].values.astype(int)
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
        fields = ["has_store", "has_abn", "has_date", "has_total"]
        field_names = ["STORE", "ABN", "DATE", "TOTAL"]

        detection_rates = []
        models = df["model"].unique()

        for model in models:
            model_df = df[df["model"] == model]
            rates = [model_df[field].mean() * 100 for field in fields]
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

        # 5. Core Score Distribution
        plt.subplot(2, 3, 5)
        for model in models:
            model_df = df[df["model"] == model]
            if len(model_df) > 0:
                scores = model_df["core_score"].value_counts().sort_index()
                plt.plot(scores.index, scores.values, marker="o", label=model, linewidth=2)

        plt.xlabel("Core Score (0-3)")
        plt.ylabel("Number of Documents")
        plt.title("Core Score Distribution")
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
                    "abn_detection_rate": float(model_df["has_abn"].mean()),
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
):
    """Main model comparison execution"""

    # Initialize components
    memory_manager = MemoryManager()
    repetition_controller = UltraAggressiveRepetitionController()
    extraction_analyzer = KeyValueExtractionAnalyzer()
    dataset_manager = DatasetManager(datasets_path)

    # Use provided model paths or defaults
    if model_paths is None:
        model_paths = DEFAULT_CONFIG["model_paths"]

    config = {
        "model_paths": model_paths,
        "extraction_prompt": DEFAULT_CONFIG["extraction_prompt"],
        "max_new_tokens": max_tokens,
        "enable_quantization": quantization,
        "test_models": models,
        "test_images": DEFAULT_CONFIG["test_images"],
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
                    raw_response = model_loaders[model_name].run_inference(
                        model, processor, config["extraction_prompt"], image, config["max_new_tokens"]
                    )
                    inference_time = time.time() - inference_start
                    total_inference_time += inference_time

                    cleaned_response = repetition_controller.clean_response(raw_response)
                    analysis = extraction_analyzer.analyze(cleaned_response, img_name)
                    analysis["inference_time"] = inference_time
                    analysis["doc_type"] = doc_type

                    extraction_results[model_name]["documents"].append(analysis)

                    if analysis["successful"]:
                        extraction_results[model_name]["successful"] += 1

                    # Show progress
                    status = "✅" if analysis["successful"] else "❌"
                    fields_detected = []
                    if analysis["has_store"]:
                        fields_detected.append("STORE")
                    if analysis["has_abn"]:
                        fields_detected.append("ABN")
                    if analysis["has_date"]:
                        fields_detected.append("DATE")
                    if analysis["has_total"]:
                        fields_detected.append("TOTAL")
                    fields_str = "|".join(fields_detected) if fields_detected else "none"

                    console.print(
                        f"   {i + 1:2d}. {img_name:<12} {status} {inference_time:.1f}s | {analysis['core_score']}/3 core | {fields_str}"
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

        # Clean up model
        console.print(f"\n🧹 Cleaning up {model_name.upper()}")
        del model, processor
        memory_manager.cleanup_gpu_memory()

    # Generate comprehensive analytics
    console.print(f"\n{'=' * 70}")
    console.print("📊 GENERATING COMPREHENSIVE ANALYTICS", style="bold green")
    console.print(f"{'=' * 70}")

    output_path = Path(output_dir)
    analyzer = ComprehensiveResultsAnalyzer(output_path)

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
            abn_rate = model_df["has_abn"].mean() * 100

            console.print(
                f"{model}: {success_rate:.1f}% success | {avg_time:.1f}s avg | {abn_rate:.1f}% ABN detection"
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
    datasets_path: str = typer.Option(..., help="Path to input datasets directory (required for KFP)"),
    output_dir: str = typer.Option("results", help="Output directory for results"),
    models: str = typer.Option("llama,internvl", help="Comma-separated list of models (llama,internvl)"),
    max_tokens: int = typer.Option(64, help="Maximum new tokens for generation"),
    quantization: bool = typer.Option(True, help="Enable 8-bit quantization for V100"),
    llama_path: str = typer.Option(None, help="Custom path to Llama model"),
    internvl_path: str = typer.Option(None, help="Custom path to InternVL model"),
):
    """Run comprehensive model comparison with analytics"""

    models_list = [m.strip() for m in models.split(",")]

    # Custom model paths if provided
    model_paths = DEFAULT_CONFIG["model_paths"].copy()
    if llama_path:
        model_paths["llama"] = llama_path
    if internvl_path:
        model_paths["internvl"] = internvl_path

    run_model_comparison(
        models=models_list,
        datasets_path=datasets_path,
        output_dir=output_dir,
        max_tokens=max_tokens,
        quantization=quantization,
        model_paths=model_paths,
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
