#!/usr/bin/env python3
"""
Unified Vision Model Comparison Script - Production Version
=========================================================

Production-ready model comparison using modular architecture with 55 standardized
field labels for Australian Tax Office document processing.

KFP (Kubeflow Pipelines) Usage:
    python model_comparison.py compare --datasets-path /mnt/datasets --output-dir /mnt/output
    python model_comparison.py compare --datasets-path /data/images --models llama --quantization
    python model_comparison.py check-environment --datasets-path /mnt/input

Local Usage:
    python model_comparison.py compare --datasets-path ./datasets --output-dir ./results
    python model_comparison.py compare --datasets-path ~/data --models internvl --max-tokens 128

Production Features:
- ✅ Modular architecture with extensible components
- ✅ 55 standardized production field labels
- ✅ Advanced field validation and extraction
- ✅ Comprehensive F1 score analysis
- ✅ Model registry for extensible model management
- ✅ Production-ready configuration system
- ✅ ATO compliance scoring
- ✅ Enhanced error handling and diagnostics
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import typer
import yaml
from rich.console import Console

from vision_processor.comparison.comparison_runner import ComparisonRunner

# Import simple configuration
from vision_processor.config import ConfigManager
from vision_processor.config.model_registry import get_model_registry

# Configure environment
console = Console()

# =============================================================================
# PRODUCTION CLI INTERFACE - USING MODULAR ARCHITECTURE
# =============================================================================

app = typer.Typer(help="Production Vision Model Comparison with Modular Architecture")


def run_production_comparison(
    datasets_path: str,
    output_dir: str,
    models: List[str],
    max_tokens: int,
    quantization: bool,
    trust_remote_code: bool,
    llama_path: Optional[str] = None,
    internvl_path: Optional[str] = None,
    config_path: str = "model_comparison.yaml",
) -> Optional[Dict[str, Any]]:
    """Run production model comparison using modular architecture."""

    try:
        # Create production configuration
        config_overrides = {
            "datasets_path": datasets_path,
            "output_dir": output_dir,
            "models": ",".join(models),
            "max_tokens": max_tokens,
            "quantization": quantization,
            "trust_remote_code": trust_remote_code,
        }

        # Add custom model paths if provided
        if llama_path:
            config_overrides["llama_path"] = llama_path
        if internvl_path:
            config_overrides["internvl_path"] = internvl_path

        # Load simple configuration with YAML file
        config = ConfigManager(yaml_file=config_path)

        # Simple validation - just check if models exist
        if not Path(datasets_path).exists():
            console.print("❌ Datasets path does not exist", style="bold red")
            return None

        # Print configuration summary
        console.print("\n📋 SIMPLE CONFIGURATION", style="bold blue")
        config.print_configuration()

        # Initialize and run comparison
        runner = ComparisonRunner(config)
        results = runner.run_comparison()

        # Print production summary
        console.print("\n🎯 PRODUCTION SUMMARY", style="bold green")
        console.print(f"✅ Total execution time: {results.total_execution_time:.1f}s")
        console.print(f"✅ Overall success rate: {results.overall_success_rate:.1%}")
        console.print(f"✅ Models tested: {len(results.models_tested)}")
        console.print(f"✅ Documents processed: {results.dataset_info.total_images}")
        console.print("✅ Core fields: 12 essential fields")

        # Export DataFrame for downstream processing
        try:
            results_df = runner.export_dataframe()
            if results_df is not None:
                csv_path = Path(output_dir) / "production_results.csv"
                results_df.to_csv(csv_path, index=False)
                console.print(f"✅ Production results exported: {csv_path}")
                console.print(
                    f"📊 DataFrame shape: {results_df.shape[0]} rows x {results_df.shape[1]} columns"
                )
                console.print(
                    "📋 Columns: model_name, image_name, processing_time, field_count, quality_rating, + all 26 field values"
                )
        except Exception as e:
            console.print(f"⚠️  Failed to export DataFrame: {e}")

        return {
            "success": True,
            "results": results,
            "config": config,
            "execution_time": results.total_execution_time,
        }

    except Exception as e:
        import traceback

        console.print(f"❌ Production comparison failed: {e}", style="bold red")
        console.print("🔍 Full traceback:", style="red")
        console.print(traceback.format_exc(), style="red")
        return {"success": False, "error": str(e)}


def validate_production_environment() -> bool:
    """Validate production environment and model registry."""
    console.print("🔍 PRODUCTION ENVIRONMENT VALIDATION", style="bold blue")

    # Check CUDA
    if torch.cuda.is_available():
        console.print(f"✅ CUDA: {torch.cuda.get_device_name(0)}")
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        console.print(f"✅ GPU Memory: {memory_gb:.1f}GB")
    else:
        console.print("⚠️  CUDA not available - using CPU")

    # Check simple schema
    console.print("✅ Simple Core Fields: 12 essential fields")

    # Check model registry
    model_registry = get_model_registry()
    available_models = model_registry.list_available_models()
    console.print(f"✅ Model Registry: {len(available_models)} models available")

    for model_name in available_models:
        info = model_registry.get_model_info(model_name)
        status = "✅" if info.get("path_exists", False) else "❌"
        console.print(
            f"   {status} {model_name}: {info.get('description', 'No description')}"
        )

    # Check dependencies
    import importlib.util

    dependencies = ["pandas", "seaborn", "sklearn"]
    missing_deps = []

    for dep in dependencies:
        if importlib.util.find_spec(dep) is None:
            missing_deps.append(dep)

    if missing_deps:
        console.print(f"❌ Missing dependencies: {', '.join(missing_deps)}")
        return False
    else:
        console.print("✅ Analysis dependencies: pandas, seaborn, sklearn")

    return len(available_models) > 0


@app.command()
def compare(
    datasets_path: str = typer.Option(
        None, help="Path to input datasets directory (default from config)"
    ),
    output_dir: str = typer.Option(
        None, help="Output directory for results (default from config)"
    ),
    models: str = typer.Option(
        None, help="Comma-separated list of models (default from config)"
    ),
    max_tokens: int = typer.Option(
        None, help="Maximum new tokens for generation (default from config)"
    ),
    quantization: bool = typer.Option(
        None, help="Enable 8-bit quantization for V100 (default from config)"
    ),
    trust_remote_code: bool = typer.Option(
        None,
        help="Allow execution of remote code for custom models (default from config)",
    ),
    llama_path: str = typer.Option(None, help="Custom path to Llama model"),
    internvl_path: str = typer.Option(None, help="Custom path to InternVL model"),
    config_path: str = typer.Option(
        "model_comparison.yaml", help="Path to model comparison configuration YAML file"
    ),
):
    """Run production model comparison with modular architecture"""

    # Load default values from config file
    try:
        config_file = Path(config_path)
        defaults = {}
        if config_file.exists():
            import yaml

            with config_file.open("r") as f:
                yaml_config = yaml.safe_load(f) or {}
                defaults = yaml_config.get("defaults", {})
    except Exception as e:
        console.print(f"⚠️  Failed to load config file: {e}")
        defaults = {}

    # Apply effective values (CLI overrides config defaults)
    effective_datasets_path = datasets_path or defaults.get("datasets_path", "datasets")
    effective_output_dir = output_dir or defaults.get("output_dir", "results")
    effective_models = models or defaults.get("models", "llama,internvl")
    effective_max_tokens = max_tokens or defaults.get("max_tokens", 256)
    effective_quantization = (
        quantization if quantization is not None else defaults.get("quantization", True)
    )
    effective_trust_remote_code = (
        trust_remote_code
        if trust_remote_code is not None
        else defaults.get("trust_remote_code", True)
    )

    models_list = [m.strip() for m in effective_models.split(",")]

    # Run production comparison
    result = run_production_comparison(
        datasets_path=effective_datasets_path,
        output_dir=effective_output_dir,
        models=models_list,
        max_tokens=effective_max_tokens,
        quantization=effective_quantization,
        trust_remote_code=effective_trust_remote_code,
        llama_path=llama_path,
        internvl_path=internvl_path,
        config_path=config_path,
    )

    # Handle results
    if result and result.get("success"):
        console.print(
            "\n🎉 Production comparison completed successfully!", style="bold green"
        )
        console.print(
            f"⏱️  Execution time: {result['execution_time']:.1f}s", style="green"
        )
    else:
        error_msg = result.get("error", "Unknown error") if result else "Unknown error"
        console.print(
            f"\n❌ Production comparison failed: {error_msg}", style="bold red"
        )
        raise typer.Exit(1)


@app.command()
def check_environment(
    datasets_path: str = typer.Option(
        "datasets", help="Path to datasets directory to check"
    ),
):
    """Check production environment and dependencies"""

    # Run production environment validation
    is_valid = validate_production_environment()

    # Check datasets
    console.print("\n📁 DATASET VALIDATION", style="bold blue")
    datasets_dir = Path(datasets_path)
    if datasets_dir.exists():
        image_count = len(list(datasets_dir.glob("*.png")))
        console.print(
            f"✅ Datasets directory: {datasets_dir} ({image_count} PNG files)"
        )

        if image_count == 0:
            console.print("⚠️  No PNG files found in datasets directory", style="yellow")
        else:
            # Sample a few images to check validity
            sample_images = list(datasets_dir.glob("*.png"))[:3]
            valid_images = 0

            for img_path in sample_images:
                try:
                    from PIL import Image

                    with Image.open(img_path) as img:
                        img.verify()
                    valid_images += 1
                except Exception:
                    pass

            console.print(
                f"✅ Sample validation: {valid_images}/{len(sample_images)} images valid"
            )
    else:
        console.print(f"❌ Datasets directory not found: {datasets_dir}")
        is_valid = False

    # Print final status
    console.print(
        f"\n{'🎉' if is_valid else '❌'} ENVIRONMENT STATUS",
        style="bold green" if is_valid else "bold red",
    )
    if is_valid:
        console.print("✅ Production environment ready for model comparison")
    else:
        console.print("❌ Environment issues detected - check the errors above")
        raise typer.Exit(1)


@app.command()
def list_models():
    """List available models in the registry"""
    console.print("🤖 AVAILABLE MODELS", style="bold blue")

    model_registry = get_model_registry()
    model_registry.print_registry_status()


@app.command()
def validate_models(
    config_path: str = typer.Option(
        "model_comparison.yaml", help="Path to configuration file"
    ),
):
    """Validate all configured models"""
    console.print("🔍 MODEL VALIDATION", style="bold blue")

    try:
        # Load configuration to get model paths
        config_file = Path(config_path)
        if config_file.exists():
            with config_file.open("r") as f:
                yaml_config = yaml.safe_load(f) or {}
                model_paths = yaml_config.get("model_paths", {})
        else:
            model_paths = {}

        model_registry = get_model_registry()
        results = model_registry.validate_all_models(model_paths)

        console.print("\n📊 Validation Results:")
        for model_name, is_valid in results.items():
            status = "✅" if is_valid else "❌"
            console.print(f"   {status} {model_name}")

        success_count = sum(results.values())
        total_count = len(results)
        console.print(
            f"\n🎯 Summary: {success_count}/{total_count} models validated successfully"
        )

        if success_count == 0:
            console.print("❌ No models are ready for comparison", style="bold red")
            raise typer.Exit(1)
        elif success_count < total_count:
            console.print("⚠️  Some models failed validation", style="yellow")
        else:
            console.print("✅ All models ready for comparison", style="bold green")

    except Exception as e:
        console.print(f"❌ Model validation failed: {e}", style="bold red")
        raise typer.Exit(1) from None


@app.command()
def show_schema():
    """Show the simple core fields schema"""
    console.print("🏷️  SIMPLE CORE FIELDS", style="bold blue")

    # Define the simple core fields directly (from base_extractor.py)
    core_fields = {
        "DATE",
        "TOTAL",
        "GST",
        "ABN",
        "SUPPLIER_NAME",
        "INVOICE_NUMBER",
        "AMOUNT",
        "DESCRIPTION",
        "BSB",
        "ACCOUNT_NUMBER",
        "BUSINESS_NAME",
        "RECEIPT_NUMBER",
    }

    console.print(f"📊 Total Core Fields: {len(core_fields)}")
    console.print("\n🎯 Core Fields for Australian Tax Documents:")

    for i, field in enumerate(sorted(core_fields), 1):
        console.print(f"   {i:2d}. {field}")

    console.print("\n💡 Note: This simplified system focuses on essential fields only")
    console.print("💡 No complex categorization - just the fields that matter most")


if __name__ == "__main__":
    # If no command is provided, default to compare
    import sys

    if len(sys.argv) == 1:
        sys.argv.append("compare")
    app()
