"""Configuration Validation Utilities
=================================

Provides validation utilities and pre-flight checks for the unified
configuration system, following fail-fast principles.
"""

import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from rich.console import Console

console = Console()


class ValidationResult:
    """Result of a validation check."""

    def __init__(
        self, passed: bool, message: str, details: Optional[Dict[str, str]] = None
    ):
        self.passed = passed
        self.message = message
        self.details = details or {}

    def print_result(self):
        """Print validation result with formatting."""
        icon = "✅" if self.passed else "❌"
        style = "green" if self.passed else "red"

        console.print(f"{icon} {self.message}", style=style)

        if not self.passed and self.details:
            for key, value in self.details.items():
                console.print(f"   💡 {key}: {value}", style="yellow")


class ConfigValidator:
    """Comprehensive configuration validator with pre-flight checks."""

    def __init__(self):
        self.results: List[ValidationResult] = []

    def validate_system_requirements(self) -> ValidationResult:
        """Validate system meets minimum requirements."""
        # Check Python version
        import sys

        python_version = f"{sys.version_info.major}.{sys.version_info.minor}"

        if sys.version_info < (3, 11):
            return ValidationResult(
                False,
                "Python version too old",
                {
                    "Current": python_version,
                    "Required": "3.11+",
                    "Solution": "Upgrade Python to 3.11 or newer",
                },
            )

        # Check CUDA availability for GPU models
        cuda_available = torch.cuda.is_available()
        cuda_version = torch.version.cuda if cuda_available else "Not available"

        return ValidationResult(
            True,
            f"System requirements met (Python {python_version}, CUDA {cuda_version})",
        )

    def validate_gpu_memory(self, required_gb: float = 16.0) -> ValidationResult:
        """Validate GPU has sufficient memory."""
        if not torch.cuda.is_available():
            return ValidationResult(
                False,
                "No GPU available",
                {
                    "Required": f"{required_gb}GB VRAM",
                    "Solution": "Use CPU mode or run on GPU-enabled system",
                },
            )

        # Get GPU memory info
        gpu_count = torch.cuda.device_count()
        total_memory_gb = 0
        memory_details = []

        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            total_memory_gb += memory_gb
            memory_details.append(f"GPU {i}: {memory_gb:.1f}GB")

        if total_memory_gb < required_gb:
            return ValidationResult(
                False,
                "Insufficient GPU memory",
                {
                    "Available": f"{total_memory_gb:.1f}GB across {gpu_count} GPU(s)",
                    "Required": f"{required_gb}GB",
                    "Details": ", ".join(memory_details),
                    "Solution": "Enable quantization or use smaller model",
                },
            )

        return ValidationResult(
            True, f"GPU memory sufficient ({total_memory_gb:.1f}GB available)"
        )

    def validate_disk_space(
        self, model_paths: List[str], required_gb: float = 50.0
    ) -> ValidationResult:
        """Validate sufficient disk space for models."""
        # Get unique mount points for model paths
        mount_points = {}
        for path in model_paths:
            try:
                stat = shutil.disk_usage(Path(path).parent)
                mount = Path(path).parent
                mount_points[str(mount)] = stat.free / (1024**3)
            except Exception:
                # If path doesn't exist, check parent directories
                parent = Path(path).parent
                while not parent.exists() and parent != parent.parent:
                    parent = parent.parent
                if parent.exists():
                    stat = shutil.disk_usage(parent)
                    mount_points[str(parent)] = stat.free / (1024**3)

        if not mount_points:
            return ValidationResult(
                False,
                "Cannot check disk space",
                {"Solution": "Ensure model directories exist"},
            )

        min_free_gb = min(mount_points.values())

        if min_free_gb < required_gb:
            return ValidationResult(
                False,
                "Insufficient disk space",
                {
                    "Available": f"{min_free_gb:.1f}GB",
                    "Required": f"{required_gb}GB",
                    "Locations": ", ".join(
                        f"{k}: {v:.1f}GB" for k, v in mount_points.items()
                    ),
                    "Solution": "Free up disk space or use different location",
                },
            )

        return ValidationResult(
            True, f"Disk space sufficient ({min_free_gb:.1f}GB free)"
        )

    def validate_model_files(
        self, model_path: str, model_name: str
    ) -> ValidationResult:
        """Validate model files exist and are complete."""
        path = Path(model_path)

        if not path.exists():
            return ValidationResult(
                False,
                f"Model path not found: {model_name}",
                {
                    "Expected": str(path.absolute()),
                    "Solution": f"Download {model_name} model to this location",
                },
            )

        # Check for common model files
        required_patterns = ["config.json", "*.safetensors", "*.bin", "tokenizer*"]

        found_files = []
        for pattern in required_patterns:
            matches = list(path.glob(pattern))
            if matches:
                found_files.extend(matches)

        if not found_files:
            return ValidationResult(
                False,
                f"No model files found for {model_name}",
                {
                    "Path": str(path),
                    "Expected files": ", ".join(required_patterns),
                    "Solution": "Ensure model is fully downloaded",
                },
            )

        # Check file sizes (warn if suspiciously small)
        total_size_gb = sum(f.stat().st_size for f in found_files) / (1024**3)

        if total_size_gb < 1.0:
            return ValidationResult(
                False,
                f"Model files too small for {model_name}",
                {
                    "Total size": f"{total_size_gb:.2f}GB",
                    "Expected": ">1GB for vision models",
                    "Solution": "Re-download model, files may be corrupted",
                },
            )

        return ValidationResult(
            True, f"Model files valid for {model_name} ({total_size_gb:.1f}GB)"
        )

    def validate_dependencies(self) -> ValidationResult:
        """Validate required Python packages are installed."""
        required_packages = {
            "torch": "PyTorch",
            "transformers": "Transformers",
            "PIL": "Pillow",
            "yaml": "PyYAML",
            "typer": "Typer",
            "rich": "Rich",
        }

        missing = []
        for package, display_name in required_packages.items():
            try:
                __import__(package)
            except ImportError:
                missing.append(display_name)

        if missing:
            return ValidationResult(
                False,
                "Missing required packages",
                {
                    "Missing": ", ".join(missing),
                    "Solution": "Run: conda env update -f environment.yml",
                },
            )

        return ValidationResult(True, "All dependencies installed")

    def validate_awk_processor(self) -> ValidationResult:
        """Validate AWK processor is available."""
        awk_script = Path("awk_markdown_processor.py")
        awk_config = Path("markdown_processing_config.yaml")

        if not awk_script.exists():
            return ValidationResult(
                False,
                "AWK processor script not found",
                {
                    "Expected": str(awk_script.absolute()),
                    "Solution": "Ensure awk_markdown_processor.py is in project root",
                },
            )

        if not awk_config.exists():
            return ValidationResult(
                False,
                "AWK processor config not found",
                {
                    "Expected": str(awk_config.absolute()),
                    "Solution": "Ensure markdown_processing_config.yaml is in project root",
                },
            )

        return ValidationResult(True, "AWK processor available")

    def run_all_validations(self, config) -> Tuple[bool, List[ValidationResult]]:
        """Run all validation checks.

        Args:
            config: UnifiedConfig instance to validate

        Returns:
            Tuple of (all_passed, results_list)
        """
        results = []

        # System requirements
        results.append(self.validate_system_requirements())

        # GPU memory (for V100 target)
        results.append(self.validate_gpu_memory(16.0))

        # Dependencies
        results.append(self.validate_dependencies())

        # AWK processor (critical for extraction)
        results.append(self.validate_awk_processor())

        # Model-specific validations
        model_paths = [cfg.path for cfg in config.models.values()]

        # Disk space
        results.append(self.validate_disk_space(model_paths, 50.0))

        # Each model's files
        for model_name, model_cfg in config.models.items():
            results.append(self.validate_model_files(model_cfg.path, model_name))

        # Check if all passed
        all_passed = all(r.passed for r in results)

        return all_passed, results

    def print_validation_summary(self, results: List[ValidationResult]):
        """Print a summary of validation results."""
        console.print("\n📋 Configuration Validation Summary", style="bold")
        console.print("=" * 50)

        passed_count = sum(1 for r in results if r.passed)
        total_count = len(results)

        # Print each result
        for result in results:
            result.print_result()

        # Summary
        console.print("\n" + "=" * 50)
        if passed_count == total_count:
            console.print(
                f"✅ All {total_count} validation checks passed!", style="bold green"
            )
        else:
            console.print(
                f"❌ {total_count - passed_count} of {total_count} validation checks failed",
                style="bold red",
            )
            console.print("💡 Fix the issues above before proceeding", style="yellow")


def run_pre_flight_checks(config) -> bool:
    """Run pre-flight checks and return success status.

    Args:
        config: UnifiedConfig instance

    Returns:
        True if all checks pass, False otherwise
    """
    validator = ConfigValidator()
    all_passed, results = validator.run_all_validations(config)
    validator.print_validation_summary(results)

    if not all_passed:
        console.print("\n🚫 Pre-flight checks failed. Exiting.", style="bold red")

    return all_passed
