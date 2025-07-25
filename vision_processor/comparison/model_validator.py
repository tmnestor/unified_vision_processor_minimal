"""Model Validator for Production Environment
========================================

Validates model loading, basic inference capabilities, and environment
compatibility before running full comparison.
"""

import gc
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image

from ..config import ConfigManager
from ..config.model_registry import ModelRegistry
from ..models.base_model import BaseVisionModel, DeviceConfig


class ModelValidator:
    """Validates models for production comparison pipeline."""

    def __init__(self, model_registry: ModelRegistry):
        """Initialize model validator.

        Args:
            model_registry: Model registry instance
        """
        self.model_registry = model_registry
        self.validation_results: Dict[str, Dict] = {}

    def validate_model_loading(
        self, model_name: str, model_path: str, processing_config: ConfigManager
    ) -> bool:
        """Validate that a model can be loaded successfully.

        Args:
            model_name: Name of the model
            model_path: Path to model files
            processing_config: Processing configuration

        Returns:
            True if model loads successfully, False otherwise
        """
        validation_result = {
            "model_name": model_name,
            "model_path": model_path,
            "start_time": time.time(),
            "success": False,
            "error_message": None,
            "load_time": 0.0,
            "memory_usage": 0.0,
            "inference_test": False,
            "capabilities_verified": False,
        }

        try:
            # Step 1: Check model registration
            registration = self.model_registry.get_model(model_name)
            if not registration:
                validation_result["error_message"] = (
                    f"Model not registered: {model_name}"
                )
                self.validation_results[model_name] = validation_result
                return False

            # Step 2: Check model path exists
            if not Path(model_path).exists():
                validation_result["error_message"] = (
                    f"Model path does not exist: {model_path}"
                )
                self.validation_results[model_name] = validation_result
                return False

            # Step 3: Attempt model loading
            load_start = time.time()

            # Get initial memory state
            initial_memory = self._get_memory_usage()

            model = self.model_registry.create_model(
                model_name,
                model_path=model_path,
                device_config=DeviceConfig.AUTO,
            )

            # Load the model
            model.load_model()
            load_time = time.time() - load_start

            # Check memory usage
            current_memory = self._get_memory_usage()
            memory_usage = current_memory - initial_memory

            validation_result["load_time"] = load_time
            validation_result["memory_usage"] = memory_usage

            # Step 4: Verify model capabilities
            capabilities = model.capabilities
            if capabilities:
                validation_result["capabilities_verified"] = True
                validation_result["capabilities"] = {
                    "supports_quantization": capabilities.supports_quantization,
                    "supports_multi_gpu": capabilities.supports_multi_gpu,
                    "memory_efficient": capabilities.memory_efficient,
                    "max_image_size": capabilities.max_image_size,
                }

            # Step 5: Basic inference test
            inference_success = self._test_basic_inference(model)
            validation_result["inference_test"] = inference_success

            # Clean up model
            model.unload_model()
            del model

            # Success if all checks pass
            validation_result["success"] = (
                inference_success and capabilities is not None
            )

        except Exception as e:
            validation_result["error_message"] = str(e)
            validation_result["success"] = False

        finally:
            # Cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            validation_result["total_time"] = (
                time.time() - validation_result["start_time"]
            )
            self.validation_results[model_name] = validation_result

        return validation_result["success"]

    def _get_memory_usage(self) -> float:
        """Get current GPU memory usage in GB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**3
        return 0.0

    def _test_basic_inference(self, model: BaseVisionModel) -> bool:
        """Test basic inference capability.

        Args:
            model: Loaded model instance

        Returns:
            True if basic inference works, False otherwise
        """
        try:
            # Create a simple test image (white 224x224 image)
            test_image = Image.new("RGB", (224, 224), color="white")

            # Simple test prompt
            test_prompt = "What do you see in this image?"

            # Run inference with short timeout
            response = model.process_image(test_image, test_prompt, max_new_tokens=32)

            # Check if we got a valid response
            if response and response.raw_text and len(response.raw_text.strip()) > 0:
                return True

        except Exception:
            pass  # Inference test failed

        return False

    def validate_all_models(
        self,
        model_names: List[str],
        model_paths: Dict[str, str],
        processing_config: ConfigManager,
    ) -> Tuple[List[str], List[str]]:
        """Validate multiple models.

        Args:
            model_names: List of model names to validate
            model_paths: Dictionary mapping model names to paths
            processing_config: Processing configuration

        Returns:
            Tuple of (valid_models, invalid_models)
        """
        valid_models = []
        invalid_models = []

        for model_name in model_names:
            model_path = model_paths.get(model_name)
            if not model_path:
                invalid_models.append(model_name)
                continue

            if self.validate_model_loading(model_name, model_path, processing_config):
                valid_models.append(model_name)
            else:
                invalid_models.append(model_name)

        return valid_models, invalid_models

    def get_validation_results(self) -> Dict[str, Dict]:
        """Get detailed validation results for all tested models.

        Returns:
            Dictionary with validation results
        """
        return self.validation_results.copy()

    def get_validation_summary(self) -> Dict[str, any]:
        """Get summary of validation results.

        Returns:
            Dictionary with validation summary
        """
        if not self.validation_results:
            return {"error": "No validation results available"}

        total_models = len(self.validation_results)
        successful_models = sum(
            1 for r in self.validation_results.values() if r["success"]
        )

        # Calculate average metrics for successful models
        successful_results = [
            r for r in self.validation_results.values() if r["success"]
        ]

        avg_load_time = 0.0
        avg_memory_usage = 0.0

        if successful_results:
            avg_load_time = sum(r["load_time"] for r in successful_results) / len(
                successful_results
            )
            avg_memory_usage = sum(r["memory_usage"] for r in successful_results) / len(
                successful_results
            )

        # Identify fastest and slowest models
        fastest_model = None
        slowest_model = None

        if successful_results:
            fastest_model = min(successful_results, key=lambda x: x["load_time"])[
                "model_name"
            ]
            slowest_model = max(successful_results, key=lambda x: x["load_time"])[
                "model_name"
            ]

        # Get error summary
        failed_models = {
            r["model_name"]: r["error_message"]
            for r in self.validation_results.values()
            if not r["success"]
        }

        return {
            "total_models_tested": total_models,
            "successful_validations": successful_models,
            "validation_success_rate": successful_models / total_models
            if total_models > 0
            else 0,
            "average_load_time": avg_load_time,
            "average_memory_usage": avg_memory_usage,
            "fastest_model": fastest_model,
            "slowest_model": slowest_model,
            "failed_models": failed_models,
            "recommendations": self._generate_recommendations(),
        }

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on validation results.

        Returns:
            List of recommendation strings
        """
        recommendations = []

        if not self.validation_results:
            recommendations.append("No validation data available")
            return recommendations

        successful_results = [
            r for r in self.validation_results.values() if r["success"]
        ]
        failed_results = [
            r for r in self.validation_results.values() if not r["success"]
        ]

        # Memory recommendations
        if successful_results:
            max_memory = max(r["memory_usage"] for r in successful_results)
            if max_memory > 12.0:  # >12GB
                recommendations.append(
                    f"High memory usage detected ({max_memory:.1f}GB). Consider enabling quantization."
                )

        # Load time recommendations
        if successful_results:
            max_load_time = max(r["load_time"] for r in successful_results)
            if max_load_time > 60.0:  # >60 seconds
                recommendations.append(
                    f"Slow model loading detected ({max_load_time:.1f}s). Consider optimizing model storage."
                )

        # Failure analysis
        if failed_results:
            common_errors = {}
            for result in failed_results:
                error = result["error_message"]
                if error:
                    error_type = error.split(":")[0]  # Get error type
                    common_errors[error_type] = common_errors.get(error_type, 0) + 1

            for error_type, count in common_errors.items():
                if count > 1:
                    recommendations.append(
                        f"Multiple models failed with {error_type} errors"
                    )

        # Success rate recommendations
        total_models = len(self.validation_results)
        success_rate = len(successful_results) / total_models if total_models > 0 else 0

        if success_rate < 0.5:
            recommendations.append(
                "Low model validation success rate. Check environment setup."
            )
        elif success_rate == 1.0:
            recommendations.append(
                "All models validated successfully. Ready for production comparison."
            )

        if not recommendations:
            recommendations.append(
                "Model validation completed successfully with no issues detected."
            )

        return recommendations

    def print_validation_report(self):
        """Print detailed validation report."""
        print("🔍 MODEL VALIDATION REPORT")
        print("=" * 50)

        summary = self.get_validation_summary()

        print(f"Total Models Tested: {summary['total_models_tested']}")
        print(f"Successful Validations: {summary['successful_validations']}")
        print(f"Success Rate: {summary['validation_success_rate']:.1%}")
        print()

        if summary["successful_validations"] > 0:
            print(f"Average Load Time: {summary['average_load_time']:.1f}s")
            print(f"Average Memory Usage: {summary['average_memory_usage']:.1f}GB")

            if summary["fastest_model"]:
                print(f"Fastest Model: {summary['fastest_model']}")
            if summary["slowest_model"]:
                print(f"Slowest Model: {summary['slowest_model']}")
            print()

        # Failed models
        if summary["failed_models"]:
            print("❌ Failed Models:")
            for model, error in summary["failed_models"].items():
                print(f"   {model}: {error}")
            print()

        # Recommendations
        print("💡 Recommendations:")
        for rec in summary["recommendations"]:
            print(f"   • {rec}")
        print()

        # Detailed results
        print("📋 Detailed Results:")
        for model_name, result in self.validation_results.items():
            status = "✅" if result["success"] else "❌"
            print(f"   {status} {model_name}:")
            print(f"      Load Time: {result['load_time']:.1f}s")
            print(f"      Memory Usage: {result['memory_usage']:.1f}GB")
            print(f"      Inference Test: {'✅' if result['inference_test'] else '❌'}")

            if not result["success"] and result["error_message"]:
                print(f"      Error: {result['error_message']}")
            print()
