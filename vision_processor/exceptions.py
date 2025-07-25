"""Custom exceptions for Vision Processor.

Provides a hierarchy of specific exceptions for better error handling
and debugging across the vision processing pipeline.
"""

from pathlib import Path
from typing import Any, Optional


class VisionProcessorError(Exception):
    """Base exception for all vision processor errors."""

    def __init__(self, message: str, details: Optional[dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ConfigurationError(VisionProcessorError):
    """Raised when configuration is invalid or missing."""

    def __init__(self, message: str, config_path: Optional[Path] = None, **kwargs):
        details = {"config_path": str(config_path)} if config_path else {}
        details.update(kwargs)
        super().__init__(message, details)


class ModelNotFoundError(VisionProcessorError):
    """Raised when a model cannot be found at the specified path."""

    def __init__(self, model_name: str, model_path: Path, **kwargs):
        message = f"Model '{model_name}' not found at: {model_path}"
        details = {"model_name": model_name, "model_path": str(model_path)}
        details.update(kwargs)
        super().__init__(message, details)


class ModelLoadError(VisionProcessorError):
    """Raised when a model fails to load."""

    def __init__(self, model_name: str, original_error: Exception, **kwargs):
        message = f"Failed to load model '{model_name}': {str(original_error)}"
        details = {
            "model_name": model_name,
            "original_error": str(original_error),
            "error_type": type(original_error).__name__,
        }
        details.update(kwargs)
        super().__init__(message, details)


class ModelInferenceError(VisionProcessorError):
    """Raised when model inference fails."""

    def __init__(
        self,
        model_name: str,
        image_path: Optional[Path] = None,
        original_error: Optional[Exception] = None,
        **kwargs,
    ):
        message = f"Inference failed for model '{model_name}'"
        if image_path:
            message += f" on image: {image_path}"
        if original_error:
            message += f" - {str(original_error)}"

        details = {"model_name": model_name}
        if image_path:
            details["image_path"] = str(image_path)
        if original_error:
            details["original_error"] = str(original_error)
            details["error_type"] = type(original_error).__name__
        details.update(kwargs)
        super().__init__(message, details)


class ImageProcessingError(VisionProcessorError):
    """Raised when image processing fails."""

    def __init__(
        self,
        image_path: Path,
        operation: str,
        original_error: Optional[Exception] = None,
        **kwargs,
    ):
        message = f"Failed to {operation} image: {image_path}"
        if original_error:
            message += f" - {str(original_error)}"

        details = {"image_path": str(image_path), "operation": operation}
        if original_error:
            details["original_error"] = str(original_error)
            details["error_type"] = type(original_error).__name__
        details.update(kwargs)
        super().__init__(message, details)


class InvalidImageError(ImageProcessingError):
    """Raised when an image is invalid or unsupported."""

    def __init__(self, image_path: Path, reason: str, **kwargs):
        super().__init__(image_path, f"validate (reason: {reason})", **kwargs)
        self.details["reason"] = reason


class ExtractionError(VisionProcessorError):
    """Raised when field extraction fails."""

    def __init__(self, message: str, extracted_text: Optional[str] = None, **kwargs):
        details = {}
        if extracted_text:
            details["extracted_text_preview"] = (
                extracted_text[:200] + "..."
                if len(extracted_text) > 200
                else extracted_text
            )
        details.update(kwargs)
        super().__init__(message, details)


class ValidationError(VisionProcessorError):
    """Raised when validation fails."""

    def __init__(self, field: str, value: Any, reason: str, **kwargs):
        message = f"Validation failed for field '{field}': {reason}"
        details = {"field": field, "value": str(value), "reason": reason}
        details.update(kwargs)
        super().__init__(message, details)


class MemoryError(VisionProcessorError):
    """Raised when memory constraints are exceeded."""

    def __init__(
        self,
        required_mb: float,
        available_mb: float,
        operation: str = "model loading",
        **kwargs,
    ):
        message = (
            f"Insufficient memory for {operation}: "
            f"required {required_mb:.1f}MB, available {available_mb:.1f}MB"
        )
        details = {
            "required_mb": required_mb,
            "available_mb": available_mb,
            "operation": operation,
        }
        details.update(kwargs)
        super().__init__(message, details)


class DeviceError(VisionProcessorError):
    """Raised when device setup or selection fails."""

    def __init__(self, device: str, reason: str, **kwargs):
        message = f"Device '{device}' error: {reason}"
        details = {"device": device, "reason": reason}
        details.update(kwargs)
        super().__init__(message, details)


class CLIError(VisionProcessorError):
    """Raised for CLI-specific errors."""

    def __init__(self, command: str, message: str, **kwargs):
        full_message = f"CLI command '{command}' error: {message}"
        details = {"command": command}
        details.update(kwargs)
        super().__init__(full_message, details)


class BatchProcessingError(VisionProcessorError):
    """Raised when batch processing encounters errors."""

    def __init__(
        self,
        total_files: int,
        failed_files: int,
        failed_paths: Optional[list[Path]] = None,
        **kwargs,
    ):
        message = f"Batch processing failed for {failed_files}/{total_files} files"
        details: dict[str, Any] = {
            "total_files": total_files,
            "failed_files": failed_files,
        }
        if failed_paths:
            details["failed_paths"] = [str(p) for p in failed_paths[:10]]  # Limit to 10
            if len(failed_paths) > 10:
                details["additional_failures"] = len(failed_paths) - 10
        details.update(kwargs)
        super().__init__(message, details)
