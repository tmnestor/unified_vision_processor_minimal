"""System Constants for Vision Processor

Contains constants for model names, system limits, and other fixed values
that should not be in YAML configuration.
"""

from typing import Final


class ModelNames:
    """Model name constants for registry and configuration access."""

    LLAMA: Final[str] = "llama"
    INTERNVL: Final[str] = "internvl"

    # All registered model names
    ALL: Final[tuple[str, ...]] = (LLAMA, INTERNVL)


class SystemLimits:
    """Hardware and system-level constants that represent fixed limitations."""

    # GPU Memory thresholds for different hardware classes
    H200_MIN_MEMORY_GB: Final[float] = 70.0  # H200 development threshold
    V100_MIN_MEMORY_GB: Final[float] = 15.0  # V100 production threshold
    MODERATE_MEMORY_GB: Final[float] = 20.0  # Moderate GPU memory
    HIGH_MEMORY_GB: Final[float] = 40.0  # High GPU memory

    # Memory calculation constants
    BYTES_TO_MB: Final[int] = 1024 * 1024
    BYTES_TO_GB: Final[int] = 1024 * 1024 * 1024


class ProcessingDefaults:
    """Default values for processing that represent system capabilities."""

    # Test parameters
    TEST_MAX_TOKENS: Final[int] = 10
    TEST_IMAGE_SIZE: Final[tuple[int, int]] = (224, 224)

    # Response analysis
    RESPONSE_PREVIEW_LENGTH: Final[int] = 50
    LOG_TRUNCATE_LENGTH: Final[int] = 500
    MAX_LOG_CHARS: Final[int] = 1000


# Export commonly used constants at module level for convenience
MODEL_LLAMA = ModelNames.LLAMA
MODEL_INTERNVL = ModelNames.INTERNVL
