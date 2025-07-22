"""Unified Configuration System with Fail-Fast Validation
====================================================

Consolidates SimpleConfig and ProductionConfig into a single, hierarchical
configuration system with clear validation and no silent failures.

Follows CLAUDE.md fail-fast principles:
- Explicit validation at startup
- No silent fallbacks
- Clear error messages with remediation steps
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv
from rich.console import Console

# Production schema legacy imports
from .production_schema_legacy import PRODUCTION_SCHEMA, FieldCategory

console = Console()


class ConfigurationError(Exception):
    """Configuration error with diagnostic information."""

    def __init__(self, message: str, diagnostics: Dict[str, Any]):
        super().__init__(message)
        self.diagnostics = diagnostics

    def print_diagnostics(self):
        """Print detailed diagnostic information."""
        console.print(f"❌ FATAL: {self.args[0]}", style="bold red")
        for key, value in self.diagnostics.items():
            console.print(f"💡 {key}: {value}", style="yellow")


@dataclass
class ModelConfig:
    """Model-specific configuration with validation."""

    path: str
    device_map: Any = field(default_factory=lambda: {"": 0})
    quantization_enabled: bool = True
    trust_remote_code: bool = True
    max_memory: Dict[int, str] = field(default_factory=lambda: {0: "15GB"})

    def validate(self, model_name: str) -> None:
        """Validate model configuration - fail fast if invalid."""
        # Check model path exists
        if not Path(self.path).exists():
            raise ConfigurationError(
                f"Model path not found for {model_name}",
                diagnostics={
                    "Expected path": Path(self.path).absolute(),
                    "Current directory": Path.cwd(),
                    "Solution": f"Download {model_name} model or update path in config",
                },
            )

        # Validate device map
        if not isinstance(self.device_map, (dict, str)):
            raise ConfigurationError(
                f"Invalid device_map for {model_name}",
                diagnostics={
                    "Current type": type(self.device_map).__name__,
                    "Expected": "dict or 'auto'",
                    "Example": '{"": 0} for single GPU or "auto" for automatic',
                },
            )


@dataclass
class ProcessingConfig:
    """Processing configuration with V100 optimization."""

    max_tokens: int = 256
    batch_size: int = 1
    enable_gradient_checkpointing: bool = True
    use_flash_attention: bool = True
    memory_limit_mb: int = 15360  # 15GB for V100

    # Repetition control (ultra-aggressive for Llama)
    repetition_control_enabled: bool = True
    repetition_word_threshold: float = 0.15
    repetition_phrase_threshold: int = 2
    repetition_max_tokens_limit: int = 384

    def validate(self) -> None:
        """Validate processing configuration."""
        if not 32 <= self.max_tokens <= 2048:
            raise ConfigurationError(
                "Invalid max_tokens configuration",
                diagnostics={
                    "Current value": self.max_tokens,
                    "Valid range": "32-2048",
                    "Recommended": "256 for V100 optimization",
                },
            )

        if not 1 <= self.batch_size <= 32:
            raise ConfigurationError(
                "Invalid batch_size configuration",
                diagnostics={
                    "Current value": self.batch_size,
                    "Valid range": "1-32",
                    "Recommended": "1 for V100 with 16GB VRAM",
                },
            )

        if self.memory_limit_mb < 1024:
            raise ConfigurationError(
                "Memory limit too low",
                diagnostics={
                    "Current value": f"{self.memory_limit_mb}MB",
                    "Minimum": "1024MB",
                    "Recommended": "15360MB for V100",
                },
            )


@dataclass
class ExtractionConfig:
    """Extraction configuration with production schema integration."""

    min_core_fields: int = field(default_factory=lambda: len(PRODUCTION_SCHEMA.get_core_fields()) // 3)
    min_total_fields: int = 3
    enable_fallback_patterns: bool = True
    enable_raw_markdown_fallback: bool = True
    strict_validation: bool = False

    priority_categories: List[FieldCategory] = field(
        default_factory=lambda: [FieldCategory.SUPPLIER, FieldCategory.FINANCIAL, FieldCategory.TEMPORAL]
    )

    def validate(self) -> None:
        """Validate extraction configuration."""
        core_fields = PRODUCTION_SCHEMA.get_core_fields()

        if self.min_core_fields > len(core_fields):
            raise ConfigurationError(
                "Invalid min_core_fields",
                diagnostics={
                    "Current value": self.min_core_fields,
                    "Total core fields": len(core_fields),
                    "Solution": f"Set to at most {len(core_fields)}",
                },
            )

        if self.min_total_fields < 1:
            raise ConfigurationError(
                "Invalid min_total_fields",
                diagnostics={
                    "Current value": self.min_total_fields,
                    "Minimum": "1",
                    "Recommended": "3 for basic extraction",
                },
            )


@dataclass
class AnalysisConfig:
    """Analysis and output configuration."""

    output_format: str = "table"  # table, json, yaml
    generate_charts: bool = True
    generate_csv: bool = True
    chart_dpi: int = 300
    log_level: str = "INFO"

    def validate(self) -> None:
        """Validate analysis configuration."""
        valid_formats = {"table", "json", "yaml"}
        if self.output_format not in valid_formats:
            raise ConfigurationError(
                "Invalid output_format",
                diagnostics={
                    "Current value": self.output_format,
                    "Valid options": ", ".join(valid_formats),
                    "Default": "table",
                },
            )

        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.log_level.upper() not in valid_levels:
            raise ConfigurationError(
                "Invalid log_level",
                diagnostics={
                    "Current value": self.log_level,
                    "Valid options": ", ".join(valid_levels),
                    "Default": "INFO",
                },
            )


class UnifiedConfig:
    """Unified configuration with fail-fast validation."""

    def __init__(self, config_file: Optional[str] = None, env_file: Optional[str] = None):
        """Initialize configuration with validation.

        Args:
            config_file: Path to YAML configuration file
            env_file: Path to .env file for environment variables
        """
        # Load environment variables first
        if env_file and Path(env_file).exists():
            load_dotenv(env_file)
        else:
            load_dotenv()  # Try default .env

        # Initialize sub-configurations
        self.models: Dict[str, ModelConfig] = {}
        self.processing = ProcessingConfig()
        self.extraction = ExtractionConfig()
        self.analysis = AnalysisConfig()

        # Load configuration file if provided
        if config_file:
            self._load_config_file(config_file)
        else:
            # Load from environment variables
            self._load_from_env()

        # Always validate at initialization
        self.validate()

    def _load_config_file(self, config_file: str) -> None:
        """Load configuration from YAML file."""
        config_path = Path(config_file)

        if not config_path.exists():
            raise ConfigurationError(
                "Configuration file not found",
                diagnostics={
                    "Expected path": config_path.absolute(),
                    "Working directory": Path.cwd(),
                    "Solution": "Create config file or use --config-file to specify path",
                },
            )

        try:
            with config_path.open() as f:
                config_data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigurationError(
                "Invalid YAML configuration file",
                diagnostics={
                    "File": config_path,
                    "Error": str(e),
                    "Solution": "Fix YAML syntax errors in config file",
                },
            ) from None

        # Load model configurations
        if "models" in config_data:
            for model_name, model_cfg in config_data["models"].items():
                self.models[model_name] = ModelConfig(**model_cfg)

        # Load processing configuration
        if "processing" in config_data:
            self.processing = ProcessingConfig(**config_data["processing"])

        # Load extraction configuration
        if "extraction" in config_data:
            self.extraction = ExtractionConfig(**config_data["extraction"])

        # Load analysis configuration
        if "analysis" in config_data:
            self.analysis = AnalysisConfig(**config_data["analysis"])

    def _load_from_env(self) -> None:
        """Load configuration from environment variables."""
        # Model configurations from env
        model_type = os.getenv("VISION_MODEL_TYPE", "internvl3")

        # Model paths with specific lookups
        if model_type == "internvl3":
            model_path = os.getenv("VISION_INTERNVL_MODEL_PATH") or os.getenv(
                "VISION_MODEL_PATH", "/home/jovyan/nfs_share/models/InternVL3-8B"
            )
        elif model_type == "llama32_vision":
            model_path = os.getenv("VISION_LLAMA_MODEL_PATH") or os.getenv(
                "VISION_MODEL_PATH", "/home/jovyan/nfs_share/models/Llama-3.2-11B-Vision-Instruct"
            )
        else:
            model_path = os.getenv("VISION_MODEL_PATH", "/path/to/models")

        # Map model type to standard names
        model_name_map = {"internvl3": "internvl", "llama32_vision": "llama"}
        model_name = model_name_map.get(model_type, model_type)

        # Create model config
        self.models[model_name] = ModelConfig(
            path=model_path,
            quantization_enabled=os.getenv("VISION_ENABLE_QUANTIZATION", "false").lower() == "true",
            trust_remote_code=os.getenv("VISION_TRUST_REMOTE_CODE", "true").lower() == "true",
        )

        # Processing configuration from env
        self.processing.max_tokens = int(os.getenv("VISION_MAX_TOKENS", "256"))
        self.processing.memory_limit_mb = int(os.getenv("VISION_MEMORY_LIMIT_MB", "15360"))
        self.processing.enable_gradient_checkpointing = (
            os.getenv("VISION_ENABLE_GRADIENT_CHECKPOINTING", "true").lower() == "true"
        )
        self.processing.use_flash_attention = (
            os.getenv("VISION_USE_FLASH_ATTENTION", "true").lower() == "true"
        )

        # Repetition control from env
        self.processing.repetition_control_enabled = (
            os.getenv("VISION_REPETITION_CONTROL_ENABLED", "true").lower() == "true"
        )
        self.processing.repetition_word_threshold = float(
            os.getenv("VISION_REPETITION_WORD_THRESHOLD", "0.15")
        )
        self.processing.repetition_phrase_threshold = int(
            os.getenv("VISION_REPETITION_PHRASE_THRESHOLD", "2")
        )
        self.processing.repetition_max_tokens_limit = int(
            os.getenv("VISION_REPETITION_MAX_TOKENS_LIMIT", "384")
        )

        # Analysis configuration from env
        self.analysis.output_format = os.getenv("VISION_OUTPUT_FORMAT", "table")
        self.analysis.log_level = os.getenv("VISION_LOG_LEVEL", "INFO")

        # Set offline mode
        if os.getenv("VISION_OFFLINE_MODE", "true").lower() == "true":
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["HF_DATASETS_OFFLINE"] = "1"

    def validate(self) -> None:
        """Validate entire configuration - fail fast on any issues."""
        # Validate each model
        for model_name, model_config in self.models.items():
            model_config.validate(model_name)

        # Validate processing
        self.processing.validate()

        # Validate extraction
        self.extraction.validate()

        # Validate analysis
        self.analysis.validate()

        console.print("✅ Configuration validated successfully", style="green")

    def get_model_config(self, model_name: str) -> ModelConfig:
        """Get configuration for specific model."""
        if model_name not in self.models:
            available = ", ".join(self.models.keys())
            raise ConfigurationError(
                f"Model '{model_name}' not configured",
                diagnostics={
                    "Available models": available,
                    "Solution": f"Add {model_name} to configuration or use one of: {available}",
                },
            )
        return self.models[model_name]

    def print_configuration(self) -> None:
        """Print current configuration for debugging."""
        console.print("\n🔧 Unified Vision Processor Configuration:", style="bold")

        # Model configurations
        console.print("\n📦 Models:", style="bold blue")
        for model_name, model_cfg in self.models.items():
            console.print(f"  {model_name}:")
            console.print(f"    Path: {model_cfg.path}")
            console.print(f"    Quantization: {model_cfg.quantization_enabled}")
            console.print(f"    Device Map: {model_cfg.device_map}")

        # Processing configuration
        console.print("\n⚙️  Processing:", style="bold blue")
        console.print(f"  Max Tokens: {self.processing.max_tokens}")
        console.print(f"  Batch Size: {self.processing.batch_size}")
        console.print(f"  Memory Limit: {self.processing.memory_limit_mb}MB")
        console.print(f"  Gradient Checkpointing: {self.processing.enable_gradient_checkpointing}")

        # Repetition control
        console.print("\n🧹 Repetition Control:", style="bold blue")
        console.print(f"  Enabled: {self.processing.repetition_control_enabled}")
        console.print(f"  Word Threshold: {self.processing.repetition_word_threshold}")
        console.print(f"  Phrase Threshold: {self.processing.repetition_phrase_threshold}")

        # Extraction configuration
        console.print("\n📋 Extraction:", style="bold blue")
        console.print(f"  Min Core Fields: {self.extraction.min_core_fields}")
        console.print(f"  Min Total Fields: {self.extraction.min_total_fields}")
        console.print(f"  Fallback Patterns: {self.extraction.enable_fallback_patterns}")

        # Analysis configuration
        console.print("\n📊 Analysis:", style="bold blue")
        console.print(f"  Output Format: {self.analysis.output_format}")
        console.print(f"  Generate Charts: {self.analysis.generate_charts}")
        console.print(f"  Log Level: {self.analysis.log_level}")

    @classmethod
    def from_cli_args(cls, **kwargs) -> "UnifiedConfig":
        """Create configuration from CLI arguments.

        Args:
            **kwargs: CLI arguments that may override configuration

        Returns:
            Configured UnifiedConfig instance
        """
        # Get base configuration file path
        config_file = kwargs.get("config_file")
        env_file = kwargs.get("env_file")

        # Create base configuration
        config = cls(config_file=config_file, env_file=env_file)

        # Apply CLI overrides
        if "model" in kwargs and kwargs["model"]:
            # Map CLI model name to config name
            model_map = {
                "llama": "llama",
                "llama32_vision": "llama",
                "internvl": "internvl",
                "internvl3": "internvl",
            }
            requested_model = model_map.get(kwargs["model"], kwargs["model"])

            # Ensure requested model is configured
            if requested_model not in config.models:
                raise ConfigurationError(
                    f"Model '{requested_model}' not available",
                    diagnostics={
                        "Requested": kwargs["model"],
                        "Available": ", ".join(config.models.keys()),
                        "Solution": "Choose an available model or add to configuration",
                    },
                )

        if "output_format" in kwargs and kwargs["output_format"]:
            config.analysis.output_format = kwargs["output_format"]

        if "quantization" in kwargs and kwargs["quantization"] is not None:
            # Apply to all models
            for model_cfg in config.models.values():
                model_cfg.quantization_enabled = kwargs["quantization"]

        # Re-validate after overrides
        config.validate()

        return config
