"""Production Configuration for Model Comparison
===========================================

Enhanced configuration system that extends the existing SimpleConfig with
production-ready features, type safety, and integration with the production
field schema.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .production_schema import PRODUCTION_SCHEMA, FieldCategory
from .simple_config import SimpleConfig


@dataclass
class ModelPathConfig:
    """Model path configuration with validation."""

    llama: str = "/home/jovyan/nfs_share/models/Llama-3.2-11B-Vision-Instruct"
    internvl: str = "/home/jovyan/nfs_share/models/InternVL3-8B"

    def __post_init__(self):
        """Validate model paths."""
        for model_name, path in self.__dict__.items():
            if not Path(path).exists():
                print(f"⚠️  Model path does not exist: {model_name} -> {path}")


@dataclass
class DeviceMapConfig:
    """Device mapping configuration for a specific model."""

    strategy: str = "single_gpu"  # single_gpu, multi_gpu, auto
    device_map: Any = field(default_factory=lambda: {"": 0})  # PyTorch device_map
    quantization_compatible: bool = True


@dataclass
class DeviceConfig:
    """Device configuration for V100 production deployment."""

    gpu_strategy: str = "single_gpu"  # single_gpu, multi_gpu, auto
    target_gpu: int = 0  # Which GPU to use for single_gpu mode
    v100_mode: bool = True  # Enable V100 production optimizations
    memory_limit_gb: int = 16  # GPU memory limit

    # Per-model device mapping configurations
    device_maps: Dict[str, DeviceMapConfig] = field(default_factory=dict)

    def get_device_map_for_model(self, model_name: str) -> Any:
        """Get device map configuration for a specific model."""
        if model_name in self.device_maps:
            return self.device_maps[model_name].device_map

        # Default single GPU configuration
        if self.gpu_strategy == "single_gpu":
            return {"": self.target_gpu}
        elif self.gpu_strategy == "auto":
            return "auto"
        else:
            return None

    def should_use_quantization_safe_loading(self, model_name: str) -> bool:
        """Check if model should use quantization-safe loading."""
        if model_name in self.device_maps:
            return self.device_maps[model_name].quantization_compatible
        return True  # Default to safe loading


@dataclass
class ProcessingConfig:
    """Processing configuration with defaults optimized for V100."""

    max_tokens: int = 256
    quantization: bool = True
    trust_remote_code: bool = True
    batch_size: int = 1
    memory_limit_mb: int = 15360  # 15GB for V100
    enable_gradient_checkpointing: bool = True
    use_flash_attention: bool = True

    def validate(self) -> bool:
        """Validate processing configuration."""
        if self.max_tokens < 32 or self.max_tokens > 2048:
            print(f"⚠️  Max tokens should be between 32-2048, got: {self.max_tokens}")
            return False

        if self.memory_limit_mb < 1024:
            print(f"⚠️  Memory limit seems low: {self.memory_limit_mb}MB")

        if self.batch_size < 1 or self.batch_size > 32:
            print(f"⚠️  Batch size should be between 1-32, got: {self.batch_size}")
            return False

        return True


@dataclass
class ExtractionConfig:
    """Field extraction configuration."""

    # Success criteria based on production schema
    min_core_fields: int = field(default_factory=lambda: len(PRODUCTION_SCHEMA.get_core_fields()) // 3)
    min_total_fields: int = 3

    # Field categories to prioritize
    priority_categories: List[FieldCategory] = field(
        default_factory=lambda: [FieldCategory.SUPPLIER, FieldCategory.FINANCIAL, FieldCategory.TEMPORAL]
    )

    # Extraction strategies
    enable_fallback_patterns: bool = True
    enable_raw_markdown_fallback: bool = True
    strict_validation: bool = False

    def get_success_criteria(self) -> Dict[str, Any]:
        """Get success criteria for extraction."""
        return {
            "min_core_fields": self.min_core_fields,
            "min_total_fields": self.min_total_fields,
            "required_fields": PRODUCTION_SCHEMA.get_required_fields(),
            "core_fields": PRODUCTION_SCHEMA.get_core_fields(),
            "priority_categories": [cat.value for cat in self.priority_categories],
        }


@dataclass
class AnalysisConfig:
    """Analysis and reporting configuration."""

    # Output formats
    generate_charts: bool = True
    generate_csv: bool = True
    generate_json: bool = True
    chart_dpi: int = 300

    # Analysis features
    calculate_f1_scores: bool = True
    generate_field_analysis: bool = True
    generate_performance_metrics: bool = True

    # Visualization settings
    chart_style: str = "default"
    color_palette: str = "husl"
    figure_size: tuple = (20, 15)


class ProductionConfig:
    """Production-ready configuration system extending SimpleConfig."""

    def __init__(self, config_file: Optional[str] = None, env_file: Optional[str] = None, **overrides):
        """Initialize production configuration.

        Args:
            config_file: Path to YAML configuration file
            env_file: Path to .env file for environment variables
            **overrides: Configuration overrides
        """
        # Initialize base configuration
        self.simple_config = SimpleConfig(env_file)

        # Load YAML configuration if provided
        self.yaml_config = {}
        if config_file and Path(config_file).exists():
            self.yaml_config = self._load_yaml_config(config_file)

        # Initialize configuration sections
        self.model_paths = self._init_model_paths(overrides)
        self.processing = self._init_processing_config(overrides)
        self.extraction = self._init_extraction_config(overrides)
        self.analysis = self._init_analysis_config(overrides)
        self.device_config = self._init_device_config(overrides)

        # Dataset and output paths
        self.datasets_path = self._get_config_value(
            "datasets_path", overrides, self.yaml_config.get("defaults", {}), "datasets"
        )
        self.output_dir = self._get_config_value(
            "output_dir", overrides, self.yaml_config.get("defaults", {}), "results"
        )

        # Models to test
        models_str = self._get_config_value(
            "models", overrides, self.yaml_config.get("defaults", {}), "llama,internvl"
        )
        self.models = [m.strip() for m in models_str.split(",")]

        # Production schema integration
        self.field_schema = PRODUCTION_SCHEMA

    def _load_yaml_config(self, config_file: str) -> Dict[str, Any]:
        """Load YAML configuration file."""
        try:
            with Path(config_file).open("r") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"⚠️  Failed to load YAML config from {config_file}: {e}")
            return {}

    def _get_config_value(self, key: str, overrides: Dict, yaml_defaults: Dict, fallback: Any) -> Any:
        """Get configuration value with priority: overrides > YAML > fallback."""
        return overrides.get(key) or yaml_defaults.get(key) or fallback

    def _init_model_paths(self, overrides: Dict) -> ModelPathConfig:
        """Initialize model path configuration."""
        yaml_paths = self.yaml_config.get("model_paths", {})

        return ModelPathConfig(
            llama=overrides.get("llama_path") or yaml_paths.get("llama") or ModelPathConfig.llama,
            internvl=overrides.get("internvl_path")
            or yaml_paths.get("internvl")
            or ModelPathConfig.internvl,
        )

    def _init_processing_config(self, overrides: Dict) -> ProcessingConfig:
        """Initialize processing configuration."""
        yaml_defaults = self.yaml_config.get("defaults", {})

        return ProcessingConfig(
            max_tokens=self._get_config_value("max_tokens", overrides, yaml_defaults, 256),
            quantization=self._get_config_value("quantization", overrides, yaml_defaults, True),
            trust_remote_code=self._get_config_value("trust_remote_code", overrides, yaml_defaults, True),
            batch_size=self._get_config_value("batch_size", overrides, yaml_defaults, 1),
            memory_limit_mb=getattr(self.simple_config, "memory_limit_mb", 15360),
            enable_gradient_checkpointing=getattr(
                self.simple_config, "enable_gradient_checkpointing", True
            ),
            use_flash_attention=getattr(self.simple_config, "use_flash_attention", True),
        )

    def _init_extraction_config(self, overrides: Dict) -> ExtractionConfig:
        """Initialize extraction configuration."""
        yaml_criteria = self.yaml_config.get("extraction", {})

        return ExtractionConfig(
            min_core_fields=overrides.get("min_core_fields")
            or yaml_criteria.get("min_core_fields")
            or len(PRODUCTION_SCHEMA.get_core_fields()) // 3,
            min_total_fields=overrides.get("min_total_fields")
            or yaml_criteria.get("min_total_fields")
            or 3,
            enable_fallback_patterns=yaml_criteria.get("enable_fallback_patterns", True),
            enable_raw_markdown_fallback=yaml_criteria.get("enable_raw_markdown_fallback", True),
            strict_validation=overrides.get("strict_validation")
            or yaml_criteria.get("strict_validation", False),
        )

    def _init_analysis_config(self, overrides: Dict) -> AnalysisConfig:
        """Initialize analysis configuration."""
        yaml_analysis = self.yaml_config.get("analysis", {})

        return AnalysisConfig(
            generate_charts=yaml_analysis.get("generate_charts", True),
            generate_csv=yaml_analysis.get("generate_csv", True),
            generate_json=yaml_analysis.get("generate_json", True),
            chart_dpi=yaml_analysis.get("chart_dpi", 300),
            calculate_f1_scores=yaml_analysis.get("calculate_f1_scores", True),
            generate_field_analysis=yaml_analysis.get("generate_field_analysis", True),
            generate_performance_metrics=yaml_analysis.get("generate_performance_metrics", True),
        )

    def _init_device_config(self, overrides: Dict) -> DeviceConfig:
        """Initialize device configuration from YAML."""
        yaml_device = self.yaml_config.get("device_config", {})

        # Parse device maps for each model
        device_maps = {}
        yaml_device_maps = yaml_device.get("device_maps", {})

        for model_name, device_map_config in yaml_device_maps.items():
            device_maps[model_name] = DeviceMapConfig(
                strategy=device_map_config.get("strategy", "single_gpu"),
                device_map=device_map_config.get("device_map", {"": 0}),
                quantization_compatible=device_map_config.get("quantization_compatible", True),
            )

        return DeviceConfig(
            gpu_strategy=yaml_device.get("gpu_strategy", "single_gpu"),
            target_gpu=yaml_device.get("target_gpu", 0),
            v100_mode=yaml_device.get("v100_mode", True),
            memory_limit_gb=yaml_device.get("memory_limit_gb", 16),
            device_maps=device_maps,
        )

    def get_prompts(self) -> Dict[str, str]:
        """Get model-specific prompts with production field integration."""
        base_prompts = self.yaml_config.get("prompts", {})

        # Get production field list for prompt generation
        production_fields = PRODUCTION_SCHEMA.get_all_fields()

        # Default prompts with production schema
        default_prompt_template = self._create_production_prompt_template(production_fields)

        return {
            "internvl": base_prompts.get("internvl", default_prompt_template),
            "llama": base_prompts.get("llama", default_prompt_template),
        }

    def _create_production_prompt_template(self, fields: List[str]) -> str:
        """Create prompt template using production fields."""
        # Group fields by category for better prompt organization
        categorized_fields = {}
        for field_name in fields:
            definition = PRODUCTION_SCHEMA.get_field_definition(field_name)
            if definition:
                category = definition.category.value
                if category not in categorized_fields:
                    categorized_fields[category] = []
                categorized_fields[category].append(field_name)

        prompt_parts = [
            "Extract data from this image in KEY-VALUE format.",
            "",
            "Required output format (extract any visible fields):",
        ]

        # Add fields grouped by category
        for category, category_fields in categorized_fields.items():
            if category_fields:
                prompt_parts.append(f"\n# {category.upper()} FIELDS:")
                for field in sorted(category_fields):
                    definition = PRODUCTION_SCHEMA.get_field_definition(field)
                    example = (
                        definition.format_example
                        if definition and definition.format_example
                        else "[value if visible]"
                    )
                    prompt_parts.append(f"{field.upper()}: {example}")

        prompt_parts.extend(
            [
                "",
                "Extract all visible text and format as KEY: VALUE pairs only.",
                "Use 'N/A' for fields that are not visible or not applicable.",
            ]
        )

        return "\n".join(prompt_parts)

    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get complete configuration for a specific model."""
        return {
            "model_path": getattr(self.model_paths, model_name),
            "processing": self.processing.__dict__,
            "extraction": self.extraction.__dict__,
            "field_schema": self.field_schema.create_universal_parser_schema(),
        }

    def validate(self) -> bool:
        """Validate entire configuration."""
        valid = True

        # Validate processing config
        if not self.processing.validate():
            valid = False

        # Validate model paths
        for model in self.models:
            if not hasattr(self.model_paths, model):
                print(f"❌ Unknown model: {model}")
                valid = False
                continue

            model_path = getattr(self.model_paths, model)
            if not Path(model_path).exists():
                print(f"❌ Model path does not exist: {model} -> {model_path}")
                valid = False

        # Validate paths
        if not Path(self.datasets_path).exists():
            print(f"⚠️  Datasets path does not exist: {self.datasets_path}")

        return valid

    def print_configuration(self):
        """Print comprehensive configuration summary."""
        print("🔧 Production Model Comparison Configuration:")
        print(f"  Models: {', '.join(self.models)}")
        print(f"  Datasets: {self.datasets_path}")
        print(f"  Output: {self.output_dir}")
        print()

        print("📁 Model Paths:")
        for model in self.models:
            if hasattr(self.model_paths, model):
                path = getattr(self.model_paths, model)
                status = "✅" if Path(path).exists() else "❌"
                print(f"  {model}: {status} {path}")
        print()

        print("⚙️  Processing Configuration:")
        print(f"  Max Tokens: {self.processing.max_tokens}")
        print(f"  Quantization: {self.processing.quantization}")
        print(f"  Trust Remote Code: {self.processing.trust_remote_code}")
        print(f"  Memory Limit: {self.processing.memory_limit_mb}MB")
        print()

        print("🎯 Extraction Configuration:")
        print(f"  Min Core Fields: {self.extraction.min_core_fields}")
        print(f"  Min Total Fields: {self.extraction.min_total_fields}")
        print(f"  Strict Validation: {self.extraction.strict_validation}")
        print()

        print("📊 Analysis Configuration:")
        print(f"  Generate Charts: {self.analysis.generate_charts}")
        print(f"  Calculate F1 Scores: {self.analysis.calculate_f1_scores}")
        print()

        print("🏷️  Production Schema:")
        print(f"  Total Fields: {len(PRODUCTION_SCHEMA.get_all_fields())}")
        print(f"  Core Fields: {len(PRODUCTION_SCHEMA.get_core_fields())}")
        print(f"  Required Fields: {len(PRODUCTION_SCHEMA.get_required_fields())}")

        # Show field distribution by category
        print("  Field Distribution:")
        for category in FieldCategory:
            fields_in_category = PRODUCTION_SCHEMA.get_fields_by_category(category)
            if fields_in_category:
                print(f"    {category.value}: {len(fields_in_category)} fields")

    def export_config(self, output_path: str):
        """Export configuration to YAML file."""
        config_dict = {
            "models": self.models,
            "datasets_path": str(self.datasets_path),
            "output_dir": str(self.output_dir),
            "model_paths": self.model_paths.__dict__,
            "processing": self.processing.__dict__,
            "extraction": self.extraction.__dict__,
            "analysis": self.analysis.__dict__,
        }

        with Path(output_path).open("w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)

        print(f"📁 Configuration exported to: {output_path}")


def create_production_config(config_file: str = "model_comparison.yaml", **overrides) -> ProductionConfig:
    """Factory function to create production configuration.

    Args:
        config_file: Path to YAML configuration file
        **overrides: Configuration overrides

    Returns:
        ProductionConfig instance
    """
    return ProductionConfig(config_file=config_file, **overrides)
