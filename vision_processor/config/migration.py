"""Configuration Migration Utilities
=================================

Helps migrate from SimpleConfig/ProductionConfig to UnifiedConfig.
"""

from pathlib import Path
from typing import Union

from rich.console import Console

from .production_config import ProductionConfig
from .simple_config import SimpleConfig
from .unified_config import UnifiedConfig

console = Console()


class ConfigMigrator:
    """Migrates old configuration to new UnifiedConfig."""

    @staticmethod
    def from_simple_config(_simple_config: SimpleConfig) -> UnifiedConfig:
        """Convert SimpleConfig to UnifiedConfig.

        Args:
            _simple_config: SimpleConfig instance (currently unused)

        Returns:
            Equivalent UnifiedConfig instance
        """
        # Create UnifiedConfig from environment (same as SimpleConfig)
        unified = UnifiedConfig(env_file=None)

        # The UnifiedConfig will load from the same environment variables
        # that SimpleConfig uses, so it should be equivalent

        console.print("✅ Migrated SimpleConfig to UnifiedConfig", style="green")
        return unified

    @staticmethod
    def from_production_config(prod_config: ProductionConfig) -> UnifiedConfig:
        """Convert ProductionConfig to UnifiedConfig.

        Args:
            prod_config: ProductionConfig instance

        Returns:
            Equivalent UnifiedConfig instance
        """
        # Create base UnifiedConfig
        unified = UnifiedConfig()

        # Migrate model configurations
        if hasattr(prod_config, "model_paths"):
            for model_name, path in prod_config.model_paths.items():
                if model_name in unified.models:
                    unified.models[model_name].path = path

        # Migrate device configurations
        if hasattr(prod_config, "device_config"):
            device_cfg = prod_config.device_config
            for model_name in unified.models:
                if hasattr(device_cfg, "get_device_map_for_model"):
                    device_map = device_cfg.get_device_map_for_model(model_name)
                    if device_map:
                        unified.models[model_name].device_map = device_map

        # Migrate processing configuration
        if hasattr(prod_config, "processing_config"):
            proc_cfg = prod_config.processing_config
            unified.processing.max_tokens = proc_cfg.max_tokens
            unified.processing.batch_size = proc_cfg.batch_size
            unified.processing.memory_limit_mb = proc_cfg.memory_limit_mb
            unified.processing.enable_gradient_checkpointing = (
                proc_cfg.enable_gradient_checkpointing
            )
            unified.processing.use_flash_attention = proc_cfg.use_flash_attention

        # Migrate extraction configuration
        if hasattr(prod_config, "extraction_config"):
            ext_cfg = prod_config.extraction_config
            unified.extraction.min_core_fields = ext_cfg.min_core_fields
            unified.extraction.min_total_fields = ext_cfg.min_total_fields
            unified.extraction.enable_fallback_patterns = (
                ext_cfg.enable_fallback_patterns
            )
            unified.extraction.enable_raw_markdown_fallback = (
                ext_cfg.enable_raw_markdown_fallback
            )
            unified.extraction.strict_validation = ext_cfg.strict_validation
            unified.extraction.priority_categories = ext_cfg.priority_categories

        # Migrate analysis configuration
        if hasattr(prod_config, "analysis_config"):
            ana_cfg = prod_config.analysis_config
            unified.analysis.generate_charts = ana_cfg.generate_charts
            unified.analysis.generate_csv = ana_cfg.generate_csv
            unified.analysis.chart_dpi = ana_cfg.chart_dpi

        # Re-validate the migrated configuration
        unified.validate()

        console.print("✅ Migrated ProductionConfig to UnifiedConfig", style="green")
        return unified

    @staticmethod
    def create_compatible_wrapper(
        config: Union[SimpleConfig, ProductionConfig, UnifiedConfig],
    ) -> UnifiedConfig:
        """Create a UnifiedConfig from any configuration type.

        This allows gradual migration by accepting any config type and
        returning a UnifiedConfig.

        Args:
            config: Any configuration type

        Returns:
            UnifiedConfig instance
        """
        if isinstance(config, UnifiedConfig):
            return config
        elif isinstance(config, SimpleConfig):
            return ConfigMigrator.from_simple_config(config)
        elif isinstance(config, ProductionConfig):
            return ConfigMigrator.from_production_config(config)
        else:
            raise ValueError(f"Unsupported configuration type: {type(config)}")


def migrate_config_file(
    old_config_path: str, new_config_path: str, config_type: str = "auto"
) -> None:
    """Migrate a configuration file to the new format.

    Args:
        old_config_path: Path to old configuration file
        new_config_path: Path where new configuration will be saved
        config_type: Type of old config ("simple", "production", or "auto")
    """
    import yaml

    old_path = Path(old_config_path)
    new_path = Path(new_config_path)

    if not old_path.exists():
        console.print(f"❌ Old config file not found: {old_path}", style="red")
        return

    # Load old configuration
    if config_type == "auto":
        # Try to detect type from content
        with old_path.open() as f:
            content = f.read()
            if "device_config:" in content and "processing_config:" in content:
                config_type = "production"
            else:
                config_type = "simple"

    console.print(f"📄 Detected config type: {config_type}", style="blue")

    # Create appropriate config object
    if config_type == "simple":
        old_config = SimpleConfig(str(old_path))
        unified = ConfigMigrator.from_simple_config(old_config)
    elif config_type == "production":
        old_config = ProductionConfig(config_file=str(old_path))
        unified = ConfigMigrator.from_production_config(old_config)
    else:
        console.print(f"❌ Unknown config type: {config_type}", style="red")
        return

    # Save new configuration
    config_dict = {
        "models": {
            name: {
                "path": cfg.path,
                "device_map": cfg.device_map,
                "quantization_enabled": cfg.quantization_enabled,
                "trust_remote_code": cfg.trust_remote_code,
                "max_memory": cfg.max_memory,
            }
            for name, cfg in unified.models.items()
        },
        "processing": {
            "max_tokens": unified.processing.max_tokens,
            "batch_size": unified.processing.batch_size,
            "enable_gradient_checkpointing": unified.processing.enable_gradient_checkpointing,
            "use_flash_attention": unified.processing.use_flash_attention,
            "memory_limit_mb": unified.processing.memory_limit_mb,
            "repetition_control_enabled": unified.processing.repetition_control_enabled,
            "repetition_word_threshold": unified.processing.repetition_word_threshold,
            "repetition_phrase_threshold": unified.processing.repetition_phrase_threshold,
            "repetition_max_tokens_limit": unified.processing.repetition_max_tokens_limit,
        },
        "extraction": {
            "min_core_fields": unified.extraction.min_core_fields,
            "min_total_fields": unified.extraction.min_total_fields,
            "enable_fallback_patterns": unified.extraction.enable_fallback_patterns,
            "enable_raw_markdown_fallback": unified.extraction.enable_raw_markdown_fallback,
            "strict_validation": unified.extraction.strict_validation,
            "priority_categories": [
                cat.value for cat in unified.extraction.priority_categories
            ],
        },
        "analysis": {
            "output_format": unified.analysis.output_format,
            "generate_charts": unified.analysis.generate_charts,
            "generate_csv": unified.analysis.generate_csv,
            "chart_dpi": unified.analysis.chart_dpi,
            "log_level": unified.analysis.log_level,
        },
    }

    with new_path.open("w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    console.print(f"✅ Migrated configuration saved to: {new_path}", style="green")
    console.print(
        "💡 Review the new configuration and adjust as needed", style="yellow"
    )
