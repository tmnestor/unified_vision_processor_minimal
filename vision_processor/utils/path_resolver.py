"""Path resolution utilities for consistent file path handling across CLI commands."""

from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..config import ConfigManager


class PathResolver:
    """Utility class for resolving relative paths against configured base paths."""

    def __init__(self, config: "ConfigManager"):
        """Initialize with configuration manager.

        Args:
            config: ConfigManager instance with base path configuration
        """
        self.config = config

    def resolve_input_path(self, path: Optional[str]) -> str:
        """Resolve input path against datasets_path base.

        Args:
            path: Input path (can be relative or absolute, or None to use config default)

        Returns:
            Resolved absolute path as string

        Raises:
            ValueError: If resolved path doesn't exist
        """
        if not path:
            # Use configured datasets_path as default
            resolved_path = Path(self.config.defaults.datasets_path)
        else:
            input_path = Path(path)
            if input_path.is_absolute():
                resolved_path = input_path
            else:
                # Relative path: resolve against configured datasets_path
                base_path = Path(self.config.defaults.datasets_path)
                resolved_path = base_path / path

        if not resolved_path.exists():
            raise ValueError(f"Path not found: {resolved_path}")

        return str(resolved_path)

    def resolve_output_path(
        self, path: Optional[str], filename: Optional[str] = None
    ) -> str:
        """Resolve output path against output_dir base.

        Args:
            path: Output path (can be relative or absolute, or None to use config default)
            filename: Optional filename to append to directory path

        Returns:
            Resolved absolute path as string
        """
        if not path:
            # Use configured output_dir as default
            resolved_path = Path(self.config.defaults.output_dir)
        else:
            output_path = Path(path)
            if output_path.is_absolute():
                resolved_path = output_path
            else:
                # Relative path: resolve against configured output_dir
                base_path = Path(self.config.defaults.output_dir)
                resolved_path = base_path / path

        # Append filename if provided
        if filename:
            resolved_path = resolved_path / filename

        return str(resolved_path)

    def ensure_output_dir(self, path: str) -> Path:
        """Ensure output directory exists and return Path object.

        Args:
            path: Directory path to create

        Returns:
            Path object for the created directory
        """
        output_dir = Path(path)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir


def create_path_resolver(yaml_file: str = "model_comparison.yaml") -> PathResolver:
    """Factory function to create PathResolver with configuration.

    Args:
        yaml_file: Path to YAML configuration file

    Returns:
        Configured PathResolver instance
    """
    from ..config import ConfigManager

    config = ConfigManager(yaml_file)
    return PathResolver(config)
