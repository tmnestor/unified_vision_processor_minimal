"""Unified logging system for Vision Processor.

Provides centralized logging with Rich console formatting, file logging,
and configurable verbosity levels to replace raw print statements.
"""

import logging

from rich.console import Console
from rich.logging import RichHandler


class VisionProcessorLogger:
    """Unified logging system for vision processor."""

    def __init__(self, config_manager=None):
        """Initialize logger with configuration.
        
        Args:
            config_manager: Configuration manager instance for settings
        """
        self.config = config_manager
        self.console = Console()
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """Configure logger with appropriate handlers."""
        logger = logging.getLogger("vision_processor")

        # Get log level from config
        log_level = self._get_log_level()
        logger.setLevel(log_level)

        # Clear existing handlers to avoid duplicates
        logger.handlers.clear()

        # Add rich handler for console output
        if self._should_use_console():
            rich_handler = RichHandler(console=self.console, show_path=False)
            rich_handler.setLevel(log_level)
            logger.addHandler(rich_handler)

        # Add file handler for production
        file_handler = logging.FileHandler("vision_processor.log")
        file_handler.setLevel(logging.WARNING)  # Only warnings/errors to file
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger

    def debug(self, message: str, **kwargs):
        """Debug level logging."""
        if self._should_use_console() and self._is_debug_mode():
            self.console.print(f"🔍 DEBUG: {message}", style="dim blue", **kwargs)
        self.logger.debug(message)

    def info(self, message: str, **kwargs):
        """Info level logging."""
        if self._should_use_console() and self._is_verbose():
            self.console.print(f"ℹ️  {message}", style="blue", **kwargs)
        self.logger.info(message)

    def warning(self, message: str, **kwargs):
        """Warning level logging."""
        if self._should_use_console():
            self.console.print(f"⚠️  {message}", style="yellow", **kwargs)
        self.logger.warning(message)

    def error(self, message: str, **kwargs):
        """Error level logging."""
        if self._should_use_console():
            self.console.print(f"❌ {message}", style="bold red", **kwargs)
        self.logger.error(message)

    def success(self, message: str, **kwargs):
        """Success message (INFO level with green formatting)."""
        if self._should_use_console():
            self.console.print(f"✅ {message}", style="green", **kwargs)
        self.logger.info(f"SUCCESS: {message}")

    def status(self, message: str, **kwargs):
        """Status message (INFO level with blue formatting)."""
        if self._should_use_console() and self._is_verbose():
            self.console.print(f"📋 {message}", style="cyan", **kwargs)
        self.logger.info(message)

    def _get_log_level(self) -> int:
        """Get logging level from config."""
        if not self.config:
            return logging.INFO

        level_str = getattr(self.config.defaults, 'log_level', 'INFO')
        return getattr(logging, level_str.upper(), logging.INFO)

    def _is_debug_mode(self) -> bool:
        """Check if debug mode is enabled."""
        return self.config and getattr(self.config.defaults, 'debug_mode', False)

    def _is_verbose(self) -> bool:
        """Check if verbose mode is enabled."""
        return self.config and getattr(self.config.defaults, 'verbose_mode', True)

    def _should_use_console(self) -> bool:
        """Check if console output should be used."""
        return self.config and getattr(self.config.defaults, 'console_output', True)