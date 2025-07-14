"""Vision Processor CLI Module."""

from .evaluation_cli import app as evaluation_app
from .simple_extract_cli import app as extract_app

__all__ = ["extract_app", "evaluation_app"]
