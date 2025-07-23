"""Repetition Control Legacy Module
====================================

Provides legacy repetition control classes for compatibility with existing models.
This is a simplified version that provides the same interface as the original.
"""

from .simple_repetition_control import SimpleRepetitionCleaner


class RepetitionController:
    """Base repetition controller."""

    def __init__(self, **kwargs):
        """Initialize repetition controller."""
        self.cleaner = SimpleRepetitionCleaner()

    def clean_response(self, response: str, image_name: str = None) -> str:
        """Clean model response to remove repetition.

        Args:
            response: Raw model response
            image_name: Image name (for compatibility, not used)

        Returns:
            Cleaned response
        """
        return self.cleaner.clean_response(response)

    def has_severe_repetition(self, text: str) -> bool:
        """Check if text has severe repetition issues."""
        return self.cleaner.has_severe_repetition(text)


class UltraAggressiveRepetitionController(RepetitionController):
    """Ultra-aggressive repetition controller for legacy compatibility."""

    def __init__(self, **kwargs):
        """Initialize ultra-aggressive repetition controller.

        Args:
            **kwargs: Configuration options (for compatibility)
        """
        super().__init__(**kwargs)

        # Legacy compatibility properties
        self.max_tokens_limit = kwargs.get("max_tokens_limit", 1000)
        self.word_threshold = kwargs.get("word_threshold", 10)
        self.phrase_threshold = kwargs.get("phrase_threshold", 5)
        self.enabled = kwargs.get("enabled", True)

    def clean_response(self, response: str, image_name: str = None) -> str:
        """Clean response with ultra-aggressive repetition removal.

        Args:
            response: Raw model response
            image_name: Image name (for logging/debugging)

        Returns:
            Cleaned response
        """
        if not self.enabled or not response:
            return response

        # Use the simple cleaner
        cleaned = self.cleaner.clean_response(response)

        # Additional aggressive cleaning for legacy compatibility
        if self.has_severe_repetition(cleaned):
            # If still has severe repetition, truncate at reasonable length
            words = cleaned.split()
            if len(words) > self.max_tokens_limit:
                cleaned = " ".join(words[: self.max_tokens_limit])

        return cleaned

    def get_stats(self) -> dict:
        """Get repetition control statistics (for compatibility)."""
        return {
            "enabled": self.enabled,
            "max_tokens_limit": self.max_tokens_limit,
            "word_threshold": self.word_threshold,
            "phrase_threshold": self.phrase_threshold,
        }
