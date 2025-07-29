"""Repetition Control Utility

Unified repetition control for all vision models to prevent repetitive outputs
and ensure clean, consistent extractions.
"""

import logging
import re
from typing import Dict

logger = logging.getLogger(__name__)


class RepetitionController:
    """Unified repetition control for vision models.
    
    Handles:
    - Token limit enforcement to prevent infinite repetition
    - Special token cleanup
    - Duplicate line removal
    - Response truncation
    - Whitespace normalization
    """

    def __init__(
        self,
        enabled: bool = True,
        word_threshold: int = 3,
        phrase_threshold: int = 2,
        max_new_tokens_limit: int = 2024,
    ):
        """Initialize repetition controller.
        
        Args:
            enabled: Whether repetition control is active
            word_threshold: Threshold for word repetition detection
            phrase_threshold: Threshold for phrase repetition detection  
            max_new_tokens_limit: Maximum tokens to prevent runaway generation
        """
        self.enabled = enabled
        self.word_threshold = word_threshold
        self.phrase_threshold = phrase_threshold
        self.max_new_tokens_limit = max_new_tokens_limit
        
        # Special tokens to clean up (common across models)
        self.cleanup_tokens = [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|eot_id|>",
            "<|reserved_special_token_",
            "<pad>",
            "</s>",
        ]
        
        logger.info(
            f"RepetitionController initialized - enabled={self.enabled}, "
            f"max_tokens_limit={self.max_new_tokens_limit}"
        )

    @classmethod
    def from_config(cls, config) -> "RepetitionController":
        """Create RepetitionController from configuration.
        
        Args:
            config: ConfigManager instance
            
        Returns:
            RepetitionController instance
        """
        if hasattr(config, "repetition_control") and config.repetition_control:
            # Get fallback_max_tokens from repetition_control config
            fallback_max_tokens = getattr(config.repetition_control, "fallback_max_tokens", 2024)
            
            return cls(
                enabled=config.repetition_control.enabled,
                word_threshold=config.repetition_control.word_threshold,
                phrase_threshold=config.repetition_control.phrase_threshold,
                max_new_tokens_limit=fallback_max_tokens,
            )
        else:
            # Default configuration with repetition control enabled
            return cls(enabled=True)

    def enforce_token_limit(self, max_tokens: int) -> int:
        """Enforce token limit to prevent repetition.
        
        Args:
            max_tokens: Requested max tokens
            
        Returns:
            Limited max tokens if repetition control is enabled
        """
        if not self.enabled:
            return max_tokens
            
        return min(max_tokens, self.max_new_tokens_limit)

    def clean_response(self, response: str) -> str:
        """Clean model response to remove repetition and artifacts.
        
        Args:
            response: Raw model response
            
        Returns:
            Cleaned response
        """
        if not response or not response.strip():
            return ""

        cleaned = response.strip()

        if self.enabled:
            # Remove special tokens
            for token in self.cleanup_tokens:
                cleaned = cleaned.replace(token, "")

            # Remove consecutive duplicate lines
            cleaned = self._remove_duplicate_lines(cleaned)

            # Remove excessive whitespace
            cleaned = re.sub(r"\s+", " ", cleaned)

            # Truncate if too long (rough character estimate)
            max_chars = self.max_new_tokens_limit * 5
            if len(cleaned) > max_chars:
                cleaned = cleaned[:max_chars] + "..."
        else:
            # Basic cleaning only
            cleaned = re.sub(r"\s+", " ", cleaned)
            if len(cleaned) > 1000:
                cleaned = cleaned[:1000] + "..."

        return cleaned.strip()

    def _remove_duplicate_lines(self, text: str) -> str:
        """Remove consecutive duplicate lines from text.
        
        Args:
            text: Input text
            
        Returns:
            Text with duplicate lines removed
        """
        lines = text.split("\\n")
        unique_lines = []
        prev_line = None

        for line in lines:
            line_cleaned = line.strip()
            if line_cleaned and line_cleaned != prev_line:
                unique_lines.append(line)
                prev_line = line_cleaned

        return "\\n".join(unique_lines)

    def detect_repetition(self, text: str) -> Dict[str, float]:
        """Detect repetition patterns in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with repetition metrics
        """
        if not text or not self.enabled:
            return {"word_repetition": 0.0, "phrase_repetition": 0.0}
            
        words = text.split()
        if len(words) < 3:
            return {"word_repetition": 0.0, "phrase_repetition": 0.0}
            
        # Calculate word repetition rate
        word_counts = {}
        for word in words:
            word_lower = word.lower().strip(".,!?;:")
            word_counts[word_lower] = word_counts.get(word_lower, 0) + 1
            
        repeated_words = sum(1 for count in word_counts.values() if count >= self.word_threshold)
        word_repetition = repeated_words / len(word_counts) if word_counts else 0.0
        
        # Calculate phrase repetition (simplified)
        phrases = []
        for i in range(len(words) - 1):
            phrase = f"{words[i]} {words[i + 1]}".lower()
            phrases.append(phrase)
            
        phrase_counts = {}
        for phrase in phrases:
            phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1
            
        repeated_phrases = sum(1 for count in phrase_counts.values() if count >= self.phrase_threshold)
        phrase_repetition = repeated_phrases / len(phrase_counts) if phrase_counts else 0.0
        
        return {
            "word_repetition": word_repetition,
            "phrase_repetition": phrase_repetition,
        }

    def is_repetitive(self, text: str, threshold: float = 0.3) -> bool:
        """Check if text is excessively repetitive.
        
        Args:
            text: Text to check
            threshold: Repetition threshold (0.0 to 1.0)
            
        Returns:
            True if text is considered repetitive
        """
        if not self.enabled:
            return False
            
        metrics = self.detect_repetition(text)
        return (
            metrics["word_repetition"] > threshold 
            or metrics["phrase_repetition"] > threshold
        )