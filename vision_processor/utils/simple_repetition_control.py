"""Simple Repetition Control
===========================

Simplified repetition cleaning that just removes obvious duplicates.
"""

import re


class SimpleRepetitionCleaner:
    """Simple repetition cleaner for model responses."""

    def __init__(self):
        # Common tokens to remove from responses
        self.cleanup_tokens = [
            '<|begin_of_text|>', '<|end_of_text|>', '<|image|>',
            '[INST]', '[/INST]', '<s>', '</s>'
        ]

    def clean_response(self, response: str) -> str:
        """Clean response by removing repetition and artifacts.

        Args:
            response: Raw model response

        Returns:
            Cleaned response
        """
        if not response:
            return ""

        # Remove special tokens
        cleaned = response
        for token in self.cleanup_tokens:
            cleaned = cleaned.replace(token, '')

        # Remove consecutive duplicate lines
        lines = cleaned.split('\n')
        unique_lines = []
        prev_line = None

        for line in lines:
            line_cleaned = line.strip()
            if line_cleaned and line_cleaned != prev_line:
                unique_lines.append(line)
                prev_line = line_cleaned

        cleaned = '\n'.join(unique_lines)

        # Remove excessive whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned)

        return cleaned.strip()

    def has_severe_repetition(self, text: str) -> bool:
        """Check if text has severe repetition issues.

        Args:
            text: Text to check

        Returns:
            True if severe repetition detected
        """
        if not text or len(text) < 50:
            return False

        # Check for same phrase repeated many times
        words = text.split()
        if len(words) < 10:
            return False

        # Look for 3-word phrases repeated 5+ times (severe repetition)
        phrase_counts = {}
        for i in range(len(words) - 2):
            phrase = ' '.join(words[i:i+3]).lower()
            phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1

        # If any phrase appears 5+ times, it's severe
        for count in phrase_counts.values():
            if count >= 5:
                return True

        return False
