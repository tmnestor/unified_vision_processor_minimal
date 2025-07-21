"""Repetition control for vision models.

This module provides sophisticated repetition detection and control for vision language models,
particularly targeting the repetition issues found in Llama-3.2-Vision models.

Classes:
    RepetitionController: Standard repetition detection and cleaning
    UltraAggressiveRepetitionController: Nuclear option for severe repetition issues
"""

import re


class RepetitionController:
    """Standard repetition detection and control for vision language models."""

    def __init__(self, word_threshold: float = 0.3, phrase_threshold: int = 3):
        """
        Initialize repetition controller.

        Args:
            word_threshold: If any word appears more than this % of total words, it's repetitive
            phrase_threshold: Minimum repetitions to trigger cleaning
        """
        self.word_threshold = word_threshold
        self.phrase_threshold = phrase_threshold

    def detect_repetitive_generation(self, text: str, min_words: int = 5) -> bool:
        """
        Detect if text contains repetitive patterns.

        Args:
            text: Text to analyze
            min_words: Minimum words required for valid content

        Returns:
            True if text appears repetitive
        """
        words = text.split()

        if len(words) < min_words:
            return True

        # Check word repetition
        word_counts = {}
        for word in words:
            word_lower = word.lower().strip(".,!?()[]{}\"'")
            if len(word_lower) > 2:
                word_counts[word_lower] = word_counts.get(word_lower, 0) + 1

        total_words = len([w for w in words if len(w.strip(".,!?()[]{}\"'")) > 2])
        if total_words > 0:
            for _word, count in word_counts.items():
                if count > total_words * self.word_threshold:
                    return True

        # Check phrase repetition
        return self._detect_phrase_repetition(text)

    def _detect_phrase_repetition(self, text: str) -> bool:
        """Detect repeated phrases in text."""
        # Check for repeated 3+ word phrases
        words = text.split()
        for i in range(len(words) - 8):  # Need at least 9 words for 3+3+3
            phrase = " ".join(words[i : i + 3]).lower()
            remainder = " ".join(words[i + 3 :]).lower()
            if phrase in remainder:
                # Count occurrences
                phrase_count = remainder.count(phrase) + 1
                if phrase_count >= self.phrase_threshold:
                    return True

        return False

    def clean_response(self, response: str, image_name: str = "") -> str:
        """
        Clean repetitive patterns from response text.

        Args:
            response: Raw model response
            image_name: Name of the image being processed (for debugging)

        Returns:
            Cleaned response with repetition removed
        """
        if not response or len(response.strip()) == 0:
            return ""

        original_length = len(response)

        # Remove consecutive word repetition
        response = self._remove_word_repetition(response)

        # Remove phrase repetition
        response = self._remove_phrase_repetition(response)

        # Clean artifacts
        response = self._clean_artifacts(response)

        final_length = len(response)
        reduction = ((original_length - final_length) / original_length * 100) if original_length > 0 else 0

        if reduction > 5:  # Only log significant reductions
            print(
                f"🧹 Repetition cleaning: {original_length} → {final_length} chars "
                f"({reduction:.1f}% reduction)"
            )

        return response.strip()

    def _remove_word_repetition(self, text: str) -> str:
        """Remove consecutive repeated words."""
        # Remove 3+ consecutive identical words
        text = re.sub(r"\b(\w+)(\s+\1){2,}", r"\1", text, flags=re.IGNORECASE)
        return text

    def _remove_phrase_repetition(self, text: str) -> str:
        """Remove repeated phrases."""
        # Remove repeated 2-5 word phrases
        for phrase_length in range(2, 6):
            pattern = (
                r"\b((?:\w+\s+){"
                + str(phrase_length - 1)
                + r"}\w+)(\s+\1){"
                + str(self.phrase_threshold - 1)
                + r",}"
            )
            text = re.sub(pattern, r"\1", text, flags=re.IGNORECASE)

        return text

    def _clean_artifacts(self, text: str) -> str:
        """Clean common artifacts from model output."""
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove excessive punctuation
        text = re.sub(r"[.]{3,}", "...", text)
        text = re.sub(r"[!]{2,}", "!", text)
        text = re.sub(r"[?]{2,}", "?", text)

        return text


class UltraAggressiveRepetitionController(RepetitionController):
    """Ultra-aggressive repetition detection and control specifically for Llama-3.2-Vision.

    This controller implements the nuclear option for severe repetition issues,
    using much stricter thresholds and aggressive cleaning strategies.
    """

    def __init__(self, word_threshold: float = 0.15, phrase_threshold: int = 2):
        """
        Initialize ultra-aggressive repetition controller.

        Args:
            word_threshold: If any word appears more than this % of total words, it's repetitive (15% vs 30%)
            phrase_threshold: Minimum repetitions to trigger cleaning (2 vs 3)
        """
        super().__init__(word_threshold, phrase_threshold)

        # Initialize AWK processor
        self.awk_processor = None
        self._init_awk_processor()

        # Known problematic patterns from Llama-3.2-Vision
        self.toxic_patterns = [
            r"THANK YOU FOR SHOPPING WITH US",
            r"All prices include GST where applicable",
            r"\\+[a-zA-Z]*\{[^}]*\}",  # LaTeX artifacts
            r"\(\s*\)",  # Empty parentheses
            r"[.-]\s*THANK YOU",  # Dash/period before thank you
        ]

    def _init_awk_processor(self):
        """Initialize AWK processor for sophisticated markdown processing."""
        try:
            # Import AWK processor (defer import to avoid circular dependencies)
            import sys
            from pathlib import Path

            # Add the root directory to path to import awk_markdown_processor
            root_dir = Path(__file__).parent.parent.parent
            sys.path.insert(0, str(root_dir))

            from awk_markdown_processor import AWKMarkdownProcessor

            # Try to initialize with config file
            config_path = root_dir / "markdown_processing_config.yaml"
            if config_path.exists():
                self.awk_processor = AWKMarkdownProcessor(str(config_path))
                print("✅ AWK processor initialized successfully")
            else:
                print(f"⚠️  AWK config not found: {config_path}")
        except Exception as e:
            print(f"⚠️  Failed to initialize AWK processor: {e}")
            self.awk_processor = None

    def detect_repetitive_generation(self, text: str, min_words: int = 3) -> bool:
        """Conservative repetition detection (BALANCED FOR FIELD EXTRACTION)."""
        words = text.split()

        # Much stricter minimum content requirement
        if len(words) < min_words:
            return True

        # Check for known toxic patterns first (these are truly problematic)
        if self._has_toxic_patterns(text):
            return True

        # More balanced word repetition check (25% threshold - business docs have repeated field names)
        word_counts = {}
        for word in words:
            word_lower = word.lower().strip(".,!?()[]{}\"'")
            if len(word_lower) > 3:  # Ignore short words like "ABN", "GST" that appear naturally
                word_counts[word_lower] = word_counts.get(word_lower, 0) + 1

        total_words = len([w for w in words if len(w.strip(".,!?()[]{}\"'")) > 3])
        if total_words > 0:
            for word, count in word_counts.items():
                # More tolerant threshold - field extraction needs repeated field names
                if count > total_words * 0.25:  # 25% vs original 15%
                    # Don't flag common business document words
                    if word not in ["receipt", "invoice", "total", "amount", "store", "date", "payment"]:
                        return True

        # More balanced phrase repetition (only flag truly repetitive phrases)
        if self._detect_balanced_phrase_repetition(text):
            return True

        return False

    def _has_toxic_patterns(self, text: str) -> bool:
        """Check for known problematic patterns."""
        for pattern in self.toxic_patterns:
            matches = re.findall(pattern, text, flags=re.IGNORECASE)
            if len(matches) >= 2:  # Even 2 occurrences is too many
                return True

        return False

    def _detect_balanced_phrase_repetition(self, text: str) -> bool:
        """Balanced phrase repetition detection for field extraction (WORKING SCRIPT LOGIC)."""
        # More lenient than aggressive - only flag truly repetitive phrases
        words = text.split()
        if len(words) < 12:  # Need substantial content for balanced check
            return False

        # Check for 4+ word phrases repeated 3+ times (more restrictive)
        for i in range(len(words) - 12):  # Need at least 12 words for 4+4+4
            phrase = " ".join(words[i : i + 4]).lower()
            remainder = " ".join(words[i + 4 :]).lower()
            if phrase in remainder:
                # Count occurrences - need 3+ for balanced detection
                phrase_count = remainder.count(phrase) + 1
                if phrase_count >= 3:  # 3+ repetitions indicates real problem
                    return True

        return False

    def _detect_aggressive_phrase_repetition(self, text: str) -> bool:
        """Ultra-aggressive phrase repetition detection."""
        # Check for 3+ word phrases repeated even twice
        words = text.split()
        for i in range(len(words) - 6):  # Need at least 6 words for 3+3
            phrase = " ".join(words[i : i + 3]).lower()
            remainder = " ".join(words[i + 3 :]).lower()
            if phrase in remainder:
                return True

        # Check sentences/segments
        segments = re.split(r"[.!?]+", text)
        segment_counts = {}

        for segment in segments:
            segment_clean = re.sub(r"\s+", " ", segment.strip().lower())
            # Much shorter minimum segment length
            if len(segment_clean) > 5:  # Was 10, now 5
                segment_counts[segment_clean] = segment_counts.get(segment_clean, 0) + 1

        # Any segment appearing twice is problematic
        for count in segment_counts.values():
            if count >= self.phrase_threshold:  # Now 2 instead of 3
                return True

        return False

    def clean_response(self, response: str, image_name: str = "") -> str:
        """Ultra-aggressive cleaning with AWK processing and competitive strategy."""
        if not response or len(response.strip()) == 0:
            return ""

        original_length = len(response)

        # Use competitive processing: try both AWK and original, choose the better result
        if self.awk_processor:
            try:
                # Try AWK processing
                awk_result = self.awk_processor.process(response, image_name)

                # Try original processing
                original_response = self._fallback_to_original_conversion(response, image_name)

                # Choose the better result based on success and field count
                if self._should_use_awk_result(awk_result, original_response, image_name):
                    return awk_result.processed_text
                else:
                    return original_response
            except Exception as e:
                print(f"⚠️  AWK processing failed for {image_name}: {e}")
                # Fall back to original processing
                pass

        # Step 1: Early truncation at first major repetition
        response = self._early_truncate_at_repetition(response)

        # Step 2: Remove toxic patterns aggressively
        response = self._remove_toxic_patterns(response)

        # Step 3: Remove safety warnings
        response = self._remove_safety_warnings(response)

        # Step 4: Ultra-aggressive repetition removal
        response = self._ultra_aggressive_word_removal(response)
        response = self._ultra_aggressive_phrase_removal(response)
        response = self._ultra_aggressive_sentence_removal(response)

        # Step 5: Clean artifacts
        response = self._clean_artifacts_aggressive(response)

        # Step 6: Final validation and truncation
        response = self._final_validation_truncate(response)

        final_length = len(response)
        reduction = ((original_length - final_length) / original_length * 100) if original_length > 0 else 0

        if reduction > 10:  # Log significant reductions
            print(
                f"🧹 Ultra-aggressive cleaning: {original_length} → {final_length} chars "
                f"({reduction:.1f}% reduction)"
            )

        return response.strip()

    def _early_truncate_at_repetition(self, text: str) -> str:
        """Truncate immediately when repetition starts."""
        # Find first occurrence of toxic patterns and truncate there
        for pattern in self.toxic_patterns:
            matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))
            if len(matches) >= 2:
                # Truncate before the second occurrence
                truncate_point = matches[1].start()
                print(f"🔪 Early truncation at repetition: {len(text)} → {truncate_point} chars")
                return text[:truncate_point]

        return text

    def _remove_toxic_patterns(self, text: str) -> str:
        """Aggressively remove known toxic patterns."""
        for pattern in self.toxic_patterns:
            # Remove ALL occurrences, not just duplicates
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)

        return text

    def _remove_safety_warnings(self, text: str) -> str:
        """Remove safety warnings."""
        safety_patterns = [
            r"I'm not able to provide.*?information\.?",
            r"I cannot provide.*?information\.?",
            r"I'm unable to.*?\.?",
            r"I can't.*?\.?",
            r"Sorry, I cannot.*?\.?",
            r".*could compromise.*privacy.*",
        ]

        for pattern in safety_patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)

        return text

    def _ultra_aggressive_word_removal(self, text: str) -> str:
        """Ultra-aggressive word repetition removal."""
        # Remove 2+ consecutive identical words (was 3+)
        text = re.sub(r"\b(\w+)(\s+\1){1,}", r"\1", text, flags=re.IGNORECASE)

        # Remove any word appearing more than 3 times total
        words = text.split()
        word_counts = {}
        for word in words:
            word_lower = word.lower().strip(".,!?()[]{}\"'")
            word_counts[word_lower] = word_counts.get(word_lower, 0) + 1

        # Rebuild text, limiting each word to max 3 occurrences
        result_words = []
        word_usage = {}

        for word in words:
            word_lower = word.lower().strip(".,!?()[]{}\"'")
            current_count = word_usage.get(word_lower, 0)

            if current_count < 3:  # Allow max 3 occurrences
                result_words.append(word)
                word_usage[word_lower] = current_count + 1

        return " ".join(result_words)

    def _ultra_aggressive_phrase_removal(self, text: str) -> str:
        """Ultra-aggressive phrase removal."""
        # Remove repeated 2-6 word phrases (expanded range)
        for phrase_length in range(2, 7):
            pattern = (
                r"\b((?:\w+\s+){" + str(phrase_length - 1) + r"}\w+)(\s+\1){1,}"  # 1+ repetitions vs 2+
            )
            text = re.sub(pattern, r"\1", text, flags=re.IGNORECASE)

        return text

    def _ultra_aggressive_sentence_removal(self, text: str) -> str:
        """Ultra-aggressive sentence removal."""
        sentences = re.split(r"[.!?]+", text)

        # Keep only first occurrence of any sentence
        seen = set()
        unique_sentences = []

        for sentence in sentences:
            sentence_clean = re.sub(r"\s+", " ", sentence.strip().lower())
            sentence_clean = re.sub(r"[^\w\s]", "", sentence_clean)  # Remove all punctuation for comparison

            if sentence_clean and len(sentence_clean) > 3:  # Very short minimum
                if sentence_clean not in seen:
                    seen.add(sentence_clean)
                    unique_sentences.append(sentence.strip())

        return ". ".join(unique_sentences)

    def _clean_artifacts_aggressive(self, text: str) -> str:
        """Aggressive artifact cleaning."""
        # Remove whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove LaTeX/markdown aggressively
        text = re.sub(r"\\+[a-zA-Z]*\{[^}]*\}", "", text)
        text = re.sub(r"\\+[a-zA-Z]+", "", text)
        text = re.sub(r"```+[^`]*```+", "", text)
        text = re.sub(r"[{}]+", "", text)

        # Remove excessive punctuation
        text = re.sub(r"[.]{2,}", ".", text)
        text = re.sub(r"[!]{2,}", "!", text)
        text = re.sub(r"[?]{2,}", "?", text)
        text = re.sub(r"[,]{2,}", ",", text)

        # Remove empty parentheses and brackets
        text = re.sub(r"\(\s*\)", "", text)
        text = re.sub(r"\[\s*\]", "", text)

        # Remove standalone punctuation
        text = re.sub(r"\s+[.,!?;:]\s+", " ", text)

        return text

    def _final_validation_truncate(self, text: str, max_length: int = 1500) -> str:
        """Final validation with conservative truncation (WORKING SCRIPT COMPATIBLE)."""
        # If still repetitive after all cleaning, be more conservative about truncation
        if self.detect_repetitive_generation(text):
            print("⚠️ Still repetitive after ultra-aggressive cleaning - conservative truncation")
            # Instead of cutting in half, just remove the most repetitive part
            # Try cutting the last quarter instead of half to preserve more content
            quarter_point = len(text) * 3 // 4  # Keep 75% instead of 50%
            truncated = text[:quarter_point]
            last_period = truncated.rfind(".")
            if last_period > quarter_point * 0.8:  # More conservative
                return truncated[: last_period + 1]
            else:
                return truncated + "..."

        # Much more generous length limit (800 was too restrictive for 15.9 fields)
        if len(text) > max_length:
            truncated = text[:max_length]
            last_period = truncated.rfind(".")
            if last_period > max_length * 0.8:  # More conservative
                return truncated[: last_period + 1]
            else:
                return truncated + "..."

        return text

    def _fallback_to_original_conversion(self, response: str, image_name: str = "") -> str:
        """Apply original cleaning strategy for comparison."""
        # Step 1: Early truncation at first major repetition
        response = self._early_truncate_at_repetition(response)

        # Step 2: Remove toxic patterns aggressively
        response = self._remove_toxic_patterns(response)

        # Step 3: Remove safety warnings
        response = self._remove_safety_warnings(response)

        # Step 4: Ultra-aggressive repetition removal
        response = self._ultra_aggressive_word_removal(response)
        response = self._ultra_aggressive_phrase_removal(response)
        response = self._ultra_aggressive_sentence_removal(response)

        # Step 5: Clean artifacts and final validation
        response = self._clean_artifacts_aggressive(response)
        response = self._final_validation_truncate(response)

        return response.strip()

    def _should_use_awk_result(self, awk_result, original_response: str, image_name: str = "") -> bool:
        """Determine if AWK result is better than original processing (EXACT WORKING SCRIPT LOGIC)."""
        if not awk_result or not awk_result.success:
            return False

        # Count fields in AWK result
        awk_field_count = len(awk_result.extracted_fields)

        # Count fields in original response (basic heuristic)
        original_field_count = len(
            [line for line in original_response.split("\n") if ":" in line and len(line.strip()) > 0]
        )

        # Decision logic with multiple criteria (EXACT WORKING SCRIPT LOGIC)
        use_awk = False
        reason = ""

        if not awk_result.success:
            # AWK failed, use original
            use_awk = False
            reason = "AWK processing failed"
        elif awk_field_count == 0:
            # AWK extracted no fields, use original
            use_awk = False
            reason = "AWK extracted 0 fields"
        elif original_field_count == 0:
            # Original extracted no fields, use AWK
            use_awk = True
            reason = "original extracted 0 fields"
        elif awk_field_count >= original_field_count + 2:
            # AWK significantly better (2+ more fields)
            use_awk = True
            reason = f"AWK much better ({awk_field_count} vs {original_field_count} fields)"
        elif original_field_count >= awk_field_count + 2:
            # Original significantly better (2+ more fields)
            use_awk = False
            reason = f"original much better ({original_field_count} vs {awk_field_count} fields)"
        elif awk_field_count >= original_field_count and awk_result.confidence_score >= 0.6:
            # AWK equal or better + high confidence (WORKING SCRIPT: 0.6 threshold!)
            use_awk = True
            reason = f"AWK better/equal with high confidence ({awk_field_count} fields, {awk_result.confidence_score:.2f} conf)"
        else:
            # Default to original for all other cases
            use_awk = False
            reason = f"default to original ({original_field_count} vs {awk_field_count} fields, {awk_result.confidence_score:.2f} conf)"

        # Log decision for debugging (matching working script)
        if use_awk:
            print(f"🏆 Using AWK result for {image_name}: {reason}")
        else:
            print(f"🏆 Using original result for {image_name}: {reason}")

        return use_awk
