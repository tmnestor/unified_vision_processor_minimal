"""Base Extractor with Shared Logic
=================================

Abstract base class that consolidates common extraction logic
from SimpleExtractionManager, ProductionExtractor, and DynamicFieldExtractor.

Follows DRY principle to eliminate code duplication.
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from rich.console import Console

from .patterns import FieldCleaner, FieldValidator, PatternLibrary

console = Console()


@dataclass
class ExtractionResult:
    """Unified result model for all extractors."""

    # Core fields
    success: bool
    fields: Dict[str, str]
    image_name: str
    model_name: str

    # Metadata
    extraction_mode: str = "unknown"  # structured, markdown, fallback
    processing_time: float = 0.0
    confidence_score: float = 0.0

    # Extraction details
    fields_extracted: int = 0
    core_fields_found: int = 0
    validation_errors: List[str] = field(default_factory=list)

    # Raw data for debugging
    raw_response: Optional[str] = None
    cleaned_response: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "success": self.success,
            "fields": self.fields,
            "image_name": self.image_name,
            "model_name": self.model_name,
            "extraction_mode": self.extraction_mode,
            "processing_time": self.processing_time,
            "confidence_score": self.confidence_score,
            "fields_extracted": self.fields_extracted,
            "core_fields_found": self.core_fields_found,
            "validation_errors": self.validation_errors,
        }

    def get_field(self, field_name: str) -> Optional[str]:
        """Get field value by name (case-insensitive)."""
        field_upper = field_name.upper()
        for key, value in self.fields.items():
            if key.upper() == field_upper:
                return value
        return None


class BaseExtractor(ABC):
    """Base class for all extractors with shared functionality."""

    def __init__(self, verbose: bool = False):
        """Initialize base extractor.

        Args:
            verbose: Enable verbose output
        """
        self.verbose = verbose
        self.pattern_library = PatternLibrary
        self.field_cleaner = FieldCleaner
        self.field_validator = FieldValidator

        # Define core fields that are commonly expected in Australian tax documents
        # These are the actual field names we extract, not schema-mapped names
        self.core_fields = {
            "DATE", "TOTAL", "GST", "ABN", "SUPPLIER_NAME",
            "INVOICE_NUMBER", "AMOUNT", "DESCRIPTION",
            "BSB", "ACCOUNT_NUMBER", "BUSINESS_NAME", "RECEIPT_NUMBER"
        }

    def extract(self, response: str, image_name: str, model_name: str) -> ExtractionResult:
        """Main extraction method.

        Args:
            response: Model response text
            image_name: Name of the image being processed
            model_name: Name of the model used

        Returns:
            ExtractionResult with extracted fields
        """
        start_time = datetime.now()

        # Clean response
        cleaned_response = self._clean_response(response)

        # Extract fields using specific implementation
        fields, extraction_mode = self._extract_fields(cleaned_response)

        # Validate and clean fields
        validated_fields = self._validate_and_clean_fields(fields)

        # Calculate metrics
        fields_extracted = len(validated_fields)
        core_fields_found = len([f for f in validated_fields if f.upper() in self.core_fields])

        # Determine success
        success = fields_extracted > 0

        # Calculate confidence score
        confidence_score = self._calculate_confidence(validated_fields, extraction_mode)

        # Create result
        result = ExtractionResult(
            success=success,
            fields=validated_fields,
            image_name=image_name,
            model_name=model_name,
            extraction_mode=extraction_mode,
            processing_time=(datetime.now() - start_time).total_seconds(),
            confidence_score=confidence_score,
            fields_extracted=fields_extracted,
            core_fields_found=core_fields_found,
            raw_response=response,
            cleaned_response=cleaned_response,
        )

        if self.verbose:
            self._print_extraction_summary(result)

        return result

    @abstractmethod
    def _extract_fields(self, response: str) -> Tuple[Dict[str, str], str]:
        """Extract fields from response - must be implemented by subclasses.

        Args:
            response: Cleaned response text

        Returns:
            Tuple of (extracted_fields, extraction_mode)
        """
        pass

    def _clean_response(self, response: str) -> str:
        """Clean response text before extraction."""
        if not response:
            return ""

        # Remove special tokens
        cleaned = response
        for token in ["<|begin_of_text|>", "<|end_of_text|>", "<|image|>", "[INST]", "[/INST]"]:
            cleaned = cleaned.replace(token, "")

        # Remove excessive whitespace
        cleaned = re.sub(r"\s+", " ", cleaned)

        # Remove markdown code blocks if present
        cleaned = re.sub(r"```[\s\S]*?```", "", cleaned)

        return cleaned.strip()

    def _validate_and_clean_fields(self, fields: Dict[str, str]) -> Dict[str, str]:
        """Validate and clean extracted fields."""
        validated = {}

        for field_name, value in fields.items():
            # Skip empty values
            if not value or value.strip() in ["", "N/A", "None", "null"]:
                continue

            # Clean field name
            clean_name = field_name.strip().upper()

            # Clean value based on field type
            clean_value = self.pattern_library.clean_field_value(clean_name, value)

            # Validate format
            is_valid, error = self.pattern_library.validate_field_format(clean_name, clean_value)
            if not is_valid and self.verbose:
                console.print(f"⚠️  Validation error for {clean_name}: {error}", style="yellow")

            validated[clean_name] = clean_value

        return validated

    def _calculate_confidence(self, fields: Dict[str, str], extraction_mode: str) -> float:
        """Calculate confidence score for extraction."""
        if not fields:
            return 0.0

        # Base score from extraction mode
        mode_scores = {"structured": 0.9, "markdown": 0.7, "fallback": 0.5, "unknown": 0.3}
        base_score = mode_scores.get(extraction_mode, 0.3)

        # Adjust based on field coverage
        field_count = len(fields)
        core_count = len([f for f in fields if f in self.core_fields])

        coverage_bonus = min(0.1, field_count / 50.0)  # Up to 0.1 bonus
        core_bonus = min(0.1, core_count / 10.0)  # Up to 0.1 bonus for core fields

        return min(1.0, base_score + coverage_bonus + core_bonus)

    def _print_extraction_summary(self, result: ExtractionResult):
        """Print extraction summary for debugging."""
        console.print(f"\n📊 Extraction Summary for {result.image_name}", style="bold")
        console.print(f"   Model: {result.model_name}")
        console.print(f"   Mode: {result.extraction_mode}")
        console.print(f"   Fields extracted: {result.fields_extracted}")
        console.print(f"   Core fields: {result.core_fields_found}")
        console.print(f"   Confidence: {result.confidence_score:.2f}")
        console.print(f"   Time: {result.processing_time:.2f}s")

        if result.validation_errors:
            console.print(f"   ⚠️  Validation errors: {len(result.validation_errors)}", style="yellow")

    # Shared utility methods

    def _extract_key_value_pairs(self, text: str) -> Dict[str, str]:
        """Extract key-value pairs from text using common patterns."""
        fields = {}

        # Pattern 1: KEY: VALUE
        pattern1 = re.compile(r"^([A-Z][A-Z\s_/-]+?):\s*(.+)$", re.MULTILINE | re.IGNORECASE)

        # Pattern 2: KEY = VALUE
        pattern2 = re.compile(r"^([A-Z][A-Z\s_/-]+?)\s*=\s*(.+)$", re.MULTILINE | re.IGNORECASE)

        # Pattern 3: KEY - VALUE
        pattern3 = re.compile(r"^([A-Z][A-Z\s_/-]+?)\s*-\s*(.+)$", re.MULTILINE | re.IGNORECASE)

        for pattern in [pattern1, pattern2, pattern3]:
            matches = pattern.findall(text)
            for key, value in matches:
                clean_key = key.strip().replace(" ", "_").replace("/", "_").replace("-", "_")
                clean_value = value.strip()
                if clean_key and clean_value:
                    fields[clean_key] = clean_value

        return fields

    def _detect_markdown_content(self, text: str) -> bool:
        """Detect if response contains markdown-formatted content."""
        markdown_indicators = [
            r"\|.*\|.*\|",  # Table rows
            r"^#+\s",  # Headers
            r"^\*\s",  # Bullet points
            r"^\d+\.\s",  # Numbered lists
            r"\*\*[^*]+\*\*",  # Bold text
            r"^---+$",  # Horizontal rules
        ]

        for pattern in markdown_indicators:
            if re.search(pattern, text, re.MULTILINE):
                return True

        return False

    def _extract_from_markdown_table(self, text: str) -> Dict[str, str]:
        """Extract fields from markdown table format."""
        fields = {}

        # Find table rows
        table_pattern = re.compile(r"\|\s*([^|]+)\s*\|\s*([^|]+)\s*\|")
        matches = table_pattern.findall(text)

        for key, value in matches:
            # Skip header rows
            if "---" in key or "---" in value:
                continue

            clean_key = key.strip().replace(" ", "_").upper()
            clean_value = value.strip()

            if clean_key and clean_value and clean_key != "FIELD" and clean_value != "VALUE":
                fields[clean_key] = clean_value

        return fields

    def _extract_banking_fields(self, text: str) -> Dict[str, str]:
        """Extract banking-specific fields using patterns."""
        fields = {}

        # BSB extraction
        bsb = self.pattern_library.extract_bsb(text)
        if bsb:
            fields["BSB"] = bsb

        # Account number (usually follows BSB)
        account_pattern = re.compile(r"(?:Account|Acc|A/C)[\s#:]*(\d{6,12})", re.IGNORECASE)
        match = account_pattern.search(text)
        if match:
            fields["ACCOUNT_NUMBER"] = match.group(1)

        return fields
