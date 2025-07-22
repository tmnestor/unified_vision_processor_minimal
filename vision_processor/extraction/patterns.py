"""Centralized Pattern Library for Field Extraction
================================================

Consolidates all regex patterns and field-specific logic that was
duplicated across SimpleExtractionManager, ProductionExtractor,
and DynamicFieldExtractor.

Follows DRY principle to eliminate pattern duplication.
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Pattern, Tuple


class FieldType(Enum):
    """Types of fields for specialized processing."""

    CURRENCY = "currency"
    DATE = "date"
    ABN = "abn"
    BSB = "bsb"
    PERCENTAGE = "percentage"
    NUMERIC = "numeric"
    TEXT = "text"
    BOOLEAN = "boolean"


@dataclass
class FieldPattern:
    """Pattern definition for a field type."""

    name: str
    pattern: Pattern[str]
    type: FieldType
    description: str
    example: str

    def extract(self, text: str) -> Optional[str]:
        """Extract field value from text using pattern."""
        match = self.pattern.search(text)
        return match.group(0) if match else None


class PatternLibrary:
    """Centralized pattern library for all field extraction."""

    # Core regex patterns used across extractors
    PATTERNS = {
        # Australian Business Number
        "ABN": re.compile(r"\b\d{2}\s?\d{3}\s?\d{3}\s?\d{3}\b"),
        # Bank State Branch
        "BSB": re.compile(r"\b\d{3}[-\s]?\d{3}\b"),
        # Currency (Australian dollars)
        "CURRENCY": re.compile(r"\$[\d,]+\.?\d*|\d+\.?\d*\s*(?:AUD|aud|dollars?)"),
        # Date formats
        "DATE_DMY": re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"),
        "DATE_MDY": re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"),
        "DATE_YMD": re.compile(r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b"),
        "DATE_LONG": re.compile(
            r"\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4}\b",
            re.IGNORECASE,
        ),
        # Percentage
        "PERCENTAGE": re.compile(r"\b\d+\.?\d*\s*%"),
        # Phone numbers (Australian)
        "PHONE": re.compile(r"(?:\+61|0)[2-9]\d{8}|\(\d{2}\)\s*\d{4}\s*\d{4}"),
        # Email
        "EMAIL": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
        # Postcode (Australian)
        "POSTCODE": re.compile(r"\b(?:NSW|VIC|QLD|SA|WA|TAS|NT|ACT)?\s*\d{4}\b"),
        # Invoice/Reference numbers
        "INVOICE_NUMBER": re.compile(
            r"(?:Invoice|Inv|Bill|Reference|Ref)[:\s#]*([A-Z0-9-]+)", re.IGNORECASE
        ),
        # GST specific
        "GST_AMOUNT": re.compile(r"GST[:\s]*\$?([\d,]+\.?\d*)", re.IGNORECASE),
        # Total amount
        "TOTAL_AMOUNT": re.compile(r"(?:Total|Amount\s+Due|Balance)[:\s]*\$?([\d,]+\.?\d*)", re.IGNORECASE),
    }

    # Field-specific patterns with metadata
    FIELD_PATTERNS = {
        "DATE": FieldPattern(
            name="DATE",
            pattern=re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"),
            type=FieldType.DATE,
            description="Document date",
            example="01/01/2024",
        ),
        "TOTAL": FieldPattern(
            name="TOTAL",
            pattern=re.compile(r"(?:Total|Amount\s+Due)[:\s]*\$?([\d,]+\.?\d*)", re.IGNORECASE),
            type=FieldType.CURRENCY,
            description="Total amount",
            example="$1,234.56",
        ),
        "GST": FieldPattern(
            name="GST",
            pattern=re.compile(r"GST[:\s]*\$?([\d,]+\.?\d*)", re.IGNORECASE),
            type=FieldType.CURRENCY,
            description="GST amount",
            example="$123.45",
        ),
        "ABN": FieldPattern(
            name="ABN",
            pattern=re.compile(r"\b\d{2}\s?\d{3}\s?\d{3}\s?\d{3}\b"),
            type=FieldType.ABN,
            description="Australian Business Number",
            example="12 345 678 901",
        ),
        "BSB": FieldPattern(
            name="BSB",
            pattern=re.compile(r"\b\d{3}[-\s]?\d{3}\b"),
            type=FieldType.BSB,
            description="Bank State Branch",
            example="123-456",
        ),
    }

    @classmethod
    def get_pattern(cls, pattern_name: str) -> Optional[Pattern[str]]:
        """Get a specific regex pattern by name."""
        return cls.PATTERNS.get(pattern_name)

    @classmethod
    def get_field_pattern(cls, field_name: str) -> Optional[FieldPattern]:
        """Get a field pattern with metadata."""
        return cls.FIELD_PATTERNS.get(field_name.upper())

    @classmethod
    def extract_currency(cls, text: str) -> Optional[str]:
        """Extract and clean currency value."""
        if not text:
            return None

        # Remove common currency symbols and text
        cleaned = re.sub(r"[$,]", "", str(text))
        cleaned = re.sub(r"\s*(AUD|aud|dollars?)\s*", "", cleaned, flags=re.IGNORECASE)

        # Extract numeric value
        match = re.search(r"\d+\.?\d*", cleaned)
        if match:
            try:
                value = float(match.group(0))
                return f"${value:,.2f}"
            except ValueError:
                pass

        return None

    @classmethod
    def extract_date(cls, text: str) -> Optional[str]:
        """Extract and normalize date."""
        if not text:
            return None

        # Try different date patterns
        for pattern_name in ["DATE_DMY", "DATE_MDY", "DATE_YMD", "DATE_LONG"]:
            pattern = cls.PATTERNS.get(pattern_name)
            if pattern:
                match = pattern.search(text)
                if match:
                    return match.group(0)

        return None

    @classmethod
    def extract_abn(cls, text: str) -> Optional[str]:
        """Extract and format ABN."""
        if not text:
            return None

        pattern = cls.PATTERNS.get("ABN")
        if pattern:
            match = pattern.search(text)
            if match:
                # Format as XX XXX XXX XXX
                digits = re.sub(r"\D", "", match.group(0))
                if len(digits) == 11:
                    return f"{digits[:2]} {digits[2:5]} {digits[5:8]} {digits[8:]}"

        return None

    @classmethod
    def extract_bsb(cls, text: str) -> Optional[str]:
        """Extract and format BSB."""
        if not text:
            return None

        pattern = cls.PATTERNS.get("BSB")
        if pattern:
            match = pattern.search(text)
            if match:
                # Format as XXX-XXX
                digits = re.sub(r"\D", "", match.group(0))
                if len(digits) == 6:
                    return f"{digits[:3]}-{digits[3:]}"

        return None

    @classmethod
    def extract_percentage(cls, text: str) -> Optional[str]:
        """Extract percentage value."""
        if not text:
            return None

        pattern = cls.PATTERNS.get("PERCENTAGE")
        if pattern:
            match = pattern.search(text)
            if match:
                return match.group(0)

        return None

    @classmethod
    def clean_field_value(cls, field_name: str, value: str) -> str:
        """Clean field value based on field type."""
        if not value:
            return ""

        # Determine field type and clean accordingly
        field_upper = field_name.upper()

        # Currency fields
        if any(curr in field_upper for curr in ["TOTAL", "AMOUNT", "GST", "SUBTOTAL", "PRICE", "COST"]):
            cleaned = cls.extract_currency(value)
            return cleaned if cleaned else value

        # Date fields
        elif any(date in field_upper for date in ["DATE", "DUE", "ISSUED"]):
            cleaned = cls.extract_date(value)
            return cleaned if cleaned else value

        # ABN field
        elif "ABN" in field_upper:
            cleaned = cls.extract_abn(value)
            return cleaned if cleaned else value

        # BSB field
        elif "BSB" in field_upper:
            cleaned = cls.extract_bsb(value)
            return cleaned if cleaned else value

        # Percentage fields
        elif any(pct in field_upper for pct in ["RATE", "PERCENTAGE", "DISCOUNT"]):
            cleaned = cls.extract_percentage(value)
            return cleaned if cleaned else value

        # Default text cleaning
        else:
            # Remove excessive whitespace
            cleaned = " ".join(value.split())
            # Remove special extraction markers
            cleaned = re.sub(r"^\*+\s*|\s*\*+$", "", cleaned)
            cleaned = re.sub(r"^[-:]+\s*|\s*[-:]+$", "", cleaned)
            return cleaned.strip()

    @classmethod
    def extract_all_patterns(cls, text: str) -> Dict[str, List[str]]:
        """Extract all recognized patterns from text."""
        results = {}

        for pattern_name, pattern in cls.PATTERNS.items():
            matches = pattern.findall(text)
            if matches:
                results[pattern_name] = matches

        return results

    @classmethod
    def validate_field_format(cls, field_name: str, value: str) -> Tuple[bool, Optional[str]]:
        """Validate field format and return (is_valid, error_message)."""
        if not value:
            return True, None

        field_pattern = cls.get_field_pattern(field_name)
        if not field_pattern:
            return True, None  # No validation for unknown fields

        # Check if value matches expected pattern
        if field_pattern.extract(value):
            return True, None
        else:
            return False, f"Value '{value}' doesn't match expected format: {field_pattern.example}"


class FieldCleaner:
    """Specialized field cleaning logic."""

    @staticmethod
    def clean_currency(value: str) -> str:
        """Clean and format currency values."""
        cleaned = PatternLibrary.extract_currency(value)
        return cleaned if cleaned else value

    @staticmethod
    def clean_date(value: str) -> str:
        """Clean and format date values."""
        cleaned = PatternLibrary.extract_date(value)
        return cleaned if cleaned else value

    @staticmethod
    def clean_abn(value: str) -> str:
        """Clean and format ABN."""
        cleaned = PatternLibrary.extract_abn(value)
        return cleaned if cleaned else value

    @staticmethod
    def clean_bsb(value: str) -> str:
        """Clean and format BSB."""
        cleaned = PatternLibrary.extract_bsb(value)
        return cleaned if cleaned else value

    @staticmethod
    def clean_text(value: str) -> str:
        """Clean general text fields."""
        # Remove excessive whitespace
        cleaned = " ".join(value.split())
        # Remove common markers
        cleaned = re.sub(r"^\*+\s*|\s*\*+$", "", cleaned)
        cleaned = re.sub(r"^[-:]+\s*|\s*[-:]+$", "", cleaned)
        return cleaned.strip()


class FieldValidator:
    """Field validation logic."""

    @staticmethod
    def validate_abn(abn: str) -> bool:
        """Validate Australian Business Number format."""
        digits = re.sub(r"\D", "", abn)
        return len(digits) == 11

    @staticmethod
    def validate_bsb(bsb: str) -> bool:
        """Validate Bank State Branch format."""
        digits = re.sub(r"\D", "", bsb)
        return len(digits) == 6

    @staticmethod
    def validate_postcode(postcode: str) -> bool:
        """Validate Australian postcode."""
        digits = re.sub(r"\D", "", postcode)
        return len(digits) == 4 and digits.isdigit()

    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format."""
        pattern = PatternLibrary.PATTERNS.get("EMAIL")
        return bool(pattern and pattern.match(email))

    @staticmethod
    def validate_currency(value: str) -> bool:
        """Validate currency format."""
        try:
            # Remove $ and commas, then try to parse as float
            cleaned = re.sub(r"[$,]", "", value)
            float(cleaned)
            return True
        except ValueError:
            return False
