"""Dynamic Field Extractor - Original Working Logic
=================================================

This module implements the exact field extraction logic from the working
model_comparison_working_backup_monday.py script that achieved 100% success rates.

Key differences from ProductionExtractor:
- Dynamic field detection (not limited to 55 production fields)
- Raw markdown fallback extraction
- Simple key-value pair counting
- Matches original working script behavior exactly
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class DynamicExtractionResult:
    """Result from dynamic field extraction matching original script format."""

    image_name: str
    model_name: str
    raw_response: str
    cleaned_response: str

    # Extracted fields (dynamic - whatever we find)
    extracted_fields: Dict[str, str]
    field_count: int

    # Success determination
    is_successful: bool
    extraction_score: int  # Number of fields found

    # Processing metadata
    processing_time: float
    using_raw_markdown: bool = False
    processing_notes: List[str] = None

    def __post_init__(self):
        if self.processing_notes is None:
            self.processing_notes = []


class DynamicFieldExtractor:
    """Dynamic field extractor using original working script logic."""

    def __init__(self, min_fields_for_success: int = 3):
        """Initialize dynamic extractor.

        Args:
            min_fields_for_success: Minimum fields needed to consider extraction successful
        """
        self.min_fields_for_success = min_fields_for_success

    def extract_fields(
        self, response: str, image_name: str, model_name: str, processing_time: float = 0.0
    ) -> DynamicExtractionResult:
        """Extract fields dynamically from response using original working logic.

        Args:
            response: Cleaned response from repetition controller
            image_name: Name of the image being processed
            model_name: Name of the model used
            processing_time: Time taken for model inference

        Returns:
            DynamicExtractionResult with fields found
        """
        # Step 1: Detect if we have structured key-value pairs
        if self._has_valid_keyvalue_pairs(response):
            # Structured extraction
            using_raw_markdown = False
            detected_fields = self._get_field_names_from_response(response)
        else:
            # Raw markdown fallback
            using_raw_markdown = True
            detected_fields = self._extract_fields_from_raw_markdown(response)

        # Step 2: Extract each detected field
        extracted_fields = {}
        field_matches = {}

        for field_name in detected_fields:
            if using_raw_markdown:
                field_detected, field_match = self._extract_field_from_raw_markdown(field_name, response)
            else:
                field_detected, field_match = self._extract_and_validate_field_simple(field_name, response)

            if field_detected and field_match:
                extracted_fields[field_name] = field_match
                field_matches[field_name.lower()] = field_match

        # Step 3: Calculate success metrics
        extraction_score = len(extracted_fields)
        is_successful = extraction_score >= self.min_fields_for_success

        return DynamicExtractionResult(
            image_name=image_name,
            model_name=model_name,
            raw_response=response,  # Already cleaned by repetition controller
            cleaned_response=response,
            extracted_fields=extracted_fields,
            field_count=extraction_score,
            is_successful=is_successful,
            extraction_score=extraction_score,
            processing_time=processing_time,
            using_raw_markdown=using_raw_markdown,
            processing_notes=[
                f"Dynamic field detection: {len(detected_fields)} fields detected",
                f"Extraction mode: {'Raw markdown' if using_raw_markdown else 'Structured key-value'}",
                f"Success: {extraction_score} fields extracted",
            ],
        )

    def _get_field_names_from_response(self, response: str) -> List[str]:
        """Dynamically extract field names from response (original logic)."""
        # Find all "FIELD:" patterns in the response
        field_pattern = r"([A-Z_]+):\\s*"
        matches = re.findall(field_pattern, response)

        # Clean and deduplicate
        fields = []
        seen = set()
        for match in matches:
            field = match.strip().upper()
            if field and field not in seen and len(field) > 1:
                fields.append(field)
                seen.add(field)

        return fields

    def _has_valid_keyvalue_pairs(self, text: str) -> bool:
        """Check if text contains valid key-value pairs after conversion (original logic)."""
        if not text or not text.strip():
            return False

        lines = text.split("\\n")
        valid_pairs = 0

        for line in lines:
            line = line.strip()
            if ":" in line and len(line.split(":", 1)) == 2:
                key, value = line.split(":", 1)
                if (
                    key.strip()
                    and value.strip()
                    and len(key.strip()) > 1
                    and len(value.strip()) > 1
                    and value.strip().upper() not in ["N/A", "NOT VISIBLE", "NONE", "UNKNOWN"]
                ):
                    valid_pairs += 1

        # Consider conversion successful if we have at least 3 valid key-value pairs
        return valid_pairs >= 3

    def _extract_and_validate_field_simple(
        self, field_name: str, response: str
    ) -> Tuple[bool, Optional[str]]:
        """Extract and validate a specific field from the response (original logic)."""
        # Try structured extraction first
        pattern = rf'(?:{field_name}|{field_name.lower()}):\\s*"?([^"\\n]+)"?'
        match = re.search(pattern, response, re.IGNORECASE)

        if match:
            value = match.group(1).strip()
            # Basic validation
            if (
                value
                and len(value) > 0
                and value.upper() not in ["N/A", "NOT VISIBLE", "NONE", "UNKNOWN", "-"]
            ):
                return True, value

        return False, None

    def _extract_fields_from_raw_markdown(self, response: str) -> List[str]:
        """Extract synthetic field names from raw markdown content (original logic)."""
        synthetic_fields = []

        # Check for specific content types and create synthetic fields
        if re.search(r"\\b\\d{2,3}[\\s-]\\d{3}[\\s-]\\d{3}[\\s-]\\d{3}\\b", response):
            synthetic_fields.append("ABN")

        if re.search(r"\\$\\d+\\.\\d{2}", response):
            synthetic_fields.append("TOTAL")
            synthetic_fields.append("SUBTOTAL")
            synthetic_fields.append("GST")

        if re.search(r"\\d{1,2}/\\d{1,2}/\\d{4}", response):
            synthetic_fields.append("DATE")

        if re.search(r"\\b[A-Z][a-z]+(?:\\s+[A-Z][a-z]+)+\\b", response):
            synthetic_fields.append("SUPPLIER")
            synthetic_fields.append("STORE")

        if re.search(r"\\b\\d{3}-\\d{3}\\b", response):
            synthetic_fields.append("BSB")

        if re.search(r"\\b\\d{6,12}\\b", response):
            synthetic_fields.append("ACCOUNT_NUMBER")

        return synthetic_fields

    def _extract_field_from_raw_markdown(
        self, field_name: str, response: str
    ) -> Tuple[bool, Optional[str]]:
        """Extract specific field from raw markdown content using pattern matching (original logic)."""
        field_name_upper = field_name.upper()

        # Define extraction patterns for each field type
        patterns = {
            "ABN": [r"\\b(\\d{2,3}[\\s-]\\d{3}[\\s-]\\d{3}[\\s-]\\d{3})\\b"],
            "TOTAL": [r"\\$?(\\d+\\.\\d{2})", r"total[:\\s]*(\\$?\\d+\\.\\d{2})"],
            "SUBTOTAL": [
                r"subtotal[:\\s]*(\\$?\\d+\\.\\d{2})",
                r"sub[\\s-]?total[:\\s]*(\\$?\\d+\\.\\d{2})",
            ],
            "GST": [r"gst[:\\s]*(\\$?\\d+\\.\\d{2})", r"tax[:\\s]*(\\$?\\d+\\.\\d{2})"],
            "DATE": [r"(\\d{1,2}/\\d{1,2}/\\d{4})", r"(\\d{1,2}-\\d{1,2}-\\d{4})"],
            "SUPPLIER": [r"^([A-Z][a-z]+(?:\\s+[A-Z][a-z]+)+)", r"supplier[:\\s]*([A-Za-z].*?)(?:\\n|$)"],
            "STORE": [r"^([A-Z][a-z]+(?:\\s+[A-Z][a-z]+)+)", r"store[:\\s]*([A-Za-z].*?)(?:\\n|$)"],
            "BSB": [r"\\b(\\d{3}-\\d{3})\\b"],
            "ACCOUNT_NUMBER": [r"\\b(\\d{6,12})\\b"],
        }

        if field_name_upper in patterns:
            for pattern in patterns[field_name_upper]:
                match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
                if match:
                    value = match.group(1).strip()
                    if value and len(value) > 0:
                        return True, value

        return False, None
