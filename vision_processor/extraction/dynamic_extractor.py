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

    # Working script compatibility - has_* fields
    field_results: Dict[str, bool] = None

    # Processing metadata
    processing_time: float
    using_raw_markdown: bool = False
    processing_notes: List[str] = None

    def __post_init__(self):
        if self.processing_notes is None:
            self.processing_notes = []
        if self.field_results is None:
            self.field_results = {}


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
        # Step 1: Field detection exactly like working script
        # Dynamically detect fields from response
        detected_fields = self._get_field_names_from_response(response)

        # Check if response is structured (contains any field patterns)
        is_structured = len(detected_fields) > 0

        # If no structured fields found, try to detect content in raw markdown format
        if not is_structured and response:
            # For raw markdown fallback, count meaningful content as successful extraction
            content_indicators = self._detect_raw_markdown_content(response)
            if content_indicators > 0:
                is_structured = True
                # Create synthetic field detection for raw markdown content
                detected_fields = self._extract_fields_from_raw_markdown(response)

        # EXACT WORKING SCRIPT LOGIC: Determine extraction method
        initial_structured_fields = self._get_field_names_from_response(response)
        using_raw_markdown = len(initial_structured_fields) == 0 and len(detected_fields) > 0

        # Step 2: Extract each detected field (EXACT WORKING SCRIPT LOGIC)
        extracted_fields = {}
        field_matches = {}
        field_results = {}

        for field_name in detected_fields:
            if using_raw_markdown:
                field_detected, field_match = self._extract_field_from_raw_markdown(field_name, response)
            else:
                field_detected, field_match = self._extract_and_validate_field_simple(field_name, response)

            # CRITICAL: Working script counts field as successful if DETECTED, not if value extracted
            field_results[f"has_{field_name.lower()}"] = field_detected
            field_matches[field_name.lower()] = field_match
            
            # Only add to extracted_fields if we have a value (for compatibility)
            if field_detected and field_match:
                extracted_fields[field_name] = field_match

        # Step 3: Calculate success metrics (EXACT WORKING SCRIPT LOGIC)
        # Count number of True values in field_results (detected fields)
        all_scores = [field_results.get(key, False) for key in field_results.keys()]
        extraction_score = sum(all_scores)
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
            field_results=field_results,  # Working script compatibility
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
        field_pattern = r"([A-Z_]+):\s*"
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

        lines = text.split("\n")
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
        pattern = rf'(?:{field_name}|{field_name.lower()}):\s*"?([^"\n]+)"?'
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
        """Extract synthetic field names from raw markdown content (WORKING SCRIPT LOGIC - SIMPLE 6 FIELDS)."""
        synthetic_fields = []

        # Check for specific content types and create synthetic fields (EXACTLY LIKE WORKING SCRIPT)
        if re.search(r"\b\d{2,3}[\s-]\d{3}[\s-]\d{3}[\s-]\d{3}\b", response):
            synthetic_fields.append("ABN")

        if re.search(r"\$\d+\.\d{2}", response):
            synthetic_fields.append("TOTAL")
            synthetic_fields.append("AMOUNT")

        if re.search(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b", response):
            synthetic_fields.append("STORE")
            synthetic_fields.append("BUSINESS_NAME")

        if re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", response):
            synthetic_fields.append("DATE")

        if re.search(r"\b\d{4,}\b", response):
            synthetic_fields.append("RECEIPT_NUMBER")

        return synthetic_fields

    def _detect_raw_markdown_content(self, response: str) -> int:
        """Detect meaningful content indicators in raw markdown format (WORKING SCRIPT LOGIC)."""
        content_score = 0

        # Look for business-specific patterns that indicate successful extraction
        if re.search(r"\b\d{2,3}[\s-]\d{3}[\s-]\d{3}[\s-]\d{3}\b", response):
            content_score += 2  # ABN pattern found

        if re.search(r"\$\d+\.\d{2}", response):
            content_score += 2  # Currency amounts found

        if re.search(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b", response):
            content_score += 1  # Business names found

        if re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", response):
            content_score += 1  # Date patterns found

        if re.search(r"\b\d{4,}\b", response):
            content_score += 1  # Potential receipt numbers

        # Any meaningful content above threshold indicates successful extraction
        return content_score

    def _extract_field_from_raw_markdown(
        self, field_name: str, response: str
    ) -> Tuple[bool, Optional[str]]:
        """Extract specific field from raw markdown content using pattern matching (WORKING SCRIPT LOGIC)."""
        field_name_upper = field_name.upper()

        if field_name_upper == "ABN":
            match = re.search(r"\b(\d{2,3}[\s-]?\d{3}[\s-]?\d{3}[\s-]?\d{3})\b", response)
            if match:
                return True, match.group(1)

        elif field_name_upper in ["TOTAL", "AMOUNT"]:
            # Look for currency amounts, prefer the largest one as total
            amounts = re.findall(r"\$(\d+\.\d{2})", response)
            if amounts:
                # Return the largest amount as the most likely total
                max_amount = max(amounts, key=lambda x: float(x))
                return True, f"${max_amount}"

        elif field_name_upper in ["STORE", "BUSINESS_NAME"]:
            # Look for title case business names
            match = re.search(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", response)
            if match:
                return True, match.group(1)

        elif field_name_upper == "DATE":
            # Look for date patterns
            match = re.search(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b", response)
            if match:
                return True, match.group(1)

        elif field_name_upper == "RECEIPT_NUMBER":
            # Look for any 4+ digit number
            match = re.search(r"\b(\d{4,})\b", response)
            if match:
                return True, match.group(1)

        return False, None
