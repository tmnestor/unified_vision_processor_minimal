"""Production-Ready Field Extractor
================================

Enhanced field extraction system that integrates the UniversalKeyValueParser
with the production field schema for Australian Tax Office document processing.
"""

import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from ..config.production_schema import PRODUCTION_SCHEMA, FieldType
from ..utils.repetition_control import UltraAggressiveRepetitionController
from .universal_key_value_parser import UniversalKeyValueParser


@dataclass
class ExtractionResult:
    """Standardized extraction result format."""

    # Basic information
    image_name: str
    model_name: str
    extraction_time: float

    # Raw data
    raw_response: str
    cleaned_response: str

    # Extracted fields
    extracted_fields: Dict[str, Any]
    field_count: int

    # Validation results
    core_fields_found: List[str]
    required_fields_found: List[str]
    missing_core_fields: List[str]
    missing_required_fields: List[str]

    # Success metrics
    is_successful: bool
    confidence_score: float

    # Field categorization
    fields_by_category: Dict[str, List[str]]

    # Analysis metadata
    has_structured_output: bool
    raw_markdown_fallback_used: bool
    awk_processing_used: bool = False
    processing_notes: List[str] = None

    def __post_init__(self):
        if self.processing_notes is None:
            self.processing_notes = []


class ProductionExtractor:
    """Production-ready field extractor with comprehensive validation and fallback strategies."""

    def __init__(
        self,
        enable_repetition_control: bool = True,
        enable_awk_processing: bool = True,
        strict_validation: bool = False,
    ):
        """Initialize production extractor.

        Args:
            enable_repetition_control: Enable aggressive repetition cleaning
            enable_awk_processing: Enable AWK-style markdown processing
            strict_validation: Use strict field validation rules
        """
        self.production_schema = PRODUCTION_SCHEMA
        self.strict_validation = strict_validation

        # Initialize parser with production schema
        parser_schema = self.production_schema.create_universal_parser_schema()
        self.universal_parser = UniversalKeyValueParser(parser_schema)

        # Initialize repetition controller
        self.repetition_controller = None
        if enable_repetition_control:
            try:
                self.repetition_controller = UltraAggressiveRepetitionController()
            except Exception as e:
                print(f"⚠️  Failed to initialize repetition controller: {e}")

        # Success criteria from production schema
        self.success_criteria = self.production_schema.get_success_criteria()

    def extract_fields(self, raw_response: str, image_name: str, model_name: str) -> ExtractionResult:
        """Extract fields from model response using production schema.

        Args:
            raw_response: Raw response text from vision model
            image_name: Name of the processed image
            model_name: Name of the model used

        Returns:
            ExtractionResult with comprehensive extraction data
        """
        start_time = time.time()

        # Step 1: Clean response using repetition controller
        cleaned_response = self._clean_response(raw_response, image_name)

        # Step 2: Extract fields using multiple strategies
        extracted_fields, processing_notes = self._extract_with_multiple_strategies(
            cleaned_response, image_name
        )

        # Step 3: Validate and categorize extracted fields
        validation_results = self._validate_extracted_fields(extracted_fields)

        # Step 4: Calculate success metrics
        success_metrics = self._calculate_success_metrics(extracted_fields, validation_results)

        # Step 5: Categorize fields by type
        fields_by_category = self._categorize_extracted_fields(extracted_fields)

        extraction_time = time.time() - start_time

        return ExtractionResult(
            image_name=image_name,
            model_name=model_name,
            extraction_time=extraction_time,
            raw_response=raw_response,
            cleaned_response=cleaned_response,
            extracted_fields=extracted_fields,
            field_count=len(extracted_fields),
            core_fields_found=validation_results["core_fields_found"],
            required_fields_found=validation_results["required_fields_found"],
            missing_core_fields=validation_results["missing_core_fields"],
            missing_required_fields=validation_results["missing_required_fields"],
            is_successful=success_metrics["is_successful"],
            confidence_score=success_metrics["confidence_score"],
            fields_by_category=fields_by_category,
            has_structured_output=self._has_structured_output(cleaned_response),
            raw_markdown_fallback_used=success_metrics["raw_markdown_fallback_used"],
            processing_notes=processing_notes,
        )

    def _clean_response(self, raw_response: str, image_name: str) -> str:
        """Clean raw response using repetition controller."""
        if not self.repetition_controller:
            return raw_response

        try:
            return self.repetition_controller.clean_response(raw_response, image_name)
        except Exception as e:
            print(f"⚠️  Repetition cleaning failed for {image_name}: {e}")
            return raw_response

    def _extract_with_multiple_strategies(
        self, cleaned_response: str, image_name: str
    ) -> Tuple[Dict[str, Any], List[str]]:
        """Extract fields using multiple strategies with fallbacks."""
        processing_notes = []

        # Strategy 1: Use UniversalKeyValueParser for structured extraction
        structured_fields = self.universal_parser.parse(cleaned_response)

        if structured_fields:
            processing_notes.append("Primary extraction: UniversalKeyValueParser succeeded")
            return structured_fields, processing_notes

        # Strategy 2: Manual pattern extraction for production fields
        processing_notes.append("Primary extraction failed, trying manual pattern extraction")
        manual_fields = self._manual_pattern_extraction(cleaned_response)

        if manual_fields:
            processing_notes.append("Manual pattern extraction succeeded")
            return manual_fields, processing_notes

        # Strategy 3: Raw markdown content analysis (fallback)
        processing_notes.append("Structured extraction failed, using raw markdown analysis")
        raw_fields = self._raw_markdown_extraction(cleaned_response)

        if raw_fields:
            processing_notes.append("Raw markdown extraction succeeded")
        else:
            processing_notes.append("All extraction strategies failed")

        return raw_fields, processing_notes

    def _manual_pattern_extraction(self, text: str) -> Dict[str, Any]:
        """Manual pattern-based extraction using production schema patterns."""
        extracted = {}

        for field_name in self.production_schema.get_all_fields():
            field_definition = self.production_schema.get_field_definition(field_name)
            if not field_definition or not field_definition.extraction_patterns:
                continue

            # Try each extraction pattern for the field
            for pattern in field_definition.extraction_patterns:
                try:
                    match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                    if match:
                        value = match.group(1).strip()
                        if self._is_valid_field_value(field_name, value):
                            extracted[field_name] = self._clean_field_value(field_name, value)
                            break
                except Exception:
                    continue  # Skip invalid patterns

        return extracted

    def _raw_markdown_extraction(self, text: str) -> Dict[str, Any]:
        """Extract fields from raw markdown content when structured extraction fails."""
        extracted = {}

        # Define patterns for common field types that can be detected in raw text
        field_patterns = {
            # Financial patterns
            "total_a_li": [r"\$(\d+\.?\d*)", r"total:\s*\$?(\d+\.?\d*)", r"amount:\s*\$?(\d+\.?\d*)"],
            "subtotal_a_li": [r"subtotal:\s*\$?(\d+\.?\d*)", r"sub[\s-]?total:\s*\$?(\d+\.?\d*)"],
            "tax_a_li": [r"tax:\s*\$?(\d+\.?\d*)", r"gst:\s*\$?(\d+\.?\d*)"],
            # Date patterns
            "date_a_li": [r"(\d{1,2}/\d{1,2}/\d{4})", r"(\d{1,2}-\d{1,2}-\d{4})"],
            "invDate_a_pgs": [r"invoice.*?(\d{1,2}/\d{1,2}/\d{4})", r"date.*?(\d{1,2}/\d{1,2}/\d{4})"],
            # ABN pattern
            "supplierABN_a_pgs": [
                r"abn:?\s*(\d{2}\s?\d{3}\s?\d{3}\s?\d{3})",
                r"(\d{2}\s\d{3}\s\d{3}\s\d{3})",
                r"australian business number:?\s*(\d{11})",
            ],
            # Supplier/business name
            "supplier_a_pgs": [
                r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)",  # Title case business names
                r"supplier:?\s*([A-Za-z].*?)(?:\n|$)",
                r"business:?\s*([A-Za-z].*?)(?:\n|$)",
            ],
        }

        for field_name, patterns in field_patterns.items():
            for pattern in patterns:
                try:
                    match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                    if match:
                        value = match.group(1).strip()
                        if self._is_valid_field_value(field_name, value):
                            extracted[field_name] = self._clean_field_value(field_name, value)
                            break
                except Exception:
                    continue

        return extracted

    def _is_valid_field_value(self, field_name: str, value: str) -> bool:
        """Check if field value is valid according to production schema."""
        if not value or not value.strip():
            return False

        # Use production schema validation if available
        if self.production_schema.validate_field_value(field_name, value):
            return True

        # Fallback validation for non-strict mode
        if not self.strict_validation:
            # Basic validation - not empty and not in common invalid values
            invalid_values = ["N/A", "NA", "NOT AVAILABLE", "NOT FOUND", "NONE", "-", "UNKNOWN"]
            return value.upper() not in invalid_values

        return False

    def _clean_field_value(self, field_name: str, value: str) -> Any:
        """Clean field value according to its type definition."""
        field_definition = self.production_schema.get_field_definition(field_name)
        if not field_definition:
            return value.strip()

        field_type = field_definition.field_type

        # Clean based on field type
        if field_type == FieldType.CURRENCY:
            return self._clean_currency_value(value)
        elif field_type == FieldType.DATE:
            return self._clean_date_value(value)
        elif field_type == FieldType.ABN:
            return self._clean_abn_value(value)
        elif field_type == FieldType.BSB:
            return self._clean_bsb_value(value)
        elif field_type == FieldType.INTEGER:
            return self._clean_integer_value(value)
        elif field_type == FieldType.NUMERIC:
            return self._clean_numeric_value(value)
        else:
            return value.strip()

    def _clean_currency_value(self, value: str) -> str:
        """Clean currency value to standard format."""
        # Remove currency symbols and clean
        cleaned = re.sub(r"[^\d.-]", "", value)
        try:
            amount = float(cleaned)
            return f"${amount:.2f}"
        except ValueError:
            return value.strip()

    def _clean_date_value(self, value: str) -> str:
        """Clean date value to DD/MM/YYYY format."""
        # Basic date cleaning - ensure DD/MM/YYYY format
        date_match = re.search(r"(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})", value)
        if date_match:
            day, month, year = date_match.groups()
            if len(year) == 2:
                year = "20" + year  # Assume 2000s for 2-digit years
            return f"{day.zfill(2)}/{month.zfill(2)}/{year}"
        return value.strip()

    def _clean_abn_value(self, value: str) -> str:
        """Clean ABN to standard XX XXX XXX XXX format."""
        digits = re.sub(r"\D", "", value)
        if len(digits) == 11:
            return f"{digits[:2]} {digits[2:5]} {digits[5:8]} {digits[8:11]}"
        return value.strip()

    def _clean_bsb_value(self, value: str) -> str:
        """Clean BSB to standard XXX-XXX format."""
        digits = re.sub(r"\D", "", value)
        if len(digits) == 6:
            return f"{digits[:3]}-{digits[3:6]}"
        return value.strip()

    def _clean_integer_value(self, value: str) -> int:
        """Extract integer value."""
        digits = re.sub(r"\D", "", value)
        try:
            return int(digits) if digits else 0
        except ValueError:
            return 0

    def _clean_numeric_value(self, value: str) -> float:
        """Extract numeric value."""
        cleaned = re.sub(r"[^\d.-]", "", value)
        try:
            return float(cleaned) if cleaned else 0.0
        except ValueError:
            return 0.0

    def _validate_extracted_fields(self, extracted_fields: Dict[str, Any]) -> Dict[str, Any]:
        """Validate extracted fields against production schema."""
        core_fields = self.production_schema.get_core_fields()
        required_fields = self.production_schema.get_required_fields()

        core_fields_found = [f for f in extracted_fields.keys() if f in core_fields]
        required_fields_found = [f for f in extracted_fields.keys() if f in required_fields]

        missing_core_fields = [f for f in core_fields if f not in extracted_fields]
        missing_required_fields = [f for f in required_fields if f not in extracted_fields]

        return {
            "core_fields_found": core_fields_found,
            "required_fields_found": required_fields_found,
            "missing_core_fields": missing_core_fields,
            "missing_required_fields": missing_required_fields,
            "total_fields_found": len(extracted_fields),
            "core_fields_ratio": len(core_fields_found) / len(core_fields) if core_fields else 0,
            "required_fields_ratio": len(required_fields_found) / len(required_fields)
            if required_fields
            else 1,
        }

    def _calculate_success_metrics(
        self, extracted_fields: Dict[str, Any], validation_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate success metrics based on extraction results."""
        criteria = self.success_criteria

        # Check success conditions
        has_min_core_fields = len(validation_results["core_fields_found"]) >= criteria["min_core_fields"]
        has_min_total_fields = validation_results["total_fields_found"] >= criteria["min_total_fields"]
        has_all_required_fields = len(validation_results["missing_required_fields"]) == 0

        # Overall success determination
        is_successful = has_min_total_fields and (has_min_core_fields or has_all_required_fields)

        # Calculate confidence score (0.0 to 1.0)
        confidence_components = [
            validation_results["core_fields_ratio"] * 0.4,  # 40% weight on core fields
            min(validation_results["total_fields_found"] / 10, 1.0)
            * 0.3,  # 30% weight on total fields (cap at 10)
            validation_results["required_fields_ratio"] * 0.3,  # 30% weight on required fields
        ]
        confidence_score = sum(confidence_components)

        # Check if raw markdown fallback was used
        raw_markdown_fallback_used = (
            validation_results["total_fields_found"] > 0 and not self._has_structured_output
        )

        return {
            "is_successful": is_successful,
            "confidence_score": confidence_score,
            "has_min_core_fields": has_min_core_fields,
            "has_min_total_fields": has_min_total_fields,
            "has_all_required_fields": has_all_required_fields,
            "raw_markdown_fallback_used": raw_markdown_fallback_used,
        }

    def _categorize_extracted_fields(self, extracted_fields: Dict[str, Any]) -> Dict[str, List[str]]:
        """Categorize extracted fields by their schema categories."""
        categories = {}

        for field_name in extracted_fields.keys():
            field_definition = self.production_schema.get_field_definition(field_name)
            if field_definition:
                category = field_definition.category.value
                if category not in categories:
                    categories[category] = []
                categories[category].append(field_name)

        return categories

    def _has_structured_output(self, text: str) -> bool:
        """Check if response contains structured KEY: VALUE pairs."""
        # Count lines that look like KEY: VALUE format
        lines = text.strip().split("\n")
        structured_lines = 0

        for line in lines:
            line = line.strip()
            if ":" in line and len(line.split(":", 1)) == 2:
                key, value = line.split(":", 1)
                if key.strip() and value.strip():
                    structured_lines += 1

        # Consider structured if at least 3 lines are in KEY: VALUE format
        return structured_lines >= 3

    def create_extraction_summary(self, result: ExtractionResult) -> Dict[str, Any]:
        """Create a summary of extraction results for analysis."""
        return {
            "image_name": result.image_name,
            "model_name": result.model_name,
            "extraction_time": result.extraction_time,
            "field_count": result.field_count,
            "is_successful": result.is_successful,
            "confidence_score": result.confidence_score,
            "core_fields_count": len(result.core_fields_found),
            "required_fields_count": len(result.required_fields_found),
            "categories_found": list(result.fields_by_category.keys()),
            "has_structured_output": result.has_structured_output,
            "fallback_used": result.raw_markdown_fallback_used,
            "processing_notes": result.processing_notes,
        }
