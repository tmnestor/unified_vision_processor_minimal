#!/usr/bin/env python3
"""
AWK-like Markdown Processor - Modular Implementation
==================================================

This module implements AWK-style text processing for markdown-to-keyvalue conversion.
It provides multi-stage processing, state machine context tracking, and configurable
pattern matching with graceful fallback chains.

Design Pattern: AWK-like BEGIN/MAIN/END processing with YAML configuration.

Usage:
    processor = AWKMarkdownProcessor("markdown_processing_config.yaml")
    result = processor.process(markdown_text, image_name="optional")
"""

import re
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List

import yaml


class ProcessingState(Enum):
    """State machine states for markdown processing context."""

    UNKNOWN = "unknown"
    TABLE_HEADER = "table_header"
    TABLE_ROW = "table_row"
    BULLET_LIST = "bullet_list"
    STRUCTURED_KV = "structured_kv"
    PLAIN_TEXT = "plain_text"
    ERROR_FALLBACK = "error_fallback"


@dataclass
class ProcessingContext:
    """Context object for tracking processing state."""

    state: ProcessingState = ProcessingState.UNKNOWN
    detected_format: str = "unknown"
    line_number: int = 0
    total_lines: int = 0
    extracted_fields: Dict[str, str] = None
    confidence_score: float = 0.0
    processing_notes: List[str] = None

    def __post_init__(self):
        if self.extracted_fields is None:
            self.extracted_fields = {}
        if self.processing_notes is None:
            self.processing_notes = []


@dataclass
class ProcessingResult:
    """Result object containing processed output and metadata."""

    processed_text: str
    extracted_fields: Dict[str, str]
    original_format: str
    final_format: str
    success: bool
    confidence_score: float
    processing_time: float
    notes: List[str]
    fallback_used: bool = False


class AWKMarkdownProcessor:
    """AWK-style markdown processor with multi-stage processing and state machine."""

    def __init__(self, config_path: str = "markdown_processing_config.yaml"):
        """Initialize processor with YAML configuration.

        Args:
            config_path: Path to YAML configuration file
        """
        self.config = self._load_config(config_path)
        self.context = ProcessingContext()
        self._validate_config()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load and validate YAML configuration."""
        config_file = Path(config_path)

        if not config_file.exists():
            raise FileNotFoundError(f"Markdown processing config not found: {config_file.absolute()}")

        with config_file.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        return config

    def _validate_config(self):
        """Validate configuration structure - FAIL FAST design."""
        required_sections = [
            "format_detection",
            "processing_stages",
            "state_machine",
            "format_processors",
            "fallback_patterns",
            "validation_rules",
        ]

        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required config section: {section}")

    def process(self, text: str, image_name: str = "") -> ProcessingResult:
        """Main processing entry point - AWK-style processing pipeline.

        Args:
            text: Markdown text to process
            image_name: Optional image name for logging

        Returns:
            ProcessingResult with processed text and metadata
        """
        start_time = time.time()
        self.context = ProcessingContext()

        try:
            # AWK-style BEGIN phase
            text = self._begin_phase(text, image_name)

            # AWK-style MAIN phase
            text = self._main_phase(text)

            # AWK-style END phase
            text = self._end_phase(text)

            # Build result
            processing_time = time.time() - start_time
            success = self._validate_final_result(text)

            return ProcessingResult(
                processed_text=text,
                extracted_fields=self.context.extracted_fields,
                original_format=self.context.detected_format,
                final_format="keyvalue" if success else "raw",
                success=success,
                confidence_score=self.context.confidence_score,
                processing_time=processing_time,
                notes=self.context.processing_notes,
                fallback_used=any("fallback" in note.lower() for note in self.context.processing_notes),
            )

        except Exception as e:
            # Emergency fallback
            processing_time = time.time() - start_time
            self.context.processing_notes.append(f"ERROR: {str(e)} - using raw text")

            return ProcessingResult(
                processed_text=text,
                extracted_fields={},
                original_format="unknown",
                final_format="raw",
                success=False,
                confidence_score=0.0,
                processing_time=processing_time,
                notes=self.context.processing_notes,
                fallback_used=True,
            )

    def _begin_phase(self, text: str, image_name: str) -> str:
        """BEGIN phase: Format detection and preparation."""
        if self.config.get("debug_settings", {}).get("log_format_detection", False):
            print(f"🔍 BEGIN: Processing {image_name} ({len(text)} chars)")

        # Normalize whitespace
        text = self._normalize_whitespace(text)

        # Detect format
        detected_format = self._detect_format(text)
        self.context.detected_format = detected_format
        self.context.processing_notes.append(f"Detected format: {detected_format}")

        # Initialize context
        self.context.total_lines = len(text.split("\n"))
        self.context.state = self._format_to_initial_state(detected_format)

        return text

    def _main_phase(self, text: str) -> str:
        """MAIN phase: Content processing with state machine."""
        # Extract by detected format
        text = self._extract_by_format(text)

        # Apply fallback patterns if needed
        if (
            len(self.context.extracted_fields)
            < self.config["validation_rules"]["minimum_extractions"]["any_format"]
        ):
            text = self._apply_fallback_patterns(text)

        # Validate extractions
        self._validate_extractions()

        return text

    def _end_phase(self, text: str) -> str:
        """END phase: Cleanup and finalization."""
        # Clean artifacts
        text = self._clean_artifacts(text)

        # Format output
        text = self._format_output(text)

        # Final validation
        self.context.confidence_score = self._calculate_confidence_score(text)

        return text

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace while preserving structure."""
        # Preserve line breaks but clean up spaces
        lines = text.split("\n")
        cleaned_lines = []

        for line in lines:
            # Remove excessive spaces but preserve single spaces
            cleaned = re.sub(r"[ \t]+", " ", line.strip())
            if cleaned:  # Only keep non-empty lines
                cleaned_lines.append(cleaned)

        return "\n".join(cleaned_lines)

    def _detect_format(self, text: str) -> str:
        """Detect markdown format using configured patterns."""
        format_scores = {}

        for format_name, format_config in self.config["format_detection"].items():
            score = 0
            for pattern in format_config["patterns"]:
                matches = re.findall(pattern, text, re.MULTILINE | re.IGNORECASE)
                score += len(matches)

            if score >= format_config["confidence_threshold"]:
                format_scores[format_name] = score

        # Return format with highest score, or plain_text as fallback
        if format_scores:
            best_format = max(format_scores.items(), key=lambda x: x[1])[0]
            if self.config.get("debug_settings", {}).get("log_format_detection", False):
                print(f"📊 Format scores: {format_scores} -> {best_format}")
            return best_format
        else:
            return "plain_text"

    def _format_to_initial_state(self, format_name: str) -> ProcessingState:
        """Convert detected format to initial processing state."""
        format_state_map = {
            "markdown_table": ProcessingState.TABLE_HEADER,
            "bullet_list": ProcessingState.BULLET_LIST,
            "structured_keyvalue": ProcessingState.STRUCTURED_KV,
            "plain_text": ProcessingState.PLAIN_TEXT,
        }
        return format_state_map.get(format_name, ProcessingState.UNKNOWN)

    def _extract_by_format(self, text: str) -> str:
        """Extract content based on detected format."""
        format_name = self.context.detected_format

        if format_name not in self.config["format_processors"]:
            self.context.processing_notes.append(f"No processor for format: {format_name}")
            return text

        processor_config = self.config["format_processors"][format_name]

        if format_name == "markdown_table":
            return self._process_markdown_table(text, processor_config)
        elif format_name == "bullet_list":
            return self._process_bullet_list(text, processor_config)
        elif format_name == "structured_keyvalue":
            return self._process_structured_keyvalue(text, processor_config)
        else:
            return self._process_plain_text(text)

    def _process_markdown_table(self, text: str, config: Dict[str, Any]) -> str:
        """Process markdown table format."""
        lines = text.split("\n")
        result_lines = []

        for line in lines:
            # Skip separator lines
            if re.match(config["patterns"]["separator"], line):
                continue

            # Process data rows
            match = re.match(config["patterns"]["data_row"], line)
            if match and len(match.groups()) >= 2:
                key = match.group(config["key_column"] + 1).strip()
                value = match.group(config["value_column"] + 1).strip()

                # Clean key and value
                for cleanup_pattern in config["cleanup_patterns"]:
                    key = re.sub(cleanup_pattern, "", key).strip()
                    value = re.sub(cleanup_pattern, "", value).strip()

                if key and value:
                    # Normalize key
                    key = self._normalize_key(key)
                    self.context.extracted_fields[key] = value
                    result_lines.append(f"{key}: {value}")

        self.context.processing_notes.append(f"Table processing: {len(result_lines)} rows extracted")
        return "\n".join(result_lines)

    def _process_bullet_list(self, text: str, config: Dict[str, Any]) -> str:
        """Process bullet list format."""
        lines = text.split("\n")
        result_lines = []

        for line in lines:
            # Try bullet patterns
            for _pattern_name, pattern in config["patterns"].items():
                match = re.match(pattern, line)
                if match and len(match.groups()) >= 2:
                    key = match.group(1).strip()
                    value = match.group(2).strip()

                    if key and value:
                        key = self._normalize_key(key)
                        self.context.extracted_fields[key] = value
                        result_lines.append(f"{key}: {value}")
                        break

        self.context.processing_notes.append(f"Bullet processing: {len(result_lines)} items extracted")
        return "\n".join(result_lines)

    def _process_structured_keyvalue(self, text: str, config: Dict[str, Any]) -> str:
        """Process structured key-value format."""
        lines = text.split("\n")
        result_lines = []

        for line in lines:
            # Try key-value patterns
            for _pattern_name, pattern in config["patterns"].items():
                match = re.match(pattern, line)
                if match and len(match.groups()) >= 2:
                    key = match.group(1).strip()
                    value = match.group(2).strip()

                    if key and value:
                        # Apply business pattern normalization for known business fields
                        normalized_key, normalized_value = self._normalize_business_field(key, value, line)
                        self.context.extracted_fields[normalized_key] = normalized_value
                        result_lines.append(f"{normalized_key}: {normalized_value}")
                        break

        self.context.processing_notes.append(f"Structured processing: {len(result_lines)} pairs extracted")
        return "\n".join(result_lines)

    def _process_plain_text(self, text: str) -> str:
        """Process plain text using fallback patterns."""
        self.context.processing_notes.append("Using plain text processing")
        return self._apply_fallback_patterns(text)

    def _normalize_key(self, key: str) -> str:
        """Normalize key according to configuration."""
        key_config = (
            self.config["format_processors"].get("structured_keyvalue", {}).get("key_normalization", {})
        )

        # Apply normalization rules
        if key_config.get("uppercase", False):
            key = key.upper()

        if key_config.get("replace_spaces"):
            key = key.replace(" ", key_config["replace_spaces"])

        if key_config.get("remove_special_chars"):
            key = re.sub(key_config["remove_special_chars"], "", key)

        return key

    def _normalize_business_field(self, key: str, value: str, full_line: str) -> tuple[str, str]:
        """Normalize business fields using business pattern matching."""

        # Check if this field matches any business patterns
        business_patterns = self.config["fallback_patterns"].get("business_patterns", {})

        # Handle ABN specifically - most important for detection improvement
        if "abn" in key.lower() or re.search(r"(\d{2}\s*\d{3}\s*\d{3}\s*\d{3}|\d{11})", value):
            # This looks like an ABN field
            for pattern in business_patterns.get("ABN", []):
                matches = re.findall(pattern, full_line, re.IGNORECASE)
                if matches:
                    # Extract the ABN number properly
                    abn_value = matches[0] if isinstance(matches[0], str) else matches[0][0]
                    return "ABN", abn_value

        # Handle other common business field normalizations
        field_mappings = {
            # Business identification
            "business_name": "STORE",
            "company_name": "STORE",
            "supplier": "STORE",
            "business_abn": "ABN",
            "company_abn": "ABN",
            "abn_number": "ABN",
            # Financial fields
            "total_amount": "TOTAL",
            "amount_total": "TOTAL",
            "final_total": "TOTAL",
            "gst_amount": "GST",
            "tax_amount": "GST",
            # Dates and identifiers
            "invoice_date": "DATE",
            "receipt_date": "DATE",
            "transaction_date": "DATE",
            "receipt_no": "RECEIPT_NUMBER",
            "invoice_no": "INVOICE_NUMBER",
            "ref_number": "RECEIPT_NUMBER",
        }

        # Normalize key name
        normalized_key = self._normalize_key(key)
        key_lower = key.lower().replace(" ", "_").replace("-", "_")

        # Apply business field mapping
        if key_lower in field_mappings:
            normalized_key = field_mappings[key_lower]

        # Additional ABN detection from value patterns
        if normalized_key == "ABN" or re.search(r"(\d{2}\s*\d{3}\s*\d{3}\s*\d{3}|\d{11})", value):
            # Ensure ABN is properly formatted
            clean_abn = re.sub(r"[^\d]", "", value)
            if len(clean_abn) == 11:
                # Format as standard ABN: XX XXX XXX XXX
                formatted_abn = f"{clean_abn[:2]} {clean_abn[2:5]} {clean_abn[5:8]} {clean_abn[8:11]}"
                return "ABN", formatted_abn

        return normalized_key, value

    def _apply_fallback_patterns(self, text: str) -> str:
        """Apply fallback patterns in tier order."""
        result_lines = []
        fallback_config = self.config["fallback_patterns"]

        # Tier 1: Business-specific patterns
        for field_name, patterns in fallback_config.get("business_patterns", {}).items():
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    # Take first match for each field
                    value = matches[0] if isinstance(matches[0], str) else matches[0][0]
                    self.context.extracted_fields[field_name] = value
                    result_lines.append(f"{field_name}: {value}")
                    break

        # Tier 2: Generic patterns (if still not enough data)
        if len(self.context.extracted_fields) < 3:
            for pattern_name, pattern in fallback_config.get("generic_patterns", {}).items():
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    values = matches[:3]  # Take up to 3 matches
                    for i, value in enumerate(values):
                        field_name = f"{pattern_name}_{i + 1}" if i > 0 else pattern_name
                        value_str = value if isinstance(value, str) else value[0]
                        self.context.extracted_fields[field_name] = value_str
                        result_lines.append(f"{field_name}: {value_str}")

        self.context.processing_notes.append(f"Fallback processing: {len(result_lines)} patterns matched")
        return "\n".join(result_lines)

    def _validate_extractions(self):
        """Validate extracted fields according to rules."""
        validation_config = self.config["validation_rules"]["field_validation"]

        for field_name, field_value in self.context.extracted_fields.items():
            if field_name in validation_config:
                field_rules = validation_config[field_name]

                # Pattern validation
                if "pattern" in field_rules:
                    if not re.match(field_rules["pattern"], field_value):
                        self.context.processing_notes.append(
                            f"Validation warning: {field_name} pattern mismatch"
                        )

                # Length validation
                if "required_length" in field_rules:
                    clean_value = re.sub(r"\\D", "", field_value)  # Remove non-digits
                    if len(clean_value) not in field_rules["required_length"]:
                        self.context.processing_notes.append(
                            f"Validation warning: {field_name} length issue"
                        )

    def _clean_artifacts(self, text: str) -> str:
        """Clean common artifacts from processed text."""
        # Remove empty lines
        lines = [line.strip() for line in text.split("\n") if line.strip()]

        # Remove duplicate key-value pairs
        seen = set()
        unique_lines = []
        for line in lines:
            if line not in seen:
                seen.add(line)
                unique_lines.append(line)

        return "\n".join(unique_lines)

    def _format_output(self, text: str) -> str:
        """Format output according to configuration."""
        output_config = self.config["output_format"]

        # If no structured output, check fallback behavior
        if not text.strip() or ":" not in text:
            fallback_behavior = output_config["fallback_behavior"]["on_no_extraction"]
            if fallback_behavior == "return_raw_content":
                # Return the original text with minimal cleaning
                return self._get_original_text_cleaned()

        return text

    def _get_original_text_cleaned(self) -> str:
        """Get minimally cleaned original text for fallback."""
        # This would need the original text - for now return placeholder
        return "# Raw markdown content preserved (original text would be here)"

    def _calculate_confidence_score(self, text: str) -> float:
        """Calculate confidence score for the processing result."""
        if not text or not self.context.extracted_fields:
            return 0.0

        num_fields = len(self.context.extracted_fields)

        # More realistic field scoring - diminishing returns after 3 fields
        if num_fields <= 3:
            field_score = num_fields * 0.15  # 0.15, 0.30, 0.45
        elif num_fields <= 6:
            field_score = 0.45 + (num_fields - 3) * 0.08  # 0.53, 0.61, 0.69
        else:
            field_score = 0.69 + (num_fields - 6) * 0.03  # 0.72, 0.75, etc., capped at 0.80

        field_score = min(field_score, 0.80)  # Cap at 80%

        # Format detection confidence - more conservative
        format_score = 0.15 if self.context.detected_format != "plain_text" else 0.05

        # Validation score - only small bonus for no warnings
        validation_score = (
            0.05 if not any("warning" in note for note in self.context.processing_notes) else 0.0
        )

        return min(field_score + format_score + validation_score, 0.95)  # Cap total at 95%

    def _validate_final_result(self, text: str) -> bool:
        """Final validation of processing result."""
        if not text.strip():
            return False

        # Check minimum extraction requirements
        min_extractions = self.config["validation_rules"]["minimum_extractions"]
        format_requirement = min_extractions.get(
            self.context.detected_format, min_extractions["any_format"]
        )

        return len(self.context.extracted_fields) >= format_requirement


# Convenience function for standalone usage
def process_markdown(
    text: str, config_path: str = "markdown_processing_config.yaml", image_name: str = ""
) -> ProcessingResult:
    """Convenience function for processing markdown text.

    Args:
        text: Markdown text to process
        config_path: Path to configuration file
        image_name: Optional image name for logging

    Returns:
        ProcessingResult with processed text and metadata
    """
    processor = AWKMarkdownProcessor(config_path)
    return processor.process(text, image_name)


if __name__ == "__main__":
    # Simple test
    test_text = """
    | Field | Value |
    |-------|--------|
    | STORE | Telstra Limited |
    | ABN | 88 088 174 781 |
    | TOTAL | $120.00 |
    """

    result = process_markdown(test_text, image_name="test")
    print(f"Success: {result.success}")
    print(f"Extracted: {result.extracted_fields}")
    print(f"Output:\\n{result.processed_text}")
