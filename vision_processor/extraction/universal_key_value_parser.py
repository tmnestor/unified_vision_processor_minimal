"""Universal KEY-VALUE parser using YAML-defined schema."""

import re
from typing import Any, Dict, List


class UniversalKeyValueParser:
    """Parse KEY-VALUE responses using YAML-defined schema."""

    def __init__(self, key_schema: dict):
        """Initialize parser with key schema.

        Args:
            key_schema: Dictionary containing required_keys, optional_keys, and key_patterns.
        """
        self.required_keys = key_schema.get("required_keys", [])
        self.optional_keys = key_schema.get("optional_keys", [])
        self.key_patterns = key_schema.get("key_patterns", {})
        self.all_keys = self.required_keys + self.optional_keys

    def parse(self, response_text: str) -> Dict[str, Any]:
        """Parse response using schema-defined keys.

        Args:
            response_text: Raw text response from model.

        Returns:
            Dictionary of extracted key-value pairs.
        """
        extracted = {}

        # Try to extract each key defined in schema
        for key in self.all_keys:
            # Try different pattern variations
            patterns = [
                rf"{key}:\s*([^\n\r]+)",  # Standard KEY: value
                rf"{key}\s*:\s*([^\n\r]+)",  # KEY : value (with spaces)
                rf"{key}:\s*\[([^\]]+)\]",  # KEY: [value]
                rf"{key}:\s*'([^']+)'",  # KEY: 'value'
                rf'{key}:\s*"([^"]+)"',  # KEY: "value"
            ]

            for pattern in patterns:
                match = re.search(pattern, response_text, re.IGNORECASE | re.MULTILINE)
                if match:
                    value = match.group(1).strip()
                    if value and value.lower() not in [
                        "",
                        "n/a",
                        "not visible",
                        "not available",
                        "none",
                    ]:
                        extracted[key] = self._clean_value(key, value)
                        break

        return extracted

    def _clean_value(self, key: str, value: str) -> Any:
        """Clean and parse value according to key type.

        Args:
            key: The key name.
            value: Raw value string.

        Returns:
            Cleaned value (string, float, or list).
        """
        # Remove common artifacts
        value = value.strip()
        value = value.rstrip(".")
        value = value.strip("\"'")

        # Handle numeric values
        if key in [
            "TOTAL",
            "GST",
            "SUBTOTAL",
            "OPENING_BALANCE",
            "CLOSING_BALANCE",
            "NIGHTLY_RATE",
            "AMOUNT",
            "UNIT_PRICE",
            "PRICE_PER_LITRE",
        ]:
            return self._extract_numeric(value)

        # Handle list values
        elif key in ["ITEMS", "QUANTITIES", "PRICES"]:
            return self._extract_list(value)

        # Handle integer values
        elif key in ["LITRES", "QUANTITY", "ATTENDEES"]:
            return self._extract_integer(value)

        # Handle date ranges
        elif key == "STATEMENT_PERIOD":
            return self._clean_date_range(value)

        # Handle ABN format
        elif key == "ABN":
            return self._clean_abn(value)

        # Handle BSB format
        elif key == "BSB":
            return self._clean_bsb(value)

        # Default: return as string
        else:
            return value.strip()

    def _extract_numeric(self, value: str) -> float:
        """Extract numeric value from string.

        Args:
            value: String containing numeric value.

        Returns:
            Float value or 0.0 if extraction fails.
        """
        # Remove currency symbols and commas
        cleaned = re.sub(r"[$,]", "", value)

        # Try to find the first numeric pattern
        match = re.search(r"-?\d+\.?\d*", cleaned)
        if match:
            try:
                return float(match.group())
            except ValueError:
                return 0.0
        return 0.0

    def _extract_integer(self, value: str) -> int:
        """Extract integer value from string.

        Args:
            value: String containing integer value.

        Returns:
            Integer value or 0 if extraction fails.
        """
        # Try to find the first integer pattern
        match = re.search(r"\d+", value)
        if match:
            try:
                return int(match.group())
            except ValueError:
                return 0
        return 0

    def _extract_list(self, value: str) -> List[str]:
        """Extract list values from string.

        Args:
            value: String containing list values.

        Returns:
            List of string values.
        """
        # Handle different list separators
        if "|" in value:
            items = value.split("|")
        elif "," in value:
            items = value.split(",")
        elif ";" in value:
            items = value.split(";")
        else:
            # Single item
            return [value.strip()]

        # Clean each item
        return [item.strip() for item in items if item.strip()]

    def _clean_date_range(self, value: str) -> str:
        """Clean date range format.

        Args:
            value: Raw date range string.

        Returns:
            Cleaned date range string.
        """
        # Ensure it has " to " format
        if " to " not in value.lower():
            # Try to find two dates
            dates = re.findall(r"\d{1,2}/\d{1,2}/\d{4}", value)
            if len(dates) >= 2:
                return f"{dates[0]} to {dates[1]}"

        return value.strip()

    def _clean_abn(self, value: str) -> str:
        """Clean ABN format to XX XXX XXX XXX.

        Args:
            value: Raw ABN string.

        Returns:
            Formatted ABN string.
        """
        # Remove all non-digits
        digits = re.sub(r"\D", "", value)

        # Format as XX XXX XXX XXX
        if len(digits) == 11:
            return f"{digits[:2]} {digits[2:5]} {digits[5:8]} {digits[8:11]}"

        # Return original if not 11 digits
        return value.strip()

    def _clean_bsb(self, value: str) -> str:
        """Clean BSB format to XXX-XXX.

        Args:
            value: Raw BSB string.

        Returns:
            Formatted BSB string.
        """
        # Remove all non-digits
        digits = re.sub(r"\D", "", value)

        # Format as XXX-XXX
        if len(digits) == 6:
            return f"{digits[:3]}-{digits[3:6]}"

        # Return original if not 6 digits
        return value.strip()

    def validate_extraction(self, extracted_data: dict) -> Dict[str, Any]:
        """Validate extraction completeness and accuracy.

        Args:
            extracted_data: Dictionary of extracted data.

        Returns:
            Dictionary with validation results.
        """
        missing_required = []
        invalid_patterns = []

        # Check required keys
        for key in self.required_keys:
            if key not in extracted_data:
                missing_required.append(key)

        # Check patterns
        for key, value in extracted_data.items():
            if key in self.key_patterns:
                if not self._matches_pattern(value, self.key_patterns[key]):
                    invalid_patterns.append(f"{key}: {value}")

        return {
            "is_valid": len(missing_required) == 0 and len(invalid_patterns) == 0,
            "missing_required": missing_required,
            "invalid_patterns": invalid_patterns,
            "extracted_count": len(extracted_data),
            "required_count": len(self.required_keys),
        }

    def _matches_pattern(self, value: Any, pattern: str) -> bool:
        """Check if value matches expected pattern.

        Args:
            value: Value to check.
            pattern: Pattern description.

        Returns:
            True if matches, False otherwise.
        """
        if pattern == "DD/MM/YYYY":
            if isinstance(value, str):
                return bool(re.match(r"^\d{1,2}/\d{1,2}/\d{4}$", value))

        elif pattern == "numeric with 2 decimals":
            if isinstance(value, (int, float)):
                return True

        elif pattern == "XX XXX XXX XXX format":
            if isinstance(value, str):
                return bool(re.match(r"^\d{2} \d{3} \d{3} \d{3}$", value))

        elif pattern == "XXX-XXX format":
            if isinstance(value, str):
                return bool(re.match(r"^\d{3}-\d{3}$", value))

        elif pattern == "numeric string":
            if isinstance(value, str):
                return bool(re.match(r"^\d+$", value))

        # Default: assume valid
        return True
