"""Production Field Schema for Australian Tax Document Processing
=====================================

Defines the standardized field schema used in production for Australian Taxation Office
document processing. Contains the complete set of 55 production labels with their
validation rules, patterns, and extraction strategies.
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class FieldType(Enum):
    """Field type classifications for different validation strategies."""

    # Australian-specific fields
    ABN = "australian_abn"  # 11-digit Australian Business Number
    BSB = "australian_bsb"  # 6-digit Bank State Branch
    DATE = "australian_date"  # DD/MM/YYYY format
    CURRENCY = "australian_currency"  # $XX.XX or AUD format

    # Generic field types
    TEXT = "text"
    TEXT_LIST = "text_list"  # Pipe-separated list
    NUMERIC = "numeric"
    INTEGER = "integer"
    EMAIL = "email"
    PHONE = "phone"
    ADDRESS = "address"

    # Special cases
    OTHER = "other"  # Catch-all field


class FieldCategory(Enum):
    """Field categories for grouping and analysis."""

    # Core business information
    SUPPLIER = "supplier"  # Business/supplier information
    FINANCIAL = "financial"  # Amounts, taxes, totals
    TEMPORAL = "temporal"  # Dates, times
    CONTACT = "contact"  # Addresses, phones, emails
    TRANSACTION = "transaction"  # Receipt numbers, payments
    LINE_ITEM = "line_item"  # Item-level details
    ACCOUNT = "account"  # Banking information
    OTHER = "other"


@dataclass
class FieldDefinition:
    """Complete definition of a production field."""

    name: str
    field_type: FieldType
    category: FieldCategory
    is_required: bool = False
    is_core: bool = False  # Core fields for success determination

    # Validation rules
    validation_patterns: List[str] = None
    invalid_values: List[str] = None
    format_example: Optional[str] = None

    # Extraction hints
    extraction_patterns: List[str] = None
    fallback_patterns: List[str] = None

    # Business rules
    description: Optional[str] = None
    ato_compliance_level: str = "standard"  # standard, high, critical

    def __post_init__(self):
        if self.validation_patterns is None:
            self.validation_patterns = []
        if self.invalid_values is None:
            self.invalid_values = ["N/A", "NA", "NOT AVAILABLE", "NOT FOUND", "NONE", "-", "UNKNOWN"]
        if self.extraction_patterns is None:
            self.extraction_patterns = []
        if self.fallback_patterns is None:
            self.fallback_patterns = []


class ProductionFieldSchema:
    """Production field schema containing all 55 standardized labels for ATO processing."""

    # Production standard labels - complete list of 55 fields
    STANDARD_LABELS = [
        "address_extra",
        "adjust_discount_a_li",
        "adjust_discount_a_pg",
        "adjust_discount_q_li",
        "adjust_discount_q_pg",
        "balance_a_li",
        "balance_a_pg",
        "balance_q_li",
        "balance_q_pg",
        "bank_acc_name_extra",
        "bank_acc_number_extra",
        "bank_bsb_extra",
        "date_a_li",
        "date_q_li",
        "desc_a_li",
        "desc_q_li",
        "due_a_li",
        "due_a_pg",
        "due_q_li",
        "due_q_pg",
        "emails_extra",
        "fee_help_a_li",
        "fee_help_q_li",
        "header_a_pg",
        "invDate_a_pgs",
        "invDate_q_pgs",
        "other",
        "payDate_a_pgs",
        "payDate_q_pgs",
        "payer_a_pgs",
        "payer_q_pgs",
        "phones_extra",
        "quantity_a_li",
        "quantity_q_li",
        "received_a_li",
        "received_a_pg",
        "received_q_li",
        "received_q_pg",
        "subtotal_a_li",
        "subtotal_a_pg",
        "subtotal_q_li",
        "subtotal_q_pg",
        "supplier_a_pgs",
        "supplier_q_pgs",
        "supplierABN_a_pgs",
        "tax_a_li",
        "tax_a_pg",
        "tax_q_li",
        "tax_q_pg",
        "total_a_li",
        "total_a_pg",
        "total_a_pgs",
        "total_q_li",
        "total_q_pg",
        "total_q_pgs",
        "unit_price_a_li",
        "unit_price_q_li",
        "website_extra",
    ]

    def __init__(self):
        """Initialize production schema with all field definitions."""
        self._field_definitions = self._create_field_definitions()
        self._fields_by_category = self._group_by_category()
        self._core_fields = self._identify_core_fields()

    def _create_field_definitions(self) -> Dict[str, FieldDefinition]:
        """Create comprehensive field definitions for all production labels."""
        definitions = {}

        # Financial fields - amounts, totals, taxes, balances
        financial_fields = [
            "adjust_discount_a_li",
            "adjust_discount_a_pg",
            "adjust_discount_q_li",
            "adjust_discount_q_pg",
            "balance_a_li",
            "balance_a_pg",
            "balance_q_li",
            "balance_q_pg",
            "received_a_li",
            "received_a_pg",
            "received_q_li",
            "received_q_pg",
            "subtotal_a_li",
            "subtotal_a_pg",
            "subtotal_q_li",
            "subtotal_q_pg",
            "tax_a_li",
            "tax_a_pg",
            "tax_q_li",
            "tax_q_pg",
            "total_a_li",
            "total_a_pg",
            "total_a_pgs",
            "total_q_li",
            "total_q_pg",
            "total_q_pgs",
            "unit_price_a_li",
            "unit_price_q_li",
            "fee_help_a_li",
            "fee_help_q_li",
        ]

        for field in financial_fields:
            definitions[field] = FieldDefinition(
                name=field,
                field_type=FieldType.CURRENCY,
                category=FieldCategory.FINANCIAL,
                is_core=field.startswith(("total_", "subtotal_", "tax_")),
                validation_patterns=[r"^\$?\d+\.?\d*$", r"^AUD\s*\d+\.?\d*$"],
                format_example="$123.45 or AUD 123.45",
                extraction_patterns=[
                    rf"{field}:\s*\$?(\d+\.?\d*)",
                    rf"{field}:\s*AUD\s*(\d+\.?\d*)",
                    rf"{field}:\s*([+-]?\$?\d+\.?\d*)",
                ],
                description=f"Financial amount for {field.replace('_', ' ')}",
                ato_compliance_level="high" if field.startswith(("total_", "tax_")) else "standard",
            )

        # Date fields
        date_fields = [
            "date_a_li",
            "date_q_li",
            "invDate_a_pgs",
            "invDate_q_pgs",
            "payDate_a_pgs",
            "payDate_q_pgs",
        ]
        for field in date_fields:
            definitions[field] = FieldDefinition(
                name=field,
                field_type=FieldType.DATE,
                category=FieldCategory.TEMPORAL,
                is_core=True,
                validation_patterns=[r"^\d{1,2}/\d{1,2}/\d{4}$", r"^\d{1,2}-\d{1,2}-\d{4}$"],
                format_example="DD/MM/YYYY or DD-MM-YYYY",
                extraction_patterns=[
                    rf"{field}:\s*(\d{{1,2}}/\d{{1,2}}/\d{{4}})",
                    rf"{field}:\s*(\d{{1,2}}-\d{{1,2}}-\d{{4}})",
                ],
                description=f"Date field for {field.replace('_', ' ')}",
                ato_compliance_level="high",
            )

        # Supplier and payer information
        entity_fields = ["supplier_a_pgs", "supplier_q_pgs", "payer_a_pgs", "payer_q_pgs"]
        for field in entity_fields:
            definitions[field] = FieldDefinition(
                name=field,
                field_type=FieldType.TEXT,
                category=FieldCategory.SUPPLIER,
                is_core=field.startswith("supplier_"),
                extraction_patterns=[rf"{field}:\s*([A-Za-z].*?)(?:\n|$)", rf"{field}:\s*([^|\n]+)"],
                description=f"Entity name for {field.replace('_', ' ')}",
                ato_compliance_level="high" if field.startswith("supplier_") else "standard",
            )

        # ABN field - critical for ATO compliance
        definitions["supplierABN_a_pgs"] = FieldDefinition(
            name="supplierABN_a_pgs",
            field_type=FieldType.ABN,
            category=FieldCategory.SUPPLIER,
            is_core=True,
            is_required=True,
            validation_patterns=[r"^\d{2}\s?\d{3}\s?\d{3}\s?\d{3}$"],
            format_example="12 345 678 901 or 12345678901",
            extraction_patterns=[
                r"supplierABN_a_pgs:\s*(\d{2}\s?\d{3}\s?\d{3}\s?\d{3})",
                r"ABN:\s*(\d{2}\s?\d{3}\s?\d{3}\s?\d{3})",
                r"(?:ABN|Australian Business Number):\s*(\d{11}|\d{2}\s\d{3}\s\d{3}\s\d{3})",
            ],
            description="Australian Business Number (11 digits)",
            ato_compliance_level="critical",
        )

        # Quantity fields
        quantity_fields = ["quantity_a_li", "quantity_q_li"]
        for field in quantity_fields:
            definitions[field] = FieldDefinition(
                name=field,
                field_type=FieldType.INTEGER,
                category=FieldCategory.LINE_ITEM,
                validation_patterns=[r"^\d+$"],
                extraction_patterns=[rf"{field}:\s*(\d+)"],
                description=f"Quantity for {field.replace('_', ' ')}",
                ato_compliance_level="standard",
            )

        # Description fields
        desc_fields = ["desc_a_li", "desc_q_li"]
        for field in desc_fields:
            definitions[field] = FieldDefinition(
                name=field,
                field_type=FieldType.TEXT,
                category=FieldCategory.LINE_ITEM,
                extraction_patterns=[rf"{field}:\s*([^|\n]+)", rf"{field}:\s*(.*?)(?:\n|$)"],
                description=f"Description for {field.replace('_', ' ')}",
                ato_compliance_level="standard",
            )

        # Due date fields
        due_fields = ["due_a_li", "due_a_pg", "due_q_li", "due_q_pg"]
        for field in due_fields:
            definitions[field] = FieldDefinition(
                name=field,
                field_type=FieldType.DATE,
                category=FieldCategory.TEMPORAL,
                validation_patterns=[r"^\d{1,2}/\d{1,2}/\d{4}$"],
                extraction_patterns=[rf"{field}:\s*(\d{{1,2}}/\d{{1,2}}/\d{{4}})"],
                description=f"Due date for {field.replace('_', ' ')}",
                ato_compliance_level="standard",
            )

        # Contact information
        definitions["address_extra"] = FieldDefinition(
            name="address_extra",
            field_type=FieldType.ADDRESS,
            category=FieldCategory.CONTACT,
            extraction_patterns=[r"address_extra:\s*([^|\n]+)", r"address:\s*([^|\n]+)"],
            description="Address information",
            ato_compliance_level="standard",
        )

        definitions["emails_extra"] = FieldDefinition(
            name="emails_extra",
            field_type=FieldType.EMAIL,
            category=FieldCategory.CONTACT,
            validation_patterns=[r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"],
            extraction_patterns=[
                r"emails_extra:\s*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})",
                r"email:\s*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})",
            ],
            description="Email address",
            ato_compliance_level="standard",
        )

        definitions["phones_extra"] = FieldDefinition(
            name="phones_extra",
            field_type=FieldType.PHONE,
            category=FieldCategory.CONTACT,
            extraction_patterns=[r"phones_extra:\s*([0-9\s\-\+\(\)]+)", r"phone:\s*([0-9\s\-\+\(\)]+)"],
            description="Phone number",
            ato_compliance_level="standard",
        )

        definitions["website_extra"] = FieldDefinition(
            name="website_extra",
            field_type=FieldType.TEXT,
            category=FieldCategory.CONTACT,
            extraction_patterns=[
                r"website_extra:\s*([www\.]?[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})",
                r"website:\s*([www\.]?[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})",
            ],
            description="Website URL",
            ato_compliance_level="standard",
        )

        # Banking information
        definitions["bank_acc_name_extra"] = FieldDefinition(
            name="bank_acc_name_extra",
            field_type=FieldType.TEXT,
            category=FieldCategory.ACCOUNT,
            extraction_patterns=[r"bank_acc_name_extra:\s*([^|\n]+)"],
            description="Bank account name",
            ato_compliance_level="standard",
        )

        definitions["bank_acc_number_extra"] = FieldDefinition(
            name="bank_acc_number_extra",
            field_type=FieldType.TEXT,
            category=FieldCategory.ACCOUNT,
            validation_patterns=[r"^\d{6,12}$"],
            extraction_patterns=[r"bank_acc_number_extra:\s*(\d{6,12})"],
            description="Bank account number",
            ato_compliance_level="standard",
        )

        definitions["bank_bsb_extra"] = FieldDefinition(
            name="bank_bsb_extra",
            field_type=FieldType.BSB,
            category=FieldCategory.ACCOUNT,
            validation_patterns=[r"^\d{3}-?\d{3}$"],
            format_example="123-456 or 123456",
            extraction_patterns=[r"bank_bsb_extra:\s*(\d{3}-?\d{3})", r"BSB:\s*(\d{3}-?\d{3})"],
            description="Bank State Branch number",
            ato_compliance_level="standard",
        )

        # Header and other fields
        definitions["header_a_pg"] = FieldDefinition(
            name="header_a_pg",
            field_type=FieldType.TEXT,
            category=FieldCategory.OTHER,
            extraction_patterns=[r"header_a_pg:\s*([^|\n]+)"],
            description="Page header information",
            ato_compliance_level="standard",
        )

        definitions["other"] = FieldDefinition(
            name="other",
            field_type=FieldType.OTHER,
            category=FieldCategory.OTHER,
            extraction_patterns=[r"other:\s*([^|\n]+)"],
            description="Other miscellaneous information",
            ato_compliance_level="standard",
        )

        return definitions

    def _group_by_category(self) -> Dict[FieldCategory, List[str]]:
        """Group fields by category for analysis purposes."""
        groups = {}
        for field_name, definition in self._field_definitions.items():
            category = definition.category
            if category not in groups:
                groups[category] = []
            groups[category].append(field_name)
        return groups

    def _identify_core_fields(self) -> List[str]:
        """Identify core fields that are essential for successful extraction."""
        return [name for name, defn in self._field_definitions.items() if defn.is_core]

    def get_field_definition(self, field_name: str) -> Optional[FieldDefinition]:
        """Get field definition by name."""
        return self._field_definitions.get(field_name)

    def get_fields_by_category(self, category: FieldCategory) -> List[str]:
        """Get all fields in a specific category."""
        return self._fields_by_category.get(category, [])

    def get_core_fields(self) -> List[str]:
        """Get list of core fields essential for extraction success."""
        return self._core_fields.copy()

    def get_required_fields(self) -> List[str]:
        """Get list of required fields."""
        return [name for name, defn in self._field_definitions.items() if defn.is_required]

    def get_all_fields(self) -> List[str]:
        """Get complete list of all production fields."""
        return self.STANDARD_LABELS.copy()

    def get_field_type(self, field_name: str) -> Optional[FieldType]:
        """Get field type for validation purposes."""
        definition = self.get_field_definition(field_name)
        return definition.field_type if definition else None

    def get_extraction_patterns(self, field_name: str) -> List[str]:
        """Get extraction patterns for a specific field."""
        definition = self.get_field_definition(field_name)
        return definition.extraction_patterns if definition else []

    def validate_field_value(self, field_name: str, value: str) -> bool:
        """Validate field value against its definition."""
        definition = self.get_field_definition(field_name)
        if not definition:
            return False

        # Check against invalid values
        if value.upper() in [v.upper() for v in definition.invalid_values]:
            return False

        # Check validation patterns
        if definition.validation_patterns:
            return any(re.match(pattern, value) for pattern in definition.validation_patterns)

        return len(value.strip()) > 0

    def get_success_criteria(self) -> Dict[str, Any]:
        """Get criteria for determining extraction success."""
        return {
            "min_core_fields": max(1, len(self._core_fields) // 3),  # At least 1/3 of core fields
            "min_total_fields": 3,  # At least 3 fields total
            "required_fields": self.get_required_fields(),
            "core_fields": self._core_fields,
            "critical_fields": [
                name
                for name, defn in self._field_definitions.items()
                if defn.ato_compliance_level == "critical"
            ],
        }

    def create_universal_parser_schema(self) -> Dict[str, Any]:
        """Create schema compatible with UniversalKeyValueParser."""
        return {
            "required_keys": self.get_required_fields(),
            "optional_keys": [f for f in self.STANDARD_LABELS if f not in self.get_required_fields()],
            "key_patterns": {
                name: defn.validation_patterns[0] if defn.validation_patterns else None
                for name, defn in self._field_definitions.items()
                if defn.validation_patterns
            },
        }


# Global instance for production use
PRODUCTION_SCHEMA = ProductionFieldSchema()
