"""Production Schema Legacy Module
==================================

Minimal production schema definition that provides the PRODUCTION_SCHEMA object
and FieldCategory enum required by legacy files. This is based on the field
references found in the codebase and model_comparison.yaml configuration.
"""

from enum import Enum
from typing import Dict, List, Optional


class FieldCategory(Enum):
    """Field categories for organizing extraction fields."""

    SUPPLIER = "supplier"
    FINANCIAL = "financial"
    TEMPORAL = "temporal"
    ITEMS = "items"
    CONTACT = "contact"
    DOCUMENT = "document"
    PAYMENT = "payment"


class ProductionSchema:
    """Production schema for document field extraction."""

    def __init__(self):
        """Initialize the production schema with field definitions."""
        self._fields = self._define_fields()
        self._categories = self._define_categories()

    def _define_fields(self) -> Dict[str, Dict]:
        """Define all production fields with metadata."""
        return {
            # Core supplier information
            "SUPPLIER": {
                "category": FieldCategory.SUPPLIER,
                "type": "text",
                "required": True,
                "core": True,
                "description": "Business/supplier name"
            },
            "ABN": {
                "category": FieldCategory.SUPPLIER,
                "type": "abn",
                "required": True,
                "core": True,
                "description": "Australian Business Number"
            },
            "BUSINESS_ADDRESS": {
                "category": FieldCategory.SUPPLIER,
                "type": "text",
                "required": False,
                "core": False,
                "description": "Business address"
            },
            "BUSINESS_PHONE": {
                "category": FieldCategory.SUPPLIER,
                "type": "text",
                "required": False,
                "core": False,
                "description": "Business phone number"
            },

            # Financial information
            "TOTAL": {
                "category": FieldCategory.FINANCIAL,
                "type": "currency",
                "required": True,
                "core": True,
                "description": "Total amount"
            },
            "SUBTOTAL": {
                "category": FieldCategory.FINANCIAL,
                "type": "currency",
                "required": False,
                "core": True,
                "description": "Subtotal amount"
            },
            "GST": {
                "category": FieldCategory.FINANCIAL,
                "type": "currency",
                "required": False,
                "core": True,
                "description": "GST amount"
            },
            "PRICES": {
                "category": FieldCategory.FINANCIAL,
                "type": "text",
                "required": False,
                "core": False,
                "description": "Item prices separated by |"
            },

            # Temporal information
            "DATE": {
                "category": FieldCategory.TEMPORAL,
                "type": "date",
                "required": True,
                "core": True,
                "description": "Document date"
            },
            "TIME": {
                "category": FieldCategory.TEMPORAL,
                "type": "text",
                "required": False,
                "core": False,
                "description": "Document time"
            },

            # Items information
            "ITEMS": {
                "category": FieldCategory.ITEMS,
                "type": "text",
                "required": False,
                "core": False,
                "description": "Item names separated by |"
            },
            "QUANTITIES": {
                "category": FieldCategory.ITEMS,
                "type": "text",
                "required": False,
                "core": False,
                "description": "Item quantities separated by |"
            },

            # Contact information
            "PAYER_NAME": {
                "category": FieldCategory.CONTACT,
                "type": "text",
                "required": False,
                "core": False,
                "description": "Customer/payer name"
            },
            "PAYER_ADDRESS": {
                "category": FieldCategory.CONTACT,
                "type": "text",
                "required": False,
                "core": False,
                "description": "Customer/payer address"
            },
            "PAYER_PHONE": {
                "category": FieldCategory.CONTACT,
                "type": "text",
                "required": False,
                "core": False,
                "description": "Customer/payer phone"
            },
            "PAYER_EMAIL": {
                "category": FieldCategory.CONTACT,
                "type": "text",
                "required": False,
                "core": False,
                "description": "Customer/payer email"
            },

            # Document information
            "INVOICE_NUMBER": {
                "category": FieldCategory.DOCUMENT,
                "type": "text",
                "required": False,
                "core": False,
                "description": "Invoice number"
            },
            "RECEIPT_NUMBER": {
                "category": FieldCategory.DOCUMENT,
                "type": "text",
                "required": False,
                "core": False,
                "description": "Receipt number"
            },
            "DOCUMENT_TYPE": {
                "category": FieldCategory.DOCUMENT,
                "type": "text",
                "required": False,
                "core": False,
                "description": "Document type"
            },
            "STATUS": {
                "category": FieldCategory.DOCUMENT,
                "type": "text",
                "required": False,
                "core": False,
                "description": "Document status"
            },

            # Payment information
            "PAYMENT_METHOD": {
                "category": FieldCategory.PAYMENT,
                "type": "text",
                "required": False,
                "core": False,
                "description": "Payment method"
            },
            "CARD_NUMBER": {
                "category": FieldCategory.PAYMENT,
                "type": "text",
                "required": False,
                "core": False,
                "description": "Card number (last 4 digits)"
            },
            "AUTH_CODE": {
                "category": FieldCategory.PAYMENT,
                "type": "text",
                "required": False,
                "core": False,
                "description": "Authorization code"
            },
        }

    def _define_categories(self) -> Dict[FieldCategory, List[str]]:
        """Define field categories mapping."""
        categories = {}
        for field_name, field_def in self._fields.items():
            category = field_def["category"]
            if category not in categories:
                categories[category] = []
            categories[category].append(field_name)
        return categories

    def get_all_fields(self) -> List[str]:
        """Get all field names."""
        return list(self._fields.keys())

    def get_core_fields(self) -> List[str]:
        """Get core field names."""
        return [field for field, def_ in self._fields.items() if def_["core"]]

    def get_required_fields(self) -> List[str]:
        """Get required field names."""
        return [field for field, def_ in self._fields.items() if def_["required"]]

    def get_fields_by_category(self, category: FieldCategory) -> List[str]:
        """Get field names by category."""
        return self._categories.get(category, [])

    def get_field_definition(self, field_name: str) -> Optional[Dict]:
        """Get field definition by name."""
        return self._fields.get(field_name)

    def is_core_field(self, field_name: str) -> bool:
        """Check if field is a core field."""
        field_def = self._fields.get(field_name)
        return field_def.get("core", False) if field_def else False

    def is_required_field(self, field_name: str) -> bool:
        """Check if field is required."""
        field_def = self._fields.get(field_name)
        return field_def.get("required", False) if field_def else False


# Global instance for legacy compatibility
PRODUCTION_SCHEMA = ProductionSchema()
