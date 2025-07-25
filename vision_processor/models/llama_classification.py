"""Llama-3.2-Vision Document Classification

Handles document type classification with hardcoded patterns and indicators.
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, Union

from PIL import Image

logger = logging.getLogger(__name__)


class LlamaDocumentClassifier:
    """Handles document classification for Llama-3.2-Vision model."""

    def __init__(self, inference_manager):
        """Initialize document classifier.
        
        Args:
            inference_manager: The inference manager instance
        """
        self.inference_manager = inference_manager
        
        # Document classification patterns
        self.fuel_indicators = [
            "13ulp", "ulp", "unleaded", "diesel", "litre", " l ", ".l ",
            "price/l", "per litre", "fuel"
        ]
        
        self.bank_indicators = [
            "account", "balance", "transaction", "deposit", "withdrawal", "bsb",
            "opening balance", "closing balance", "statement period", "account number",
            "sort code", "debit", "credit", "available balance", "current balance"
        ]
        
        # Regex patterns for document detection
        self.fuel_quantity_pattern = r"\d+\.\d{2,3}\s*l\b|\d+\s*litre"
        self.bank_account_pattern = r"\d{3}-\d{3}\s+\d{4,10}|\bBSB\b|\baccount\s+number\b"
        
        # Classification prompt template
        self.classification_prompt = """<|image|>Analyze document structure and format. Classify based on layout patterns:

- fuel_receipt: Contains fuel quantities (L, litres), price per unit
- tax_invoice: Formal invoice layout, tax calculations
- receipt: Product lists, subtotals, retail format
- bank_statement: Account numbers, transaction records
- unknown: Cannot determine format

Output document type only."""

    def classify_document(
        self,
        image_path: Union[str, Path, Image.Image],
    ) -> Dict[str, Any]:
        """Classify document type using Llama-3.2-Vision.

        Args:
            image_path: Path to image file or PIL Image

        Returns:
            Classification result dictionary with document_type, confidence, etc.
        """
        try:
            # Use inference manager to get model response
            response = self.inference_manager.process_image(
                image_path, self.classification_prompt, do_sample=False
            )
            response_lower = response.raw_text.lower()

            # Parse classification response with improved detection
            response_text = response.raw_text.lower()

            # Check for fuel indicators in the OCR text
            has_fuel_content = any(
                indicator in response_text for indicator in self.fuel_indicators
            )

            # Look for quantity patterns that indicate fuel
            has_fuel_quantity = bool(re.search(self.fuel_quantity_pattern, response_text))

            # Check for bank statement indicators in the OCR text
            has_bank_content = any(
                indicator in response_text for indicator in self.bank_indicators
            )

            # Look for account number patterns (Australian BSB + Account format)
            has_bank_account = bool(
                re.search(self.bank_account_pattern, response_text, re.IGNORECASE)
            )

            # Classification logic with confidence scoring
            doc_type, confidence = self._determine_document_type(
                response_lower, has_fuel_content, has_fuel_quantity,
                has_bank_content, has_bank_account
            )

            return {
                "document_type": doc_type,
                "confidence": confidence,
                "classification_response": response.raw_text,
                "is_business_document": self._is_business_document(doc_type, confidence),
            }

        except Exception as e:
            logger.error(f"Document classification failed: {e}")
            return {
                "document_type": "unknown",
                "confidence": 0.0,
                "classification_response": f"Error: {str(e)}",
                "is_business_document": False,
            }

    def _determine_document_type(
        self, response_lower: str, has_fuel_content: bool, has_fuel_quantity: bool,
        has_bank_content: bool, has_bank_account: bool
    ) -> tuple[str, float]:
        """Determine document type based on patterns and content.
        
        Args:
            response_lower: Lowercase response text
            has_fuel_content: Whether fuel indicators were found
            has_fuel_quantity: Whether fuel quantity patterns were found
            has_bank_content: Whether bank indicators were found
            has_bank_account: Whether bank account patterns were found
            
        Returns:
            Tuple of (document_type, confidence)
        """
        # Priority 1: Explicit fuel receipt detection
        if "fuel_receipt" in response_lower or "fuel receipt" in response_lower:
            return "fuel_receipt", 0.90
            
        # Priority 2: Content-based fuel detection (overrides other classifications)
        elif has_fuel_content or has_fuel_quantity:
            logger.info("Overriding classification to fuel_receipt based on content indicators")
            return "fuel_receipt", 0.95
            
        # Priority 3: General fuel keywords
        elif "fuel" in response_lower or "petrol" in response_lower:
            return "fuel_receipt", 0.85
            
        # Priority 4: Explicit tax invoice detection
        elif "tax_invoice" in response_lower or "tax invoice" in response_lower:
            return "tax_invoice", 0.85
            
        # Priority 5: Tax + invoice keywords
        elif "tax" in response_lower and "invoice" in response_lower:
            return "tax_invoice", 0.80
            
        # Priority 6: Explicit bank statement detection
        elif "bank_statement" in response_lower or "bank statement" in response_lower:
            return "bank_statement", 0.90
            
        # Priority 7: Content-based bank detection (overrides other classifications)
        elif has_bank_content or has_bank_account:
            logger.info("Overriding classification to bank_statement based on content indicators")
            return "bank_statement", 0.95
            
        # Priority 8: General bank keywords
        elif "bank" in response_lower:
            return "bank_statement", 0.75
            
        # Priority 9: Receipt detection
        elif "receipt" in response_lower:
            return "receipt", 0.75
            
        # Priority 10: General invoice (default to tax_invoice)
        elif "invoice" in response_lower:
            return "tax_invoice", 0.70
            
        # Default: Unknown
        else:
            return "unknown", 0.50

    def _is_business_document(self, doc_type: str, confidence: float) -> bool:
        """Determine if document is a business document.
        
        Args:
            doc_type: Classified document type
            confidence: Classification confidence
            
        Returns:
            True if document is considered a business document
        """
        business_types = [
            "receipt", "tax_invoice", "fuel_receipt", 
            "bank_statement", "invoice"
        ]
        return doc_type in business_types and confidence > 0.7