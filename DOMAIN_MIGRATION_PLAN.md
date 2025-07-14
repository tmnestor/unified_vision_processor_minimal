# Domain-Specific Field Schema Migration Guide

## Overview

This document outlines the migration path from the current generic document processing schema (18 fields) to a comprehensive domain-specific schema with 57 standardized fields plus an "other" category for non-conforming data.

## Current vs Target Schema

### Current Schema (18 Fields)
```yaml
required_keys: [DATE, STORE, TOTAL, GST, ABN]
optional_keys: [ITEMS, QUANTITIES, PRICES, RECEIPT_NUMBER, PAYMENT_METHOD, 
               DOCUMENT_TYPE, ADDRESS, PHONE, TIME, CARD_NUMBER, AUTH_CODE, STATUS, 
               SUBTOTAL]
```

### Target Schema (57 + Other)
```yaml
# Domain-specific classification with hierarchical organization
field_categories:
  business_identification:
    fields: [BUSINESS_NAME, ABN, ACN, TRADING_NAME, ENTITY_TYPE]
  transaction_core:
    fields: [DATE, TIME, TOTAL_AMOUNT, GST_AMOUNT, NET_AMOUNT]
  payment_details:
    fields: [PAYMENT_METHOD, CARD_TYPE, TERMINAL_ID, AUTH_CODE, REFERENCE_NUMBER]
  document_metadata:
    fields: [DOCUMENT_TYPE, INVOICE_NUMBER, RECEIPT_NUMBER, DOCUMENT_STATUS]
  # ... (53 more fields organized by category)
  other_data:
    fields: [EXTRACTED_TEXT_OTHER]  # Catch-all for non-standard information
```

## Implementation Strategy

### Phase 1: Schema Architecture Update

#### 1.1 Enhanced Configuration System
```yaml
# vision_processor/config/domain_schema.yaml
domain_config:
  schema_version: "2.0"
  total_fields: 57
  
  field_categories:
    business_identification:
      required: true
      fields:
        - BUSINESS_NAME
        - ABN
        - ACN
        - TRADING_NAME
        - ENTITY_TYPE
        
    transaction_core:
      required: true  
      fields:
        - DATE
        - TIME
        - TOTAL_AMOUNT
        - GST_AMOUNT
        - NET_AMOUNT
        
    payment_details:
      required: false
      fields:
        - PAYMENT_METHOD
        - CARD_TYPE
        - TERMINAL_ID
        - AUTH_CODE
        - REFERENCE_NUMBER
        
    # ... additional categories
    
  fallback_handling:
    other_category: "EXTRACTED_TEXT_OTHER"
    confidence_threshold: 0.7
    max_other_length: 500
```

#### 1.2 Backwards Compatibility Mapping
```yaml
# vision_processor/config/field_migration.yaml
field_mappings:
  # Current -> Domain-specific mappings
  STORE: BUSINESS_NAME
  TOTAL: TOTAL_AMOUNT
  GST: GST_AMOUNT
  SUBTOTAL: NET_AMOUNT
  RECEIPT_NUMBER: DOCUMENT_REFERENCE
  PAYMENT_METHOD: PAYMENT_TYPE
  
deprecated_fields:
  - ITEMS  # -> Parse into line item fields
  - QUANTITIES  # -> Parse into line item fields  
  - PRICES  # -> Parse into line item fields
```

### Phase 2: Parser Enhancement

#### 2.1 Extended Universal Parser
```python
# vision_processor/extraction/domain_key_value_parser.py
class DomainKeyValueParser(UniversalKeyValueParser):
    """Enhanced parser for 57-field domain-specific extraction."""
    
    def __init__(self, config_path: str = "domain_schema.yaml"):
        super().__init__()
        self.domain_schema = self._load_domain_schema(config_path)
        self.other_category = self.domain_schema["fallback_handling"]["other_category"]
        
    def parse_response(self, response: str) -> Dict[str, Any]:
        """Parse with domain-specific fields and other category."""
        # Standard field extraction
        extracted_fields = super().parse_response(response)
        
        # Domain-specific field mapping
        mapped_fields = self._map_to_domain_fields(extracted_fields)
        
        # Capture unmapped content in "other" category
        other_content = self._extract_other_content(response, mapped_fields)
        if other_content:
            mapped_fields[self.other_category] = other_content
            
        return mapped_fields
        
    def _extract_other_content(self, response: str, mapped_fields: Dict) -> str:
        """Extract content that doesn't fit standard fields."""
        # Implementation for identifying and capturing non-standard data
        pass
```

#### 2.2 Intelligent Field Classification
```python
# vision_processor/extraction/field_classifier.py
class DomainFieldClassifier:
    """Classifies extracted text into appropriate domain fields."""
    
    def __init__(self):
        self.field_patterns = self._load_field_patterns()
        self.business_name_patterns = self._load_business_patterns()
        
    def classify_extracted_text(self, text: str) -> Dict[str, str]:
        """Classify text into domain-specific fields."""
        classifications = {}
        
        # Business identification
        if self._is_business_name(text):
            classifications["BUSINESS_NAME"] = text
        elif self._is_abn(text):
            classifications["ABN"] = text
            
        # Transaction details
        elif self._is_amount(text):
            classifications.update(self._classify_amount(text))
            
        # Document metadata  
        elif self._is_document_reference(text):
            classifications["DOCUMENT_REFERENCE"] = text
            
        else:
            # Assign to other category
            classifications["CANDIDATE_OTHER"] = text
            
        return classifications
```

### Phase 3: Model Prompt Enhancement

#### 3.1 Domain-Specific Prompts
```yaml
# vision_processor/config/prompts.yaml - Enhanced
model_prompts:
  internvl3:
    domain_extraction_prompt: |
      <|image|>Extract business document information using these EXACT field names.
      
      BUSINESS_NAME: [business or store name]
      ABN: [11-digit business number]
      ACN: [9-digit company number]
      TRADING_NAME: [if different from business name]
      
      DATE: [transaction date DD/MM/YYYY]
      TIME: [transaction time HH:MM]
      TOTAL_AMOUNT: [final amount including tax]
      GST_AMOUNT: [tax amount]
      NET_AMOUNT: [amount before tax]
      
      PAYMENT_TYPE: [VISA/MASTERCARD/EFTPOS/CASH/etc]
      CARD_TYPE: [if card payment]
      TERMINAL_ID: [POS terminal identifier]
      AUTH_CODE: [authorization code]
      REFERENCE_NUMBER: [transaction reference]
      
      # ... (52 more fields)
      
      EXTRACTED_TEXT_OTHER: [any visible text not fitting above categories]
      
      Format: FIELD_NAME: [extracted value]
      Only extract visible text. Use exact field names above.
```

#### 3.2 Conditional Field Extraction
```python
# vision_processor/extraction/conditional_extractor.py
class ConditionalFieldExtractor:
    """Extract fields based on document type and business rules."""
    
    def extract_conditional_fields(self, base_extraction: Dict, document_type: str) -> Dict:
        """Apply business rules for conditional field extraction."""
        enhanced_extraction = base_extraction.copy()
        
        # Document-type specific rules
        if document_type == "TAX_INVOICE":
            enhanced_extraction.update(self._extract_tax_invoice_fields(base_extraction))
        elif document_type == "RECEIPT":
            enhanced_extraction.update(self._extract_receipt_fields(base_extraction))
            
        # Business rule validation
        enhanced_extraction = self._apply_business_rules(enhanced_extraction)
        
        return enhanced_extraction
```

### Phase 4: Evaluation System Migration

#### 4.1 Enhanced Ground Truth Schema
```csv
# evaluation_ground_truth_domain.csv
image_file,BUSINESS_NAME,ABN,ACN,DATE,TIME,TOTAL_AMOUNT,GST_AMOUNT,NET_AMOUNT,PAYMENT_TYPE,DOCUMENT_TYPE,DOCUMENT_REFERENCE,TERMINAL_ID,AUTH_CODE,EXTRACTED_TEXT_OTHER,...(44 more fields)
image14.png,SPOTLIGHT,10306488435,,11/07/2022,15:53,22.45,2.04,20.41,VISA,TAX_INVOICE,503152,T12345,206031,Additional store hours information,...
```

#### 4.2 Domain-Specific Evaluation Metrics
```python
# vision_processor/evaluation/domain_evaluator.py
class DomainExtractionEvaluator(ExtractionEvaluator):
    """Evaluator for 57-field domain schema."""
    
    def __init__(self, domain_schema_path: str):
        super().__init__()
        self.domain_fields = self._load_domain_fields(domain_schema_path)
        self.field_weights = self._calculate_field_importance()
        
    def _calculate_field_accuracy(self, extracted: str, ground_truth: str, field_type: str) -> float:
        """Enhanced accuracy calculation with domain-specific rules."""
        base_accuracy = super()._calculate_field_accuracy(extracted, ground_truth, field_type)
        
        # Apply domain-specific weighting
        field_weight = self.field_weights.get(field_type, 1.0)
        
        # Special handling for other category
        if field_type == "EXTRACTED_TEXT_OTHER":
            return self._evaluate_other_category(extracted, ground_truth)
            
        return base_accuracy * field_weight
        
    def _evaluate_other_category(self, extracted: str, ground_truth: str) -> float:
        """Evaluate other category using semantic similarity."""
        # Implementation for fuzzy matching of miscellaneous content
        pass
```

### Phase 5: Migration Implementation

#### 5.1 Migration Script
```python
# scripts/migrate_to_domain_schema.py
class SchemaMigrationTool:
    """Tool to migrate from generic to domain-specific schema."""
    
    def migrate_configuration(self):
        """Update prompts.yaml with domain fields."""
        # Backup current configuration
        # Load domain schema
        # Generate new prompts with 57 fields
        # Update evaluation ground truth
        
    def migrate_existing_data(self, data_path: str):
        """Convert existing extraction results to new schema."""
        # Load existing results
        # Apply field mappings
        # Generate other category content
        # Save migrated data
        
    def validate_migration(self):
        """Ensure migration preserves data integrity."""
        # Compare extraction results before/after
        # Validate field coverage
        # Check other category content quality
```

#### 5.2 Backwards Compatibility Layer
```python
# vision_processor/config/compatibility_manager.py
class SchemaCompatibilityManager:
    """Manage compatibility between schema versions."""
    
    def __init__(self):
        self.current_version = "2.0"  # Domain-specific
        self.legacy_version = "1.0"   # Generic
        
    def convert_legacy_results(self, legacy_results: Dict) -> Dict:
        """Convert legacy 18-field results to 57-field format."""
        domain_results = {}
        
        # Apply field mappings
        for legacy_field, value in legacy_results.items():
            domain_field = self._map_legacy_field(legacy_field)
            if domain_field:
                domain_results[domain_field] = value
            else:
                # Add to other category
                self._add_to_other_category(domain_results, legacy_field, value)
                
        return domain_results
```

## Migration Timeline

### Week 1-2: Schema Design & Configuration
- Define all 57 domain fields with categories
- Create `domain_schema.yaml` configuration
- Design field mapping system
- Plan backwards compatibility approach

### Week 3-4: Parser Enhancement  
- Implement `DomainKeyValueParser`
- Build field classification system
- Add "other" category handling
- Create conditional extraction rules

### Week 5-6: Model Integration
- Update prompts for 57-field extraction
- Test domain-specific prompts with both models
- Optimize field extraction accuracy
- Implement conditional field logic

### Week 7-8: Evaluation System
- Migrate ground truth data to 57-field schema
- Enhance evaluation metrics for domain fields
- Add other category evaluation methods
- Create migration validation tools

### Week 9-10: Testing & Optimization
- Comprehensive testing with domain schema
- Performance optimization for 57 fields
- Migration script development
- Documentation and deployment preparation

## Configuration Migration

### Environment Variables
```bash
# New domain-specific configuration
VISION_SCHEMA_VERSION=2.0
VISION_DOMAIN_FIELDS_COUNT=57
VISION_ENABLE_OTHER_CATEGORY=true
VISION_OTHER_CATEGORY_MAX_LENGTH=500
VISION_FIELD_CLASSIFICATION_THRESHOLD=0.7

# Backwards compatibility
VISION_LEGACY_SCHEMA_SUPPORT=true
VISION_AUTO_MIGRATE_LEGACY_RESULTS=true
```

### CLI Updates
```bash
# Enhanced CLI for domain schema
python -m vision_processor.cli.domain_extract_cli extract receipt.jpg --schema-version 2.0

# Migration tools
python -m vision_processor.cli.migration_cli migrate-schema --from 1.0 --to 2.0
python -m vision_processor.cli.migration_cli validate-migration --data-path results/

# Domain-specific evaluation
python -m vision_processor.cli.evaluation_cli compare --schema domain --fields 57
```

## Benefits of Migration

### Enhanced Data Capture
- **57 standardized fields** instead of 18 generic fields
- **Hierarchical field organization** for better data structure
- **Other category** ensures no information loss
- **Domain-specific validation** rules

### Improved Processing Accuracy
- **Field-specific extraction** patterns
- **Business rule validation** 
- **Conditional field logic** based on document type
- **Weighted evaluation** metrics for field importance

### Production Readiness
- **Backwards compatibility** with existing systems
- **Migration tools** for seamless transition
- **Enhanced evaluation** framework
- **Comprehensive documentation** and testing

This migration provides a robust foundation for domain-specific document processing while maintaining the flexibility and performance of the existing system.