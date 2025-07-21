# Production Schema Documentation

## Overview

The Production Field Schema is a comprehensive standardized field definition system for Australian Tax Office (ATO) document processing. It defines 55 production labels with their validation rules, extraction patterns, and business logic for automated document field extraction from Australian tax documents.

## Architecture

### Core Components

1. **Field Types** - Classification system for validation strategies
2. **Field Categories** - Grouping system for analysis and organization  
3. **Field Definitions** - Complete specifications for each production field
4. **Production Schema** - Main orchestrator containing all field definitions

## Field Types (`FieldType` Enum)

The system uses strongly-typed field classifications to enable appropriate validation and extraction strategies:

### Australian-Specific Types
- `ABN` - 11-digit Australian Business Number validation
- `BSB` - 6-digit Bank State Branch validation  
- `DATE` - DD/MM/YYYY Australian date format
- `CURRENCY` - $XX.XX or AUD format validation

### Generic Types
- `TEXT` - Standard text fields
- `TEXT_LIST` - Pipe-separated list values
- `NUMERIC` - Decimal numbers
- `INTEGER` - Whole numbers
- `EMAIL` - Email address validation
- `PHONE` - Phone number patterns
- `ADDRESS` - Address field handling

### Special Cases
- `OTHER` - Catch-all for miscellaneous fields

## Field Categories (`FieldCategory` Enum)

Fields are organized into logical business categories:

- **SUPPLIER** - Business/supplier information (names, ABN)
- **FINANCIAL** - Amounts, taxes, totals, balances
- **TEMPORAL** - Dates, times, due dates
- **CONTACT** - Addresses, phones, emails, websites
- **TRANSACTION** - Receipt numbers, payment information
- **LINE_ITEM** - Item-level details (quantities, descriptions)
- **ACCOUNT** - Banking information (BSB, account numbers)
- **OTHER** - Miscellaneous fields

## Field Definition Structure

Each production field is defined using the `FieldDefinition` dataclass:

```python
@dataclass
class FieldDefinition:
    name: str                           # Field identifier
    field_type: FieldType              # Type classification
    category: FieldCategory            # Business category
    is_required: bool = False          # ATO requirement flag
    is_core: bool = False             # Core field for success determination
    
    # Validation Rules
    validation_patterns: List[str]     # Regex patterns for validation
    invalid_values: List[str]         # Values to reject
    format_example: str               # User-facing format example
    
    # Extraction Hints
    extraction_patterns: List[str]    # Regex patterns for field extraction
    fallback_patterns: List[str]     # Alternative extraction patterns
    
    # Business Rules
    description: str                  # Human-readable description
    ato_compliance_level: str        # ATO compliance importance
```

## The 55 Standard Production Labels

The schema defines exactly 55 standardized field labels used across all ATO document processing:

### Financial Fields (Currency amounts)
```
adjust_discount_a_li, adjust_discount_a_pg, adjust_discount_q_li, adjust_discount_q_pg
balance_a_li, balance_a_pg, balance_q_li, balance_q_pg  
received_a_li, received_a_pg, received_q_li, received_q_pg
subtotal_a_li, subtotal_a_pg, subtotal_q_li, subtotal_q_pg
tax_a_li, tax_a_pg, tax_q_li, tax_q_pg
total_a_li, total_a_pg, total_a_pgs, total_q_li, total_q_pg, total_q_pgs
unit_price_a_li, unit_price_q_li
fee_help_a_li, fee_help_q_li
```

### Date Fields 
```
date_a_li, date_q_li
invDate_a_pgs, invDate_q_pgs  
payDate_a_pgs, payDate_q_pgs
due_a_li, due_a_pg, due_q_li, due_q_pg
```

### Entity Information
```
supplier_a_pgs, supplier_q_pgs
payer_a_pgs, payer_q_pgs
supplierABN_a_pgs (critical for ATO compliance)
```

### Line Item Details
```
quantity_a_li, quantity_q_li
desc_a_li, desc_q_li
```

### Contact Information
```
address_extra, emails_extra, phones_extra, website_extra
```

### Banking Fields
```
bank_acc_name_extra, bank_acc_number_extra, bank_bsb_extra
```

### Other Fields
```
header_a_pg, other
```

## Field Naming Convention

The production labels follow a systematic naming convention:

- `{field}_{scope}_{level}`
- **field**: The business concept (total, tax, supplier, etc.)
- **scope**: 
  - `a` = actual/amount
  - `q` = quantity  
- **level**:
  - `li` = line item level
  - `pg` = page level  
  - `pgs` = page summary level
- **extra**: Contact/additional information fields

Examples:
- `total_a_li` = Total amount at line item level
- `supplier_a_pgs` = Supplier name at page summary level
- `tax_q_pg` = Tax quantity at page level

## ATO Compliance Levels

Fields are classified by ATO compliance importance:

### Critical Level
- `supplierABN_a_pgs` - Required for tax compliance
- Must be present for successful ATO processing

### High Level  
- Financial totals (`total_*`, `tax_*`)
- Date fields (invoice dates, payment dates)
- Supplier information
- Essential for audit and verification

### Standard Level
- Supporting details and contact information
- Helpful but not strictly required for compliance

## Core Fields and Success Criteria

The schema identifies **core fields** essential for successful extraction:

- Date fields (all `*Date*` patterns)
- Supplier information (`supplier_*`)
- Financial totals (`total_*`, `subtotal_*`, `tax_*`)
- ABN field (`supplierABN_a_pgs`)

**Success Criteria:**
- Minimum 1/3 of core fields must be extracted
- At least 3 total fields must be extracted  
- All required fields must be present
- Critical ATO compliance fields validated

## Validation and Extraction Patterns

Each field definition includes comprehensive regex patterns:

### Validation Patterns
Used to verify extracted values meet format requirements:
```python
# ABN validation
validation_patterns=[r"^\d{2}\s?\d{3}\s?\d{3}\s?\d{3}$"]

# Currency validation  
validation_patterns=[r"^\$?\d+\.?\d*$", r"^AUD\s*\d+\.?\d*$"]

# Date validation
validation_patterns=[r"^\d{1,2}/\d{1,2}/\d{4}$", r"^\d{1,2}-\d{1,2}-\d{4}$"]
```

### Extraction Patterns
Used to locate and extract field values from document text:
```python
# Financial field extraction
extraction_patterns=[
    rf"{field}:\s*\$?(\d+\.?\d*)",           # Standard format
    rf"{field}:\s*AUD\s*(\d+\.?\d*)",        # AUD format
    rf"{field}:\s*([+-]?\$?\d+\.?\d*)",      # With signs
]

# Date extraction
extraction_patterns=[
    rf"{field}:\s*(\d{{1,2}}/\d{{1,2}}/\d{{4}})",    # DD/MM/YYYY
    rf"{field}:\s*(\d{{1,2}}-\d{{1,2}}-\d{{4}})",    # DD-MM-YYYY
]
```

## Invalid Values

The schema defines common invalid values to reject:
```python
invalid_values = [
    "N/A", "NA", "NOT AVAILABLE", "NOT FOUND", 
    "NONE", "-", "UNKNOWN", "NULL", "EMPTY"
]
```

## Usage Examples

### Getting Field Information
```python
from vision_processor.config.production_schema import PRODUCTION_SCHEMA

# Get field definition
defn = PRODUCTION_SCHEMA.get_field_definition("total_a_li")
print(f"Type: {defn.field_type}")
print(f"Category: {defn.category}")
print(f"Is Core: {defn.is_core}")

# Get fields by category
financial_fields = PRODUCTION_SCHEMA.get_fields_by_category(FieldCategory.FINANCIAL)
print(f"Financial fields: {financial_fields}")

# Get core fields for success determination
core_fields = PRODUCTION_SCHEMA.get_core_fields()
print(f"Core fields: {core_fields}")
```

### Field Validation
```python
# Validate field value
is_valid = PRODUCTION_SCHEMA.validate_field_value("supplierABN_a_pgs", "12 345 678 901")
print(f"ABN valid: {is_valid}")  # True

is_valid = PRODUCTION_SCHEMA.validate_field_value("total_a_li", "$123.45")  
print(f"Total valid: {is_valid}")  # True
```

### Extraction Pattern Usage
```python
# Get extraction patterns for a field
patterns = PRODUCTION_SCHEMA.get_extraction_patterns("total_a_li")
for pattern in patterns:
    print(f"Pattern: {pattern}")
```

### Success Criteria
```python
# Get extraction success criteria
criteria = PRODUCTION_SCHEMA.get_success_criteria()
print(f"Min core fields: {criteria['min_core_fields']}")
print(f"Min total fields: {criteria['min_total_fields']}")
print(f"Required fields: {criteria['required_fields']}")
```

## Integration with Processing Pipeline

The Production Schema integrates with the entire document processing pipeline:

1. **Field Extraction** - Provides regex patterns for finding field values
2. **Validation** - Ensures extracted values meet business rules
3. **Success Determination** - Defines criteria for successful extraction
4. **ATO Compliance** - Enforces tax office requirements  
5. **Analysis & Metrics** - Enables performance evaluation by category
6. **Quality Control** - Identifies problematic extractions

## Relationship to Dynamic Extractor

**Important Note:** The current system uses a dual approach:

- **Production Schema** - Defines the official 55 standardized labels for production deployment
- **Dynamic Extractor** - Uses flexible field detection for development and testing (fields like "DATE", "TOTAL", "AMOUNT")

The Dynamic Extractor is used for:
- Model comparison and evaluation
- Development and testing
- Handling diverse document formats
- Field discovery and analysis

The Production Schema is used for:
- Final production deployment
- ATO compliance validation  
- Standardized reporting
- Business rule enforcement

This dual approach allows flexibility during development while maintaining strict standards for production deployment.

## Benefits

1. **Standardization** - Consistent field definitions across all processing
2. **Validation** - Automatic quality control for extracted values
3. **Compliance** - Built-in ATO requirements and audit trails
4. **Flexibility** - Configurable patterns and rules per field type
5. **Analysis** - Category-based performance metrics and reporting
6. **Maintainability** - Centralized field definitions and business rules

The Production Schema serves as the authoritative source of truth for all field-related processing in the Australian tax document processing system.