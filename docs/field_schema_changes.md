# Field Schema Change Management

## Overview

When production requirements change, the information extraction field schema must be updated to accommodate new document types, additional data points, or changing business requirements. This document provides a comprehensive guide for safely implementing field schema changes in the Unified Vision Document Processing Architecture.

## System Design Philosophy

The architecture follows a **single source of truth** principle where field definitions are centralized in the YAML configuration and dynamically propagated throughout the system. However, some components require manual updates for optimal functionality.

## Impact Analysis

### Primary Affected Components

#### 1. Configuration Files (CRITICAL)
- **model_comparison.yaml** - Main production configuration
- **model_comparison_local.yaml** - Local development configuration
- **Impact**: Core field definitions and prompts
- **Risk Level**: HIGH - System breaks without these updates

#### 2. Metrics and Analysis (IMPORTANT)
- **simple_metrics.py** - Hardcoded field weights
- **comparison_runner.py** - Field-specific analysis logic
- **Impact**: Analysis accuracy and completeness
- **Risk Level**: MEDIUM - Analysis may be incomplete but processing continues

#### 3. Documentation (MAINTENANCE)
- **README.md** - Example field lists and documentation
- **batch_results_creation.md** - Field structure examples
- **batch_to_csv_analysis.md** - DataFrame column references
- **Impact**: User guidance and examples
- **Risk Level**: LOW - Outdated documentation but system functions

#### 4. Ground Truth Data (VALIDATION)
- **annotations/*.csv** - Evaluation and test data
- **Impact**: Model validation and testing accuracy
- **Risk Level**: MEDIUM - Cannot validate new fields without updated ground truth

### Dynamically Handled Components (NO CHANGES NEEDED)

#### ✅ Automatic Adaptation
- **extraction_manager.py** - Uses `config.get_expected_fields()`
- **batch_to_dataframe.py** - Dynamically creates DataFrame columns
- **extract_cli.py** - Field validation uses dynamic schema
- **config_manager.py** - Parses fields from prompt using regex

## Change Implementation Process

### Phase 1: Configuration Updates (REQUIRED)

#### 1. Update Main Configuration
**File**: `model_comparison.yaml`

```yaml
# BEFORE - Original 26 fields
extraction_prompt: |
  REQUIRED OUTPUT FORMAT (output ALL lines exactly as shown):
  DOCUMENT_TYPE: [value or N/A]
  SUPPLIER: [value or N/A]
  ABN: [11-digit Australian Business Number or N/A]
  # ... 26 total fields

# AFTER - Example with new fields
extraction_prompt: |
  REQUIRED OUTPUT FORMAT (output ALL lines exactly as shown):
  DOCUMENT_TYPE: [value or N/A]
  SUPPLIER: [value or N/A]
  ABN: [11-digit Australian Business Number or N/A]
  VENDOR_ID: [internal vendor identifier or N/A]           # NEW FIELD
  TAX_RATE: [applicable tax rate percentage or N/A]        # NEW FIELD
  CURRENCY: [currency code (USD, AUD, etc.) or N/A]        # NEW FIELD
  # ... other existing fields
  
  # Update field count in instructions
  ABSOLUTELY CRITICAL: Output EXACTLY 29 lines using ONLY the keys listed above.
  Include ALL 29 keys listed above even if value is N/A.
```

**Critical Requirements**:
- Each field must follow exact format: `FIELD_NAME: [description or N/A]`
- Update total field count in instructions
- Maintain consistent field naming (UPPER_CASE with underscores)
- Update prompts for both models if they differ

#### 2. Update Local Configuration
**File**: `model_comparison_local.yaml`

Apply identical changes to maintain development/production parity.

### Phase 2: Analysis Component Updates (RECOMMENDED)

#### 1. Update Field Weights
**File**: `vision_processor/analysis/simple_metrics.py`

```python
# BEFORE - Original field weights
FIELD_WEIGHTS = {
    "DATE": 1.0,
    "SUPPLIER": 1.0,
    "ABN": 1.0,
    # ... existing fields
}

# AFTER - Add new fields with appropriate weights
FIELD_WEIGHTS = {
    "DATE": 1.0,
    "SUPPLIER": 1.0,
    "ABN": 1.0,
    "VENDOR_ID": 1.0,        # NEW: Standard business field
    "TAX_RATE": 1.2,         # NEW: Higher weight for financial accuracy
    "CURRENCY": 0.8,         # NEW: Lower weight for international docs
    # ... existing fields
}
```

**Weight Assignment Guidelines**:
- **1.0**: Standard business fields (names, addresses, dates)
- **1.2-1.5**: Critical financial fields (totals, tax rates, account numbers)
- **0.8-0.9**: Contextual fields (currency, document metadata)
- **0.5-0.7**: Optional fields (websites, secondary contact info)

#### 2. Update Quality Thresholds
**File**: `model_comparison.yaml`

```yaml
# Update thresholds to account for new field count
quality_thresholds:
  excellent: 18    # was 12 for 26 fields, now ~62% of 29 fields
  good: 12         # was 8, now ~41% of 29 fields  
  fair: 8          # was 5, now ~28% of 29 fields
  poor: 0          # unchanged
```

### Phase 3: Documentation Updates (MAINTENANCE)

#### 1. Update Main Documentation
**File**: `README.md`

- Update field count references (26 → new count)
- Update example field lists
- Update DataFrame structure descriptions
- Update CLI usage examples

#### 2. Update Technical Documentation
**Files**: `docs/batch_results_creation.md`, `docs/batch_to_csv_analysis.md`

- Update JSON structure examples
- Update field descriptions
- Update extraction statistics examples
- Update DataFrame column counts

### Phase 4: Ground Truth Data Updates (VALIDATION)

#### 1. Extend Existing Annotations
**Files**: `annotations/*.csv`

For new fields that can be annotated from existing images:
```csv
# Add new field annotations
words,prediction,annotation
JB Hi-Fi,,SUPPLIER
"Tax Rate: 10%",,TAX_RATE
"AUD",,CURRENCY
```

#### 2. Create New Test Cases
For fields requiring new document types:
- Source documents with new field types
- Create comprehensive annotations
- Ensure balanced field representation

## Testing Strategy

### Phase 1: Configuration Validation

#### 1. Syntax Validation
```bash
# Test YAML syntax
python -c "import yaml; yaml.safe_load(open('model_comparison.yaml'))"

# Test field extraction
python -c "
from vision_processor.config import ConfigManager
config = ConfigManager('model_comparison.yaml')
fields = config.get_expected_fields()
print(f'Extracted {len(fields)} fields: {fields}')
"
```

#### 2. Field Count Verification
```bash
# Verify expected field count matches prompt instructions
python -c "
from vision_processor.config import ConfigManager
config = ConfigManager('model_comparison.yaml')
fields = config.get_expected_fields()
prompt = config.extraction_prompt
import re
count_in_prompt = re.search(r'EXACTLY (\d+) lines', prompt)
if count_in_prompt:
    expected_count = int(count_in_prompt.group(1))
    actual_count = len(fields)
    print(f'Expected: {expected_count}, Actual: {actual_count}')
    assert expected_count == actual_count, 'Field count mismatch!'
else:
    print('Warning: No explicit field count found in prompt')
"
```

### Phase 2: Processing Validation

#### 1. Single Document Test
```bash
# Test with single image to verify new field extraction
python -m vision_processor.cli.extract_cli extract test_image.png --model llama --yaml-file model_comparison.yaml --debug
```

#### 2. Batch Processing Test
```bash
# Test batch processing with updated schema
python -m vision_processor.cli.extract_cli batch ./test_datasets/ --model llama --yaml-file model_comparison.yaml
```

#### 3. DataFrame Conversion Test
```bash
# Test DataFrame creation with new fields
python -m vision_processor.cli.batch_to_csv_cli convert batch_results.json --yaml-file model_comparison.yaml --info
```

### Phase 3: Analysis Validation

#### 1. Metrics Calculation Test
```python
# Test updated metrics with new fields
from vision_processor.analysis.simple_metrics import InformationExtractionCalculator
from vision_processor.config import ConfigManager

config = ConfigManager('model_comparison.yaml')
expected_fields = config.get_expected_fields()

# Verify all expected fields have weights
from vision_processor.analysis.simple_metrics import FIELD_WEIGHTS
missing_weights = [f for f in expected_fields if f not in FIELD_WEIGHTS]
if missing_weights:
    print(f"Warning: Missing weights for fields: {missing_weights}")
```

#### 2. Quality Threshold Test
```python
# Test quality assessment with new field counts
import json
with open('batch_results.json') as f:
    results = json.load(f)

for result in results[:3]:  # Test first 3 results
    field_count = len([k for k, v in result['extracted_fields'].items() if v != 'N/A'])
    print(f"Image: {result['filename']}, Fields extracted: {field_count}")
    
    # Manual quality assessment
    if field_count >= 18: quality = "excellent"
    elif field_count >= 12: quality = "good"  
    elif field_count >= 8: quality = "fair"
    else: quality = "poor"
    print(f"Quality: {quality}")
```

## Deployment Process

### Stage 1: Development Environment

#### 1. Update Local Configuration
```bash
# Update local config first
cp model_comparison.yaml model_comparison_local.yaml
# Edit model_comparison_local.yaml with local paths
```

#### 2. Local Testing
```bash
# Test with local configuration
python -m vision_processor.cli.extract_cli extract test_image.png --model llama --yaml-file model_comparison_local.yaml

# Test batch processing
python -m vision_processor.cli.extract_cli batch ./small_test_set/ --model llama --yaml-file model_comparison_local.yaml
```

### Stage 2: Staging Environment

#### 1. Deploy Configuration Changes
```bash
# Deploy updated YAML to staging
cp model_comparison.yaml /staging/config/
```

#### 2. Validation Testing
```bash
# Run comprehensive tests on staging
python -m vision_processor.cli.extract_cli batch ./staging_test_set/ --model llama
python -m vision_processor.cli.batch_to_csv_cli convert batch_results.json --yaml-file model_comparison.yaml
```

#### 3. Performance Testing
```bash
# Monitor processing times and memory usage
python -m vision_processor.cli.extract_cli batch ./large_test_set/ --model llama --verbose
```

### Stage 3: Production Deployment

#### 1. Blue-Green Deployment
- Deploy to blue environment first
- Validate with production traffic sample
- Switch traffic if validation succeeds

#### 2. Monitoring
- Track field extraction rates for new fields
- Monitor processing times for performance regression
- Watch error logs for schema-related issues

#### 3. Rollback Plan
```bash
# Keep previous configuration for quick rollback
cp model_comparison.yaml model_comparison_backup_$(date +%Y%m%d).yaml
```

## Common Pitfalls and Solutions

### 1. Field Count Mismatch

**Problem**: Prompt instructions don't match actual field count
```
ABSOLUTELY CRITICAL: Output EXACTLY 26 lines using ONLY the keys listed above.
# But actually have 29 fields now
```

**Solution**: Always update field count in prompt instructions
```
ABSOLUTELY CRITICAL: Output EXACTLY 29 lines using ONLY the keys listed above.
```

**Prevention**: Automated testing that verifies field count consistency

### 2. Inconsistent Field Naming

**Problem**: Mixed naming conventions
```yaml
PAYER_NAME: [value or N/A]     # Correct: UPPER_CASE
payer_email: [value or N/A]    # Wrong: lower_case
PAYER-PHONE: [value or N/A]    # Wrong: hyphens
```

**Solution**: Enforce consistent UPPER_CASE with underscores
**Prevention**: Use naming convention linter or validation script

### 3. Missing Field Weights

**Problem**: New fields added to YAML but not to FIELD_WEIGHTS
```python
# simple_metrics.py missing new fields
FIELD_WEIGHTS = {
    "SUPPLIER": 1.0,
    # Missing: VENDOR_ID, TAX_RATE, CURRENCY
}
```

**Impact**: New fields ignored in quality metrics
**Solution**: Add all new fields to FIELD_WEIGHTS
**Prevention**: Automated check that compares fields in YAML vs metrics

### 4. Outdated Ground Truth

**Problem**: Evaluation fails because ground truth expects old fields
```csv
# annotations/test.csv still has old field structure
SUPPLIER,ABN,TOTAL
# Missing new fields: VENDOR_ID,TAX_RATE,CURRENCY
```

**Impact**: Cannot validate new field extraction accuracy
**Solution**: Update or recreate ground truth data
**Prevention**: Version ground truth data with schema versions

### 5. Documentation Lag

**Problem**: Documentation shows old field examples
```markdown
DataFrame Structure:
- Header: [image, DOCUMENT_TYPE, SUPPLIER, ABN, ...] (27 columns: image + 26 fields)
# Should be: (30 columns: image + 29 fields)
```

**Impact**: User confusion and incorrect expectations
**Solution**: Update all documentation references
**Prevention**: Documentation review checklist

## Validation Checklist

### Pre-Deployment Checklist

- [ ] **Configuration Files Updated**
  - [ ] model_comparison.yaml extraction_prompt updated
  - [ ] model_comparison_local.yaml matches production
  - [ ] Field count in prompt instructions matches actual fields
  - [ ] All fields follow naming conventions

- [ ] **Code Updates Applied**
  - [ ] FIELD_WEIGHTS includes all new fields
  - [ ] Quality thresholds adjusted for new field count
  - [ ] No hardcoded field references missed

- [ ] **Documentation Updated**
  - [ ] README.md field counts and examples
  - [ ] Technical documentation updated
  - [ ] API documentation reflects changes

- [ ] **Testing Completed**
  - [ ] Single document extraction works
  - [ ] Batch processing succeeds
  - [ ] DataFrame conversion handles new fields
  - [ ] Metrics calculation includes new fields
  - [ ] Performance benchmarks acceptable

- [ ] **Ground Truth Prepared**
  - [ ] Test annotations include new fields
  - [ ] Evaluation datasets updated
  - [ ] Quality assessment baselines established

### Post-Deployment Checklist

- [ ] **Production Validation**
  - [ ] Sample documents processed successfully
  - [ ] New fields being extracted as expected
  - [ ] No processing errors or crashes
  - [ ] Performance within acceptable limits

- [ ] **Monitoring Active**
  - [ ] Field extraction rates tracked
  - [ ] Processing time monitoring
  - [ ] Error rate monitoring
  - [ ] Quality metrics dashboards updated

- [ ] **Rollback Ready**
  - [ ] Previous configuration backed up
  - [ ] Rollback procedure tested
  - [ ] Monitoring alerts configured

## Advanced Scenarios

### 1. Conditional Fields

Some fields may only apply to certain document types:

```yaml
# Document-type-specific instructions
extraction_prompt: |
  Extract data from this business document. Output ALL fields below with their exact keys.
  Use "N/A" if field is not visible, not present, or not applicable to this document type.
  
  # Universal fields (all documents)
  DOCUMENT_TYPE: [value or N/A]
  SUPPLIER: [value or N/A]
  
  # Invoice-specific fields (use N/A for receipts/statements)
  INVOICE_NUMBER: [invoice number or N/A]
  DUE_DATE: [payment due date or N/A]
  
  # Bank statement-specific fields (use N/A for invoices/receipts)
  ACCOUNT_NUMBER: [account number or N/A]
  STATEMENT_PERIOD: [statement date range or N/A]
```

### 2. Field Dependencies

Some fields may have validation dependencies:

```python
# Example validation logic for dependent fields
def validate_extracted_fields(fields):
    # If DOCUMENT_TYPE is "BANK_STATEMENT", require bank fields
    if fields.get("DOCUMENT_TYPE") == "BANK_STATEMENT":
        required_bank_fields = ["ACCOUNT_NUMBER", "STATEMENT_PERIOD"]
        for field in required_bank_fields:
            if fields.get(field) == "N/A":
                # Flag for manual review
                fields[f"{field}_CONFIDENCE"] = "low"
```

### 3. Field Versioning

For large schema changes, consider versioning:

```yaml
# Schema version tracking
schema_version: "2.1"

# Migration notes
schema_changes:
  "2.1":
    - "Added VENDOR_ID field"
    - "Added TAX_RATE field" 
    - "Added CURRENCY field"
  "2.0":
    - "Restructured bank statement fields"
    - "Added transaction details"
```

## Future Considerations

### 1. Automated Schema Management

**Goal**: Reduce manual updates and human error

**Approach**:
- Schema definition files separate from prompts
- Code generation for metrics and validation
- Automated documentation updates

### 2. Dynamic Field Weights

**Goal**: Automatically adjust field importance based on extraction success

**Approach**:
- Machine learning-based weight optimization
- Historical extraction rate analysis
- Business impact weighting

### 3. Multi-Schema Support

**Goal**: Support different document types with different field sets

**Approach**:
- Document type detection
- Schema selection based on document type
- Unified analysis across schema types

## Conclusion

Field schema changes are a critical aspect of production maintenance that require careful planning and systematic execution. The system's design with dynamic field extraction minimizes the impact of changes, but manual updates to analysis components and documentation remain necessary.

**Key Success Factors**:

1. **Systematic Approach**: Follow the defined process phases
2. **Comprehensive Testing**: Validate all components before deployment
3. **Documentation Discipline**: Keep all documentation current
4. **Monitoring**: Track field extraction performance post-deployment
5. **Rollback Readiness**: Always have a quick recovery plan

By following this guide, field schema changes can be implemented safely and efficiently, ensuring continued system reliability while adapting to evolving business requirements.