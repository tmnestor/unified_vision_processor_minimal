# Ground Truth Generation for Model Evaluation

This document describes the comprehensive ground truth generation system that enables proper evaluation and comparison of vision models (Llama-3.2-Vision and InternVL3) for document processing tasks.

## Overview

The ground truth generation system solves the critical evaluation problem: **how to measure model performance without known correct answers**. Previously, model comparison relied on subjective visual inspection of extraction results. Now, we can generate precise ground truth data and calculate objective accuracy metrics.

## Architecture

### Core Components

1. **`generate_invoices_with_ground_truth.py`** - Main CLI script with multiple commands
2. **`vision_processor/utils/ground_truth_generator.py`** - Core ground truth generation logic
3. **`synthetic_invoice_generator.py`** - Synthetic invoice creation (existing)
4. **Integration with evaluation CLI** - Seamless workflow for model comparison

### Field Mapping

The system maps synthetic invoice data to the 25 expected fields from `model_comparison.yaml`:

```yaml
expected_fields:
  - DOCUMENT_TYPE    # → document_type
  - SUPPLIER         # → business_name
  - ABN              # → abn
  - PAYER_NAME       # → payer_name
  - PAYER_ADDRESS    # → payer_address
  - PAYER_PHONE      # → payer_phone
  - PAYER_EMAIL      # → payer_email
  - INVOICE_DATE     # → date
  - DUE_DATE         # → due_date
  - GST              # → gst (formatted as $X.XX)
  - TOTAL            # → total (formatted as $X.XX)
  - SUBTOTAL         # → subtotal (formatted as $X.XX)
  - QUANTITIES       # → quantities (pipe-separated)
  - PRICES           # → prices (pipe-separated)
  - DESCRIPTIONS     # → items_list
  - BUSINESS_ADDRESS # → business_address
  - BUSINESS_PHONE   # → business_phone
  - BSB_NUMBER       # → bsb
  - BANK_ACCOUNT_NUMBER # → bank_account
  # ... plus fields set to N/A for invoices
```

## Usage Guide

### 1. Generate New Evaluation Datasets

Create synthetic invoices WITH ground truth for comprehensive evaluation:

```bash
# Generate 10 mixed business type invoices
python generate_invoices_with_ground_truth.py generate-with-synthetic --count 10

# Generate 5 specific business type invoices
python generate_invoices_with_ground_truth.py generate-with-synthetic \
    --count 5 \
    --business-type retail \
    --output-dir ./evaluation_retail

# Generate with custom item range
python generate_invoices_with_ground_truth.py generate-with-synthetic \
    --count 20 \
    --min-items 2 \
    --max-items 8 \
    --business-type mixed
```

**Output Structure:**
```
evaluation_dataset/
├── ground_truth.csv      # CSV with all field mappings
├── images/              # Generated invoice images
│   ├── synthetic_invoice_001_retail.png
│   ├── synthetic_invoice_002_fuel.png
│   └── ...
└── json/               # Detailed JSON records
    ├── synthetic_invoice_001_retail.json
    └── ...
```

### 2. Reconstruct Ground Truth from Existing Images

Create ground truth for existing synthetic invoices in `/datasets`:

```bash
# Automatic mode - use model agreement for ground truth
python generate_invoices_with_ground_truth.py reconstruct /Users/tod/Desktop/datasets --auto

# Interactive mode - manual verification of key fields
python generate_invoices_with_ground_truth.py reconstruct /Users/tod/Desktop/datasets

# Include hardcoded Shell example
python generate_invoices_with_ground_truth.py reconstruct /Users/tod/Desktop/datasets \
    --shell-example \
    --output-dir ./existing_ground_truth
```

**Process:**
1. **Dual Extraction**: Both Llama and InternVL process each image
2. **Field Comparison**: System compares extraction results
3. **Ground Truth Creation**: Uses agreement logic:
   - Both models agree + non-N/A → Use agreed value
   - Critical fields (SUPPLIER, ABN, TOTAL, GST, INVOICE_DATE) → Prefer non-N/A
   - Disagreement → Mark as N/A
4. **Manual Override**: Interactive mode allows verification/correction

### 3. Verify Ground Truth Quality

Inspect and validate generated ground truth files:

```bash
# Show sample data and field coverage statistics
python generate_invoices_with_ground_truth.py verify ground_truth.csv
```

**Sample Output:**
```
Ground truth file: ground_truth.csv
Total records: 10

Sample record:
  SUPPLIER: Shell Australia
  ABN: 36 643 730 685
  TOTAL: $71.67
  GST: $6.52
  ...

Field coverage:
  SUPPLIER: 10/10 (100.0%)
  ABN: 10/10 (100.0%)
  TOTAL: 10/10 (100.0%)
  GST: 10/10 (100.0%)
  INVOICE_DATE: 10/10 (100.0%)
  ...
```

### 4. Quick Testing

Generate small test datasets for development:

```bash
# Generate 3 test invoices (retail, fuel, professional)
python generate_invoices_with_ground_truth.py quick-test
```

## Integration with Model Evaluation

### Run Comprehensive Model Comparison

Once ground truth is generated, use the evaluation CLI:

```bash
# Compare both models on ground truth dataset
python -m vision_processor.cli.evaluation_cli compare ground_truth.csv --images-dir ./images

# Single model evaluation
python -m vision_processor.cli.evaluation_cli compare ground_truth.csv \
    --images-dir ./images \
    --models internvl3

# With detailed output
python -m vision_processor.cli.evaluation_cli compare ground_truth.csv \
    --images-dir ./images \
    --verbose
```

### Evaluation Metrics

The system calculates:

- **Overall Accuracy**: Percentage of correctly extracted fields
- **Per-Field Accuracy**: Individual field extraction success rates
- **Field Coverage**: Which fields are most/least successfully extracted
- **Fuzzy Matching**: Smart comparison for addresses, amounts, lists

**Fuzzy Matching Logic:**
- **Amounts**: Numeric equivalence within $0.01
- **Addresses**: 70% word overlap threshold
- **Lists**: Set comparison for quantities, prices, descriptions

## Best Practices

### 1. Dataset Size Recommendations

- **Development**: 3-5 invoices (quick-test)
- **Testing**: 10-20 invoices per business type
- **Production Evaluation**: 50+ invoices with mixed types

### 2. Ground Truth Quality Assurance

```bash
# Always verify after generation
python generate_invoices_with_ground_truth.py verify ground_truth.csv

# Check for missing images
python -m vision_processor.cli.evaluation_cli validate-ground-truth ground_truth.csv --images-dir ./images
```

### 3. Business Type Distribution

For comprehensive evaluation, ensure balanced representation:

```bash
# Generate balanced dataset
python generate_invoices_with_ground_truth.py generate-with-synthetic \
    --count 40 \
    --business-type mixed  # Ensures random distribution
```

### 4. Iterative Improvement

1. **Generate small test set** → Verify quality
2. **Run initial evaluation** → Identify weak fields
3. **Generate larger dataset** → Focus on problem areas
4. **Continuous evaluation** → Track improvements

## Troubleshooting

### Common Issues

**ImportError for typer/rich:**
```bash
# Ensure conda environment is activated
conda activate unified_vision_processor
```

**OpenMP Error:**
```bash
# Set environment variable
export KMP_DUPLICATE_LIB_OK=TRUE
```

**Model Loading Issues:**
- Ensure GPU memory is sufficient
- Check model paths in configuration
- Verify CUDA/MPS availability

### Validation Commands

```bash
# Check ground truth file structure
head -n 5 ground_truth.csv

# Verify image-ground truth alignment
python generate_invoices_with_ground_truth.py verify ground_truth.csv

# Test evaluation pipeline
python -m vision_processor.cli.evaluation_cli validate-ground-truth ground_truth.csv
```

## Advanced Usage

### Custom Field Mapping

To modify field mappings, edit `vision_processor/utils/ground_truth_generator.py`:

```python
def save_ground_truth(self, image_filename: str, invoice_data: Dict) -> Dict[str, str]:
    ground_truth = {
        "CUSTOM_FIELD": invoice_data.get("new_field", "N/A"),
        # ... other mappings
    }
```

### Batch Processing

```bash
# Process multiple directories
for dir in dataset1 dataset2 dataset3; do
    python generate_invoices_with_ground_truth.py reconstruct $dir --auto --output-dir ground_truth_$dir
done
```

### Integration with CI/CD

```bash
#!/bin/bash
# evaluation_pipeline.sh

# Generate test dataset
python generate_invoices_with_ground_truth.py quick-test

# Run evaluation
python -m vision_processor.cli.evaluation_cli compare quick_test_dataset/ground_truth.csv \
    --images-dir quick_test_dataset/images \
    --quiet

# Check for regressions
python scripts/check_accuracy_threshold.py evaluation_results.json
```

## Future Enhancements

### Planned Features

1. **Ground Truth Editor**: GUI for manual correction
2. **Confidence Scoring**: Model confidence integration
3. **Active Learning**: Identify uncertain predictions for manual review
4. **Multi-Modal Ground Truth**: Support for text + bounding boxes

### Extension Points

- **Custom Business Types**: Add new invoice templates
- **Field Validators**: Custom validation logic per field
- **Export Formats**: JSONL, COCO, YOLO label formats
- **Metrics Dashboard**: Web interface for evaluation results

---

## Summary

The ground truth generation system transforms model evaluation from subjective visual inspection to objective, metrics-driven analysis. It enables:

✅ **Accurate Performance Measurement** - Precise accuracy percentages  
✅ **Field-Level Analysis** - Identify specific extraction weaknesses  
✅ **Model Comparison** - Objective Llama vs InternVL comparison  
✅ **Regression Testing** - Detect performance degradation  
✅ **Dataset Creation** - Generate evaluation benchmarks  

This foundation enables data-driven optimization of the vision document processing pipeline.