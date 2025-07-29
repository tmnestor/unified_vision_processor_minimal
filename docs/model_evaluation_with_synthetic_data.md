# Model Evaluation with Synthetic Ground Truth Data

This document explains how to use the vision processor evaluation system to compare InternVL3 and Llama-3.2-Vision models using synthetic ground truth data.

## Overview

The evaluation system provides comprehensive model comparison capabilities using known ground truth data. It measures extraction accuracy across all 25 fields defined in `model_comparison.yaml` and generates detailed performance reports.

## Prerequisites

### Environment Setup
```bash
# Activate the conda environment
conda activate unified_vision_processor

# Set OpenMP workaround (if needed on macOS)
export KMP_DUPLICATE_LIB_OK=TRUE
```

### Required Files
- `test_synthetic/evaluation_ground_truth.csv` - Ground truth data
- `test_synthetic/*.png` - Synthetic document images
- `model_comparison.yaml` - Field definitions and model configuration

## Evaluation Methods

### 1. CLI-Based Evaluation (Recommended)

#### Single Model Benchmarking
```bash
# Evaluate InternVL model
python -m vision_processor.cli.evaluation_cli benchmark test_synthetic --model internvl

# Evaluate Llama model
python -m vision_processor.cli.evaluation_cli benchmark test_synthetic --model llama
```

#### Model Comparison
```bash
# Compare both models automatically
python -m vision_processor.cli.evaluation_cli compare test_synthetic/evaluation_ground_truth.csv

# Compare specific models
python -m vision_processor.cli.evaluation_cli compare test_synthetic/evaluation_ground_truth.csv --models internvl,llama
```

### 2. Programmatic Evaluation

#### Basic Usage
```python
from vision_processor.evaluation.evaluator import ExtractionEvaluator

# Initialize evaluator
evaluator = ExtractionEvaluator(
    ground_truth_csv='test_synthetic/evaluation_ground_truth.csv',
    images_dir='test_synthetic',
    output_dir='evaluation_results'
)

# Compare models
results = evaluator.compare_models(['internvl', 'llama'])

# Generate report
evaluator.generate_report(results)
```

#### Single Model Evaluation
```python
# Evaluate single model
results = evaluator.evaluate_model('internvl')

print(f"Accuracy: {results['avg_accuracy']:.1%}")
print(f"Processing Time: {results['avg_processing_time']:.1f}s")
print(f"Success Rate: {results['success_rate']:.1%}")
```

## Evaluation Metrics

### Overall Performance Metrics
- **Average Accuracy**: Percentage of correctly extracted fields
- **Success Rate**: Percentage of successfully processed images
- **Average Processing Time**: Time per document in seconds
- **Fields Extracted**: Average number of fields extracted per document

### Field-Wise Accuracy
Individual accuracy scores for each of the 25 extraction fields:

#### Core Document Fields
- `DOCUMENT_TYPE`, `SUPPLIER`, `ABN`
- `PAYER_NAME`, `PAYER_ADDRESS`, `PAYER_PHONE`, `PAYER_EMAIL`
- `INVOICE_DATE`, `DUE_DATE`

#### Financial Fields
- `GST`, `TOTAL`, `SUBTOTAL`
- `QUANTITIES`, `PRICES`
- `SUPPLIER_WEBSITE`

#### Business Information
- `BUSINESS_ADDRESS`, `BUSINESS_PHONE`

#### Banking Fields (Bank Statements Only)
- `BANK_NAME`, `BSB_NUMBER`, `BANK_ACCOUNT_NUMBER`
- `ACCOUNT_HOLDER`, `STATEMENT_PERIOD`
- `OPENING_BALANCE`, `CLOSING_BALANCE`, `DESCRIPTIONS`

## Understanding Results

### Console Output Example
```
📊 EVALUATION REPORT
============================================================

Model Performance Overview
┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃ Model        ┃ Success Rate┃ Avg Accuracy┃ Avg Speed (s)┃ Fields/Image┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
│ INTERNVL3    │ 100.0%      │ 92.3%       │ 8.2s         │ 23.1        │
│ LLAMA32_VISION│ 100.0%      │ 87.6%       │ 12.4s        │ 21.8        │
└──────────────┴─────────────┴─────────────┴──────────────┴─────────────┘

🏆 WINNERS:
🎯 Best Accuracy: INTERNVL3 (92.3%)
⚡ Fastest: INTERNVL3 (8.2s)
```

### Field-Wise Comparison Example  
```
📋 Field-wise Accuracy Comparison
┏━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┓
┃ Field             ┃ INTERNVL3 ┃ LLAMA32_VISION ┃
┡━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━┩
│ DOCUMENT_TYPE     │ 100.0%    │ 95.0%          │
│ SUPPLIER          │ 95.0%     │ 90.0%          │
│ ABN               │ 100.0%    │ 85.0%          │
│ GST               │ 90.0%     │ 80.0%          │
│ TOTAL             │ 95.0%     │ 85.0%          │
│ BANK_NAME         │ 100.0%    │ 50.0%          │
└───────────────────┴───────────┴────────────────┘
```

## Output Files

The evaluation system generates several output files in the specified output directory:

### Generated Files
```
evaluation_results/
├── comparison_results.json          # Raw comparison data
├── internvl_results.json           # InternVL detailed results  
├── llama_results.json              # Llama detailed results
└── evaluation_report.md            # Comprehensive markdown report
```

### JSON Results Structure
```json
{
  "internvl": {
    "model_type": "internvl",
    "total_images": 10,
    "successful": 10,  
    "failed": 0,
    "success_rate": 1.0,
    "avg_accuracy": 0.923,
    "avg_processing_time": 8.2,
    "avg_fields_extracted": 23.1,
    "field_wise_accuracy": {
      "DOCUMENT_TYPE": 1.0,
      "SUPPLIER": 0.95,
      "ABN": 1.0,
      // ... all 25 fields
    },
    "detailed_results": [
      // Individual image results
    ]
  }
}
```

## Document Type Analysis

### Mixed Document Evaluation
The synthetic dataset includes different document types that test various aspects of model performance:

#### Invoice Documents (8/10)
- **TAX INVOICE**: Standard business invoices
- **RECEIPT**: Fuel and retail receipts
- Tests: Financial calculations, business information, customer details

#### Bank Statement Documents (2/10)  
- **BANK STATEMENT**: Complete banking information
- Tests: Banking fields, account details, transaction data
- Challenge: Different layout and field requirements

### Document-Specific Performance
```python
# Analyze performance by document type
for image_file, result in results['internvl']['detailed_results']:
    doc_type = result['ground_truth']['DOCUMENT_TYPE']
    accuracy = result['overall_accuracy']
    print(f"{doc_type}: {accuracy:.1%} accuracy")
```

## Performance Benchmarking

### Speed Analysis
- **Processing Time**: Time to extract all fields from one document
- **Throughput**: Documents processed per minute
- **Memory Usage**: GPU memory consumption during processing

### Accuracy Analysis
- **Field Completeness**: Percentage of required fields extracted
- **Financial Accuracy**: Correctness of monetary calculations
- **Text Recognition**: Accuracy of OCR and text extraction
- **Layout Understanding**: Ability to locate fields in different layouts

## Advanced Evaluation Options

### Custom Test Sets
```python
# Evaluate on specific images
test_images = ['synthetic_invoice_003.png', 'synthetic_invoice_007.png']  # Bank statements only
results = evaluator.compare_models(['internvl', 'llama'], test_images)
```

### Field Subset Analysis
```python
# Focus on financial fields only
financial_fields = ['GST', 'TOTAL', 'SUBTOTAL', 'QUANTITIES', 'PRICES']
# Custom analysis code here
```

### Error Analysis
```python
# Analyze failed extractions
for result in results['internvl']['detailed_results']:
    if 'error' in result:
        print(f"Failed: {result['image_file']} - {result['error']}")
    elif result['overall_accuracy'] < 0.8:
        print(f"Low accuracy: {result['image_file']} - {result['overall_accuracy']:.1%}")
```

## Troubleshooting

### Common Issues

#### Memory Errors
```bash
# Reduce batch size or use CPU fallback
export CUDA_VISIBLE_DEVICES=""  # Force CPU mode
```

#### Model Loading Issues
```bash
# Check model paths in model_comparison.yaml
# Ensure models are downloaded and accessible
python -c "from transformers import AutoModel; print('Models accessible')"
```

#### CSV Format Issues
```bash
# Verify CSV structure
python -c "
import pandas as pd
df = pd.read_csv('test_synthetic/evaluation_ground_truth.csv')
print(f'Columns: {len(df.columns)}')
print(f'Rows: {len(df)}')
assert len(df.columns) == 26  # 25 fields + image_file
"
```

### Debug Mode
```python
# Enable detailed logging
evaluator = ExtractionEvaluator(
    ground_truth_csv='test_synthetic/evaluation_ground_truth.csv',
    images_dir='test_synthetic',
    output_dir='debug_output'
)

# Check field loading
print(f"Fields loaded: {evaluator.extraction_fields}")
print(f"Ground truth images: {len(evaluator.ground_truth)}")
```

## Performance Optimization

### Hardware Recommendations
- **GPU**: NVIDIA with 16GB+ VRAM (V100, A100, H100)
- **RAM**: 32GB+ system memory
- **Storage**: SSD for faster image loading

### Batch Processing
```python
# Process multiple test sets
test_sets = ['test_synthetic', 'validation_set', 'production_test']
for test_set in test_sets:
    evaluator = ExtractionEvaluator(
        ground_truth_csv=f'{test_set}/evaluation_ground_truth.csv',
        images_dir=test_set,
        output_dir=f'results_{test_set}'
    )
    results = evaluator.compare_models(['internvl', 'llama'])
    evaluator.generate_report(results)
```

## Model Comparison Insights

### Expected Performance Characteristics

#### InternVL3
- **Strengths**: High accuracy, good layout understanding
- **Speed**: Generally faster processing
- **Banking Fields**: Strong performance on structured data
- **Challenge**: Requires `trust_remote_code=True`

#### Llama-3.2-Vision  
- **Strengths**: Good general document understanding
- **Markdown Output**: May return full document markdown (requires post-processing)
- **Speed**: Slower but thorough processing
- **Challenge**: May need prompt tuning for specific fields

### Decision Criteria
- **Accuracy Priority**: Choose model with higher field-wise accuracy
- **Speed Priority**: Consider processing time requirements
- **Deployment**: Factor in security requirements (`trust_remote_code`)
- **Maintenance**: Consider model update frequency and support

## Integration with Production

### Validation Pipeline
```python
def validate_model_performance(model_name, min_accuracy=0.85):
    evaluator = ExtractionEvaluator(
        ground_truth_csv='test_synthetic/evaluation_ground_truth.csv',
        images_dir='test_synthetic',
        output_dir=f'validation_{model_name}'
    )
    
    results = evaluator.evaluate_model(model_name)
    
    if results['avg_accuracy'] >= min_accuracy:
        print(f"✅ {model_name} passed validation ({results['avg_accuracy']:.1%})")
        return True
    else:
        print(f"❌ {model_name} failed validation ({results['avg_accuracy']:.1%})")
        return False

# Validate both models
internvl_ready = validate_model_performance('internvl')
llama_ready = validate_model_performance('llama')
```

### Continuous Evaluation
```bash
# Schedule regular evaluations
# crontab entry example:
# 0 2 * * * cd /path/to/vision_comparison && python -m vision_processor.cli.evaluation_cli compare test_synthetic/evaluation_ground_truth.csv
```

This evaluation framework provides comprehensive model comparison capabilities with synthetic ground truth data, enabling data-driven decisions for production model selection.