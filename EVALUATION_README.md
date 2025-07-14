# Vision Model Evaluation System

## Overview

This evaluation system provides comprehensive performance comparison between **InternVL3-8B** and **Llama-3.2-11B-Vision** models for Australian business document processing. The system evaluates key-value extraction accuracy across 18 standardized fields using ground truth data from real receipt images.

## 🏆 Key Results Summary

Based on comprehensive testing with 5 diverse Australian receipts:

| Model | Overall Accuracy | Processing Speed | Fields Extracted | Success Rate |
|-------|-----------------|------------------|------------------|--------------|
| **InternVL3-8B** | **45.6%** | **5.2s** | **15.4/image** | **100%** |
| Llama-3.2-Vision | 19.1% | 13.2s | 4.2/image | 100% |

**Winner: InternVL3-8B** - 2.4x better accuracy, 2.5x faster processing

## 🎯 Evaluation Fields

The system evaluates extraction accuracy across 18 standardized fields:

### Core Fields
- `DATE` - Transaction date (DD/MM/YYYY format)
- `STORE` - Business name
- `TOTAL` - Total amount including GST
- `GST` - GST amount
- `ABN` - Australian Business Number

### Transaction Details
- `SUBTOTAL` - Amount before GST
- `ITEMS` - Product names (pipe-separated)
- `QUANTITIES` - Item quantities (pipe-separated)
- `PRICES` - Individual prices (pipe-separated)
- `RECEIPT_NUMBER` - Receipt/invoice number
- `PAYMENT_METHOD` - Payment type (VISA, EFTPOS, etc.)

### Document Metadata
- `DOCUMENT_TYPE` - RECEIPT or TAX INVOICE
- `ADDRESS` - Business address
- `PHONE` - Business phone number
- `TIME` - Transaction time
- `CARD_NUMBER` - Masked card number
- `AUTH_CODE` - Authorization code
- `STATUS` - Transaction status

## 🔧 Technical Architecture

### System Components

```
vision_processor/
├── evaluation/                        # Evaluation module
│   ├── __init__.py                   # Module exports
│   └── evaluator.py                  # ExtractionEvaluator class
├── cli/
│   └── evaluation_cli.py             # CLI commands
├── extraction/
│   └── simple_extraction_manager.py  # Working extraction pipeline
├── config/
│   ├── simple_config.py             # Model configuration
│   └── prompts.yaml                 # Evaluation field schema

Root Level:
├── evaluate_models.py                # Simple evaluation script
├── evaluation_ground_truth.csv       # Test data with expected values
└── datasets/                         # Receipt images (image14.png, etc.)

Results Output:
├── evaluation_results/
│   ├── comparison_results.json       # Detailed comparison data
│   ├── internvl3_results.json        # InternVL3 detailed results
│   ├── llama32_vision_results.json   # Llama detailed results
│   └── evaluation_report.md          # Human-readable report
```

### Evaluation Methodology

1. **Ground Truth Loading**: Loads expected field values from CSV
2. **Model Processing**: Uses working `SimpleExtractionManager` with model-specific prompts
3. **Field Extraction**: `UniversalKeyValueParser` extracts structured data
4. **Accuracy Calculation**: Field-by-field comparison with tolerance for numeric values
5. **Aggregation**: Overall accuracy, processing time, and field-wise performance metrics

### Accuracy Calculation Logic

```python
# Numeric fields (GST, TOTAL, SUBTOTAL)
tolerance = 0.01  # 1 cent tolerance
accuracy = 1.0 if abs(extracted - ground_truth) < tolerance else 0.0

# List fields (ITEMS, QUANTITIES, PRICES)
accuracy = exact_matches / total_items

# String fields (STORE, ADDRESS)
accuracy = 1.0 if exact_match else 0.8 if partial_match else 0.0

# Date fields
accuracy = 1.0 if normalized_dates_match else 0.0
```

## 📋 Setup Instructions

### Prerequisites

```bash
# Activate conda environment
conda activate unified_vision_processor

# Verify model access
python -c "from vision_processor.models.internvl_model import InternVLModel; print('InternVL OK')"
python -c "from vision_processor.models.llama_model import LlamaVisionModel; print('Llama OK')"
```

### Environment Configuration

The system uses environment variables for model paths:

```bash
# Remote development machine
export VISION_INTERNVL_MODEL_PATH="/home/jovyan/nfs_share/models/InternVL3-8B"
export VISION_LLAMA_MODEL_PATH="/home/jovyan/nfs_share/models/Llama-3.2-11B-Vision"

# Local development (Hugging Face)
export VISION_INTERNVL_MODEL_PATH="OpenGVLab/InternVL2-8B"
export VISION_LLAMA_MODEL_PATH="meta-llama/Llama-3.2-11B-Vision-Instruct"
```

## 🚀 Usage

### Basic Evaluation

```bash
# Simple evaluation script
python evaluate_models.py

# Using the CLI module
python -m vision_processor.cli.evaluation_cli compare evaluation_ground_truth.csv

# With custom options
python -m vision_processor.cli.evaluation_cli compare \
    evaluation_ground_truth.csv \
    --images-dir datasets \
    --models internvl3,llama32_vision \
    --output-dir results

# Results saved to evaluation_results/ directory
```

### Advanced Usage

```bash
# Benchmark single model performance
python -m vision_processor.cli.evaluation_cli benchmark datasets --model internvl3

# Validate ground truth data
python -m vision_processor.cli.evaluation_cli validate-ground-truth evaluation_ground_truth.csv

# Using as Python module
python -c "
from vision_processor.evaluation import ExtractionEvaluator
evaluator = ExtractionEvaluator('evaluation_ground_truth.csv', 'datasets')
results = evaluator.compare_models(['internvl3', 'llama32_vision'])
evaluator.generate_report(results)
"
```

### Output Files

- `comparison_results.json` - Complete evaluation data
- `evaluation_report.md` - Human-readable markdown report
- `{model}_results.json` - Per-model detailed results

### Console Output Example

```
🏁 Starting Model Comparison Evaluation
📝 Using model-specific prompts from prompts.yaml

🔬 Evaluating INTERNVL3 model...
✅ internvl3: 45.6% accuracy

🔬 Evaluating LLAMA32_VISION model...  
✅ llama32_vision: 19.1% accuracy

📊 EVALUATION REPORT
============================================================
│ INTERNVL3      │    100.0%    │    45.6%     │     5.2s      │     15.4     │
│ LLAMA32_VISION │    100.0%    │    19.1%     │     13.2s     │     4.2      │
```

## 📊 Detailed Results Analysis

### InternVL3-8B Strengths
- **Perfect extraction**: DOCUMENT_TYPE (100%), STORE (100%), TOTAL (100%)
- **Reliable metadata**: PAYMENT_METHOD (80%), TIME (80%)
- **Consistent performance**: High field count across all receipts
- **Speed**: 2.5x faster than Llama

### Llama-3.2-Vision Issues
- **Repetition truncation**: Ultra-aggressive cleaning reduces output to fragments
- **Format problems**: Outputs unstructured text instead of KEY: VALUE format
- **Missing fields**: Fails to extract many required fields
- **Safety mode**: Complex prompts trigger content restrictions

### Common Challenges
- **DATE formatting**: Both models struggle with DD/MM/YYYY vs DD-MM-YYYY
- **Ground truth mismatches**: Models read actual receipt data that differs from CSV
- **List parsing**: ITEMS, QUANTITIES, PRICES require consistent pipe separation

## 🔍 Ground Truth Data

### Test Images Coverage

| Image | Store | Document Type | Key Features |
|-------|-------|---------------|--------------|
| image14.png | SPOTLIGHT | TAX INVOICE | Complete fields, card payment |
| image65.png | THE GOOD GUYS | RECEIPT | Missing ABN, EFTPOS payment |
| image71.png | DAN MURPHY'S | TAX INVOICE | Many items, complex layout |
| image74.png | DAVID JONES | RECEIPT | Minimal fields, PayPal payment |
| image76.png | BCFAUSTRALIA | TAX INVOICE | Full metadata, address/phone |

### Adding New Test Images

1. Add image to `datasets/` directory
2. Update `evaluation_ground_truth.csv`:

```csv
image_new.png,DD/MM/YYYY,STORE NAME,ABN,GST,TOTAL,SUBTOTAL,item1|item2,1|1,price1|price2,receipt_num,PAYMENT,TYPE,address,phone,time,card,auth,status
```

3. Run evaluation to include new image

## 🛠 Troubleshooting

### Common Issues

#### Model Loading Errors
```
Failed to load model: Incorrect path_or_model_id
```
**Solution**: Verify environment variables and model paths

#### Missing Ground Truth
```
❌ Ground truth file not found: evaluation_ground_truth.csv
```
**Solution**: Ensure CSV file exists in project root

#### Conda Environment Issues
```
ModuleNotFoundError: No module named 'vision_processor'
```
**Solution**: Activate conda environment and verify installation

#### Date Accuracy Always 0%
```
Error calculating accuracy for DATE: bad character range
```
**Solution**: Fixed in latest version - regex pattern corrected

### Debug Mode

Enable debug output for detailed troubleshooting:

```python
# Uncomment debug lines in evaluate_extraction_performance.py
self.console.print(f"🔍 DEBUG: Extracted {len(extracted_data)} fields")
```

## 🔄 System Integration

### Using with Existing CLI

The evaluation system integrates with the working CLI tools:

```bash
# Single image extraction (for comparison)
python -m vision_processor.cli.simple_extract_cli datasets/image14.png --model internvl3
python -m vision_processor.cli.simple_extract_cli datasets/image14.png --model llama32_vision

# Model comparison
python -m vision_processor.cli.simple_extract_cli datasets/image14.png --compare
```

### Prompt Configuration

The system uses model-specific prompts from `prompts.yaml`:

```yaml
model_prompts:
  internvl3:
    prompt: |
      <|image|>Extract information from this Australian business document...
      
  llama32_vision:
    prompt: |
      <|image|>Extract data from this receipt in KEY-VALUE format...
```

## 🚀 Future Improvements

### Planned Enhancements

1. **Extended Ground Truth**: Add more diverse receipt types and edge cases
2. **Confidence Scoring**: Include model confidence in accuracy calculations  
3. **Error Analysis**: Detailed categorization of extraction failures
4. **Batch Processing**: Support for large-scale evaluation datasets
5. **OCR Comparison**: Compare against traditional OCR + NLP approaches

### Research Opportunities

1. **Prompt Optimization**: A/B testing of different prompt strategies
2. **Multi-modal Analysis**: Combine text and visual feature analysis
3. **Domain Adaptation**: Fine-tuning for Australian business documents
4. **Real-time Evaluation**: Stream processing performance metrics

## 📈 Performance Baselines

### Hardware Requirements

- **Development**: 2x H200 GPUs (multi-GPU training/testing)
- **Production**: Single V100 16GB (optimized inference)
- **Memory**: 64GB RAM recommended for large batch processing

### Processing Benchmarks

| Model | GPU Memory | Processing Time/Image | Throughput |
|-------|------------|----------------------|------------|
| InternVL3-8B | ~8GB VRAM | 5.2s | 11.5 images/min |
| Llama-3.2-Vision | ~12GB VRAM | 13.2s | 4.5 images/min |

## 🏅 Conclusion

The evaluation system demonstrates **InternVL3-8B's clear superiority** for Australian business document processing:

- **Higher accuracy** across all field types
- **Faster processing** for production workflows  
- **More reliable extraction** with fewer edge case failures
- **Better format compliance** for structured data output

This comprehensive evaluation framework provides the foundation for production model selection and ongoing performance monitoring in document processing workflows.

---

**Last Updated**: 2025-07-14  
**Version**: 1.0  
**Tested Models**: InternVL3-8B, Llama-3.2-11B-Vision  
**Test Dataset**: 5 Australian business receipts