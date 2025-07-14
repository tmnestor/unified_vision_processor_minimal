# Vision Model Key-Value Extraction Evaluation System

## Overview

This document describes the design, implementation, and usage of a comprehensive evaluation system for comparing the key-value extraction performance of vision language models, specifically designed for evaluating **InternVL3** and **Llama-3.2-Vision** models on Australian business documents.

## System Architecture

### 🎯 **Design Goals**

1. **Scientific Rigor**: Objective, repeatable evaluation methodology
2. **Comprehensive Metrics**: Multi-dimensional performance analysis
3. **Practical Applicability**: Real-world document processing scenarios
4. **Ease of Use**: Simple workflow for adding test data and running evaluations
5. **Actionable Insights**: Clear recommendations for model selection

### 🏗️ **Core Components**

```
evaluation_system/
├── evaluation_ground_truth.csv           # Ground truth data repository
├── evaluate_extraction_performance.py    # Main evaluation engine
├── add_ground_truth.py                  # Interactive data entry tool
├── evaluation_results/                  # Generated evaluation outputs
│   ├── internvl3_results.json          # Detailed InternVL results
│   ├── llama32_vision_results.json     # Detailed Llama results
│   ├── comparison_results.json         # Side-by-side comparison
│   └── evaluation_report.md            # Comprehensive report
└── datasets/                           # Test images directory
    ├── image14.png
    ├── image15.png
    └── ...
```

## Data Structure Design

### 📊 **Ground Truth CSV Schema**

The evaluation system uses a standardized CSV format to store ground truth data:

```csv
image_file,DATE,STORE,ABN,GST,TOTAL,SUBTOTAL,ITEMS,QUANTITIES,PRICES,RECEIPT_NUMBER,PAYMENT_METHOD,DOCUMENT_TYPE,ADDRESS,PHONE,TIME,CARD_NUMBER,AUTH_CODE,STATUS
```

#### **Field Categories:**

1. **Core Financial Data**: `DATE`, `STORE`, `GST`, `TOTAL`, `SUBTOTAL`
2. **Australian Business Info**: `ABN`, `ADDRESS`, `PHONE`
3. **Transaction Details**: `RECEIPT_NUMBER`, `PAYMENT_METHOD`, `TIME`, `CARD_NUMBER`, `AUTH_CODE`, `STATUS`
4. **Item Information**: `ITEMS`, `QUANTITIES`, `PRICES` (pipe-separated format: `item1|item2|item3`)
5. **Document Metadata**: `DOCUMENT_TYPE`

#### **Data Format Conventions:**

- **Dates**: DD/MM/YYYY format (Australian standard)
- **Currency**: Numeric values only (e.g., `22.45` not `$22.45`)
- **Lists**: Pipe-separated (`|`) for multiple items
- **Missing Data**: Empty strings for unavailable fields
- **Text**: Exact case-sensitive matching where applicable

### 🔍 **Example Ground Truth Entry:**

```csv
image14.png,11-07-2022,SPOTLIGHT,10 306 488 435,2.04,22.45,20.41,"Apples (kg)|Tea Bags (box)|Free Range Eggs (d)|Dishwashing Liquid|Bananas","1|1|1|1|1","3.96|4.53|4.71|3.79|3.42",503152,VISA,TAX INVOICE,858 Park Rd Brisbane NT 817,(07) 3161 9999,3:53 PM,XXXX-XXXX-XXXX-4978,206031,APPROVED
```

## Evaluation Methodology

### 🧠 **Extraction Strategy**

The system uses **regex-based key-value extraction** from model responses:

```python
key_patterns = {
    "DATE": r"DATE:\s*([^\n\r|]+)",
    "STORE": r"STORE:\s*([^\n\r|]+)",
    "ABN": r"ABN:\s*([^\n\r|]+)",
    "GST": r"GST:\s*\$?([0-9.]+)",
    "TOTAL": r"TOTAL:\s*\$?([0-9.]+)",
    # ... additional patterns
}
```

### 📏 **Accuracy Calculation Methods**

#### **1. Numeric Fields (GST, TOTAL, SUBTOTAL)**
```python
def numeric_accuracy(extracted, ground_truth):
    tolerance = 0.01  # $0.01 tolerance
    return 1.0 if abs(float(extracted) - float(ground_truth)) < tolerance else 0.0
```

#### **2. List Fields (ITEMS, QUANTITIES, PRICES)**
```python
def list_accuracy(extracted, ground_truth):
    ext_items = extracted.split("|")
    gt_items = ground_truth.split("|")
    
    if len(ext_items) != len(gt_items):
        return 0.0
    
    matches = sum(1 for e, g in zip(ext_items, gt_items) if e.lower() == g.lower())
    return matches / len(gt_items)
```

#### **3. Text Fields (STORE, ABN, etc.)**
```python
def text_accuracy(extracted, ground_truth):
    if extracted.lower() == ground_truth.lower():
        return 1.0  # Perfect match
    elif extracted.lower() in ground_truth.lower() or ground_truth.lower() in extracted.lower():
        return 0.8  # Partial match
    else:
        return 0.0  # No match
```

#### **4. Date Fields**
```python
def date_accuracy(extracted, ground_truth):
    # Extract numeric components only
    ext_date = re.sub(r"[^\d-/]", "", extracted)
    gt_date = re.sub(r"[^\d-/]", "", ground_truth)
    return 1.0 if ext_date == gt_date else 0.0
```

### 📊 **Performance Metrics**

#### **Individual Model Metrics:**
- **Overall Accuracy**: Average accuracy across all fields
- **Field-wise Accuracy**: Accuracy for each specific field type
- **Success Rate**: Percentage of images processed without errors
- **Processing Speed**: Average time per image (seconds)
- **Extraction Coverage**: Average number of fields successfully extracted
- **Response Quality**: Response length and confidence scores

#### **Comparative Metrics:**
- **Head-to-Head Accuracy**: Direct model comparison
- **Speed Comparison**: Processing time ratios
- **Reliability Analysis**: Error rates and failure modes
- **Field-specific Winners**: Best model for each field type

## Implementation Details

### 🔧 **Core Classes**

#### **ExtractionEvaluator**
```python
class ExtractionEvaluator:
    def __init__(self, ground_truth_csv, images_dir, output_dir):
        # Initialize evaluation environment
        
    def evaluate_model(self, model_type, prompt, test_images):
        # Single model evaluation
        
    def compare_models(self, prompt, models, test_images):
        # Multi-model comparison
        
    def generate_report(self, comparison_results):
        # Comprehensive reporting
```

#### **Key Methods:**

1. **`_extract_key_values(response_text)`**: Uses regex patterns to extract structured data
2. **`_calculate_field_accuracy(extracted, ground_truth, field_type)`**: Type-specific accuracy calculation
3. **`_evaluate_single_image(model, image_file, prompt)`**: Single image processing and evaluation
4. **`_save_markdown_report(comparison_results, output_file)`**: Detailed report generation

### 🎛️ **Configuration Integration**

The system integrates with the existing `vision_processor` configuration:

```python
# Model initialization
config = SimpleConfig()
config.model_type = model_type  # "internvl3" or "llama32_vision"
model = ModelFactory.create_model(config)

# Automatic repetition control for Llama-3.2-Vision
# UltraAggressiveRepetitionController applied automatically
```

### 📝 **Standardized Prompt**

The evaluation uses a standardized KEY-VALUE extraction prompt:

```
<|image|>Extract information from this Australian business document and return in KEY-VALUE format.

CRITICAL: Use EXACT KEY-VALUE format below - NO JSON, NO other formats.

DATE: [document date in DD/MM/YYYY format]
STORE: [business name in capitals]
ABN: [Australian Business Number if visible]
GST: [GST amount]
TOTAL: [total amount including GST]
SUBTOTAL: [subtotal before GST if visible]
ITEMS: [item1 | item2 | item3]
QUANTITIES: [qty1 | qty2 | qty3]
PRICES: [price1 | price2 | price3]
RECEIPT_NUMBER: [receipt/invoice number if visible]
PAYMENT_METHOD: [payment method if visible]
TIME: [time if visible]

Return ONLY the key-value pairs above. Skip any keys where information is not available.
```

## Usage Guide

### 🚀 **Quick Start**

#### **Step 1: Prepare Environment**
```bash
export KMP_DUPLICATE_LIB_OK=TRUE
source /opt/homebrew/Caskroom/miniforge/base/bin/activate unified_vision_processor
```

#### **Step 2: Add Ground Truth Data**
```bash
python add_ground_truth.py
```

**Interactive workflow:**
1. Enter image filename (e.g., `image15.png`)
2. Fill in fields with actual values from the image
3. Use pipe separators for multiple items: `item1|item2|item3`
4. Skip optional fields by pressing Enter
5. Save when complete

#### **Step 3: Run Evaluation**
```bash
python evaluate_extraction_performance.py
```

**Automated process:**
1. Loads ground truth data
2. Initializes both InternVL3 and Llama-3.2-Vision models
3. Processes all test images
4. Calculates comprehensive metrics
5. Generates detailed reports

### 📊 **Output Analysis**

#### **Console Output Example:**
```
🏁 Starting Model Comparison Evaluation
📝 Prompt: <|image|>Extract information from this Australian business document...

🔬 Evaluating INTERNVL3 model...
✅ internvl3: 87.3% accuracy

🔬 Evaluating LLAMA32_VISION model...
🧹 Ultra-aggressive cleaning: 2883 → 381 chars (86.8% reduction)
✅ llama32_vision: 92.1% accuracy

📊 EVALUATION REPORT
============================================================

Model Performance Overview
┏━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃ Model         ┃ Success Rate┃ Avg Accuracy┃ Avg Speed (s)┃ Fields/Image┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
│ INTERNVL3     │    100.0%   │    87.3%    │     8.2s     │     9.4     │
│ LLAMA32_VISION│    100.0%   │    92.1%    │    18.3s     │    10.8     │
└───────────────┴─────────────┴─────────────┴──────────────┴─────────────┘

🏆 WINNERS:
🎯 Best Accuracy: LLAMA32_VISION (92.1%)
⚡ Fastest: INTERNVL3 (8.2s)
```

#### **Generated Files:**

1. **`evaluation_results/internvl3_results.json`**
   - Detailed per-image results
   - Field-wise accuracy breakdown
   - Processing times and metadata

2. **`evaluation_results/llama32_vision_results.json`**
   - Complete Llama evaluation data
   - Repetition control effectiveness metrics
   - Error analysis

3. **`evaluation_results/comparison_results.json`**
   - Side-by-side model comparison
   - Statistical analysis
   - Performance deltas

4. **`evaluation_results/evaluation_report.md`**
   - Human-readable comprehensive report
   - Recommendations and insights
   - Field-wise performance analysis

### 🎯 **Advanced Usage**

#### **Custom Model Testing**
```python
evaluator = ExtractionEvaluator("evaluation_ground_truth.csv", "datasets")

# Test specific images
test_images = ["image14.png", "image15.png"]
results = evaluator.compare_models(prompt, models=["internvl3"], test_images=test_images)
```

#### **Custom Prompts**
```python
# Test different prompt strategies
simple_prompt = "<|image|>Extract the store name and total amount."
detailed_prompt = "<|image|>Provide detailed extraction with confidence scores."

results1 = evaluator.compare_models(simple_prompt)
results2 = evaluator.compare_models(detailed_prompt)
```

#### **Field-Specific Analysis**
```python
# Focus on specific fields
financial_fields = ["GST", "TOTAL", "SUBTOTAL"]
for field in financial_fields:
    accuracy = results["internvl3"]["field_wise_accuracy"][field]
    print(f"{field}: {accuracy:.1%}")
```

## Design Rationale

### 🎯 **Why Regex-Based Extraction?**

1. **Model-Agnostic**: Works with any model output format
2. **Robust**: Handles variations in spacing and formatting
3. **Debuggable**: Clear patterns for troubleshooting
4. **Fast**: Efficient processing for batch evaluation
5. **Flexible**: Easy to add new field types

### 📊 **Why Field-Specific Accuracy?**

1. **Granular Insights**: Identify model strengths/weaknesses
2. **Domain-Specific**: Different fields have different criticality
3. **Optimization Target**: Focus improvement efforts
4. **Production Guidance**: Select models based on use case priorities

### 🔄 **Why Comparative Evaluation?**

1. **Objective Analysis**: Remove subjective bias
2. **Trade-off Identification**: Speed vs accuracy analysis
3. **Context-Aware**: Different models excel in different scenarios
4. **Future-Proof**: Easy to add new models for comparison

## Expected Outcomes

### 📈 **Performance Insights**

Based on the design and initial testing, expected findings:

#### **Llama-3.2-Vision Strengths:**
- ✅ Higher extraction accuracy (especially with repetition control)
- ✅ Better handling of complex layouts
- ✅ Built-in OCR capabilities
- ❌ Slower processing speed
- ❌ Requires repetition control

#### **InternVL3 Strengths:**
- ✅ Faster processing (2-3x speed advantage)
- ✅ More stable output (no repetition issues)
- ✅ Better multi-GPU scaling
- ❌ Lower extraction accuracy for complex documents
- ❌ No specialized OCR integration

### 🎯 **Decision Framework**

The evaluation system provides data for informed model selection:

```
IF accuracy_priority > speed_priority:
    CHOOSE Llama-3.2-Vision (with repetition control)
ELIF speed_priority > accuracy_priority:
    CHOOSE InternVL3
ELIF mixed_requirements:
    ANALYZE field-wise performance for specific use case
```

## Extensibility

### 🔧 **Adding New Models**

```python
# Add to compare_models call
models = ["internvl3", "llama32_vision", "new_model_type"]

# Implement model in ModelFactory
class NewVisionModel(BaseVisionModel):
    # Implementation details
```

### 📊 **Adding New Metrics**

```python
def calculate_confidence_accuracy(extracted_conf, ground_truth_conf):
    # Custom confidence-based accuracy
    pass

# Add to _evaluate_single_image
result["confidence_accuracy"] = calculate_confidence_accuracy(...)
```

### 🎯 **Adding New Field Types**

```python
# Add to key_patterns
"NEW_FIELD": r"NEW_FIELD:\s*([^\n\r|]+)",

# Add to _calculate_field_accuracy
elif field_type == "NEW_FIELD":
    return custom_field_accuracy(extracted, ground_truth)
```

## Conclusion

This evaluation system provides a **comprehensive, scientific, and practical framework** for comparing vision language model performance on key-value extraction tasks. The design prioritizes:

1. **Accuracy and Reliability**: Multiple validation methods and error handling
2. **Ease of Use**: Simple workflow for adding data and running evaluations  
3. **Actionable Insights**: Clear recommendations for model selection
4. **Scalability**: Easy to extend with new models, metrics, and field types
5. **Production Readiness**: Integration with existing vision_processor architecture

The system enables **data-driven decision making** for vision model selection in production environments, specifically optimized for Australian business document processing workflows.

---

**Next Steps:**
1. Collect ground truth data for 10-20 diverse test images
2. Run comprehensive evaluation on both models
3. Analyze results to determine optimal model for production use
4. Fine-tune model configurations based on evaluation insights