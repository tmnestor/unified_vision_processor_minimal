# Data Storage Architecture for Vision Model Evaluation

## Overview

This document describes how extracted keys and values are currently stored in the unified vision processor system and provides recommendations for enhancing the architecture to support extensive downstream evaluations of model information extraction capabilities.

## Current Storage Architecture

### 1. Individual Document Level Storage

Each document's analysis results are stored as a **Python dictionary** with the following structure:

```python
document_result = {
    "img_name": "image01.png",
    "response": "DATE: 08/07/2025\nSUPPLIER: David Miller\n...",  # Full cleaned text
    "is_structured": True,                    # Boolean - contains recognizable structure
    "extraction_score": 15,                  # Integer - number of fields detected
    "successful": True,                      # Boolean - meets minimum field criteria
    "inference_time": 2.3,                  # Float - seconds taken for inference
    "doc_type": "TAX_INVOICE",              # String - classified document type
    
    # Dynamic field detection flags (added for each detected field)
    "has_date": True,
    "has_supplier": True, 
    "has_abn": False,
    "has_gst": True,
    "has_total": True,
    # ... additional has_* fields as detected
}
```

**Key Characteristics:**
- **Dynamic field detection**: Fields are detected automatically from model responses
- **Boolean flags**: Only tracks presence/absence, not actual values
- **Flexible structure**: Can accommodate varying field types across document types

### 2. Model-Level Aggregation

Results across all documents for each model are stored in a **nested dictionary structure**:

```python
extraction_results = {
    "llama": {
        "documents": [doc1_dict, doc2_dict, doc3_dict, ...],  # List of document results
        "successful": 15,                                     # Count of successful extractions
        "total_time": 245.3,                                 # Total inference time (seconds)
        "avg_time": 2.1                                      # Average time per document
    },
    "internvl": {
        "documents": [doc1_dict, doc2_dict, doc3_dict, ...],
        "successful": 18,
        "total_time": 198.7,
        "avg_time": 1.8
    }
    # Additional models...
}
```

**Benefits:**
- **Model comparison**: Easy to compare performance across models
- **Aggregated metrics**: Summary statistics readily available
- **Scalable**: Can accommodate any number of models

### 3. Analysis-Ready DataFrame

For comprehensive analysis, the nested dictionaries are flattened into a **pandas DataFrame**:

```python
# Example DataFrame structure
df_columns = [
    'model',           # 'LLAMA' or 'INTERNVL'
    'image',           # 'image01.png'
    'doc_type',        # 'TAX_INVOICE', 'BANK_STATEMENT', etc.
    'inference_time',  # 2.3
    'is_structured',   # True/False
    'extraction_score',# 15
    'successful',      # True/False
    'has_date',        # True/False
    'has_supplier',    # True/False
    'has_abn',         # True/False
    # ... all detected fields as boolean columns
]
```

**Advantages:**
- **Statistical analysis**: Compatible with sklearn, seaborn, matplotlib
- **Cross-model comparison**: Easy filtering and grouping
- **Export capability**: Direct CSV export for external analysis

## Current Export Capabilities

The system currently exports three types of analysis files:

### 1. Detailed Results (`detailed_results.csv`)
- **Content**: Full DataFrame with all documents and detected fields
- **Format**: CSV with boolean field detection flags
- **Use case**: Comprehensive field-level analysis across models

### 2. F1 Scores (`f1_scores.json`)
- **Content**: Precision, recall, and F1 scores for each field and model
- **Format**: Nested JSON structure
- **Use case**: Statistical performance comparison

```json
{
    "LLAMA": {
        "has_date": {"f1": 0.95, "precision": 0.92, "recall": 0.98},
        "has_abn": {"f1": 0.23, "precision": 0.18, "recall": 0.31}
    },
    "INTERNVL": {
        "has_date": {"f1": 0.88, "precision": 0.85, "recall": 0.91},
        "has_abn": {"f1": 0.34, "precision": 0.29, "recall": 0.42}
    }
}
```

### 3. Summary Results (`summary_results.json`)
- **Content**: High-level performance metrics per model
- **Format**: JSON with aggregated statistics
- **Use case**: Executive summary and model selection

```json
{
    "models_tested": ["llama", "internvl"],
    "total_documents": 114,
    "overall_performance": {
        "LLAMA": {
            "success_rate": 0.72,
            "avg_inference_time": 2.1,
            "structured_output_rate": 0.89,
            "abn_detection_rate": 0.23
        }
    }
}
```

## Current Limitations for Downstream Evaluation

### ❌ Missing Capabilities

1. **No Raw Extracted Values**
   - Only boolean `has_*` flags stored
   - Cannot analyze accuracy of extracted content
   - Cannot perform value-level error analysis

2. **No Ground Truth Integration**
   - No systematic storage of known correct values
   - Cannot calculate value-level accuracy metrics
   - Cannot perform automated quality assessment

3. **No Field-Level Value Analysis**
   - Cannot analyze common extraction errors
   - Cannot identify systematic biases in value extraction
   - Cannot perform semantic similarity analysis

4. **No Confidence Scoring**
   - No per-field confidence metrics
   - Cannot prioritize high-confidence extractions
   - Cannot implement confidence-based filtering

5. **Limited Error Analysis**
   - Cannot categorize types of extraction errors
   - Cannot identify problematic document patterns
   - Cannot perform failure mode analysis

## Recommended Enhancements for Downstream Evaluation

### Enhanced Document Result Structure

```python
enhanced_document_result = {
    # Current fields (maintained for backward compatibility)
    "img_name": "image01.png",
    "model": "llama",
    "inference_time": 2.3,
    "is_structured": True,
    "extraction_score": 15,
    "successful": True,
    "doc_type": "TAX_INVOICE",
    
    # ENHANCEMENT 1: Field detection list
    "fields_detected": ["DATE", "SUPPLIER", "TOTAL", "ABN", "GST"],
    "fields_missing": ["INVOICE_NUMBER", "PAYMENT_METHOD"],
    
    # ENHANCEMENT 2: Actual extracted values
    "extracted_values": {
        "DATE": "08/07/2025",
        "SUPPLIER": "David Miller",
        "TOTAL": "$28.27",
        "ABN": None,  # Detected as field but no value extracted
        "GST": "$2.57"
    },
    
    # ENHANCEMENT 3: Ground truth for comparison
    "ground_truth": {
        "DATE": "08/07/2025",
        "SUPPLIER": "David Miller", 
        "TOTAL": "$28.27",
        "ABN": "12345678901",  # Known correct value
        "GST": "$2.57",
        "INVOICE_NUMBER": "INV-65301",
        "PAYMENT_METHOD": "VISA"
    },
    
    # ENHANCEMENT 4: Field-level confidence scores
    "field_confidence": {
        "DATE": 0.95,
        "SUPPLIER": 0.87,
        "TOTAL": 0.92,
        "GST": 0.89
    },
    
    # ENHANCEMENT 5: Value-level accuracy
    "value_accuracy": {
        "DATE": {"exact_match": True, "semantic_similarity": 1.0},
        "SUPPLIER": {"exact_match": True, "semantic_similarity": 1.0},
        "TOTAL": {"exact_match": True, "semantic_similarity": 1.0},
        "ABN": {"exact_match": False, "semantic_similarity": 0.0}
    },
    
    # ENHANCEMENT 6: Error classification
    "extraction_errors": {
        "missing_fields": ["INVOICE_NUMBER", "PAYMENT_METHOD"],
        "incorrect_values": [],
        "format_errors": [],
        "hallucinated_fields": []
    },
    
    # ENHANCEMENT 7: Raw model response (for debugging)
    "raw_response": "INVOICE_DATE: 08/07/2025\nDUE_DATE: 18/08/2025\n...",
    "cleaned_response": "DATE: 08/07/2025\nSUPPLIER: David Miller\n..."
}
```

### Enhanced Analysis Capabilities

With the enhanced structure, the following downstream analyses become possible:

#### 1. Value-Level Accuracy Analysis
```python
# Calculate exact match accuracy per field
field_accuracy = df.groupby(['model', 'field'])['exact_match'].mean()

# Analyze semantic similarity distributions
similarity_stats = df.groupby('model')['semantic_similarity'].describe()
```

#### 2. Error Pattern Analysis
```python
# Identify most commonly missed fields
common_missing = df['missing_fields'].explode().value_counts()

# Analyze extraction error patterns by document type
error_patterns = df.groupby(['model', 'doc_type'])['extraction_errors'].apply(analyze_error_types)
```

#### 3. Confidence-Based Performance
```python
# Performance by confidence threshold
high_conf_accuracy = df[df['field_confidence'] > 0.8]['exact_match'].mean()

# Confidence calibration analysis
calibration_analysis = analyze_confidence_calibration(df['field_confidence'], df['exact_match'])
```

#### 4. Model Comparison Metrics
```python
# Comprehensive model comparison
comparison_metrics = {
    'field_detection_f1': calculate_field_f1(df),
    'value_extraction_accuracy': calculate_value_accuracy(df),
    'inference_speed': df.groupby('model')['inference_time'].mean(),
    'confidence_calibration': calculate_calibration(df)
}
```

### Implementation Roadmap

#### Phase 1: Value Storage Enhancement
1. Modify `ConfigurableKeyValueExtractionAnalyzer` to store actual extracted values
2. Update DataFrame creation to include value columns
3. Enhance CSV export with value data

#### Phase 2: Ground Truth Integration
1. Create ground truth annotation system
2. Implement ground truth loading and validation
3. Add value comparison logic

#### Phase 3: Advanced Metrics
1. Implement confidence scoring system
2. Add semantic similarity calculations
3. Create error classification framework

#### Phase 4: Enhanced Analysis Tools
1. Build value-level analysis functions
2. Create advanced visualization tools
3. Implement comprehensive reporting system

## Current File Locations

- **Main analysis logic**: `model_comparison.py` (lines 410-502)
- **DataFrame creation**: `model_comparison.py` (lines 720-747)
- **Export functionality**: `model_comparison.py` (lines 907-943)
- **Configuration**: `model_comparison.yaml`

## Conclusion

The current storage architecture provides a solid foundation for field detection analysis but requires enhancements to support comprehensive downstream evaluation of information extraction capabilities. The recommended enhancements would enable:

- **Value-level accuracy assessment**
- **Confidence-based performance analysis** 
- **Detailed error pattern identification**
- **Semantic similarity evaluation**
- **Ground truth comparison capabilities**

These improvements would transform the system from a field detection analyzer into a comprehensive information extraction evaluation platform suitable for research and production deployment decisions.