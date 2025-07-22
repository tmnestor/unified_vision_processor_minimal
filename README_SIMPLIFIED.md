# Simplified Vision Processor

## Overview

This is the **simplified version** of the Unified Vision Document Processing system, refactored to remove unnecessary complexity while maintaining all core functionality.

## Key Simplifications

### What Was Removed ❌
- 55-field production schema → 12 simple core fields
- 7-step pipeline orchestrator → Primary + AWK fallback
- Complex F1/precision/recall metrics → Simple success rate
- Aggressive repetition control → Basic duplicate removal

### What Was Fixed ✅
- `core_fields_found` now shows actual values (was always 0)
- Field matching uses actual extracted names (DATE, TOTAL) not schema names
- Cleaner, more maintainable code

## Quick Start

### 1. Setup Environment
```bash
conda activate unified_vision_processor
export KMP_DUPLICATE_LIB_OK=TRUE  # Fix OpenMP issues
```

### 2. Run Model Comparison
```bash
python simple_comparison.py compare \
  --datasets-path ./datasets \
  --output-dir ./results \
  --models llama,internvl
```

### 3. View Results
- `results/simple_results.csv` - CSV with all metrics
- `results/simple_comparison_results.json` - Full JSON results

## Core Components

### Extraction Pipeline (`simple_pipeline.py`)
1. **Primary extraction** - Extract fields from model response
2. **AWK fallback** - Process markdown content if primary fails

### Core Fields (12 fields)
```python
CORE_FIELDS = {
    "DATE", "TOTAL", "GST", "ABN", "SUPPLIER_NAME",
    "INVOICE_NUMBER", "AMOUNT", "DESCRIPTION",
    "BSB", "ACCOUNT_NUMBER", "BUSINESS_NAME", "RECEIPT_NUMBER"
}
```

### Simple Metrics
- **Success rate** - % of successful extractions
- **Fields per image** - Average fields extracted
- **Core fields per image** - Average core fields found
- **Processing time** - Average inference time

## Results Example

```csv
model,image,field_count,core_fields_found,successful
llama,image01.png,17,5,True
llama,image02.png,25,6,True
internvl,image01.png,19,6,True
internvl,image02.png,23,6,True
```

## Architecture

```
vision_processor/
├── extraction/
│   ├── patterns.py              # Centralized regex patterns
│   ├── base_extractor.py        # Shared extraction logic
│   ├── simple_pipeline.py       # Primary + AWK pipeline
│   └── dynamic_extractor.py     # Dynamic field detection
├── analysis/
│   └── simple_metrics.py        # Simple success metrics
├── utils/
│   └── simple_repetition_control.py  # Basic cleaning
└── comparison/
    └── simple_runner.py         # Model comparison runner
```

## Legacy Files

The following files are kept for reference but not used:
- `model_comparison_legacy.py` - Original complex comparison
- `model_comparison_working_backup_monday.py` - Working backup

## Key Differences from Original

| Original | Simplified |
|----------|------------|
| 55 predefined fields | 12 core fields |
| Complex schema mapping | Direct field names |
| 7-step pipeline | 2-step pipeline |
| F1/precision/recall | Success rate |
| Multiple analyzers | Single metrics calculator |
| `core_fields_found=0` | Actual core field counts |

## Benefits

1. **Simpler** - Much easier to understand and maintain
2. **Accurate** - Core fields are counted correctly
3. **Faster** - Less complex processing
4. **Cleaner** - Removed unnecessary abstraction layers
5. **Working** - Fixed the core issue with field counting

The system now does exactly what it needs to do, nothing more, nothing less.
