# Unified Vision Processor

A flexible vision document processing system with model-agnostic architecture supporting both InternVL3 and Llama-3.2-Vision models. Features dynamic field extraction, YAML-driven configuration, and comprehensive evaluation framework.

## 🎯 Overview

This system provides a simplified, production-ready solution for extracting structured data from document images using state-of-the-art vision language models.

### Key Features

- **Dynamic Field Extraction**: Automatically detects and extracts fields without predefined schemas
- **Model Agnostic**: Supports InternVL3-8B and Llama-3.2-Vision with identical interfaces
- **YAML Configuration**: All settings and prompts managed through YAML files
- **Memory Optimized**: 8-bit quantization support for V100 16GB GPUs
- **Comprehensive Evaluation**: Built-in model comparison and benchmarking tools

## 🏗️ Architecture

The system implements a streamlined extraction pipeline:

```
Document Image → Unified Prompt → Vision Model → Key-Value Extraction → Validation → Results
```

### Core Components

- **ModelFactory**: Dynamic model instantiation based on configuration
- **DynamicExtractor**: Flexible field detection and extraction
- **SimpleExtractionManager**: Orchestrates the extraction process
- **UniversalKeyValueParser**: Parses structured data from model outputs
- **ComparisonRunner**: Evaluates and compares model performance

## 📦 Project Structure

```
unified_vision_processor_minimal/
├── model_comparison.yaml          # Main configuration file
├── model_comparison.py            # Primary comparison script
├── environment.yml                # Conda environment specification
├── requirements.txt               # Python dependencies
├── .env                          # Environment variables (optional)
├── 
├── datasets/                      # Test images (25 images)
│   ├── image01.png through image25.png
├── 
├── vision_processor/              # Main package
│   ├── config/                   # Configuration management
│   │   ├── model_factory.py     # Model instantiation
│   │   ├── model_registry.py    # Model registration
│   │   ├── prompts.yaml         # Extraction prompts and schemas
│   │   └── simple_config.py     # Configuration classes
│   │
│   ├── models/                   # Model implementations
│   │   ├── base_model.py        # Abstract interface
│   │   ├── internvl_model.py    # InternVL3 wrapper
│   │   ├── llama_model.py       # Llama-3.2-Vision wrapper
│   │   └── model_utils.py       # Shared utilities
│   │
│   ├── extraction/               # Extraction logic
│   │   ├── dynamic_extractor.py # Dynamic field detection
│   │   ├── simple_pipeline.py   # Processing pipeline
│   │   ├── patterns.py          # Regex patterns
│   │   └── universal_key_value_parser.py
│   │
│   ├── comparison/               # Model comparison
│   │   ├── comparison_runner.py # Cross-model evaluation
│   │   └── model_validator.py   # Validation logic
│   │
│   ├── analysis/                 # Metrics and reporting
│   │   └── simple_metrics.py    # Performance analysis
│   │
│   ├── cli/                      # Command-line interfaces
│   │   ├── simple_extract_cli.py
│   │   └── evaluation_cli.py
│   │
│   └── utils/                    # Utilities
│       ├── memory_monitor.py    # GPU memory tracking
│       └── repetition_control.py # Output deduplication
│
├── backup/                       # Legacy code (not in git)
└── docs/                         # Documentation
```

## 🚀 Quick Start

### Prerequisites

1. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate unified_vision_processor
```

2. Configure model paths in `model_comparison.yaml`:
```yaml
model_paths:
  llama: "/path/to/Llama-3.2-11B-Vision-Instruct"
  internvl: "/path/to/InternVL3-8B"
```

### Basic Usage

```bash
# Run model comparison on test dataset
python model_comparison.py

# Extract from single document
python -m vision_processor.cli.simple_extract_cli extract datasets/image14.png

# Compare models on single document
python -m vision_processor.cli.simple_extract_cli compare datasets/image14.png
```

## ⚙️ Configuration

The system uses YAML-based configuration with two main files:

### model_comparison.yaml
Primary configuration for model paths, device settings, and prompts:
```yaml
# Model paths
model_paths:
  llama: "/path/to/Llama-3.2-11B-Vision-Instruct"
  internvl: "/path/to/InternVL3-8B"

# Device configuration
device_config:
  gpu_strategy: "single_gpu"
  v100_mode: true
  memory_limit_gb: 16

# Extraction settings  
min_fields_for_success: 1
defaults:
  max_tokens: 256
  quantization: true
```

### prompts.yaml
Field extraction schemas and unified prompts for both models:
```yaml
key_schema:
  required_keys:
    - DATE
    - STORE
    - TOTAL
    - GST
    - ABN
  optional_keys:
    - ITEMS
    - QUANTITIES
    - PRICES
    # ... additional fields
```

## 📊 CLI Commands

### Document Extraction
```bash
# Single document extraction
python -m vision_processor.cli.simple_extract_cli extract image.jpg

# Extract with model override
python -m vision_processor.cli.simple_extract_cli extract image.jpg --model llama32_vision

# Extract with output format override  
python -m vision_processor.cli.simple_extract_cli extract image.jpg --output-format json

# Batch processing
python -m vision_processor.cli.simple_extract_cli batch ./datasets/ --output-dir ./results/
```

### Model Evaluation
```bash
# Run full comparison
python model_comparison.py

# Evaluate specific models
python -m vision_processor.cli.evaluation_cli compare evaluation_ground_truth.csv \
    --models internvl3,llama32_vision \
    --output-dir results

# Benchmark single model
python -m vision_processor.cli.evaluation_cli benchmark datasets --model internvl3
```

## 🔑 Field Extraction

The system dynamically extracts fields based on the schema defined in `prompts.yaml`. Fields are categorized as:

**Required Fields**: Must be present for successful extraction
- DATE, STORE, TOTAL, GST, ABN

**Optional Fields**: Extracted when present in the document
- ITEMS, QUANTITIES, PRICES, RECEIPT_NUMBER, PAYMENT_METHOD, etc.

Fields can be added or modified by editing `prompts.yaml` without changing any code.

## 🧪 Testing & Evaluation

### Running Tests
```bash
# Test single extraction
python -m vision_processor.cli.simple_extract_cli extract datasets/image14.png

# Run complete model comparison
python model_comparison.py

# Validate configuration
python -m vision_processor.cli.simple_extract_cli config-info
```

### Evaluation Metrics
The system evaluates models based on:
- **Field Extraction Rate**: Number of fields successfully extracted
- **Processing Speed**: Time per document
- **Success Rate**: Percentage of documents meeting minimum field threshold
- **Memory Usage**: GPU memory consumption

## 📈 Model Support

Both models use the identical prompt defined in `model_comparison.yaml`, ensuring consistent behavior and fair comparison.

### InternVL3-8B
- Advanced multi-modal vision-language model
- Requires `trust_remote_code=True` for loading
- Supports 8-bit quantization for memory optimization
- Performance varies with document complexity

### Llama-3.2-Vision-Instruct (11B)
- Robust vision-language model with large context
- Often returns complete markdown for complex images instead of KEY: VALUE pairs
- May repeat outputs due to GPU-related bug in certain configurations
- System includes repetition control and markdown fallback extraction
- Compatible with standard transformers framework

## 🔧 Advanced Configuration

### Memory Optimization
For V100 16GB deployment:
```yaml
device_config:
  v100_mode: true
  memory_limit_gb: 16
  
defaults:
  quantization: true  # Enable 8-bit quantization
```

### Unified Prompt System
Both models use the same prompt defined in `model_comparison.yaml`:
```yaml
prompts:
  internvl: |
    <|image|>Extract data from this business document in KEY-VALUE format.
    
    Output format:
    DATE: [date from receipt]
    SUPPLIER: [SUPPLIER name]
    ABN: [11-digit Australian Business Number if visible]
    GST: [GST amount]
    TOTAL: [total amount]
    # ... additional fields
    
    Extract all visible text and format as KEY: VALUE pairs only.
    
  llama: |
    # Identical prompt to internvl above
```

## 🚀 Production Deployment

### Requirements
- **Hardware**: V100 GPU (16GB) or better
- **Memory**: 64GB system RAM recommended
- **Storage**: ~50GB for models
- **Python**: 3.11+

### Optimization Tips
1. Enable quantization for memory-constrained environments
2. Use single GPU mode for V100 deployment
3. Configure appropriate batch sizes based on GPU memory
4. Monitor GPU memory usage with built-in tools

## 📚 Documentation

- **README.md**: This file
- **CLAUDE.md**: Development guidelines and project notes
- **model_comparison.yaml**: Configuration reference
- **prompts.yaml**: Field schema documentation

## 🤝 Contributing

1. Follow the development workflow in CLAUDE.md
2. Run `ruff check . --fix` before committing
3. Test changes with both InternVL3 and Llama models
4. Update documentation for new features

## 📝 License

This project is designed for research and evaluation purposes. Model licenses:
- InternVL3: Apache 2.0
- Llama-3.2-Vision: Llama Community License

---

**Status**: Production Ready  
**Version**: 2.0.0  
**Last Updated**: 2025-07-22  
**Models**: InternVL3-8B, Llama-3.2-11B-Vision