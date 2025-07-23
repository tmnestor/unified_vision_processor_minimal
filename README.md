# Vision Model Comparison Framework

A pure model comparison system designed to answer the critical question: **"Which vision model better understands business documents?"** This framework provides unbiased, reproducible comparisons between InternVL3 and Llama-3.2-Vision models on real-world document understanding tasks.

## 🎯 Overview

This codebase implements a fair, transparent comparison methodology for evaluating vision language models on business document understanding. By preserving raw model outputs and avoiding processing bias, it enables objective assessment of each model's capabilities.

### Core Principles

- **Unbiased Comparison**: Raw model outputs preserved without favoring any specific format
- **Identical Prompts**: Both models receive exactly the same instructions
- **No Post-Processing**: Model responses analyzed as-is, without extraction logic that could introduce bias
- **Reproducible Results**: Complete output preservation enables re-analysis and validation
- **Production Constraints**: Designed for real-world deployment on V100 16GB GPUs

## 🏗️ Comparison Architecture

The framework implements a pure comparison pipeline:

```
Document Image → Unified Prompt → Vision Model → Raw Output Capture → Analysis → Comparison Report
```

### Key Components for Fair Comparison

- **ComparisonRunner**: Orchestrates unbiased model evaluation across all test images
- **Analysis Dictionary**: Preserves complete model outputs without interpretation
- **Model Registry**: Ensures identical configuration for both models
- **Memory Monitor**: Tracks resource usage for performance comparison
- **Unified Prompts**: Guarantees both models receive identical instructions

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

## 🚀 Quick Start - Model Comparison

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

### Primary Usage - Running Model Comparisons

```bash
# Run the main comparison - this is the core purpose of the codebase
python model_comparison.py

# This will:
# 1. Load both vision models sequentially (V100-optimized)
# 2. Process all 25 test images with each model
# 3. Capture raw, unprocessed outputs from each model
# 4. Display side-by-side comparison of what each model "sees"
# 5. Generate performance metrics and analysis
```

### Additional Tools

```bash
# Compare models on a specific document
python -m vision_processor.cli.simple_extract_cli compare datasets/image14.png

# Test individual model (for debugging)
python -m vision_processor.cli.simple_extract_cli extract datasets/image14.png --model llama32_vision
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

## 🔍 Pure Model Comparison Architecture

### The Analysis Dictionary

At the heart of the comparison system is the `analysis_dict`, a carefully designed data structure that captures the complete, unbiased output from each vision model. This approach ensures fair comparison by preserving raw model capabilities without imposing any post-processing that might favor one model over another.

#### Structure and Purpose

```python
analysis_dict = {
    "img_name": image_path.name,           # Name of the image file being processed
    "raw_response": response.raw_text,     # Complete, unmodified output from the model
    "model_name": model_name,              # Which model produced this result (llama/internvl)
    "processing_time": response.processing_time,  # How long the model took to process
    "response_length": len(response.raw_text),   # Character count of the response
    "successful": True,                    # Always True for raw comparison (no validation)
    "timestamp": datetime.now().isoformat(),      # When this processing occurred
}
```

#### Key Benefits

1. **Complete Output Preservation**: The `raw_response` field stores exactly what each model generates, including:
   - Properly formatted KEY: VALUE pairs
   - Markdown-formatted responses
   - OCR fallback outputs (marked with `<OCR/>`)
   - Any model-specific formatting quirks

2. **Fair Comparison**: By avoiding any extraction or parsing logic in the data collection phase, we ensure:
   - No bias toward models that output specific formats
   - Complete visibility into what each model "sees" in documents
   - Ability to analyze different response strategies

3. **Performance Tracking**: Processing time and response length provide insights into:
   - Model efficiency on different document types
   - Trade-offs between speed and output quality
   - Resource utilization patterns

4. **Traceability**: The combination of image name, model name, and timestamp enables:
   - Reproducible comparisons
   - Debugging of model behavior on specific images
   - Historical analysis of model improvements

#### Usage in the Pipeline

The `analysis_dict` flows through the comparison pipeline as follows:

1. **Collection Phase**: Each model processes each image, generating an `analysis_dict`
2. **Aggregation**: Dictionaries are collected in `extraction_results[model_name]`
3. **Analysis**: The raw data feeds into performance metrics and comparison reports
4. **Export**: Complete results can be exported for further analysis

This architecture exemplifies the principle of "measure first, interpret later" - by preserving complete model outputs, researchers can apply different analysis approaches without re-running expensive model inference.

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

**Purpose**: Model Comparison Framework  
**Status**: Production Ready  
**Version**: 2.0.0  
**Last Updated**: 2025-07-23  
**Core Question**: "Which vision model better understands business documents?"  
**Models Under Comparison**: InternVL3-8B vs Llama-3.2-11B-Vision