# Unified Vision Document Processing Architecture

A production-ready system for document field extraction using vision-language models, with support for both InternVL3 and Llama-3.2-Vision models.

## 🎯 Overview

This system provides model-agnostic document processing with dynamic field detection, focusing on Australian tax documents and business records. The architecture emphasizes simplicity, maintainability, and production deployment on V100 GPUs.

## 🚀 Key Features

- **Model Agnostic**: Seamless switching between InternVL3 and Llama-3.2-Vision models
- **Production Ready**: Optimized for V100 16GB GPU deployment with 8-bit quantization
- **Dynamic Field Extraction**: Configurable field schemas with automatic prompt generation
- **Unified Configuration**: Single YAML source of truth for all settings
- **Robust Parsing**: Handles various model output formats with fallback strategies
- **Memory Efficient**: Built-in memory monitoring and cleanup

## 📦 Project Structure

```
vision_processor/
├── config/                    # Configuration management
│   ├── config_manager.py     # Unified configuration system
│   ├── config_models.py      # Configuration data classes
│   ├── model_registry.py     # Model factory and registration
│   └── __init__.py
├── models/                    # Model implementations
│   ├── base_model.py         # Abstract base class
│   ├── internvl_model.py     # InternVL3 implementation
│   ├── llama_model.py        # Llama-3.2-Vision implementation
│   ├── model_utils.py        # Shared utilities
│   └── __init__.py
├── extraction/                # Field extraction logic
│   ├── extraction_manager.py  # Main extraction pipeline
│   └── __init__.py
├── comparison/                # Model comparison tools
│   ├── comparison_runner.py  # Cross-model evaluation
│   ├── model_validator.py    # Validation logic
│   └── __init__.py
├── cli/                       # Command-line interfaces
│   ├── simple_extract_cli.py # Single extraction CLI
│   ├── evaluation_cli.py     # Evaluation CLI
│   └── __init__.py
├── evaluation/                # Evaluation framework
│   ├── evaluator.py          # Model evaluation
│   └── __init__.py
├── analysis/                  # Metrics and analysis
│   ├── simple_metrics.py     # Performance metrics
│   └── __init__.py
├── utils/                     # Utilities
│   ├── memory_monitor.py     # GPU memory tracking
│   └── __init__.py
├── exceptions.py              # Custom exceptions
└── __init__.py

# Project root
├── model_comparison.yaml      # Main configuration file
├── model_comparison.py        # Primary comparison script
├── environment.yml            # Conda environment
├── requirements.txt           # Python dependencies
├── datasets/                  # Test images
└── backup/                    # Legacy code (archived)
```

## 🚀 Quick Start

### Prerequisites

1. **Create conda environment**:
```bash
conda env create -f environment.yml
conda activate unified_vision_processor
```

2. **Configure model paths** in `model_comparison.yaml`:
```yaml
model_paths:
  llama: "/path/to/Llama-3.2-11B-Vision-Instruct"
  internvl: "/path/to/InternVL3-8B"
```

### Basic Usage

```bash
# Run full model comparison
python model_comparison.py

# Extract from single document
python -m vision_processor.cli.simple_extract_cli extract datasets/image14.png

# Compare models on specific document
python -m vision_processor.cli.simple_extract_cli compare datasets/image14.png --models llama,internvl

# Extract with specific model
python -m vision_processor.cli.simple_extract_cli extract datasets/image14.png --model llama

# Batch processing
python -m vision_processor.cli.simple_extract_cli batch ./datasets/ --output-dir ./results/
```

### Verbosity Control

All CLI commands support runtime verbosity control:

```bash
# Quiet mode (minimal output, errors/warnings only)
python model_comparison.py compare --quiet
python -m vision_processor.cli.simple_extract_cli extract image.jpg --quiet

# Verbose mode (detailed status messages)
python model_comparison.py compare --verbose
python -m vision_processor.cli.evaluation_cli compare ground_truth.csv --verbose

# Debug mode (full diagnostic output)
python model_comparison.py compare --debug
python -m vision_processor.cli.simple_extract_cli extract image.jpg --debug
```

## ⚙️ Configuration

The system uses a unified YAML configuration system with fail-fast validation:

### model_comparison.yaml

```yaml
# Model paths
model_paths:
  llama: "/path/to/Llama-3.2-11B-Vision-Instruct"
  internvl: "/path/to/InternVL3-8B"

# Default settings
defaults:
  datasets_path: "datasets"
  quantization: true
  output_dir: "results"
  models: "llama,internvl"
  verbose_mode: false      # Enable detailed status messages
  debug_mode: false        # Enable debug-level output
  console_output: true     # Enable Rich console formatting
  log_level: "INFO"        # ERROR, WARNING, INFO, DEBUG

# Memory configuration
memory_config:
  v100_limit_gb: 16.0
  safety_margin: 0.85

# Device configuration
device_config:
  gpu_strategy: "single_gpu"
  v100_mode: true
  memory_limit_gb: 16
  device_maps:
    llama:
      strategy: "single_gpu"
      device_map: {"": 0}
      quantization_compatible: true
    internvl:
      strategy: "single_gpu"
      device_map: {"": 0}
      quantization_compatible: true

# Field extraction configuration
expected_fields:
  - DOCUMENT_TYPE
  - SUPPLIER
  - ABN
  - PAYER_NAME
  - PAYER_ADDRESS
  # ... additional fields (26 total)

# Extraction prompt (dynamically generated from expected_fields)
extraction_prompt: |
  Extract data from this business document. Output each field with its exact key:
  # Automatically includes all expected_fields
```

### Key Configuration Features

- **Single Source of Truth**: All settings in YAML files
- **Fail-Fast Validation**: Configuration errors caught at startup
- **Dynamic Prompts**: Automatically generated from `expected_fields`
- **Model-Agnostic**: Same configuration works for both models
- **Production Optimized**: V100 GPU constraints built-in
- **Unified Logging**: Configurable verbosity with CLI override support

## 🔑 Field Extraction

The system dynamically extracts 26 document fields:

**Core Fields**: DOCUMENT_TYPE, SUPPLIER, ABN, TOTAL, GST
**Financial**: SUBTOTAL, INVOICE_DATE, DUE_DATE, PAYMENT_METHOD
**Business**: PAYER_NAME, PAYER_ADDRESS, BUSINESS_ADDRESS, BUSINESS_PHONE
**Banking**: BANK_NAME, BSB_NUMBER, BANK_ACCOUNT_NUMBER, ACCOUNT_HOLDER
**Itemization**: ITEMS, QUANTITIES, PRICES
**Additional**: RECEIPT_NUMBER, CARD_NUMBER, AUTH_CODE, STATUS, etc.

### Adding New Fields

Simply update the `expected_fields` list in `model_comparison.yaml`:

```yaml
expected_fields:
  - DOCUMENT_TYPE
  - SUPPLIER
  - YOUR_NEW_FIELD  # Automatically included in prompts
  # ... existing fields
```

No code changes required - prompts are generated dynamically.

## 📋 Logging & Verbosity Control

The system provides a unified logging system with configurable verbosity levels:

### Logging Levels
- **ERROR**: Critical failures and exceptions (always shown)
- **WARNING**: Non-fatal issues, missing paths, fallbacks (always shown)
- **INFO**: Important status updates, completion messages (verbose mode)
- **DEBUG**: Detailed processing info, internal state (debug mode)

### CLI Verbosity Flags

All CLI commands support these verbosity flags:

| Flag | Short | Description | Output Level |
|------|-------|-------------|--------------|
| `--quiet` | `-q` | Minimal output, disable Rich formatting | ERROR, WARNING only |
| (default) | | Standard output with Rich formatting | ERROR, WARNING only |
| `--verbose` | `-v` | Detailed status messages | ERROR, WARNING, INFO |
| `--debug` | | Full diagnostic output | All levels |

### Configuration Options

Control logging behavior via `model_comparison.yaml`:

```yaml
defaults:
  verbose_mode: false      # Show INFO-level messages
  debug_mode: false        # Show DEBUG-level messages  
  console_output: true     # Enable Rich console formatting
  log_level: "INFO"        # Minimum log level

logging:
  file_logging: true       # Enable file logging
  log_file: "vision_processor.log"
  max_log_size: "10MB"     # Log rotation size
  backup_count: 3          # Number of backup logs
```

### Usage Examples

```bash
# Quiet processing (automation-friendly)
python model_comparison.py compare --quiet

# Standard processing with Rich output (default)
python -m vision_processor.cli.simple_extract_cli extract image.jpg

# Verbose processing with detailed status
python -m vision_processor.cli.evaluation_cli compare data.csv --verbose

# Debug processing with full diagnostics
python model_comparison.py compare --debug
```

## 🧪 Model Support

### InternVL3-8B
- **Capabilities**: High-resolution image processing, robust parsing
- **Requirements**: `trust_remote_code=True`
- **Memory**: ~8GB with 8-bit quantization
- **Strengths**: Accurate field extraction, good with complex layouts

### Llama-3.2-Vision-Instruct (11B)
- **Capabilities**: Large context, consistent formatting
- **Requirements**: Standard transformers loading
- **Memory**: ~11GB with 8-bit quantization
- **Strengths**: Reliable output format, good instruction following

Both models use identical prompts and processing pipelines for fair comparison.

## 📊 Performance Metrics

The system tracks comprehensive performance metrics:

### Extraction Metrics
- **Fields Extracted**: Count of successfully extracted fields per document
- **Field Success Rate**: Percentage of documents with each field extracted
- **Value Completion**: Percentage of extracted fields with actual values (not "N/A")

### Performance Metrics
- **Processing Time**: Time per document (including model loading)
- **Memory Usage**: Peak GPU memory consumption
- **Response Length**: Character count of model outputs

### Quality Thresholds
- **Excellent**: 12+ fields extracted (nearly half of possible fields)
- **Good**: 8-11 fields extracted (solid extraction)
- **Fair**: 5-7 fields extracted (basic extraction)
- **Poor**: <5 fields extracted (needs optimization)

## 🔧 Production Deployment

### V100 GPU Optimization

The system is optimized for V100 16GB deployment:

```yaml
device_config:
  v100_mode: true
  memory_limit_gb: 16
  gpu_strategy: "single_gpu"

defaults:
  quantization: true  # Essential for V100
```

### Memory Management

- **8-bit Quantization**: Reduces memory usage by ~50%
- **Memory Monitoring**: Built-in GPU memory tracking
- **Automatic Cleanup**: Memory freed after each inference
- **Safety Margins**: 85% memory usage limit to prevent OOM

### Deployment Tips

1. **Enable quantization** for memory-constrained environments
2. **Monitor GPU memory** during initial runs
3. **Use single GPU mode** for V100 deployment
4. **Adjust batch sizes** based on available memory

## 🏗️ Architecture Highlights

### Configuration System
- **ConfigManager**: Unified configuration with fail-fast validation
- **Structured Objects**: Type-safe configuration data classes
- **No Legacy Support**: Clean migration from old SimpleConfig

### Model Loading
- **Factory Pattern**: Model registry for clean instantiation
- **Error Handling**: Comprehensive error messages with solutions
- **Memory Validation**: Pre-loading memory compliance checks

### Extraction Pipeline
- **Simple Pipeline**: Straightforward document → fields extraction
- **Robust Parsing**: Multiple parsing strategies with fallbacks
- **Universal Parser**: Handles various model output formats

## 🧪 Testing & Validation

```bash
# Test extraction pipeline
python -m vision_processor.cli.simple_extract_cli extract datasets/image14.png

# Validate configuration
python -m vision_processor.cli.simple_extract_cli config-info

# Run model comparison
python model_comparison.py

# Evaluate model performance
python -m vision_processor.cli.evaluation_cli compare evaluation_ground_truth.csv
```

## 🛠️ Development

### Code Quality
- **Ruff Integration**: Automated linting and formatting
- **Type Hints**: Comprehensive type annotations
- **Documentation**: Clear module and function documentation

### Development Workflow
```bash
# Install development dependencies
conda env create -f environment.yml

# Run quality checks
ruff check . --fix --ignore ARG001,ARG002,F841
ruff format .

# Test changes
python model_comparison.py
```

### Adding New Models

1. **Implement BaseVisionModel** interface in `models/`
2. **Register model** in `config/model_registry.py`
3. **Add configuration** in `model_comparison.yaml`
4. **Test integration** with existing pipeline

## 📚 Documentation

- **README.md**: This overview and usage guide
- **CLAUDE.md**: Development guidelines and project context
- **model_comparison.yaml**: Configuration reference with comments
- **Code Documentation**: Comprehensive docstrings throughout

## 🤝 Contributing

1. Follow development guidelines in `CLAUDE.md`
2. Run `ruff check . --fix` before committing
3. Test with both InternVL3 and Llama models
4. Update documentation for new features
5. Use conservative refactoring principles

## 📈 Recent Improvements

### Configuration System Refactoring
- ✅ **Unified ConfigManager**: Replaced complex SimpleConfig with clean architecture
- ✅ **Fail-Fast Validation**: Configuration errors caught at startup with clear messages
- ✅ **Legacy Cleanup**: Old configuration files moved to `backup/` directories
- ✅ **Type Safety**: Structured configuration objects replace dynamic dict access

### Logging System Implementation
- ✅ **Unified Logging**: Replaced 84+ raw print statements with structured logging system
- ✅ **CLI Verbosity Control**: Added `--verbose`, `--debug`, `--quiet` flags to all commands
- ✅ **Rich Console Output**: Color-coded messages with emojis for better UX
- ✅ **File Logging**: Production-ready logging with rotation and configurable levels
- ✅ **Runtime Configuration**: CLI flags override YAML defaults for flexible control

### Code Organization
- ✅ **Conservative Refactoring**: llama_model.py organized with clear sections
- ✅ **Clear Module Structure**: Well-defined `__init__.py` files with proper exports
- ✅ **Consistent Imports**: Standardized relative import patterns
- ✅ **API Boundaries**: Clear public interfaces defined in `__all__`

## 📝 License

- **Project**: Research and evaluation purposes
- **InternVL3**: Apache 2.0 License
- **Llama-3.2-Vision**: Llama Community License

---

**Purpose**: Production Document Processing  
**Status**: Production Ready  
**Architecture**: Unified Vision Processor  
**Focus**: Maintainable, Model-Agnostic Document Field Extraction  
**Target**: V100 GPU Deployment with Memory Optimization