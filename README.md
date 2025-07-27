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
│   ├── extract_cli.py        # Document extraction CLI
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

## 📋 CLI Usage

The system provides comprehensive command-line interfaces for document processing, model comparison, and evaluation.

### Core Commands

#### 1. Single Document Extraction
```bash
# Basic extraction (requires explicit model selection)
python -m vision_processor.cli.extract_cli extract image14.png --model llama

# Extract with specific model
python -m vision_processor.cli.extract_cli extract image14.png --model internvl

# Extract with output format override
python -m vision_processor.cli.extract_cli extract image14.png --output-format json

# Custom YAML configuration
python -m vision_processor.cli.extract_cli extract image14.png --yaml-file custom_config.yaml
```

#### 2. Model Comparison
```bash
# Compare default models on single document
python -m vision_processor.cli.extract_cli compare image14.png

# Compare specific models
python -m vision_processor.cli.extract_cli compare image14.png --models llama,internvl

# Comparison with custom config
python -m vision_processor.cli.extract_cli compare image14.png --yaml-file production.yaml
```

#### 3. Batch Processing
```bash
# Batch process all images in directory
python -m vision_processor.cli.extract_cli batch ./datasets/

# Batch with custom output directory
python -m vision_processor.cli.extract_cli batch ./datasets/ --output-dir ./batch_results/

# Batch with specific model
python -m vision_processor.cli.extract_cli batch ./datasets/ --model internvl
```

#### 4. Configuration Information
```bash
# View current configuration and paths
python -m vision_processor.cli.extract_cli config-info

# Check configuration with custom YAML
python -m vision_processor.cli.extract_cli config-info --yaml-file production.yaml

# View configuration with verbose details
python -m vision_processor.cli.extract_cli config-info --verbose
```

### Model Comparison Script

#### 5. Full Model Comparison Pipeline
```bash
# Run complete model comparison with default settings
python model_comparison.py

# Model comparison with CLI overrides
python model_comparison.py compare --datasets-path ./test_images/ --output-dir ./comparison_results/

# Compare specific models only
python model_comparison.py compare --models llama --quantization

# Environment validation
python model_comparison.py check-environment

# Model validation
python model_comparison.py validate-models
```

### Evaluation Commands

#### 6. Model Performance Evaluation
```bash
# Compare models against ground truth
python -m vision_processor.cli.evaluation_cli compare ground_truth.csv

# Evaluation with custom paths
python -m vision_processor.cli.evaluation_cli compare ground_truth.csv --images-dir ./test_set/ --output-dir ./eval_results/

# Benchmark single model performance
python -m vision_processor.cli.evaluation_cli benchmark ./datasets/ --model llama --iterations 5

# Validate ground truth data
python -m vision_processor.cli.evaluation_cli validate-ground-truth ground_truth.csv --images-dir ./datasets/
```

### Path Resolution

The system uses intelligent path resolution for both inputs and outputs:

#### Input Path Resolution
```bash
# Relative paths resolve against configured datasets_path
python -m vision_processor.cli.extract_cli extract image14.png
# Resolves to: {datasets_path}/image14.png

# Absolute paths used as-is
python -m vision_processor.cli.extract_cli extract /full/path/to/image.png
# Uses: /full/path/to/image.png
```

#### Output Path Resolution
```bash
# Relative output paths resolve against configured output_dir
python -m vision_processor.cli.extract_cli batch ./datasets/ --output-dir results
# Creates: {output_dir}/results/

# Absolute output paths used as-is
python -m vision_processor.cli.extract_cli batch ./datasets/ --output-dir /full/path/output
# Creates: /full/path/output/
```

### Environment-Specific Usage

#### Local Development
```bash
# Use local configuration with Desktop paths
python -m vision_processor.cli.extract_cli extract image14.png --yaml-file model_comparison_local.yaml

# Local batch processing
python -m vision_processor.cli.extract_cli batch datasets/ --output-dir results/
```

#### Remote/Production (KFP)
```bash
# Use production configuration with NFS paths
python -m vision_processor.cli.extract_cli extract image14.png --yaml-file model_comparison.yaml

# Production batch processing
python -m vision_processor.cli.extract_cli batch datasets/ --output-dir production_results/
```

### Model Selection Requirement

**IMPORTANT**: All CLI commands now require explicit model selection. No default model is assumed.

```bash
# ❌ This will fail - no model specified
python -m vision_processor.cli.extract_cli extract image14.png

# ✅ This works - model explicitly specified  
python -m vision_processor.cli.extract_cli extract image14.png --model llama
```

**Error when no model specified:**
```
❌ FATAL: No model specified
💡 Available models: ['llama', 'internvl']
💡 Fix: Add --model flag to CLI command
💡 Example: --model llama or --model internvl
```

### Advanced Usage

#### Error Diagnosis
```bash
# Debug configuration issues
python -m vision_processor.cli.extract_cli config-info --debug

# Debug extraction problems
python -m vision_processor.cli.extract_cli extract problem_image.png --debug

# Debug model comparison
python model_comparison.py compare --debug
```

#### Performance Testing
```bash
# Quick performance test
python -m vision_processor.cli.evaluation_cli benchmark datasets/ --model llama --iterations 3

# Memory usage monitoring
python model_comparison.py check-environment --verbose

# Validate all configured models
python model_comparison.py validate-models
```

#### Automation-Friendly Usage
```bash
# Quiet mode for scripts (only errors/warnings)
python -m vision_processor.cli.extract_cli batch datasets/ --quiet

# JSON output for processing
python -m vision_processor.cli.extract_cli extract image.png --output-format json --quiet

# Exit codes for CI/CD
python model_comparison.py validate-models --quiet
echo $?  # 0 for success, 1 for failure
```

### Verbosity Control

All CLI commands support runtime verbosity control:

```bash
# Quiet mode (minimal output, errors/warnings only)
python model_comparison.py compare --quiet
python -m vision_processor.cli.extract_cli extract image.jpg --quiet

# Verbose mode (detailed status messages)
python model_comparison.py compare --verbose
python -m vision_processor.cli.evaluation_cli compare ground_truth.csv --verbose

# Debug mode (full diagnostic output)
python model_comparison.py compare --debug
python -m vision_processor.cli.extract_cli extract image.jpg --debug
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
python -m vision_processor.cli.extract_cli extract image.jpg

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

## 📊 Repetition Control Analysis

### Current Implementation: Post-Processing Approach

The system currently uses **post-processing repetition control** rather than real-time `repetition_penalty` parameters during generation. This design choice is based on technical research and practical considerations:

### **Research Findings on Repetition Control Efficiency**

#### **Repetition Penalty (Real-time)**
- **Performance Impact**: Causes 15.4% throughput decrease in vLLM at high request rates¹
- **Memory Overhead**: Adds computational cost at every token generation step
- **Minimal Impact**: Negligible overhead at low request rates (memory-bound conditions)²

#### **Post-Processing Approach (Current)**
- **Batch Efficiency**: Can process multiple documents simultaneously
- **Sophisticated Detection**: Capable of detecting complex repetition patterns³
- **Separation of Concerns**: Generation and repetition control are independent processes
- **Industrial Applications**: Proven to meet real-time performance requirements⁴

### **Llama 3.2 Vision Specific Considerations**

Research reveals several technical challenges with Llama 3.2 Vision models:

1. **Known Compatibility Issues**: Reported errors when adding `repetition_penalty` to `generation_config` with Llama-3.2-11B-Vision-Instruct⁵
2. **GPU Utilization Problems**: Llama 3.2 Vision models sometimes fail to utilize GPU properly, falling back to CPU-only processing⁶
3. **Memory Constraints**: Users report CUDA out-of-memory errors specifically with Llama 3.2 Vision models, even with quantization⁷

### **Design Rationale**

The post-processing approach was implemented to:
- **Avoid GPU Compatibility Issues**: Circumvent known bugs with repetition penalty in Llama 3.2 Vision
- **Ensure V100 Compatibility**: Minimize GPU memory overhead for 16GB VRAM constraints
- **Maintain Reliability**: Provide consistent results across different model configurations
- **Enable Advanced Detection**: Support sophisticated repetition pattern recognition

### **Future Optimization Opportunities**

1. **Hybrid Approach**: Light repetition penalty (1.1-1.3) + simplified post-processing
2. **Model-Specific Tuning**: Different strategies for InternVL3 vs Llama-3.2-Vision
3. **Performance Monitoring**: Benchmark generation time vs post-processing overhead
4. **Memory Optimization**: Test repetition penalty impact on V100 memory usage

### **References**

1. *vLLM vs TensorRT-LLM: Understanding Sampling Methods and Their Performance Impact* - SqueezeBits Blog
2. *Navigating the Nuances of Text Generation: How to Control LLM Outputs* - Neural Magic
3. *Code Copycat Conundrum: Demystifying Repetition in LLM-based Code Generation* - arXiv:2504.12608v1
4. *Customizing LLM Output: Post-Processing Techniques* - Neptune.ai
5. *Llama 3.2, inference error with "repetition_penalty" in generation_config* - Hugging Face Transformers Issue #34304
6. *llama3.2-vision doesn't utilize my GPU* - Ollama Issue #8310
7. *unsloth/Llama-3.2-11B-Vision-Instruct CUDA error: out of memory* - Unsloth Issue #1572

## 📊 Performance Metrics

The system tracks comprehensive performance metrics with intelligent extraction method detection:

### Extraction Metrics
- **Fields Extracted**: Count of successfully extracted fields per document
- **Field Success Rate**: Percentage of documents with each field extracted
- **Value Completion**: Percentage of extracted fields with actual values (not "N/A")
- **Extraction Method**: Intelligent detection of processing approach used per document

### Extraction Method Classification

The system automatically detects and reports the extraction method used for each document:

#### **Clean Processing Methods**
- **`clean_extraction`**: Optimal - Clean KEY: value format, no post-processing needed
- **`clean_with_markdown`**: Good - Clean format with minor markdown formatting
- **`clean_with_repetition`**: Acceptable - Clean format but contains repetitive content

#### **Fallback Processing Methods**  
- **`markdown_fallback`**: Moderate - Primarily markdown content requiring parsing
- **`repetition_control`**: Challenging - Response needed repetition cleanup
- **`complex_parsing`**: Difficult - Complex response requiring advanced parsing
- **`error`**: Failed - Processing failed, no data extracted

### Detection Algorithm

The extraction method is determined by analyzing model responses:

```python
# Format Analysis: Ratio of KEY: value lines to total lines
clean_ratio = clean_lines / total_lines

# Content Analysis: Detection of formatting issues
has_markdown = presence of #, **, _, |, -, * formatting
has_repetition = words appearing >3 times (excluding common words)

# Classification Logic
if clean_ratio > 0.8:
    if has_markdown: return "clean_with_markdown"
    elif has_repetition: return "clean_with_repetition" 
    else: return "clean_extraction"
elif has_markdown: return "markdown_fallback"
elif has_repetition: return "repetition_control"
else: return "complex_parsing"
```

### Performance Metrics
- **Processing Time**: Time per document (including model loading)
- **Memory Usage**: Peak GPU memory consumption
- **Response Length**: Character count of model outputs
- **Method Distribution**: Percentage breakdown of extraction methods used

### Quality Thresholds
- **Excellent**: 12+ fields extracted (nearly half of possible fields)
- **Good**: 8-11 fields extracted (solid extraction)
- **Fair**: 5-7 fields extracted (basic extraction)
- **Poor**: <5 fields extracted (needs optimization)

### Batch Results Analysis

When using `batch` command, results are saved to `batch_results.json` with detailed extraction method information:

```json
{
  "filename": "invoice_001.png",
  "extracted_fields": { "DOCUMENT_TYPE": "Tax Invoice", ... },
  "processing_time": 2.34,
  "model_confidence": 0.95,
  "extraction_method": "clean_extraction"
}
```

This enables analysis of:
- Which documents process cleanly vs require post-processing
- Model-specific response quality patterns  
- Performance optimization opportunities
- Success rate by extraction method type

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
python -m vision_processor.cli.extract_cli extract datasets/image14.png

# Validate configuration
python -m vision_processor.cli.extract_cli config-info

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