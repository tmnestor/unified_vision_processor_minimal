# Unified Vision Document Processing Architecture

A production-ready system for document field extraction using vision-language models, with support for both InternVL3 and Llama-3.2-Vision models.

## 🎯 Overview

This system provides model-agnostic document processing with dynamic field detection, focusing on Australian tax documents and business records. The architecture emphasizes simplicity, maintainability, and production deployment on V100 GPUs.

## 🚀 Key Features

- **Model Agnostic**: Seamless switching between InternVL3 and Llama-3.2-Vision models
- **Production Ready**: Optimized for V100 16GB GPU deployment with 8-bit quantization
- **Unified Interface**: All functionality through `model_comparison.py` - no complex module paths
- **Simplified Workflow**: Intuitive `compare` → `visualize` command structure
- **Dynamic Field Extraction**: Configurable field schemas with automatic prompt generation
- **Unified Configuration**: Single YAML source of truth for all settings
- **KFP Compatible**: Persistent storage support for Kubeflow Pipelines
- **Robust Parsing**: Handles various model output formats with fallback strategies
- **Memory Efficient**: Built-in memory monitoring and cleanup
- **DataFrame Integration**: Convert batch results to pandas DataFrames for analysis

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
│   ├── batch_to_csv_cli.py   # Batch results to DataFrame CLI
│   └── __init__.py
├── evaluation/                # Evaluation framework
│   ├── evaluator.py          # Model evaluation
│   └── __init__.py
├── analysis/                  # Metrics and analysis
│   ├── simple_metrics.py     # Performance metrics
│   └── __init__.py
├── utils/                     # Utilities
│   ├── memory_monitor.py     # GPU memory tracking
│   ├── batch_to_dataframe.py # Batch results to DataFrame converter
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

### Unified Interface - All Commands Through `model_comparison.py`

The architecture has been simplified to provide all functionality through a single, intuitive interface:

### Primary Workflow

#### 1. Model Comparison
```bash
# Run complete model comparison with default settings
python model_comparison.py compare

# Model comparison with custom paths
python model_comparison.py compare --datasets-path ./test_images/ --output-dir ./results/

# Compare specific models only
python model_comparison.py compare --models llama --quantization
```

#### 2. Generate Visualizations (NEW SIMPLIFIED APPROACH)
```bash
# Generate visualizations from existing results
python model_comparison.py visualize

# Generate with custom ground truth file
python model_comparison.py visualize --ground-truth-csv evaluation_ground_truth.csv

# Generate with custom paths
python model_comparison.py visualize --images-dir ./test_images/ --output-dir ./results/
```

#### 3. Complete Workflow Examples

**After (Intuitive):**
```bash
python model_comparison.py compare
python model_comparison.py visualize
```

**KFP-Compatible:**
```bash
# KFP Pipeline Step 1: Run Comparison
python model_comparison.py compare --datasets-path /mnt/datasets --output-dir /mnt/output

# KFP Pipeline Step 2: Generate Visualizations (SIMPLIFIED!)
python model_comparison.py visualize --ground-truth-csv evaluation_ground_truth.csv --output-dir /mnt/output
```

### Advanced Commands

#### 4. Environment Management
```bash
# Check production environment and dependencies
python model_comparison.py check-environment

# Validate all configured models
python model_comparison.py validate-models

# List available models in registry
python model_comparison.py list-models

# Show core field schema
python model_comparison.py show-schema
```

### Legacy CLI Commands (Still Available)

#### 5. Single Document Extraction
```bash
# Basic extraction (requires explicit model selection)
python -m vision_processor.cli.extract_cli extract image14.png --model llama

# Extract with specific model
python -m vision_processor.cli.extract_cli extract image14.png --model internvl

# Extract with output format override
python -m vision_processor.cli.extract_cli extract image14.png --output-format json

#### 6. Model Comparison (Legacy)
```bash
# Compare default models on single document
python -m vision_processor.cli.extract_cli compare image14.png

# Compare specific models
python -m vision_processor.cli.extract_cli compare image14.png --models llama,internvl
```

#### 7. Batch Processing (Legacy)
```bash
# Batch process all images in directory
python -m vision_processor.cli.extract_cli batch ./datasets/

# Batch with custom output directory
python -m vision_processor.cli.extract_cli batch ./datasets/ --output-dir ./batch_results/

# Batch with specific model
python -m vision_processor.cli.extract_cli batch ./datasets/ --model internvl
```

#### 8. Batch Results Analysis
```bash
# Convert batch_results.json to CSV DataFrame (one row per image) - REQUIRES --model
python -m vision_processor.cli.batch_to_csv_cli convert batch_results.json --yaml-file model_comparison.yaml --model llama

# Convert with custom output path
python -m vision_processor.cli.batch_to_csv_cli convert batch_results.json --yaml-file model_comparison.yaml --model internvl --output results_dataframe.csv

# Analyze batch results without saving CSV
python -m vision_processor.cli.batch_to_csv_cli analyze batch_results.json --yaml-file model_comparison.yaml --model llama

# Keep "N/A" strings instead of converting to NaN
python -m vision_processor.cli.batch_to_csv_cli convert batch_results.json --yaml-file model_comparison.yaml --model llama --keep-na

# Show info only (no CSV file created)
python -m vision_processor.cli.batch_to_csv_cli convert batch_results.json --yaml-file model_comparison.yaml --model llama --info
```

**DataFrame Structure:**
- **Header**: `[image, DOCUMENT_TYPE, SUPPLIER, ABN, ...]` (27 columns: image + 26 fields)
- **Rows**: One row per image with extracted field values
- **Output**: `batch_results_{model}_dataframe.csv` (e.g., `batch_results_llama_dataframe.csv`)
- **Missing values**: `None/NaN` (default, better for analysis) or `"N/A"` strings (with `--keep-na`)

**Direct Python usage:**
```python
from vision_processor.utils.batch_to_dataframe import batch_results_to_dataframe

# Convert to DataFrame
df = batch_results_to_dataframe("batch_results.json")
print(df.shape)  # (n_images, 27)

# Save to CSV
df.to_csv("extracted_fields.csv", index=False)

# Analyze extraction completeness
missing_per_field = df.isnull().sum()
overall_completion = (df.count().sum() / df.size) * 100
```

#### 9. Configuration Information (Legacy)
```bash
# View current configuration and paths
python -m vision_processor.cli.extract_cli config-info

# Check configuration with custom YAML
python -m vision_processor.cli.extract_cli config-info --yaml-file production.yaml
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
# Memory usage monitoring
python model_comparison.py check-environment --verbose

# Validate all configured models
python model_comparison.py validate-models

# Quick performance comparison
python model_comparison.py compare --models llama
```

#### Automation-Friendly Usage
```bash
# Quiet mode for scripts (only errors/warnings)
python model_comparison.py compare --quiet
python model_comparison.py visualize --quiet

# Exit codes for CI/CD
python model_comparison.py validate-models --quiet
echo $?  # 0 for success, 1 for failure
```

### Verbosity Control

All commands support runtime verbosity control:

```bash
# Quiet mode (minimal output, errors/warnings only)
python model_comparison.py compare --quiet
python model_comparison.py visualize --quiet

# Verbose mode (detailed status messages)
python model_comparison.py compare --verbose
python model_comparison.py visualize --verbose

# Debug mode (full diagnostic output)
python model_comparison.py compare --debug
python model_comparison.py visualize --debug
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
python model_comparison.py visualize --verbose

# Debug processing with full diagnostics
python model_comparison.py compare --debug
```

## 🎯 Production Usage Patterns

### KFP (Kubeflow Pipelines) Integration

For production KFP environments with persistent storage:

**After (Intuitive):**
```bash
# KFP Pipeline Step 1: Run Comparison  
python model_comparison.py compare --datasets-path /mnt/datasets --output-dir /mnt/output

# KFP Pipeline Step 2: Generate Visualizations (SIMPLIFIED!)
python model_comparison.py visualize --ground-truth-csv evaluation_ground_truth.csv --output-dir /mnt/output
```

**KFP Storage Requirements:**
- Use persistent volume mounts (e.g., `/mnt/*`, `/data/*`, `/home/jovyan/nfs_share/*`)
- Avoid pod-local storage (`/tmp/*`, `/app/*`, `/workspace/*`)
- All generated data persists across pipeline steps and pod restarts

### Local Development

**After (Intuitive):**
```bash
python model_comparison.py compare
python model_comparison.py visualize
```

**Architecture Benefits:**
- ✅ **Unified Interface**: All functionality through one script
- ✅ **No Complex Paths**: Eliminates `python -m vision_processor.cli.evaluation_cli`
- ✅ **Smart Logic**: Uses existing results when available
- ✅ **Consistent CLI**: Same flags and configuration throughout

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

## 📊 Dynamic Visualizations

The system includes a sophisticated visualization module that dynamically generates publication-ready charts and heatmaps for model performance analysis. All visualizations are configuration-driven with no hardcoded field names or thresholds.

### Key Features

- **Dynamic Field Discovery**: Automatically loads field definitions from `model_comparison.yaml`
- **Configuration-Driven**: Thresholds, weights, and categories from YAML configuration
- **Automatic Scaling**: Charts adapt to any number of fields and models
- **Business-Importance Aware**: Field categorization based on configured weights
- **Publication-Ready**: Professional styling with matplotlib and seaborn

### Generated Visualizations

#### 1. Field-Wise Accuracy Heatmap
Shows accuracy for each field across all models with dynamic scaling:

```bash
# Generate all visualizations (simplified)
python model_comparison.py visualize

# Generate with custom ground truth file
python model_comparison.py visualize --ground-truth-csv evaluation_ground_truth.csv

# Generate with custom output directory
python model_comparison.py visualize --output-dir ./custom_viz_results/
```

**Features:**
- Dynamically scales to any number of fields (currently 26+ fields)
- Color-coded performance (red = poor, green = excellent)
- High-priority fields highlighted with gold borders
- Automatic figure sizing based on field count

#### 2. Model Performance Dashboard
Comprehensive 2x2 dashboard showing:
- Overall accuracy with configurable quality thresholds
- Processing speed with performance benchmarks
- Success rate comparison
- Average fields extracted vs. total possible

#### 3. Field Category Analysis
Categorizes fields by business importance (from configuration weights):
- **High Priority**: Fields with weight > 1.1 (critical business fields)
- **Standard**: Fields with weight = 1.0 (important fields)  
- **Lower Priority**: Fields with weight < 1.0 (supplementary fields)

#### 4. V100 VRAM Usage Comparison
Shows estimated VRAM usage for V100 GPU compliance:
- **Left Chart**: VRAM usage in GB with V100 limit lines (16GB limit, 13.6GB safety)
- **Right Chart**: VRAM utilization percentages with threshold indicators
- **Compliance Status**: Visual indicators (✅ Safe, ⚠️ Margin, ❌ Over) for each model
- **Dynamic Estimates**: Uses model-specific memory estimation methods

### CLI Integration

The visualization system integrates with the unified command interface:

```bash
# Complete workflow: comparison + visualizations
python model_comparison.py compare
python model_comparison.py visualize

# Quick visualization from existing results
python model_comparison.py visualize --ground-truth-csv evaluation_ground_truth.csv

# Custom paths for visualization
python model_comparison.py visualize --images-dir ./test_set/ --output-dir ./viz_results/
```

### Output Files

All visualizations are saved to `visualizations/` directory:

```
visualizations/
├── field_accuracy_heatmap_26fields.png     # Dynamic heatmap
├── model_performance_dashboard.png         # 2x2 performance dashboard  
├── field_category_analysis.png            # Category-based analysis
├── v100_vram_usage_comparison.png         # V100 VRAM compliance chart
└── dynamic_model_comparison_report.html   # HTML summary report
```

### Dynamic Configuration Benefits

The visualization system exemplifies the "no hardcoding" principle:

```yaml
# All visualization behavior driven by model_comparison.yaml
expected_fields:
  - DOCUMENT_TYPE    # Automatically included in all charts
  - SUPPLIER         # Field discovery from extraction_prompt
  - ABN              # No code changes needed for new fields

field_weights:
  DOCUMENT_TYPE: 1.2  # High priority (gold border in heatmap)
  SUPPLIER: 1.0       # Standard priority
  CARD_NUMBER: 0.8    # Lower priority

quality_thresholds:
  excellent: 12       # Green threshold line in dashboard
  good: 8            # Used for color coding
  
speed_thresholds:
  very_fast: 15.0    # Performance benchmark lines
  fast: 25.0         # Automatically added to speed charts
```

### Integration with Model Comparison

The visualization system works seamlessly with the unified workflow:

```bash
# Complete workflow (recommended)
python model_comparison.py compare
python model_comparison.py visualize

# One-step visualization from stored results
python model_comparison.py visualize --ground-truth-csv evaluation_ground_truth.csv
```

### Professional Output

All charts use consistent professional styling:
- Clean color schemes with accessibility considerations
- Proper typography and spacing
- High-resolution PNG output (300 DPI)
- Responsive sizing for different field counts
- HTML summary reports with embedded visualizations

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

## 📊 Ground Truth Dataset Usage

The system uses a comprehensive ground truth evaluation dataset to provide objective, quantitative performance metrics for model comparison.

### **Current Ground Truth Dataset**
- **Location**: `/synthetic_receipt_generator_with_ground_truth/evaluation_data/evaluation_ground_truth.csv`
- **Size**: 20 synthetic invoices/receipts with corresponding PNG images
- **Format**: CSV with `image_file` column + 25 field columns
- **Coverage**: Synthetic Australian business documents (invoices, receipts, bank statements)

### **How Ground Truth is Used**

#### **1. Data Loading**
```python
# Loads CSV file with image_file as key, all other columns as field values
ground_truth = {}
with Path(self.ground_truth_csv).open("r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        image_file = row.pop("image_file")  # Use image filename as key
        # Convert numeric fields (GST, TOTAL, SUBTOTAL) to float
        ground_truth[image_file] = row
```

#### **2. Field-by-Field Accuracy Calculation**

**Different comparison strategies per field type:**

- **Numeric Fields (GST, TOTAL, SUBTOTAL)**:
  - Converts to float, allows 0.01 tolerance
  - `return 1.0 if abs(extracted_num - ground_truth_num) < 0.01 else 0.0`

- **List Fields (ITEMS, QUANTITIES, PRICES)**:
  - Splits on `|` separator, compares item-by-item
  - `matches / total_items` accuracy ratio

- **Date Fields**:
  - Extracts numeric date components, exact match required
  - `return 1.0 if extracted_date == gt_date else 0.0`

- **String Fields (Default)**:
  - Exact match = 1.0 accuracy
  - Partial match (substring) = 0.8 accuracy  
  - No match = 0.0 accuracy

#### **3. Per-Image Evaluation**
```python
# For each test image:
gt_data = self.ground_truth.get(image_file, {})  # Get ground truth row
extracted_data = result.extracted_fields        # Get model extraction results

# Calculate accuracy for each field
for field in self.extraction_fields:
    gt_value = gt_data.get(field, "")
    ext_value = extracted_data.get(field, "")
    field_accuracies[field] = self._calculate_field_accuracy(ext_value, gt_value, field)

# Overall accuracy = average of all field accuracies
overall_accuracy = sum(field_accuracies.values()) / len(field_accuracies)
```

#### **4. Aggregate Metrics Generation**
- **Per-Model Averages**: Accuracy, processing time, fields extracted
- **Field-wise Performance**: Individual accuracy per field across all images  
- **Success Rates**: Percentage of images processed without errors
- **Comparative Analysis**: Side-by-side model performance

### **Current Evaluation Workflow**

```bash
# Generate evaluation with ground truth
python model_comparison.py visualize --ground-truth-csv evaluation_ground_truth.csv
```

**What happens:**
1. **Loads 20 synthetic images** from evaluation dataset
2. **Runs both models** (Llama-3.2-Vision + InternVL3) on each image
3. **Compares extracted fields** against ground truth values  
4. **Calculates field-wise accuracy** using specialized comparison logic
5. **Generates visualizations** showing model performance differences

### **Ground Truth Dataset Structure**

**Example record:**
```csv
image_file,DOCUMENT_TYPE,SUPPLIER,ABN,TOTAL,GST,...
synthetic_invoice_001.png,RECEIPT,ALDI Australia,74 878 226 893,$46.68,$4.24,...
```

**Field Coverage (25 fields):**
- **Document metadata**: DOCUMENT_TYPE, SUPPLIER, ABN
- **Financial data**: TOTAL, GST, SUBTOTAL, QUANTITIES, PRICES  
- **Contact info**: PAYER_NAME, PAYER_ADDRESS, BUSINESS_PHONE
- **Banking data**: BANK_NAME, BSB_NUMBER, ACCOUNT_HOLDER (for bank statements)
- **Dates**: INVOICE_DATE, DUE_DATE

### **Evaluation Benefits**

✅ **Quantitative Accuracy**: Precise field-by-field accuracy measurement  
✅ **Model Comparison**: Direct performance comparison between models  
✅ **Field Analysis**: Identifies which fields each model handles best  
✅ **Synthetic Data**: Controlled, consistent test dataset  
✅ **Specialized Logic**: Handles numeric tolerance, list matching, date parsing

The ground truth evaluation provides objective, measurable performance metrics rather than subjective assessment, enabling data-driven model selection and optimization.

## 🧪 Testing & Validation

```bash
# Validate environment and models
python model_comparison.py check-environment
python model_comparison.py validate-models

# Run model comparison
python model_comparison.py compare

# Generate performance evaluation and visualizations
python model_comparison.py visualize --ground-truth-csv evaluation_ground_truth.csv

# Test single document extraction (legacy)
python -m vision_processor.cli.extract_cli extract datasets/image14.png --model llama
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

### Architecture Simplification (Latest)
- ✅ **Unified Command Interface**: All functionality through `model_comparison.py` script
- ✅ **Simplified Workflow**: Intuitive `compare` → `visualize` command structure
- ✅ **Eliminated Complex Paths**: No more `python -m vision_processor.cli.evaluation_cli`
- ✅ **KFP Integration**: Streamlined pipeline commands for production deployment
- ✅ **Smart Logic**: Visualizations auto-detect existing results vs. running fresh evaluation

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