# Vision Processor CLI Usage Guide

Complete reference for using the Vision Processor command-line interfaces with real examples and workflows.

## Table of Contents

- [Overview](#overview)
- [Main Model Comparison CLI](#main-model-comparison-cli)
- [Evaluation CLI](#evaluation-cli)
- [Configuration](#configuration)
- [Common Workflows](#common-workflows)
- [Troubleshooting](#troubleshooting)

## Overview

The Vision Processor provides two main CLI tools:

1. **`model_comparison.py`** - Primary interface for model comparison and visualization
2. **`vision_processor.cli.evaluation_cli`** - Detailed evaluation and ground truth analysis

All examples use paths from the default configuration in `model_comparison.yaml`.

## Main Model Comparison CLI

### Basic Commands

#### Run Model Comparison
```bash
# Compare all configured models on default dataset
python model_comparison.py compare

# Compare with custom dataset path
python model_comparison.py compare \
  --datasets-path /home/jovyan/nfs_share/tod/datasets \
  --output-dir /home/jovyan/nfs_share/tod/output

# Compare specific models only
python model_comparison.py compare --models "llama,internvl"

# Compare with custom parameters
python model_comparison.py compare \
  --datasets-path /path/to/images \
  --output-dir /path/to/results \
  --models "llama" \
  --max-tokens 1024 \
  --quantization \
  --trust-remote-code
```

#### Generate Visualizations
```bash
# Generate all visualizations from existing results
python model_comparison.py visualize \
  --ground-truth-csv /path/to/ground_truth.csv \
  --output-dir /home/jovyan/nfs_share/tod/output

# Generate visualizations from specific images directory
python model_comparison.py visualize \
  --images-dir /home/jovyan/nfs_share/tod/datasets \
  --output-dir /home/jovyan/nfs_share/tod/output

# Generate with verbose output
python model_comparison.py visualize \
  --ground-truth-csv /path/to/ground_truth.csv \
  --verbose
```

#### Environment Validation
```bash
# Check system environment and dependencies
python model_comparison.py check-environment

# Check with specific dataset path
python model_comparison.py check-environment \
  --datasets-path /home/jovyan/nfs_share/tod/datasets

# Check with verbose output
python model_comparison.py check-environment --verbose
```

#### Model Management
```bash
# List available models
python model_comparison.py list-models

# Validate all configured models
python model_comparison.py validate-models

# Validate with specific config file
python model_comparison.py validate-models \
  --config-path /path/to/custom_config.yaml

# Show field schema
python model_comparison.py show-schema
```

### Advanced Options

#### Logging Control
```bash
# Verbose output
python model_comparison.py compare --verbose

# Debug mode (maximum detail)
python model_comparison.py compare --debug

# Quiet mode (minimal output)
python model_comparison.py compare --quiet
```

#### Custom Model Paths
```bash
# Override model paths
python model_comparison.py compare \
  --llama-path /custom/path/to/Llama-3.2-11B-Vision-Instruct \
  --internvl-path /custom/path/to/InternVL3-8B
```

#### Configuration Override
```bash
# Use custom configuration file
python model_comparison.py compare \
  --config-path /path/to/custom_config.yaml
```

## Evaluation CLI

The evaluation CLI provides a structured 3-step workflow for ground truth analysis.

### Step 1: Validate Ground Truth

```bash
# Validate ground truth CSV against images
python -m vision_processor.cli.evaluation_cli validate-ground-truth \
  /home/jovyan/nfs_share/tod/datasets/ground_truth.csv

# Validate with custom images directory
python -m vision_processor.cli.evaluation_cli validate-ground-truth \
  /path/to/ground_truth.csv \
  --images-dir /path/to/images

# Validate with verbose output
python -m vision_processor.cli.evaluation_cli validate-ground-truth \
  /path/to/ground_truth.csv \
  --verbose
```

### Step 2: Compare Models

```bash
# Compare all models against ground truth
python -m vision_processor.cli.evaluation_cli compare \
  /home/jovyan/nfs_share/tod/datasets/ground_truth.csv

# Compare specific models
python -m vision_processor.cli.evaluation_cli compare \
  /path/to/ground_truth.csv \
  --models "llama,internvl" \
  --images-dir /path/to/images \
  --output-dir /path/to/results

# Compare without visualizations
python -m vision_processor.cli.evaluation_cli compare \
  /path/to/ground_truth.csv \
  --no-visualizations

# Compare with verbose output
python -m vision_processor.cli.evaluation_cli compare \
  /path/to/ground_truth.csv \
  --verbose
```

### Step 3: Generate Visualizations

```bash
# Auto-detect results file and generate all visualizations
python -m vision_processor.cli.evaluation_cli visualize

# Generate from specific results file
python -m vision_processor.cli.evaluation_cli visualize \
  /home/jovyan/nfs_share/tod/output/comparison_results.json

# Generate with ground truth for accuracy analysis
python -m vision_processor.cli.evaluation_cli visualize \
  /home/jovyan/nfs_share/tod/output/comparison_results.json \
  --ground-truth-csv /path/to/ground_truth.csv

# Generate only specific visualization types
python -m vision_processor.cli.evaluation_cli visualize \
  /path/to/results.json \
  --format heatmaps

python -m vision_processor.cli.evaluation_cli visualize \
  /path/to/results.json \
  --format charts

python -m vision_processor.cli.evaluation_cli visualize \
  /path/to/results.json \
  --format html

# Generate all formats
python -m vision_processor.cli.evaluation_cli visualize \
  /path/to/results.json \
  --format all
```

### Optional: Benchmark Single Model

```bash
# Benchmark default model
python -m vision_processor.cli.evaluation_cli benchmark \
  /home/jovyan/nfs_share/tod/datasets

# Benchmark specific model
python -m vision_processor.cli.evaluation_cli benchmark \
  /path/to/images \
  --model llama \
  --iterations 5

# Benchmark with custom output
python -m vision_processor.cli.evaluation_cli benchmark \
  /path/to/images \
  --model internvl \
  --output-file /path/to/benchmark_results.json \
  --verbose
```

## Configuration

### Default Paths (from model_comparison.yaml)

```yaml
defaults:
  datasets_path: "/home/jovyan/nfs_share/tod/datasets"
  output_dir: "/home/jovyan/nfs_share/tod/output"
  models: "llama,internvl"
  max_tokens: 2048
  quantization: true
  trust_remote_code: true

model_paths:
  llama: "/home/jovyan/nfs_share/models/Llama-3.2-11B-Vision-Instruct"
  internvl: "/home/jovyan/nfs_share/models/InternVL3-8B"
```

### Environment Variables

```bash
# Set up conda environment
conda activate unified_vision_processor

# Optional: Set CUDA device
export CUDA_VISIBLE_DEVICES=0
```

## Common Workflows

### Complete Evaluation Workflow

```bash
# 1. Validate your data
python -m vision_processor.cli.evaluation_cli validate-ground-truth \
  /home/jovyan/nfs_share/tod/datasets/ground_truth.csv

# 2. Run comparison
python -m vision_processor.cli.evaluation_cli compare \
  /home/jovyan/nfs_share/tod/datasets/ground_truth.csv

# 3. Generate visualizations
python -m vision_processor.cli.evaluation_cli visualize
```

### Quick Model Testing

```bash
# Test all models on dataset
python model_comparison.py compare

# Generate visualizations from results
python model_comparison.py visualize \
  --ground-truth-csv /path/to/ground_truth.csv
```

### Single Model Analysis

```bash
# Compare only one model
python model_comparison.py compare --models "llama"

# Benchmark specific model
python -m vision_processor.cli.evaluation_cli benchmark \
  /home/jovyan/nfs_share/tod/datasets \
  --model llama \
  --iterations 3
```

### Custom Dataset Processing

```bash
# Process custom dataset
python model_comparison.py compare \
  --datasets-path /path/to/custom/images \
  --output-dir /path/to/custom/output

# Validate custom ground truth
python -m vision_processor.cli.evaluation_cli validate-ground-truth \
  /path/to/custom/ground_truth.csv \
  --images-dir /path/to/custom/images
```

## Output Files

### Model Comparison Outputs

Generated in the configured output directory:

- **`production_results.csv`** - Extraction results in CSV format
- **`comparison_results.json`** - Complete comparison results
- **`comparison_results_full.json`** - Detailed results with metadata
- **`vision_processor.log`** - System logs

### Visualization Outputs

Generated in `{output_dir}/visualizations/`:

- **`field_accuracy_heatmap_25fields.png`** - Field accuracy heatmap
- **`model_performance_dashboard.png`** - Performance comparison charts
- **`interactive_report.html`** - Interactive HTML report
- **`model_comparison_report.html`** - Detailed comparison report

### Evaluation Outputs

Generated by evaluation CLI:

- **`evaluation_report.md`** - Markdown evaluation report
- **`benchmark_results.json`** - Performance benchmarking results
- **`field_analysis.json`** - Detailed field-by-field analysis

## Troubleshooting

### Common Issues

#### Model Not Found
```bash
# Check available models
python model_comparison.py list-models

# Validate model paths
python model_comparison.py validate-models
```

#### Dataset Issues
```bash
# Check dataset directory
python model_comparison.py check-environment \
  --datasets-path /your/dataset/path

# Validate ground truth
python -m vision_processor.cli.evaluation_cli validate-ground-truth \
  /path/to/ground_truth.csv
```

#### Memory Issues
```bash
# Use quantization for lower memory usage
python model_comparison.py compare --quantization

# Process with single model
python model_comparison.py compare --models "llama"
```

#### Configuration Issues
```bash
# Check current configuration
python model_comparison.py show-schema

# Validate environment
python model_comparison.py check-environment --verbose
```

### Debug Mode

Enable debug output for detailed troubleshooting:

```bash
# Debug model comparison
python model_comparison.py compare --debug

# Debug evaluation
python -m vision_processor.cli.evaluation_cli compare \
  /path/to/ground_truth.csv \
  --debug

# Debug visualization
python -m vision_processor.cli.evaluation_cli visualize --debug
```

### Log Files

Check log files for detailed error information:

```bash
# View current logs
tail -f /home/jovyan/nfs_share/tod/output/vision_processor.log

# Check for specific errors
grep "ERROR" /home/jovyan/nfs_share/tod/output/vision_processor.log
```

## Help Commands

Get help for any command:

```bash
# Main CLI help
python model_comparison.py --help

# Specific command help
python model_comparison.py compare --help
python model_comparison.py visualize --help

# Evaluation CLI help
python -m vision_processor.cli.evaluation_cli --help
python -m vision_processor.cli.evaluation_cli compare --help
python -m vision_processor.cli.evaluation_cli visualize --help
```

---

For more information, see:
- [Model Evaluation Documentation](model_evaluation_with_synthetic_data.md)
- [Configuration Reference](../model_comparison.yaml)
- [Troubleshooting Guide](troubleshooting.md)