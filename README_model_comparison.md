# Model Comparison Script - Production Guide

## 🚀 Quick Start for KFP/V100 Production

This script converts the Jupyter notebook `model_comparison.ipynb` into a production-ready Python script optimized for V100 GPU environments and Kubeflow Pipelines (KFP).

### Essential Commands

```bash
# KFP Production Usage (most common)
python model_comparison.py compare \
  --datasets-path /mnt/input-data \
  --output-dir /mnt/output-results \
  --models llama,internvl

# Single model test
python model_comparison.py compare \
  --datasets-path /data/images \
  --output-dir /data/results \
  --models llama \
  --max-tokens 64

# Environment check before running
python model_comparison.py check-environment \
  --datasets-path /mnt/input-data
```

---

## 📋 Complete Usage Guide

### Required Parameters

- `--datasets-path`: **REQUIRED** - Path to directory containing PNG images
  - KFP: `/mnt/input-data`, `/data/images`
  - Local: `./datasets`, `~/data/receipts`

### Optional Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--output-dir` | `results` | Where to save all outputs |
| `--models` | `llama,internvl` | Comma-separated model list |
| `--max-tokens` | `64` | Max tokens for generation |
| `--quantization` | `True` | Enable 8-bit quantization |
| `--llama-path` | `/home/jovyan/nfs_share/models/Llama-3.2-11B-Vision` | Custom Llama path |
| `--internvl-path` | `/home/jovyan/nfs_share/models/InternVL3-8B` | Custom InternVL path |

---

## 🎯 Production Examples

### KFP Pipeline with Custom Models
```bash
python model_comparison.py compare \
  --datasets-path /mnt/input-data \
  --output-dir /mnt/output-results \
  --models llama,internvl \
  --max-tokens 64 \
  --quantization \
  --llama-path /models/Llama-3.2-11B-Vision \
  --internvl-path /models/InternVL3-8B
```

### Single Model Performance Test
```bash
python model_comparison.py compare \
  --datasets-path /data/receipts \
  --output-dir /data/analysis \
  --models internvl \
  --max-tokens 128 \
  --quantization false
```

### Environment Validation
```bash
python model_comparison.py check-environment \
  --datasets-path /mnt/data
```

---

## 📊 Output Files Generated

The script creates these files in your `--output-dir`:

### 1. Visualizations
- **`performance_analysis.png`** - 6-panel visualization:
  - Field detection rates (STORE, ABN, DATE, TOTAL)
  - F1 score heatmap
  - Inference time distribution
  - Success rate by document type
  - Core score distribution
  - Structured output rates

### 2. Data Files
- **`detailed_results.csv`** - Complete per-image results
- **`f1_scores.json`** - Precision, recall, F1 for each field
- **`summary_results.json`** - High-level performance metrics

### 3. Console Output
- Real-time progress tracking
- Memory usage monitoring
- Field-by-field detection results
- Final performance summary

---

## 🔧 Technical Details

### V100 Optimizations
- **8-bit quantization** reduces memory from ~22GB to ~11GB
- **Sequential model loading** prevents OOM errors
- **Aggressive memory cleanup** between models and images
- **Single GPU placement** avoids multi-GPU CUDA errors

### Australian Business Focus
- **ABN detection**: 11-digit patterns (XX XXX XXX XXX)
- **Date formats**: DD/MM/YYYY Australian standard
- **Currency**: AUD formatting ($X.XX)
- **Success criteria**: 2/3 core fields (STORE, DATE, TOTAL)

### Model Support
- **Llama 3.2-11B-Vision**: Uses proven prompt patterns to bypass safety mode
- **InternVL3-8B**: Comprehensive warning suppression
- **Both models**: Unified KEY-VALUE extraction prompt

---

## 🚨 Troubleshooting

### Common Issues and Solutions

#### 1. CUDA Out of Memory
```bash
# Enable quantization (default)
--quantization true

# Reduce token limit
--max-tokens 32

# Test single model first
--models llama
```

#### 2. Missing Dependencies
```bash
# Check environment first
python model_comparison.py check-environment

# Expected output:
# ✅ Pandas: 2.x.x
# ✅ Seaborn: 0.x.x
# ✅ Scikit-learn: 1.x.x
```

#### 3. No Images Found
```bash
# Verify datasets path
python model_comparison.py check-environment --datasets-path /your/path

# Expected: "✅ Datasets directory found: /your/path with X PNG files"
```

#### 4. Model Loading Failures
```bash
# Check custom model paths
--llama-path /correct/path/to/Llama-3.2-11B-Vision
--internvl-path /correct/path/to/InternVL3-8B

# Test single model
--models llama  # or --models internvl
```

#### 5. Llama Safety Mode Triggers
The script uses proven prompts that bypass Llama's safety restrictions:
- Short, structured prompts
- KEY-VALUE format requests
- `do_sample=False` for deterministic generation

---

## 📈 Understanding Results

### Success Metrics
- **Success Rate**: Documents with ≥2/3 core fields detected
- **Core Fields**: STORE, DATE, TOTAL (essential for business documents)
- **ABN Detection**: Bonus field, only expected in specific images
- **F1 Scores**: Precision/recall balance for each field

### Expected Performance
Based on previous testing:
- **InternVL3**: ~54.5% success rate, ~1.5s per document
- **Llama 3.2**: ~27.3% success rate, ~5-6s per document
- **Memory**: ~8-12GB with quantization, ~20GB without

### Production Recommendations
The script automatically recommends the best model based on:
1. Overall accuracy (success rate)
2. Inference speed
3. ABN detection performance
4. Memory efficiency

---

## 🛠️ Environment Setup

### Prerequisites
```bash
# Activate conda environment
conda activate unified_vision_processor

# Verify required packages
python -c "import torch, transformers, pandas, seaborn, sklearn"

# Check CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Expected Dataset Structure
```
/your/datasets/path/
├── image1.png
├── image14.png
├── image23.png
├── image39.png     # Expected to have ABN
├── image45.png
├── image65.png
├── image71.png     # Expected to have ABN
├── image74.png
├── image76.png     # Expected to have ABN
├── image205.png
└── ...
```

---

## 🔍 Debugging Mode

For detailed debugging, modify the script to add verbose logging:

```python
# Add to the top of the script
import logging
logging.basicConfig(level=logging.DEBUG)

# Or use rich console for detailed output
console.print(f"[DEBUG] Processing {img_name}...", style="dim")
```

---

## 📞 Emergency Recovery

If the script fails mid-execution:

1. **Check output directory**: Partial results may be saved
2. **Memory cleanup**: Restart Python kernel
3. **Single model test**: Use `--models llama` or `--models internvl`
4. **Environment check**: Run `check-environment` command
5. **Reduce scope**: Test with fewer images first

---

## 🎯 Performance Expectations

### V100 16GB Typical Performance
- **Model Loading**: 15-30 seconds per model
- **Inference**: 1-6 seconds per document
- **Memory Usage**: 8-12GB with quantization
- **Total Runtime**: 5-15 minutes for 10 documents
- **Output Size**: ~2-5MB (visualizations + data)

### KFP Pipeline Integration
The script is designed for seamless KFP integration:
- ✅ No interactive dependencies
- ✅ Configurable input/output paths
- ✅ Comprehensive error handling
- ✅ Structured JSON outputs
- ✅ Memory management for ephemeral environments

---

## 📚 Further Reading

- **Notebook Version**: `model_comparison.ipynb` (original interactive version)
- **CUDA Fixes**: `LLAMA_32_VISION_CUDA_ERRORS.md` (detailed error solutions)
- **Environment Setup**: `environment.yml` (conda dependencies)
