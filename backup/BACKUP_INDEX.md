# Legacy Files Backup

This directory contains legacy scripts, test files, and artifacts that were moved during the evaluation system reorganization.

## 📁 Moved Files

### Legacy Evaluation Scripts
- `add_ground_truth.py` - Interactive ground truth data entry tool
- `update_image_references.py` - Script to update image file references
- `EVALUATION_SYSTEM_DESIGN.md` - Original evaluation system design document

### Debug and Testing Scripts
- `debug_model_outputs.py` - Debug script for model output analysis
- `debug_llama_prompt.py` - Llama-specific prompt debugging
- `example_usage.py` - Legacy usage examples
- `minimal_model_test.ipynb` - Jupyter notebook for model testing

### Classification Scripts (Legacy)
- `compare_classification_performance.py` - Document classification comparison
- `test_individual_classification.py` - Individual classification testing
- `test_internvl_comparison.py` - InternVL comparison testing
- `classification_comparison_results.json` - Classification results data

### Legacy Directories
- `output/` - Old output directory structure
  - `comparisons/` - Model comparison results
  - `evaluations/` - Evaluation outputs
  - `predictions/` - Model predictions
  - `strategy_analysis/` - Analysis results
- `ground_truth/` - Old ground truth data structure

### Package Configuration (Incompatible with KFP)
- `setup.py` - Traditional package setup (moved due to KFP pipeline incompatibility)

## 🔄 Migration to New System

These files have been replaced by the new modular evaluation system:

### Old → New Structure
```
Legacy Files                     →  New Module Structure
─────────────────────────────────   ──────────────────────────────────
add_ground_truth.py             →  vision_processor/cli/evaluation_cli.py
update_image_references.py      →  Built into ExtractionEvaluator
debug_model_outputs.py          →  Built-in debug mode in evaluator
compare_classification_*        →  vision_processor/evaluation/
example_usage.py                →  evaluate_models.py
EVALUATION_SYSTEM_DESIGN.md     →  EVALUATION_README.md
```

### Key Improvements in New System
1. **Modular Architecture**: Proper Python package structure
2. **CLI Integration**: Professional command-line interface
3. **Better Organization**: Clear separation of concerns
4. **Enhanced Features**: Benchmarking, validation, reporting
5. **Production Ready**: Error handling, logging, documentation

## 🗂️ File Purposes (For Reference)

### `add_ground_truth.py`
- Interactive CLI for adding ground truth data
- **Replaced by**: CLI validation and management tools

### `debug_model_outputs.py`
- Simple script to test model outputs
- **Replaced by**: Built-in debug mode in `ExtractionEvaluator`

### `compare_classification_performance.py`
- Document classification comparison between models
- **Replaced by**: Comprehensive evaluation in `evaluator.py`

### `output/` Directory Structure
- Legacy file organization for results
- **Replaced by**: `evaluation_results/` with standardized structure

## 🚀 Using the New System

Instead of legacy scripts, use:

```bash
# Model comparison (replaces multiple legacy scripts)
python evaluate_models.py

# CLI interface (replaces add_ground_truth.py)
python -m vision_processor.cli.evaluation_cli compare evaluation_ground_truth.csv

# Validation (replaces update_image_references.py)
python -m vision_processor.cli.evaluation_cli validate-ground-truth evaluation_ground_truth.csv

# Benchmarking (new feature)
python -m vision_processor.cli.evaluation_cli benchmark datasets --model internvl3
```

## 📝 Notes

- All functionality from legacy scripts has been preserved and enhanced
- The new system provides better error handling and user experience
- Legacy files are kept for reference but should not be used going forward
- The new modular structure supports easier maintenance and extension

---

**Backup Created**: 2025-07-14  
**Reason**: Evaluation system reorganization into proper module structure  
**Status**: Legacy - use new vision_processor.evaluation module instead