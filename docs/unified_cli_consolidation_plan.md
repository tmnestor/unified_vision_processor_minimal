# Unified CLI Consolidation Plan

**Objective**: Eliminate CLI confusion by consolidating all command-line interfaces into a single, clean entry point: `python -m vision_processor <command>`

## Current State (Bloated)

**Multiple Confusing Entry Points:**
- `python model_comparison.py` (6 commands)
- `python -m vision_processor.cli.evaluation_cli` (4 commands)  
- `python -m vision_processor.cli.extract_cli` (4 commands)
- `python -m vision_processor.cli.batch_to_csv_cli` (1 command)

**Problems:**
- Dual interfaces for same functionality (compare, visualize)
- Inconsistent parameter patterns
- Scattered documentation
- User confusion about which CLI to use

## Target State (Clean)

**Single Entry Point:**
```bash
python -m vision_processor <command>
```

**Unified Command Structure:**

### Core Workflows
- `compare` - Model comparison with auto-detection of ground truth
- `visualize` - Generate charts/reports from results or ground truth
- `extract` - Single image extraction
- `batch` - Batch processing directory

### Evaluation Workflows  
- `evaluate` - Ground truth evaluation (current evaluation_cli compare)
- `benchmark` - Performance testing
- `validate-ground-truth` - CSV/image validation

### System Management
- `check` - Environment validation
- `models` - List/validate models  
- `config` - Show configuration
- `schema` - Show field schema

### Utilities
- `convert` - Batch results to CSV

## Configuration Philosophy

**Single Source of Truth: `model_comparison.yaml`**
- All defaults come from YAML configuration
- CLI parameters override YAML values
- No complex auto-detection or fallbacks
- Fail fast with explicit error messages

**Example Override Pattern:**
```bash
# Uses YAML defaults
python -m vision_processor compare

# Overrides YAML datasets_path
python -m vision_processor compare --datasets-path /custom/path

# Overrides YAML models
python -m vision_processor compare --models llama
```

## Implementation Plan

### Phase 1: Create Unified Entry Point

**Create `vision_processor/__main__.py`:**
```python
import typer
from .cli.unified_cli import app

if __name__ == "__main__":
    app()
```

**Create `vision_processor/cli/unified_cli.py`:**
- Single Typer app with all commands
- Import core logic from existing modules (not CLI wrappers)
- Consistent parameter handling
- ConfigManager-based configuration loading

### Phase 2: Migrate Core Functionality

**Smart Command Consolidation:**

1. **`compare` Command**
   ```bash
   # Auto-detects ground truth presence
   python -m vision_processor compare
   python -m vision_processor compare --ground-truth-csv file.csv  # Enables evaluation mode
   ```

2. **`visualize` Command** 
   ```bash  
   # Auto-detects input type
   python -m vision_processor visualize                            # Uses latest results
   python -m vision_processor visualize results.json              # From specific results
   python -m vision_processor visualize --ground-truth-csv gt.csv # Ground truth analysis
   ```

3. **All Other Commands**
   - Migrate logic from existing CLI modules
   - Standardize on ConfigManager for all configuration
   - Use consistent parameter names across commands

### Phase 3: Clean Break Implementation

**Delete Old CLI Files:**
- Remove `model_comparison.py` (script)
- Remove `vision_processor/cli/evaluation_cli.py`
- Remove `vision_processor/cli/extract_cli.py` 
- Remove `vision_processor/cli/batch_to_csv_cli.py`

**No Backward Compatibility:**
- No deprecation warnings
- No legacy command support
- Clean error messages if old commands attempted

**Fail Fast Approach:**
```bash
$ python model_comparison.py compare
ERROR: model_comparison.py has been removed.
USE: python -m vision_processor compare

$ python -m vision_processor.cli.evaluation_cli compare
ERROR: evaluation_cli has been removed.  
USE: python -m vision_processor evaluate ground_truth.csv
```

### Phase 4: Update All References

**Documentation Updates:**
- `docs/cli_usage_guide.md` - Replace with unified commands
- `README.md` - Update all CLI examples
- `unified_setup.sh` - Replace aliases with unified commands

**Alias Updates:**
```bash
# Old (remove)
alias vp-compare='python model_comparison.py compare'
alias vp-eval-compare='python -m vision_processor.cli.evaluation_cli compare'

# New (unified)
alias vp-compare='python -m vision_processor compare'
alias vp-evaluate='python -m vision_processor evaluate'
alias vp-visualize='python -m vision_processor visualize'
```

## Command Mapping

### From model_comparison.py
- `compare` → `python -m vision_processor compare`
- `visualize` → `python -m vision_processor visualize` 
- `check-environment` → `python -m vision_processor check`
- `list-models` → `python -m vision_processor models --list`
- `validate-models` → `python -m vision_processor models --validate`
- `show-schema` → `python -m vision_processor schema`

### From evaluation_cli.py
- `compare ground_truth.csv` → `python -m vision_processor evaluate ground_truth.csv`
- `visualize results.json` → `python -m vision_processor visualize results.json`
- `validate-ground-truth gt.csv` → `python -m vision_processor validate-ground-truth gt.csv`
- `benchmark images/` → `python -m vision_processor benchmark images/`

### From extract_cli.py  
- `extract image.png` → `python -m vision_processor extract image.png`
- `batch images/` → `python -m vision_processor batch images/`
- `config-info` → `python -m vision_processor config`

### From batch_to_csv_cli.py
- `convert batch.json` → `python -m vision_processor convert batch.json`

## Configuration Principles

**YAML-First Approach:**
1. Load `model_comparison.yaml` as base configuration
2. Apply CLI parameter overrides
3. Validate final configuration
4. Fail fast with clear error messages for missing/invalid config

**No Auto-Detection Bloat:**
- No complex fallback chains
- No "smart" path detection
- Explicit configuration required
- Clear error messages when config missing

**Example Error Messages:**
```bash
$ python -m vision_processor compare
ERROR: No datasets_path configured
FIX: Set datasets_path in model_comparison.yaml OR use --datasets-path

$ python -m vision_processor compare --models nonexistent  
ERROR: Model 'nonexistent' not found in configuration
AVAILABLE: llama, internvl
FIX: Update model_paths in model_comparison.yaml OR use valid model name
```

## Benefits

**User Experience:**
- Single command to remember: `python -m vision_processor`
- Consistent parameter patterns across all commands
- Clear, unified help system
- Logical command grouping

**Code Quality:**
- Eliminate duplicate functionality
- Remove legacy compatibility code
- Single configuration path (ConfigManager)
- Consistent error handling

**Maintenance:**
- One CLI to maintain instead of four
- Centralized documentation
- Simplified testing
- Reduced code bloat

## Implementation Timeline

1. **Week 1**: Create `unified_cli.py` with core commands (compare, visualize, extract)
2. **Week 1**: Add system commands (check, models, config, schema)  
3. **Week 2**: Add evaluation commands (evaluate, benchmark, validate-ground-truth)
4. **Week 2**: Add utility commands (batch, convert)
5. **Week 2**: Delete old CLI files and update all documentation
6. **Week 2**: Update unified_setup.sh and test complete system

## Success Criteria

- [x] Single entry point: `python -m vision_processor <command>`
- [x] All existing functionality accessible through unified interface
- [x] model_comparison.yaml as single source of truth
- [x] Clean parameter override system
- [x] No legacy code or backward compatibility
- [x] Clear, fail-fast error messages
- [x] Updated documentation and setup scripts
- [x] Simplified codebase with reduced bloat

---

**This plan eliminates CLI confusion, reduces codebase complexity, and creates a clean, maintainable unified interface.**