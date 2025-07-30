# Remote Testing Guide: Unified Vision Processor

## Overview

This guide provides comprehensive testing procedures for the Unified Vision Processor on your remote development environment (2x H200 GPU system). The testing follows a logical progression from basic environment validation to full model comparison workflows.

## Environment Context

- **Local Development**: Mac M1 (code editing, planning, git operations)
- **Remote Development**: 2x H200 GPU system (testing, training, evaluation)  
- **Production Target**: Single V100 GPU (16GB VRAM) with 64GB RAM
- **Synchronization**: Git workflow between local and remote environments

## Prerequisites

Before starting any tests, ensure you have:

1. **Repository Access**: Current unified CLI consolidation implementation
2. **Conda Environment**: `unified_vision_processor` environment activated
3. **Configuration**: `model_comparison.yaml` properly configured
4. **Model Paths**: InternVL3 and Llama-3.2-Vision models accessible
5. **Dataset**: Image files in configured datasets directory

## Testing Sequence

### Phase 1: Environment Validation

#### Step 1.1: System Check
**Purpose**: Validate the complete environment setup including hardware, dependencies, and configuration.

```bash
# Activate environment and run comprehensive system check
conda activate unified_vision_processor
python -m vision_processor check
```

**Expected Output Indicators**:
- ✅ CUDA available with H200 detection (139.7GB memory)
- ✅ GPU count: 2
- ✅ All required dependencies installed (PyTorch, Transformers, etc.)
- ✅ Configuration file loaded successfully
- ✅ Datasets directory exists with image count
- ✅ Model registry shows 2 models registered

**Troubleshooting**:
- If CUDA not detected: Check driver installation and environment
- If configuration missing: Verify `model_comparison.yaml` exists and is valid
- If datasets empty: Ensure images are in the configured datasets path

#### Step 1.2: Model Registry Validation
**Purpose**: Verify both models are properly registered and paths are accessible.

```bash
# List and validate all registered models
python -m vision_processor models --list --validate
```

**Expected Output Indicators**:
- ✅ 2 models registered (llama, internvl)
- ✅ Model descriptions and configurations shown
- ✅ Both models marked as "Available"
- ✅ Validation passes for both models

**Critical Issues to Watch**:
- ❌ "No model path configured" - indicates YAML configuration issue
- ❌ Path not accessible - model files missing or permissions issue
- ❌ "With issues: 2" - both models have configuration problems

#### Step 1.3: Configuration Display
**Purpose**: Verify all configuration sections are properly loaded and accessible.

```bash
# Display comprehensive configuration details
python -m vision_processor config --verbose
```

**Expected Output Indicators**:
- ✅ Configuration loaded successfully
- ✅ Model paths for both llama and internvl shown
- ✅ Device configuration with H200 detection
- ✅ Memory configuration with V100 production settings
- ✅ All configuration sections populated

#### Step 1.4: Field Schema Validation
**Purpose**: Confirm field extraction schema is properly configured.

```bash
# Display extraction field schema
python -m vision_processor schema
```

**Expected Output Indicators**:
- ✅ Field schema loaded with expected field count
- ✅ Fields displayed in table format
- ✅ Schema statistics showing field characteristics
- ✅ Usage recommendations provided

### Phase 2: Single Model Testing

#### Step 2.1: Llama Model Test
**Purpose**: Test Llama-3.2-Vision model in isolation to verify loading and basic functionality.

```bash
# Extract from single image using Llama model
python -m vision_processor extract /home/jovyan/nfs_share/tod/datasets/image14.png --model llama
```

**Expected Output Indicators**:
- ✅ Model loading completes successfully (< 60 seconds)
- ✅ V100 compliance validation passes
- ✅ 8-bit quantization applied for production compatibility
- ✅ Inference completes with processing time shown
- ✅ All 25+ fields extracted in KEY: VALUE format
- ✅ Confidence score displayed (from YAML configuration)

**Performance Expectations**:
- **Loading Time**: < 60 seconds on H200 system
- **Inference Time**: 15-60 seconds per image
- **Memory Usage**: < 13.5GB (within V100 limits)
- **Field Extraction**: 15-25 out of 25 possible fields

#### Step 2.2: InternVL Model Test
**Purpose**: Test InternVL3 model in isolation to verify loading and basic functionality.

```bash
# Extract from same image using InternVL model
python -m vision_processor extract /home/jovyan/nfs_share/tod/datasets/image14.png --model internvl
```

**Expected Output Indicators**:
- ✅ Model loading completes successfully
- ✅ Multi-GPU optimization applied (if available)
- ✅ Highlight detection support enabled
- ✅ Processing completes with performance metrics
- ✅ High-quality field extraction (typically > Llama)

**Performance Expectations**:
- **Loading Time**: < 45 seconds (typically faster than Llama)
- **Inference Time**: 10-30 seconds per image
- **Field Extraction**: 18-25 out of 25 possible fields
- **Confidence Score**: Higher average confidence than Llama

### Phase 3: Batch Processing

#### Step 3.1: Small Batch Test
**Purpose**: Test batch processing capabilities with a small dataset to validate pipeline stability.

```bash
# Process small batch (first few images) to test pipeline
python -m vision_processor batch /home/jovyan/nfs_share/tod/datasets --model llama --max-images 3
```

**Expected Output Indicators**:
- ✅ Batch processing initiates successfully
- ✅ Progress tracking shows individual image processing
- ✅ Results saved to configured output directory
- ✅ Summary statistics displayed
- ✅ No memory leaks or crashes during processing

**Performance Monitoring**:
- **Memory Stability**: No significant memory growth between images
- **Processing Speed**: Consistent timing across images
- **Error Rate**: < 10% failed images (network/file issues acceptable)

#### Step 3.2: Model Comparison Preparation
**Purpose**: Verify model comparison infrastructure before full comparison.

```bash
# Test model comparison with minimal dataset
python -m vision_processor compare --datasets-path /home/jovyan/nfs_share/tod/datasets --models llama,internvl --max-tokens 1024
```

**Expected Output Indicators**:
- ✅ Both models loaded successfully in sequence
- ✅ Comparison pipeline executes without errors
- ✅ Results saved in structured format
- ✅ Performance metrics calculated for both models
- ✅ Memory usage tracked and within limits

### Phase 4: Full System Integration

#### Step 4.1: Complete Model Comparison
**Purpose**: Execute full model comparison across entire dataset with comprehensive analysis.

```bash
# Run complete model comparison with logging
python -m vision_processor compare --verbose | tee "comparison_$(date +%Y%m%d_%H%M%S).log"
```

**Expected Output Indicators**:
- ✅ Full dataset processing for both models
- ✅ Comprehensive performance analysis
- ✅ Field extraction comparison statistics
- ✅ Processing speed benchmarks
- ✅ Memory usage analysis
- ✅ Results exported in multiple formats

**Performance Benchmarks**:
- **Dataset Processing**: All 24 images processed successfully
- **Model Performance**: Clear differentiation between models
- **Processing Speed**: Total runtime < 30 minutes for full comparison
- **Field Accuracy**: Both models achieve > 60% overall field extraction

#### Step 4.2: Visualization Generation
**Purpose**: Test visualization and reporting capabilities.

```bash
# Generate comprehensive visualizations from comparison results
python -m vision_processor visualize --browser
```

**Expected Output Indicators**:
- ✅ Charts generated successfully
- ✅ Model performance comparisons displayed
- ✅ Field extraction analysis visualized
- ✅ Memory usage charts created
- ✅ Browser opens with complete report (if available)

### Phase 5: Advanced Workflows

#### Step 5.1: Ground Truth Evaluation (if available)
**Purpose**: Test evaluation capabilities with ground truth data.

```bash
# Validate ground truth CSV format
python -m vision_processor validate-ground-truth /path/to/ground_truth.csv

# Run evaluation if ground truth is available
python -m vision_processor evaluate /path/to/ground_truth.csv
```

**Expected Output Indicators** (if ground truth available):
- ✅ Ground truth validation passes
- ✅ Evaluation metrics calculated
- ✅ Accuracy analysis per field
- ✅ Model performance comparison with statistical significance

#### Step 5.2: Performance Benchmarking
**Purpose**: Dedicated performance testing for optimization analysis.

```bash
# Run performance benchmark
python -m vision_processor benchmark /home/jovyan/nfs_share/tod/datasets --iterations 2
```

**Expected Output Indicators**:
- ✅ Multiple iterations complete successfully
- ✅ Consistent performance across iterations
- ✅ Detailed timing analysis
- ✅ Memory usage profiling
- ✅ Performance optimization recommendations

### Phase 6: System Utilities

#### Step 6.1: Results Processing
**Purpose**: Test utility functions for result analysis and conversion.

```bash
# Convert batch results to CSV format
python -m vision_processor convert /path/to/batch_results.json --output results_analysis.csv

# Display conversion info without saving
python -m vision_processor convert /path/to/batch_results.json --info
```

#### Step 6.2: Configuration Verification
**Purpose**: Final verification of all configuration aspects.

```bash
# Display complete configuration summary
python -m vision_processor config

# Show field schema for reference
python -m vision_processor schema --format json

# Final system check
python -m vision_processor check
```

## Test Results Validation

### Success Criteria

#### Environment Validation ✅
- [ ] All dependencies installed and accessible
- [ ] CUDA detection working with H200 hardware
- [ ] Configuration file loaded without errors
- [ ] Model registry shows both models available
- [ ] Dataset directory accessible with images

#### Model Functionality ✅
- [ ] Both models load successfully within time limits
- [ ] Single image extraction works for both models
- [ ] Field extraction produces expected format
- [ ] Memory usage stays within V100 limits (< 16GB)
- [ ] Processing times are reasonable (< 60s per image)

#### System Integration ✅
- [ ] Batch processing handles multiple images
- [ ] Model comparison completes successfully
- [ ] Results saved in correct formats
- [ ] Visualizations generate without errors
- [ ] All CLI commands execute properly

#### Performance Benchmarks ✅
- [ ] Full dataset processing < 30 minutes
- [ ] Field extraction rate > 60% overall
- [ ] Memory usage stable across batch processing
- [ ] Error rate < 10% for technical issues
- [ ] Consistent performance across multiple runs

### Common Issues and Solutions

#### Model Loading Issues
```bash
# Issue: CUDA out of memory
# Solution: Check quantization settings in YAML
# Verify: memory_config.v100_limit_gb and quantization: true

# Issue: Model path not found
# Solution: Verify model_paths in configuration
# Check: ls -la /home/jovyan/nfs_share/models/
```

#### Configuration Problems
```bash
# Issue: Configuration file not found
# Solution: Ensure model_comparison.yaml exists in working directory
# Check: pwd && ls -la model_comparison.yaml

# Issue: Device configuration errors
# Solution: Verify device_config section in YAML
# Check: GPU availability with nvidia-smi
```

#### Processing Failures
```bash
# Issue: Image processing errors
# Solution: Check image file formats and accessibility
# Verify: Supported formats (PNG, JPG, JPEG, WebP)

# Issue: Field extraction inconsistencies
# Solution: Review extraction_prompt in YAML configuration
# Check: Expected field names match schema
```

## Monitoring and Debugging

### Performance Monitoring
```bash
# Monitor GPU usage during processing
watch -n 1 nvidia-smi

# Check memory usage patterns
python -m vision_processor check --verbose

# Monitor processing logs
tail -f /path/to/output/vision_processor.log
```

### Debug Mode
```bash
# Enable debug logging for detailed analysis
python -m vision_processor compare --debug --verbose

# Check configuration in debug mode
python -m vision_processor config --debug
```

## Expected Test Duration

| Phase | Estimated Time | Critical Path |
|-------|---------------|---------------|
| Phase 1: Environment | 5-10 minutes | System validation |
| Phase 2: Single Models | 10-15 minutes | Model loading/testing |
| Phase 3: Batch Processing | 15-20 minutes | Pipeline validation |
| Phase 4: Full Integration | 20-30 minutes | Complete comparison |
| Phase 5: Advanced Workflows | 10-15 minutes | Optional features |
| Phase 6: Utilities | 5-10 minutes | Final verification |
| **Total Testing Time** | **65-100 minutes** | **Full validation** |

## Success Metrics

Upon completion of all testing phases, you should have:

1. **Validated Environment**: Complete system working with both models
2. **Performance Benchmarks**: Processing speed and accuracy metrics
3. **Comparison Results**: Detailed analysis of model performance
4. **Visual Reports**: Charts and graphs showing model comparison
5. **Production Readiness**: Confidence in V100 deployment capability

## Next Steps After Testing

1. **Performance Analysis**: Review benchmark results for optimization opportunities
2. **Model Selection**: Choose optimal model based on accuracy vs. speed requirements
3. **Production Deployment**: Use V100-compatible settings for production
4. **Documentation**: Update configuration based on testing results
5. **Monitoring Setup**: Implement production monitoring based on test insights

This comprehensive testing approach ensures the Unified Vision Processor is fully validated and ready for production deployment while maintaining compatibility with your target V100 hardware constraints.