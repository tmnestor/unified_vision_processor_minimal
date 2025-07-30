#!/bin/bash

# unified_setup.sh - Vision Processor Setup Script
# Usage: source unified_setup.sh [working_directory] [conda_env_name]
#
# This script sets up the unified vision processor environment for:
# - Mac M1 (local development and planning)
# - 2x H200 GPU system (development/training) 
# - Single V100 GPU (production target)
# - YAML-based configuration with model comparison and evaluation pipelines

# Set permissions for SSH and Kaggle (if they exist)
[ -f "/home/jovyan/.ssh/id_ed25519" ] && chmod 600 /home/jovyan/.ssh/id_ed25519
[ -f "/home/jovyan/nfs_share/tod/.kaggle/kaggle.json" ] && chmod 600 /home/jovyan/nfs_share/tod/.kaggle/kaggle.json

# Configure git to use SSH instead of HTTPS for GitHub
if [ -f "/home/jovyan/.ssh/id_ed25519" ]; then
    echo "🔑 Setting up git SSH authentication..."
    
    # Set git remote to use SSH if currently using HTTPS
    CURRENT_REMOTE=$(git remote get-url origin 2>/dev/null || echo "")
    if [[ "$CURRENT_REMOTE" == https://github.com/* ]]; then
        SSH_REMOTE=$(echo "$CURRENT_REMOTE" | sed 's|https://github.com/|git@github.com:|')
        git remote set-url origin "$SSH_REMOTE"
        echo "✅ Updated git remote from HTTPS to SSH: $SSH_REMOTE"
    elif [[ "$CURRENT_REMOTE" == git@github.com:* ]]; then
        echo "✅ Git already configured for SSH: $CURRENT_REMOTE"
    fi
    
    # Test SSH connection
    if ssh -T git@github.com -o StrictHostKeyChecking=no -o ConnectTimeout=10 2>&1 | grep -q "successfully authenticated"; then
        echo "✅ SSH authentication to GitHub working"
    else
        echo "⚠️ SSH authentication test failed - you may need to add the SSH key to GitHub"
        echo "   Add this key to GitHub: https://github.com/settings/ssh/new"
        [ -f "/home/jovyan/.ssh/id_ed25519.pub" ] && echo "   Public key:" && cat /home/jovyan/.ssh/id_ed25519.pub
    fi
fi

# Default configuration for unified vision processor
DEFAULT_DIR="$HOME/nfs_share/tod/unified_vision_processor_minimal"
DEFAULT_ENV="unified_vision_processor"

# Parse arguments
WORK_DIR=${1:-$DEFAULT_DIR}
CONDA_ENV=${2:-$DEFAULT_ENV}

# Print header
echo "========================================================"
echo "🔬 Unified Vision Document Processing System"
echo "🚀 Setting up environment: $CONDA_ENV"
echo "========================================================"

# Change to working directory
if [ -d "$WORK_DIR" ]; then
    cd "$WORK_DIR"
    echo "✅ Changed directory to: $(pwd)"
else
    echo "❌ Error: Directory $WORK_DIR does not exist"
    echo "   Expected: unified_vision_processor project directory"
    return 1
fi

# Initialize conda
if [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/conda/etc/profile.d/conda.sh"
    echo "✅ Conda initialized"
    
    # Try to activate the conda environment
    if conda activate "$CONDA_ENV" 2>/dev/null; then
        echo "✅ Activated conda environment: $CONDA_ENV"
    else
        echo "⚠️ Conda environment '$CONDA_ENV' not found"
        echo "   Creating environment from environment.yml..."
        
        if [ -f "environment.yml" ]; then
            echo "📦 Installing dependencies (this may take a few minutes)..."
            if conda env create -f environment.yml; then
                echo "✅ Environment created successfully"
                conda activate "$CONDA_ENV"
                echo "✅ Activated new environment: $CONDA_ENV"
            else
                echo "❌ Failed to create environment from environment.yml"
                echo "   Available environments:"
                conda env list
                return 1
            fi
        else
            echo "❌ environment.yml not found in current directory"
            return 1
        fi
    fi
else
    echo "❌ Error: Conda initialization file not found"
    return 1
fi

# Set up PYTHONPATH for package access (no pip install needed)
# Append to existing PYTHONPATH to avoid overwriting other paths, but avoid duplicates
CURRENT_DIR="$(pwd)"
if [ -z "$PYTHONPATH" ]; then
    export PYTHONPATH="$CURRENT_DIR"
    echo "✅ Set PYTHONPATH to: $CURRENT_DIR"
elif [[ ":$PYTHONPATH:" != *":$CURRENT_DIR:"* ]]; then
    export PYTHONPATH="$CURRENT_DIR:$PYTHONPATH"
    echo "✅ Added project to PYTHONPATH: $CURRENT_DIR"
    echo "   Full PYTHONPATH: $PYTHONPATH"
else
    echo "✅ Project already in PYTHONPATH: $CURRENT_DIR"
    echo "   Current PYTHONPATH: $PYTHONPATH"
fi

# Check YAML configuration file
if [ -f "model_comparison.yaml" ]; then
    echo "✅ Found model_comparison.yaml configuration"
    echo "   All settings configured via YAML (no .env needed)"
    
    # Show key configuration paths
    DATASETS_PATH=$(python -c "import yaml; c=yaml.safe_load(open('model_comparison.yaml')); print(c.get('defaults', {}).get('datasets_path', 'Not set'))" 2>/dev/null || echo "Not set")
    OUTPUT_DIR=$(python -c "import yaml; c=yaml.safe_load(open('model_comparison.yaml')); print(c.get('defaults', {}).get('output_dir', 'Not set'))" 2>/dev/null || echo "Not set")
    
    echo "   📁 Datasets path: $DATASETS_PATH"
    echo "   📁 Output directory: $OUTPUT_DIR"
else
    echo "⚠️ No model_comparison.yaml found"
    echo "   This file contains all model paths and configuration"
    echo "   Required sections:"
    echo "   - defaults: datasets_path, output_dir, models"
    echo "   - model_paths: llama, internvl paths"
    echo "   - device_config: GPU and memory settings"
fi

# Detect hardware environment and suggest optimizations
echo ""
echo "🔍 Hardware Detection:"
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_COUNT=$(nvidia-smi -L | wc -l)
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    echo "   GPUs detected: $GPU_COUNT"
    echo "   GPU memory: ${GPU_MEMORY}MB"
    
    if [ "$GPU_COUNT" -gt 1 ]; then
        echo "   💡 Multi-GPU setup detected - optimized for development"
        echo "      Recommended: VISION_ENABLE_MULTI_GPU=true"
    elif [ "$GPU_MEMORY" -lt 20000 ]; then
        echo "   💡 Single GPU with limited memory - optimized for production"
        echo "      Recommended: VISION_ENABLE_QUANTIZATION=true"
    fi
else
    echo "   CPU-only environment detected"
    echo "   💡 Consider using Mac M1 for code editing, GPU system for training"
fi

# Set up useful aliases for vision processor workflows
alias vp-compare='python model_comparison.py compare'
alias vp-visualize='python model_comparison.py visualize'
alias vp-validate='python model_comparison.py validate-models'
alias vp-check='python model_comparison.py check-environment'
alias vp-models='python model_comparison.py list-models'

# Evaluation CLI aliases
alias vp-eval='python -m vision_processor.cli.evaluation_cli'
alias vp-eval-compare='python -m vision_processor.cli.evaluation_cli compare'
alias vp-eval-viz='python -m vision_processor.cli.evaluation_cli visualize'
alias vp-eval-validate='python -m vision_processor.cli.evaluation_cli validate-ground-truth'

# Quick comparison with logging
alias vp-run='python model_comparison.py compare --verbose | tee "output_$(date +%Y%m%d_%H%M%S).txt"'

echo ""
echo "✅ Set up CLI shortcuts:"
echo "   📊 Main Commands:"
echo "   - vp-compare:     Run model comparison"
echo "   - vp-visualize:   Generate charts and reports"
echo "   - vp-validate:    Validate model configurations"
echo "   - vp-check:       Check system environment"
echo "   - vp-models:      List available models"
echo ""
echo "   🔬 Evaluation Commands:"
echo "   - vp-eval:        Full evaluation CLI help"
echo "   - vp-eval-compare: Compare against ground truth"
echo "   - vp-eval-viz:    Generate visualizations"
echo "   - vp-eval-validate: Validate ground truth CSV"
echo ""
echo "   🚀 Quick Actions:"
echo "   - vp-run:         Compare with verbose logging"

# Verify installation
echo ""
echo "🔍 Verifying installation:"

# Check Python packages
if python -c "import vision_processor" 2>/dev/null; then
    echo "   ✅ vision_processor package accessible"
else
    echo "   ❌ vision_processor package not accessible"
    echo "      PYTHONPATH: $PYTHONPATH"
fi

# Check key dependencies
echo "   📦 Checking dependencies:"
python -c "import torch; print(f'   ✅ PyTorch: {torch.__version__}')" 2>/dev/null || echo "   ❌ PyTorch not available"
python -c "import transformers; print(f'   ✅ Transformers: {transformers.__version__}')" 2>/dev/null || echo "   ❌ Transformers not available"
python -c "import typer; print(f'   ✅ Typer: {typer.__version__}')" 2>/dev/null || echo "   ❌ Typer not available"
python -c "import yaml; print('   ✅ PyYAML: available')" 2>/dev/null || echo "   ❌ PyYAML not available"

# Check CUDA availability
if python -c "import torch; print(f'   ✅ CUDA available: {torch.cuda.is_available()}')" 2>/dev/null; then
    python -c "import torch; print(f'   ✅ CUDA devices: {torch.cuda.device_count()}')" 2>/dev/null
else
    echo "   ⚠️ CUDA not available or PyTorch not installed"
fi

echo ""
echo "🎯 Quick Start Examples:"
echo "   # Run complete model comparison"
echo "   vp-compare"
echo ""
echo "   # Compare with custom dataset"
echo "   vp-compare --datasets-path /path/to/images --models llama,internvl"
echo ""
echo "   # Generate visualizations from results"
echo "   vp-visualize --ground-truth-csv /path/to/ground_truth.csv"
echo ""
echo "   # Evaluation workflow (3 steps)"
echo "   vp-eval-validate ground_truth.csv"
echo "   vp-eval-compare ground_truth.csv"
echo "   vp-eval-viz"
echo ""
echo "   # Check system and models"
echo "   vp-check && vp-validate"
echo ""
echo "📋 Current Environment:"
echo "   - Working directory: $(pwd)"
echo "   - Python: $(which python)"
echo "   - Conda environment: $CONDA_ENV"
echo "   - PYTHONPATH: $PYTHONPATH"

echo ""
echo "========================================================"
echo "🚀 Unified Vision Processor Ready!"
echo "========================================================"
echo "Remember to run with 'source' to preserve environment:"
echo "source unified_setup.sh [directory] [environment]"
echo ""
echo "📖 Documentation:"
echo "   - CLI Guide: docs/cli_usage_guide.md"
echo "   - Model Evaluation: docs/model_evaluation_with_synthetic_data.md"
echo ""
echo "🔧 Test the setup:"
echo "   vp-check    # Validate environment"
echo "   vp-models   # List available models"
echo "   vp-compare  # Run comparison (if models configured)"
echo ""
echo "🚀 Production workflow:"
echo "   vp-run      # Compare with logging"
echo "========================================================"
