#!/bin/bash

# unified_setup.sh - Simplified Vision Processor Setup Script
# Usage: source unified_setup.sh [working_directory] [conda_env_name]
#
# This script sets up the simplified vision processor environment for:
# - Mac M1 (local development)
# - 2x H200 GPU system (development/training) 
# - Single V100 GPU (production target)
# - Simplified single-step processing with .env configuration

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
echo "🔬 Simplified Vision Document Processing System"
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

# Load .env file if it exists (for model paths and configuration)
if [ -f ".env" ]; then
    # Check if .env has export statements
    if grep -q "^export" .env; then
        source .env
        echo "✅ Sourced .env file with configuration"
    else
        echo "⚠️ .env file found but no export statements"
        echo "   The CLI will still load .env variables automatically"
    fi
else
    echo "⚠️ No .env file found"
    echo "   Create one with your model paths and configuration:"
    echo "   cat >> .env << 'EOF'"
    echo "   # Model configuration"
    echo "   VISION_MODEL_TYPE=internvl3"
    echo "   VISION_MODEL_PATH=/path/to/InternVL3-8B"
    echo "   VISION_DEVICE_CONFIG=auto"
    echo "   VISION_OUTPUT_FORMAT=yaml"
    echo "   VISION_ENABLE_QUANTIZATION=true"
    echo "   VISION_OFFLINE_MODE=true"
    echo "   # Optional: export PYTHONPATH for persistence"
    echo "   export PYTHONPATH=$(pwd):\\\$PYTHONPATH"
    echo "   EOF"
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

# Set up useful aliases for simplified vision processor
alias svp-extract='python -m vision_processor.cli.simple_extract_cli extract'
alias svp-batch='python -m vision_processor.cli.simple_extract_cli batch'
alias svp-compare='python -m vision_processor.cli.simple_extract_cli compare'
alias svp-config='python -m vision_processor.cli.simple_extract_cli config-info'
alias svp-help='python -m vision_processor.cli.simple_extract_cli --help'

echo ""
echo "✅ Set up CLI shortcuts:"
echo "   - svp-extract:  Extract from single document"
echo "   - svp-batch:    Batch process directory"
echo "   - svp-compare:  Compare models"
echo "   - svp-config:   Show configuration"
echo "   - svp-help:     Show help"

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
echo "🎯 Quick Start:"
echo "   # Extract from a single document"
echo "   svp-extract datasets/image25.png --model internvl3"
echo ""
echo "   # Batch process directory"
echo "   svp-batch datasets/ --output-dir results/ --model internvl3"
echo ""
echo "   # Compare models on single document"
echo "   svp-compare datasets/image25.png --models internvl3,llama32_vision"
echo ""
echo "   # Show current configuration"
echo "   svp-config"
echo ""
echo "📋 Current Environment:"
echo "   - Working directory: $(pwd)"
echo "   - Python: $(which python)"
echo "   - Conda environment: $CONDA_ENV"
echo "   - PYTHONPATH: $PYTHONPATH"

echo ""
echo "========================================================"
echo "🚀 Simplified Vision Processor Ready!"
echo "========================================================"
echo "Remember to run with 'source' to preserve environment:"
echo "source unified_setup.sh [directory] [environment]"
echo ""
echo "Test the setup with:"
echo "svp-config  # Show configuration"
echo "python test_simple_extraction.py  # Run tests"
echo "========================================================"

alias runvision='git pull && reset && rm output_????????_??????.txt 2>/dev/null; python model_comparison.py compare \
  --datasets-path ./datasets --output-dir ./results --models llama,internvl | tee "output_$(date +%Y%m%d_%H%M%S).txt"'
