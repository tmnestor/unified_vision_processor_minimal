#!/bin/bash

# unified_setup.sh - Unified Vision Processor Setup Script
# Usage: source unified_setup.sh [working_directory] [conda_env_name]
#
# This script sets up the unified vision processor environment for:
# - Mac M1 (local development)
# - 2x H200 GPU system (development/training) 
# - Single V100 GPU (production target)

# Set permissions for SSH and Kaggle (if they exist)
[ -f "/home/jovyan/.ssh/id_ed25519" ] && chmod 600 /home/jovyan/.ssh/id_ed25519
[ -f "/home/jovyan/nfs_share/tod/.kaggle/kaggle.json" ] && chmod 600 /home/jovyan/nfs_share/tod/.kaggle/kaggle.json

# Default configuration for unified vision processor
DEFAULT_DIR="$HOME/nfs_share/tod/unified_vision_processor_minimal"
DEFAULT_ENV="unified_vision_processor"

# Parse arguments
WORK_DIR=${1:-$DEFAULT_DIR}
CONDA_ENV=${2:-$DEFAULT_ENV}

# Print header
echo "========================================================"
echo "🔬 Unified Vision Document Processing Architecture"
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
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
echo "✅ Added project to PYTHONPATH: $(pwd)"

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
    echo "   # Model paths"
    echo "   VISION_INTERNVL_MODEL_PATH=/path/to/InternVL3-8B"
    echo "   VISION_LLAMA_MODEL_PATH=/path/to/Llama-3.2-11B-Vision"
    echo "   # Configuration"
    echo "   VISION_MODEL_TYPE=internvl3"
    echo "   VISION_PROCESSING_PIPELINE=7step"
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
        echo "      Recommended: VISION_MULTI_GPU_DEV=true"
    elif [ "$GPU_MEMORY" -lt 20000 ]; then
        echo "   💡 Single GPU with limited memory - optimized for production"
        echo "      Recommended: VISION_ENABLE_8BIT_QUANTIZATION=true"
    fi
else
    echo "   CPU-only environment detected"
    echo "   💡 Consider using Mac M1 for code editing, GPU system for training"
fi

# Set up useful aliases for unified vision processor
alias uvp-process='python -m vision_processor.cli.unified_cli process'
alias uvp-batch='python -m vision_processor.cli.unified_cli batch'
alias uvp-compare='python -m vision_processor.cli.unified_cli compare'
alias uvp-evaluate='python -m vision_processor.cli.unified_cli evaluate'
alias uvp-help='python -m vision_processor.cli.unified_cli --help'

echo ""
echo "✅ Set up CLI shortcuts:"
echo "   - uvp-process:  Process single document"
echo "   - uvp-batch:    Batch process directory"
echo "   - uvp-compare:  Compare models"
echo "   - uvp-evaluate: SROIE evaluation"
echo "   - uvp-help:     Show help"

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
python -c "import cv2; print(f'   ✅ OpenCV: {cv2.__version__}')" 2>/dev/null || echo "   ❌ OpenCV not available"

# Check CUDA availability
if python -c "import torch; print(f'   ✅ CUDA available: {torch.cuda.is_available()}')" 2>/dev/null; then
    python -c "import torch; print(f'   ✅ CUDA devices: {torch.cuda.device_count()}')" 2>/dev/null
else
    echo "   ⚠️ CUDA not available or PyTorch not installed"
fi

echo ""
echo "🎯 Quick Start:"
echo "   # Process a single document"
echo "   uvp-process datasets/image25.png --model internvl3"
echo ""
echo "   # Batch process directory"
echo "   uvp-batch datasets/ --model internvl3 --output results/"
echo ""
echo "   # Compare models"
echo "   uvp-compare datasets/ ground_truth/ --models internvl3,llama32_vision"
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
echo "========================================================"