#!/usr/bin/env bash
#
# ==============================================================================
#  Mask2Former Semantic Segmentation Setup Script
# ==============================================================================
#  This script prepares the environment for the semantic segmentation part of
#  the project. It activates the existing virtual environment and downloads
#  the correct pre-trained Swin-Base model weights required for training.
# ==============================================================================

set -euo pipefail # Exit on error, undefined variable, or pipe failure

# --- Get Project Root Directory ---
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# --- Helper Functions for Colored Output ---
print_status() { echo -e "\033[1;34m$1\033[0m"; }
print_success() { echo -e "\033[1;32m ✅  $1\033[0m"; }
print_error() { echo -e "\033[1;31m ❌  $1\033[0m"; }

echo "============================================================"
echo " 🚀  Starting Setup for Semantic Segmentation"
echo "============================================================"
echo "Project root: $PROJECT_ROOT"
echo ""

# --- 1. Activate and Verify Existing Environment ---
print_status "[1/2] Activating and verifying existing Python environment..."

VENV_PATH="$PROJECT_ROOT/mask2former-env/bin/activate"
if [ -f "$VENV_PATH" ]; then
    source "$VENV_PATH"
    print_success "Virtual environment activated."
else
    print_error "Virtual environment not found at '$VENV_PATH'."
    print_error "Please run the main setup script (./scripts/setup.sh) first."
    exit 1
fi

# Quick verification
if python -c "import torch; assert torch.cuda.is_available()" && \
   python -c "import detectron2" &> /dev/null; then
    print_success "PyTorch, CUDA, and Detectron2 are correctly installed."
else
    print_error "Environment verification failed. Please re-run ./scripts/setup.sh."
    exit 1
fi

# --- 2. Download Pretrained Model Weights for Semantic Segmentation ---
print_status "[2/2] Downloading pretrained Swin-Base model weights..."

MODELS_DIR="$PROJECT_ROOT/models"
mkdir -p "$MODELS_DIR"

# CORRECTED: Swin-Base weights for ADE20K semantic segmentation
WEIGHTS_URL="https://dl.fbaipublicfiles.com/maskformer/mask2former/ade20k/semantic/maskformer2_swin_base_IN21k_384_bs16_160k_res640/model_final_7e47bf.pkl"
SEMANTIC_WEIGHTS_FILE="$MODELS_DIR/maskformer2_swin_base_sem_seg_ade20k.pkl"

if [ ! -f "$SEMANTIC_WEIGHTS_FILE" ]; then
    echo "      Downloading Mask2Former Swin-Base weights for semantic segmentation..."
    wget -q --show-progress -O "$SEMANTIC_WEIGHTS_FILE" "$WEIGHTS_URL"
    if [ $? -eq 0 ]; then
        print_success "Weights downloaded successfully to: $SEMANTIC_WEIGHTS_FILE"
    else
        print_error "Failed to download weights. Please check your internet connection or the URL."
        exit 1
    fi
else
    print_success "Swin-Base semantic segmentation weights already exist."
fi

echo ""
echo "============================================================"
echo " ✅  SEMANTIC SEGMENTATION SETUP COMPLETE"
echo "============================================================"
echo " The system is now ready for the corrosion detection task."
echo ""
echo " 🎯  Next steps:"
echo "   1. Prepare your semantic dataset (corrosion images and masks)."
echo "   2. Run the training script: ./scripts/train_semantic.sh"
echo "   3. Evaluate the trained model: ./scripts/eval_semantic.sh"
echo "   4. Analyze corrosion areas: ./scripts/calculate_corrosion_areas.sh"
echo "============================================================"