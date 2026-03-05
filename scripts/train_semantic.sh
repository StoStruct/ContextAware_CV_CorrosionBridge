#!/usr/bin/env bash

set -euo pipefail

# Get project root directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "🚀 Starting Mask2Former Semantic Segmentation Training"
echo "Project root: $PROJECT_ROOT"

# Function to print colored output
print_status() { echo -e "\033[1;34m$1\033[0m"; }
print_success() { echo -e "\033[1;32m✅ $1\033[0m"; }
print_error() { echo -e "\033[1;31m❌ $1\033[0m"; }

# Check if virtual environment exists and activate it
if [ -d "$PROJECT_ROOT/mask2former-env" ]; then
    print_status "Activating Python environment..."
    source "$PROJECT_ROOT/mask2former-env/bin/activate"
else
    print_error "Virtual environment not found at $PROJECT_ROOT/mask2former-env"
    print_error "Please run setup.sh first to create the environment"
    exit 1
fi

# Set environment variables for detectron2 and Mask2Former
# Handle case where PYTHONPATH might not be set
if [ -z "${PYTHONPATH:-}" ]; then
    export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/Mask2Former"
else
    export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/Mask2Former:$PYTHONPATH"
fi
export DETECTRON2_DATASETS="$PROJECT_ROOT/datasets"

# Configuration
CONFIG_FILE="$PROJECT_ROOT/configs/maskformer2_swin_base_semantic.yaml"
OUTPUT_DIR="$PROJECT_ROOT/output/semantic_segmentation"
PRETRAINED_MODEL="$PROJECT_ROOT/models/maskformer2_swin_base_sem_seg_ade20k.pkl"

# Validate configuration file exists
if [ ! -f "$CONFIG_FILE" ]; then
    print_error "Config file not found: $CONFIG_FILE"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check if pretrained model exists
if [ ! -f "$PRETRAINED_MODEL" ]; then
    print_error "Pretrained model not found: $PRETRAINED_MODEL"
    echo "Please download the Mask2Former Swin-Base semantic segmentation pretrained weights:"
    echo "wget https://dl.fbaipublicfiles.com/maskformer/mask2former/ade20k/semantic/maskformer2_swin_base_IN21k_384_bs16_160k/model_final_be7c4e.pkl -O $PRETRAINED_MODEL"
    exit 1
fi

# Check GPU availability
if ! command -v nvidia-smi &> /dev/null; then
    print_error "nvidia-smi not found. GPU training may not work."
else
    print_status "GPU Status:"
    nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader,nounits | head -1
fi

print_status "Configuration Summary:"
print_success " Config file: $(basename "$CONFIG_FILE")"
print_success " Output directory: $OUTPUT_DIR"
print_success " Pretrained model: $(basename "$PRETRAINED_MODEL")"
print_success " Dataset path: $DETECTRON2_DATASETS/semantic"

# Check if semantic dataset exists
SEMANTIC_TRAIN_DIR="$DETECTRON2_DATASETS/semantic/train"
SEMANTIC_VAL_DIR="$DETECTRON2_DATASETS/semantic/val"

if [ ! -d "$SEMANTIC_TRAIN_DIR" ]; then
    print_error "Semantic training dataset not found: $SEMANTIC_TRAIN_DIR"
    echo "Please prepare your semantic segmentation dataset with the following structure:"
    echo "$DETECTRON2_DATASETS/semantic/"
    echo "├── train/"
    echo "│   ├── images/"
    echo "│   └── labelme_annotations/"
    echo "└── val/"
    echo "    ├── images/"
    echo "    └── labelme_annotations/"
    exit 1
fi

if [ ! -d "$SEMANTIC_VAL_DIR" ]; then
    print_error "Semantic validation dataset not found: $SEMANTIC_VAL_DIR"
    exit 1
fi

# Create training command
TRAINING_CMD=(
    python3 "$PROJECT_ROOT/src/train_semantic.py"
    --config-file "$CONFIG_FILE"
    --num-gpus 1
    --resume
    OUTPUT_DIR "$OUTPUT_DIR"
    MODEL.WEIGHTS "$PRETRAINED_MODEL"
)

# Start training
print_status "Starting semantic segmentation training..."
echo "Command: ${TRAINING_CMD[*]}"
echo ""

if "${TRAINING_CMD[@]}"; then
    echo ""
    print_success "Semantic segmentation training completed successfully!"
    print_success "Model saved to: $OUTPUT_DIR"
    echo ""
    echo "🎯 Next steps:"
    echo " • Monitor training: tail -f $OUTPUT_DIR/log.txt"
    echo " • View training plots: ls $OUTPUT_DIR/history/"
    echo " • View visualizations: ls $OUTPUT_DIR/inference/visualizations/"
    echo " • Evaluate model: Use evaluation scripts after training"
else
    echo ""
    print_error "Semantic segmentation training failed!"
    echo "Check the logs above for error details."
    exit 1
fi