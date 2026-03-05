#!/usr/bin/env bash
set -euo pipefail

# Get project root directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "🚀 Starting Mask2Former Training"
echo "Project root: $PROJECT_ROOT"

# Function to print colored output
print_status() { echo -e "\033[1;34m$1\033[0m"; }
print_success() { echo -e "\033[1;32m✅ $1\033[0m"; }
print_error() { echo -e "\033[1;31m❌ $1\033[0m"; }

# Default configuration
DEFAULT_CONFIG="configs/maskformer2_swin_base.yaml"
CONFIG_FILE=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --config-file)
      CONFIG_FILE="$2"
      shift 2
      ;;
    --help|-h)
      echo "Simple Mask2Former Training Script"
      echo ""
      echo "Usage: $0 [options]"
      echo ""
      echo "Options:"
      echo "  --config-file PATH   Path to configuration file"
      echo "  --help, -h          Show this help message"
      echo ""
      echo "Examples:"
      echo "  $0 --config-file configs/maskformer2_swin_base.yaml"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Set default config if not provided
if [ -z "$CONFIG_FILE" ]; then
  CONFIG_FILE="$DEFAULT_CONFIG"
  print_status "Using default config: $CONFIG_FILE"
fi

# Make config file path absolute
if [[ ! "$CONFIG_FILE" = /* ]]; then
  CONFIG_FILE="$PROJECT_ROOT/$CONFIG_FILE"
fi

# Basic validation
if [ ! -f "$CONFIG_FILE" ]; then
  print_error "Config file not found: $CONFIG_FILE"
  exit 1
fi

# Activate virtual environment if it exists
if [ -d "$PROJECT_ROOT/mask2former-env" ]; then
  print_status "Activating Python environment..."
  source "$PROJECT_ROOT/mask2former-env/bin/activate"
fi

# Set up output directory (simple timestamp-based)
OUTPUT_DIR="$PROJECT_ROOT/output/instance_segmentation"
mkdir -p "$OUTPUT_DIR"

print_status "Configuration:"
print_success "  Config file: $(basename "$CONFIG_FILE")"
print_success "  Output directory: $OUTPUT_DIR"

# Check if GPU is available
if ! command -v nvidia-smi &> /dev/null; then
  print_error "nvidia-smi not found. GPU training may not work."
else
  print_status "GPU Status:"
  nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader,nounits | head -1
fi

# Create training command
TRAINING_CMD=(
  python3 "$PROJECT_ROOT/src/train_net.py"
  --config-file "$CONFIG_FILE"
  --num-gpus 1
  --resume
  OUTPUT_DIR "$OUTPUT_DIR"
)

# Start training
print_status "Starting training..."
echo "Command: ${TRAINING_CMD[*]}"
echo ""

if "${TRAINING_CMD[@]}"; then
  echo ""
  print_success "Training completed successfully!"
  print_success "Model saved to: $OUTPUT_DIR"
  
  echo ""
  echo "🎯 Next steps:"
  echo "  • Monitor logs: tail -f $OUTPUT_DIR/log.txt"
  echo "  • Evaluate model: ./scripts/eval.sh --config-file $CONFIG_FILE"
  echo "  • View tensorboard: tensorboard --logdir $OUTPUT_DIR"
else
  echo ""
  print_error "Training failed!"
  echo "Check the logs above for error details."
  exit 1
fi