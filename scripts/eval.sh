#!/usr/bin/env bash
set -euo pipefail

# --- Final, Simplified Mask2Former Evaluation Script ---

# Get project root directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=================================================================="
echo "🔍 MASK2FORMER COMPREHENSIVE EVALUATION"
echo "=================================================================="

# --- Helper Functions for colored output ---
print_status() { echo -e "\033[1;34m$1\033[0m"; }
print_success() { echo -e "\033[1;32m✅ $1\033[0m"; }
print_warning() { echo -e "\033[1;33m⚠️ $1\033[0m"; }
print_error() { echo -e "\033[1;31m❌ $1\033[0m"; }

# --- Main Execution ---
print_status "🔧 Activating Python environment..."
if [ -f "$PROJECT_ROOT/mask2former-env/bin/activate" ]; then
    source "$PROJECT_ROOT/mask2former-env/bin/activate"
else
    print_error "Python environment not found. Run setup.sh first."
    exit 1
fi
print_success "Environment activated."

print_status "🔍 Looking for trained model in 'output/instance_segmentation/'..."
TRAINING_OUTPUT_DIR="$PROJECT_ROOT/output/instance_segmentation"

# Use model_final.pth, then fall back to model_best.pth
if [ -f "$TRAINING_OUTPUT_DIR/model_final.pth" ]; then
    MODEL_WEIGHTS="$TRAINING_OUTPUT_DIR/model_final.pth"
    print_success "Using final model: $MODEL_WEIGHTS"
elif [ -f "$TRAINING_OUTPUT_DIR/model_best.pth" ]; then
    MODEL_WEIGHTS="$TRAINING_OUTPUT_DIR/model_best.pth"
    print_warning "Could not find model_final.pth. Using best model instead: $MODEL_WEIGHTS"
else
    print_error "No 'model_final.pth' or 'model_best.pth' found in '$TRAINING_OUTPUT_DIR'."
    exit 1
fi

LOG_FILE="$TRAINING_OUTPUT_DIR/log.txt"
if [ ! -f "$LOG_FILE" ]; then
    print_warning "Could not find log.txt in $TRAINING_OUTPUT_DIR. Training curve plots will be skipped."
    touch "$LOG_FILE" # Create an empty file to prevent script failure
fi

CONFIG_FILE="$PROJECT_ROOT/configs/maskformer2_swin_base.yaml"
OUTPUT_DIR="$PROJECT_ROOT/output/instance_segmentation/evaluation"
mkdir -p "$OUTPUT_DIR"

print_status "🚀 Starting comprehensive evaluation on (val, test)..."
echo "Results will be saved to: $OUTPUT_DIR"

python3 "$PROJECT_ROOT/src/eval_net.py" \
    --config-file "$CONFIG_FILE" \
    --model-weights "$MODEL_WEIGHTS" \
    --output-dir "$OUTPUT_DIR" \
    --training-log "$LOG_FILE" \
    --opts MODEL.WEIGHTS "$MODEL_WEIGHTS"

if [ $? -eq 0 ]; then
    echo ""
    print_success "🎉 Comprehensive evaluation completed successfully!"
    print_success "📊 All plots and visualizations saved to: $OUTPUT_DIR"
    
    echo ""
    echo "=================================================================="
    print_status "📋 VALIDATION RESULTS (instance_val)"
    echo "=================================================================="
    if [ -f "$OUTPUT_DIR/instance_val/evaluation_summary_report.md" ]; then
        cat "$OUTPUT_DIR/instance_val/evaluation_summary_report.md"
    else
        print_error "Validation report not found at $OUTPUT_DIR/instance_val/evaluation_summary_report.md"
    fi
    
    echo ""
    echo "=================================================================="
    print_status "📋 TEST RESULTS (instance_test)"
    echo "=================================================================="
    if [ -f "$OUTPUT_DIR/instance_test/evaluation_summary_report.md" ]; then
        cat "$OUTPUT_DIR/instance_test/evaluation_summary_report.md"
    else
        print_error "Test report not found at $OUTPUT_DIR/instance_test/evaluation_summary_report.md"
    fi
    
    echo "=================================================================="
else
    echo ""
    print_error "❌ Evaluation script failed. Please check the errors above."
    echo "=================================================================="
    exit 1
fi