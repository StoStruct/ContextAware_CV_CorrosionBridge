#!/usr/bin/env bash

set -eo pipefail

# Get the project root directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Use the environment 
source "$PROJECT_ROOT/mask2former-env/bin/activate"

# Set environment variables for detectron2
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/Mask2Former:$PYTHONPATH"
export DETECTRON2_DATASETS="$PROJECT_ROOT/datasets"

# Configuration
CONFIG_FILE="$PROJECT_ROOT/configs/maskformer2_swin_base_semantic.yaml"
MODEL_WEIGHTS="$PROJECT_ROOT/output/semantic_segmentation/model_final.pth"
# DATASET variable removed - Python script now handles both val and test
OUTPUT_DIR="$PROJECT_ROOT/output/semantic_segmentation/evaluation"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run comprehensive evaluation
echo "Running comprehensive semantic segmentation evaluation for (val, test)..."
python "$PROJECT_ROOT/src/eval_semantic.py" \
    --config-file "$CONFIG_FILE" \
    --model-weights "$MODEL_WEIGHTS" \
    --output-dir "$OUTPUT_DIR" \
    --model-name "Mask2Former-Swin-Base-Semantic" \
    --opts MODEL.WEIGHTS "$MODEL_WEIGHTS"

# Generate HTML report if pandoc is available
for SPLIT in "val" "test"; do
    DATASET="corrosion_${SPLIT}"
    SPLIT_OUTPUT_DIR="$OUTPUT_DIR/$DATASET"
    
    if [ -f "$SPLIT_OUTPUT_DIR/evaluation_summary.md" ]; then
        if command -v pandoc >/dev/null 2>&1; then
            echo "Generating HTML report for $DATASET..."
            pandoc "$SPLIT_OUTPUT_DIR/evaluation_summary.md" -o "$SPLIT_OUTPUT_DIR/evaluation_summary.html" --standalone --metadata title="Mask2Former Semantic ($DATASET) Evaluation Report"
            echo "HTML report generated at $SPLIT_OUTPUT_DIR/evaluation_summary.html"
        else
            echo "Pandoc not found. Skipping HTML report generation for $DATASET."
            echo "To generate an HTML report, install pandoc: sudo apt-get install pandoc"
        fi
    fi
done


echo "====================================="
echo "Evaluation complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "====================================="

# Display a summary of results
for SPLIT in "val" "test"; do
    DATASET="corrosion_${SPLIT}"
    SPLIT_OUTPUT_DIR="$OUTPUT_DIR/$DATASET"
    
    if [ -f "$SPLIT_OUTPUT_DIR/evaluation_summary.md" ]; then
        echo ""
        echo "Summary of Results ($DATASET):"
        echo "-------------------"
        cat "$SPLIT_OUTPUT_DIR/evaluation_summary.md"
    fi
done