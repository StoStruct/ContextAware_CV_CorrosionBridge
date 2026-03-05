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
SEMANTIC_CONFIG="$PROJECT_ROOT/configs/maskformer2_swin_base_semantic.yaml"
SEMANTIC_WEIGHTS="$PROJECT_ROOT/output/semantic_segmentation/model_final.pth"
INSTANCE_CONFIG="$PROJECT_ROOT/configs/maskformer2_swin_base.yaml"
# Fix the path to the instance segmentation model weights
INSTANCE_WEIGHTS="$PROJECT_ROOT/output/instance_segmentation/model_final.pth"

OUTPUT_DIR="$PROJECT_ROOT/output/corrosion_area_analysis"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run corrosion area calculation for (val, test)
echo "Calculating corrosion area ratios and comparing with Ground Truth..."
python "$PROJECT_ROOT/src/calculate_corrosion_areas.py" \
  --semantic-config "$SEMANTIC_CONFIG" \
  --semantic-weights "$SEMANTIC_WEIGHTS" \
  --instance-config "$INSTANCE_CONFIG" \
  --instance-weights "$INSTANCE_WEIGHTS" \
  --output-dir "$OUTPUT_DIR" \
  --output-format "all" \
  --min-confidence 0.5 \
  --detailed-tables \
  --simple-visualization

echo "====================================="
echo "Area calculation complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "====================================="

# Display a summary of results for both splits
for SPLIT in "val" "test"; do
    DATASET="corrosion_${SPLIT}"
    SPLIT_OUTPUT_DIR="$OUTPUT_DIR/$DATASET"
    
    if [ -f "$SPLIT_OUTPUT_DIR/reports/area_analysis_summary.md" ]; then
      echo ""
      echo "Summary of Results ($DATASET):"
      echo "----------------------"
      cat "$SPLIT_OUTPUT_DIR/reports/area_analysis_summary.md" | head -20
      echo "..."
      echo "(See full report in $SPLIT_OUTPUT_DIR/reports/area_analysis_summary.md)"
    fi
done