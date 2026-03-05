#!/usr/bin/env bash

set -eo pipefail

# Get the project root directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Use the environment
source "$PROJECT_ROOT/mask2former-env/bin/activate"

# Set environment variables
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/Mask2Former:$PYTHONPATH"
export DETECTRON2_DATASETS="$PROJECT_ROOT/datasets"

# Configuration files
SEMANTIC_CONFIG="$PROJECT_ROOT/configs/maskformer2_swin_base_semantic.yaml"
SEMANTIC_WEIGHTS="$PROJECT_ROOT/output/semantic_segmentation/model_final.pth"
INSTANCE_CONFIG="$PROJECT_ROOT/configs/maskformer2_swin_base.yaml"
INSTANCE_WEIGHTS="$PROJECT_ROOT/output/instance_segmentation/model_final.pth"

# Ground truth paths (REMOVED - Python script now handles this)
# INSTANCE_GT_JSON="$PROJECT_ROOT/datasets/instance/val/annotations.json"
# SEMANTIC_GT_DIR="$PROJECT_ROOT/datasets/semantic/val/annotations/sem_seg"
# IMAGES_DIR="$PROJECT_ROOT/datasets/instance/val/images"

# Output
OUTPUT_DIR="$PROJECT_ROOT/output/error_analysis"
# DATASET_NAME="corrosion_val" (REMOVED)

# Parameters
IOU_THRESHOLD=0.5
MIN_CONFIDENCE=0.5

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo ""
echo "============================================================="
echo -e "${BLUE}ERROR ANALYSIS WITH IOU-BASED MATCHING (val & test)${NC}"
echo "============================================================="
echo ""

# Validate files (only models and configs)
echo "Validating required model files..."

if [ ! -f "$SEMANTIC_CONFIG" ]; then
    echo -e "${RED}❌ Semantic config not found: $SEMANTIC_CONFIG${NC}"
    exit 1
fi

if [ ! -f "$SEMANTIC_WEIGHTS" ]; then
    echo -e "${RED}❌ Semantic weights not found: $SEMANTIC_WEIGHTS${NC}"
    exit 1
fi

if [ ! -f "$INSTANCE_CONFIG" ]; then
    echo -e "${RED}❌ Instance config not found: $INSTANCE_CONFIG${NC}"
    exit 1
fi

if [ ! -f "$INSTANCE_WEIGHTS" ]; then
    echo -e "${RED}❌ Instance weights not found: $INSTANCE_WEIGHTS${NC}"
    exit 1
fi

echo -e "${GREEN}✅ All required model files found${NC}"
echo ""

# Summary (removed file counts)
echo "Data Summary:"
echo "  - IoU threshold: $IOU_THRESHOLD"
echo "  - Confidence threshold: $MIN_CONFIDENCE"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo -e "${YELLOW}This script will:${NC}"
echo "  1. Run instance segmentation model to get predictions"
echo "  2. Run semantic segmentation model to get predictions"
echo "  3. Load ground truth for both tasks (for 'val' and 'test' splits)"
echo "  4. Match GT and predicted instances using IoU (≥ $IOU_THRESHOLD)"
echo "  5. Calculate corrosion errors ONLY for matched pairs"
echo "  6. Track false negatives (missed GT) and false positives (extra predictions)"
echo "  7. Generate comprehensive reports and visualizations for both splits"
echo ""

read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

echo ""
echo "Starting analysis..."
echo ""

# Run the error analysis (updated arguments)
python "$PROJECT_ROOT/src/error_analysis.py" \
    --semantic-config "$SEMANTIC_CONFIG" \
    --semantic-weights "$SEMANTIC_WEIGHTS" \
    --instance-config "$INSTANCE_CONFIG" \
    --instance-weights "$INSTANCE_WEIGHTS" \
    --output-dir "$OUTPUT_DIR" \
    --iou-threshold "$IOU_THRESHOLD" \
    --min-confidence "$MIN_CONFIDENCE" \
    --output-format "all" \
    --generate-visualizations

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "============================================================="
    echo -e "${GREEN}✅ ERROR ANALYSIS COMPLETE!${NC}"
    echo "============================================================="
    echo ""
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "📊 Key Output Files (generated for 'val' and 'test' subdirectories):"
    echo "  - matching_statistics.json: Instance matching details"
    echo "  - element_level_errors.csv: Per-element error data"
    echo "  - error_analysis.xlsx: Comprehensive Excel report"
    echo "    • Element Level Errors sheet"
    echo "    • Overall Statistics sheet"
    echo "    • Matching Statistics sheet"
    echo "    • False Negatives sheet (GT elements not detected)"
    echo "    • False Positives sheet (Predictions with no GT match)"
    echo "  - error_analysis_plots.png: Error visualizations"
    echo "  - visualizations/: Per-image comparison images (2 per image):"
    echo "    • overview_*.png: 2x2 grid (semantic + instances)"
    echo "    • elementwise_*.png: Side-by-side element-wise corrosion (KEY!)"
    echo ""
    
    # Display summary for both splits
    for SPLIT in "val" "test"; do
        SPLIT_OUTPUT_DIR="$OUTPUT_DIR/$SPLIT"
        if [ -f "$SPLIT_OUTPUT_DIR/reports/error_analysis_summary.md" ]; then
            echo "Summary Preview ($SPLIT):"
            echo "-------------------"
            head -40 "$SPLIT_OUTPUT_DIR/reports/error_analysis_summary.md"
            echo "..."
            echo ""
            echo "Full report: $SPLIT_OUTPUT_DIR/reports/error_analysis_summary.md"
        fi
    done
    
    echo ""
    echo -e "${BLUE}📈 Next Steps:${NC}"
    echo "  1. Review Excel reports (in $OUTPUT_DIR/val/ and $OUTPUT_DIR/test/) for detailed errors"
    echo "  2. Check False Negatives/Positives sheets"
    echo "  3. Examine visualizations for specific failure cases"
    echo ""
else
    echo "============================================================="
    echo -e "${RED}❌ ANALYSIS FAILED!${NC}"
    echo "============================================================="
    echo "Check error messages above for details."
    exit 1
fi