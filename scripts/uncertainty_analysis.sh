#!/usr/bin/env bash
set -e # Exit on error

# --- Project Setup ---
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# --- Set Environment Variables ---
echo "🔧 Setting PYTHONPATH..."
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/Mask2Former:${PYTHONPATH:-}"
export DETECTRON2_DATASETS="$PROJECT_ROOT/datasets"
echo "PYTHONPATH set to: $PYTHONPATH"
echo ""

# --- Activate Environment ---
echo "🐍 Activating Python environment..."
if [ -f "$PROJECT_ROOT/mask2former-env/bin/activate" ]; then
    source "$PROJECT_ROOT/mask2former-env/bin/activate"
    echo "✅ Environment activated."
else
    echo "❌ Python environment not found. Run setup.sh first."
    exit 1
fi
echo ""

# --- Configuration ---
INSTANCE_CONFIG="$PROJECT_ROOT/configs/maskformer2_swin_base.yaml"
INSTANCE_WEIGHTS="$PROJECT_ROOT/output/instance_segmentation/model_final.pth"

SEMANTIC_CONFIG="$PROJECT_ROOT/configs/maskformer2_swin_base_semantic.yaml"
SEMANTIC_WEIGHTS="$PROJECT_ROOT/output/semantic_segmentation/model_final.pth"

OUTPUT_DIR="$PROJECT_ROOT/output/uncertainty_analysis"
# Recommended higher samples for publication quality stats
NUM_MC_SAMPLES=50
NUM_PROPAGATION_TRIALS=1000
CONFIDENCE=0.5

# --- Run Integrated Pipeline ---
echo "=================================================="
echo "🚀 STARTING EPISTEMIC UNCERTAINTY ANALYSIS"
echo "=================================================="

# Ensure script is executable
chmod +x "$PROJECT_ROOT/src/run_full_uncertainty_pipeline.py"

python "$PROJECT_ROOT/src/run_full_uncertainty_pipeline.py" \
    --instance-config "$INSTANCE_CONFIG" \
    --instance-weights "$INSTANCE_WEIGHTS" \
    --semantic-config "$SEMANTIC_CONFIG" \
    --semantic-weights "$SEMANTIC_WEIGHTS" \
    --output-dir "$OUTPUT_DIR" \
    --num-mc-samples $NUM_MC_SAMPLES \
    --num-propagation-trials $NUM_PROPAGATION_TRIALS \
    --confidence-threshold $CONFIDENCE

echo ""
echo "=========================================="
echo "✅ PIPELINE COMPLETE"
echo "=========================================="
echo "Final reports and high-resolution plots saved to: $OUTPUT_DIR"
echo "Check 'boxplot_instance_uncertainty.png' and 'boxplot_semantic_uncertainty.png' in split directories."