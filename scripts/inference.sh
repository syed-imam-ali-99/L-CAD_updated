#!/bin/bash
# Instance-aware inference script for L-CAD with SAM masks
# Usage: ./scripts/inference.sh

set -e  # Exit on error

echo "=========================================="
echo "L-CAD Instance-Aware Inference (with SAM)"
echo "=========================================="
echo ""

# Check if config.yaml exists
if [ ! -f "config.yaml" ]; then
    echo "Error: config.yaml not found!"
    echo "Please copy config.yaml.example to config.yaml and update the paths."
    exit 1
fi

# Check if model checkpoint exists
MODEL_PATH=$(python -c "from config import cfg; print(cfg.largedecoder_checkpoint)")
if [ ! -f "$MODEL_PATH" ]; then
    echo "Warning: Model checkpoint not found at $MODEL_PATH"
    echo "Please update config.yaml with the correct path to largedecoder checkpoint."
    echo "See README.md for download links."
    exit 1
fi

# Check if SAM masks exist
SAM_MASKS_DIR=$(python -c "from config import cfg; print(cfg.sam_select_masks_dir)")
if [ ! -d "$SAM_MASKS_DIR" ]; then
    echo "Warning: SAM masks directory not found at $SAM_MASKS_DIR"
    echo "Please generate SAM masks first using: ./scripts/generate_sam_masks.sh"
    exit 1
fi

echo "Starting instance-aware inference..."
echo "Model: $MODEL_PATH"
echo "Output will be saved to image_log/"
echo ""

python inference.py

echo ""
echo "Inference completed! Check image_log/ for results."
