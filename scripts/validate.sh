#!/bin/bash
# Validation script for L-CAD on COCO validation set
# Usage: ./scripts/validate.sh

set -e  # Exit on error

echo "=========================================="
echo "L-CAD Validation Script"
echo "=========================================="
echo ""

# Check if config.yaml exists
if [ ! -f "config.yaml" ]; then
    echo "Error: config.yaml not found!"
    echo "Please copy config.yaml.example to config.yaml and update the paths."
    exit 1
fi

# Check if model checkpoint exists
MODEL_PATH=$(python -c "from config import cfg; print(cfg.resume_checkpoint)")
if [ ! -f "$MODEL_PATH" ]; then
    echo "Warning: Model checkpoint not found at $MODEL_PATH"
    echo "Please update config.yaml with the correct path or download the model."
    exit 1
fi

# Check if COCO validation dataset exists
COCO_DIR=$(python -c "from config import cfg; print(cfg.coco_img_dir)")
if [ ! -d "$COCO_DIR" ]; then
    echo "Warning: COCO dataset directory not found at $COCO_DIR"
    echo "Please update config.yaml with the correct path to COCO dataset."
    exit 1
fi

echo "Starting validation on COCO val set..."
echo "Model: $MODEL_PATH"
echo "Dataset: $COCO_DIR"
echo ""

python colorization_main.py

echo ""
echo "Validation completed!"
