#!/bin/bash
# Training script for L-CAD
# Usage: ./scripts/train.sh

set -e  # Exit on error

echo "=========================================="
echo "L-CAD Training Script"
echo "=========================================="
echo ""

# Check if config.yaml exists
if [ ! -f "config.yaml" ]; then
    echo "Error: config.yaml not found!"
    echo "Please copy config.yaml.example to config.yaml and update the paths."
    echo "Command: cp config.yaml.example config.yaml"
    exit 1
fi

# Check if init model exists
INIT_MODEL=$(python -c "from config import cfg; print(cfg.init_model_path)")
if [ ! -f "$INIT_MODEL" ]; then
    echo "Warning: Init model not found at $INIT_MODEL"
    echo "Please download the init model and place it in the models directory."
    echo "See README.md for download links."
    exit 1
fi

echo "Starting training..."
echo "Configuration loaded from config.yaml"
echo ""

python colorization_main.py -t

echo ""
echo "Training completed!"
