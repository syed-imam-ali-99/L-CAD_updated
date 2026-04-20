#!/bin/bash
# Basic inference script for L-CAD with language prompts (no SAM masks)
# Usage: ./scripts/inference_basic.sh

set -e  # Exit on error

echo "=========================================="
echo "L-CAD Basic Inference (Language Prompts)"
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
    echo "See README.md for download links."
    exit 1
fi

echo "Starting basic inference..."
echo "Model: $MODEL_PATH"
echo "Output will be saved to image_log/"
echo ""

python colorization_main.py -m

echo ""
echo "Inference completed! Check image_log/ for results."
