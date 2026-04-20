#!/bin/bash
# Resume training script for L-CAD
# Usage: ./scripts/train_resume.sh

set -e  # Exit on error

echo "=========================================="
echo "L-CAD Resume Training Script"
echo "=========================================="
echo ""

# Check if config.yaml exists
if [ ! -f "config.yaml" ]; then
    echo "Error: config.yaml not found!"
    echo "Please copy config.yaml.example to config.yaml and update the paths."
    exit 1
fi

echo "Resuming training from checkpoint..."
echo "Configuration loaded from config.yaml"
echo ""

python colorization_main.py -t -r

echo ""
echo "Training completed!"
