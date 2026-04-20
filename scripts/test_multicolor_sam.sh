#!/bin/bash
# Multi-instance colorization with SAM masks
# Usage: ./scripts/test_multicolor_sam.sh

set -e  # Exit on error

echo "=========================================="
echo "L-CAD Multi-Instance Test (with SAM)"
echo "=========================================="
echo ""

# Check if config.yaml exists
if [ ! -f "config.yaml" ]; then
    echo "Error: config.yaml not found!"
    echo "Please copy config.yaml.example to config.yaml and update the paths."
    exit 1
fi

# Check if SAM masks exist
SAM_MASKS_DIR=$(python -c "from config import cfg; print(cfg.sam_select_masks_dir)")
if [ ! -d "$SAM_MASKS_DIR" ]; then
    echo "Warning: SAM masks directory not found at $SAM_MASKS_DIR"
    echo "Please generate SAM masks first using: ./scripts/generate_sam_masks.sh"
    exit 1
fi

echo "Starting multi-instance colorization with SAM masks..."
echo "Output will be saved to image_log/"
echo ""

python colorization_main.py -m -s

echo ""
echo "Testing completed! Check image_log/ for results."
