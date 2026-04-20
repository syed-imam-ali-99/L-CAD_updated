#!/bin/bash
# Setup script for L-CAD project
# Usage: ./scripts/setup.sh

set -e  # Exit on error

echo "=========================================="
echo "L-CAD Project Setup"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"
if [[ ! "$PYTHON_VERSION" =~ ^3\.9 ]]; then
    echo "Warning: Python 3.9 is recommended. Current version: $PYTHON_VERSION"
fi
echo ""

# Install dependencies
echo "Installing dependencies from requirements.txt..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo "Dependencies installed successfully!"
else
    echo "Error: requirements.txt not found!"
    exit 1
fi
echo ""

# Setup configuration
echo "Setting up configuration..."
if [ ! -f "config.yaml" ]; then
    if [ -f "config.yaml.example" ]; then
        cp config.yaml.example config.yaml
        echo "✓ config.yaml created from config.yaml.example"
        echo ""
        echo "IMPORTANT: Please edit config.yaml and update the following paths:"
        echo "  - models.largedecoder_checkpoint"
        echo "  - datasets.coco.img_dir"
        echo "  - Any other paths specific to your environment"
        echo ""
    else
        echo "Error: config.yaml.example not found!"
        exit 1
    fi
else
    echo "✓ config.yaml already exists"
    echo ""
fi

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p models
mkdir -p image_log
mkdir -p sam_mask/masks
mkdir -p sam_mask/seg_img
mkdir -p sam_mask/select_masks
echo "✓ Directories created"
echo ""

# Make scripts executable
echo "Making scripts executable..."
chmod +x scripts/*.sh
echo "✓ Scripts are now executable"
echo ""

echo "=========================================="
echo "Setup completed!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Edit config.yaml and update paths for your environment"
echo "2. Download model checkpoints (see README.md for links)"
echo "3. Place checkpoints in the models/ directory"
echo "4. Run validation: ./scripts/validate.sh"
echo "   or training: ./scripts/train.sh"
echo ""
