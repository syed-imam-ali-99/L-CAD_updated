#!/bin/bash
# SAM mask generation script
# Usage: ./scripts/generate_sam_masks.sh

set -e  # Exit on error

echo "=========================================="
echo "L-CAD SAM Mask Generation"
echo "=========================================="
echo ""

# Check if SAM model exists
SAM_MODEL="models/sam_vit_h_4b8939.pth"
if [ ! -f "$SAM_MODEL" ]; then
    echo "Error: SAM model checkpoint not found at $SAM_MODEL"
    echo "Please download the SAM model checkpoint."
    echo "Download link: https://github.com/facebookresearch/segment-anything"
    exit 1
fi

# Check if sam_mask/make_mask.py exists
if [ ! -f "sam_mask/make_mask.py" ]; then
    echo "Error: sam_mask/make_mask.py not found!"
    echo "Please ensure the SAM mask generation script exists."
    exit 1
fi

echo "IMPORTANT: Before running this script:"
echo "1. Edit sam_mask/make_mask.py and add your image filenames to img_list"
echo "2. Place your images in the example/ directory"
echo ""
read -p "Have you edited img_list in sam_mask/make_mask.py? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Please edit sam_mask/make_mask.py first, then run this script again."
    exit 1
fi

echo ""
echo "Generating SAM masks..."
echo "Output will be saved to:"
echo "  - Masks: sam_mask/masks/"
echo "  - Visualizations: sam_mask/seg_img/"
echo ""

cd sam_mask
python make_mask.py

echo ""
echo "=========================================="
echo "SAM mask generation completed!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Review the generated masks in sam_mask/seg_img/"
echo "2. Manually select relevant masks for each image"
echo "3. Copy selected masks to sam_mask/select_masks/{image_name}/"
echo "4. Create test pairs in sam_mask/pairs.json"
echo "   Format: [[\"image.jpg\", \"desc1, desc2, desc3\"], ...]"
echo ""
echo "Then run inference with: ./scripts/inference.sh"
