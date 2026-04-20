#!/bin/bash
# Batch-based inference script
# Generates SAM masks and runs colorization in batches to save storage

set -e  # Exit on error

echo "=========================================="
echo "L-CAD Batch-Based Inference Pipeline"
echo "=========================================="
echo ""

# Configuration
SAM_CHECKPOINT="${SAM_CHECKPOINT:-models/sam_vit_h_4b8939.pth}"
MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-models/auto_weight.ckpt}"
IMG_DIR="/data/swarnim/DATA/swarnim/coco5k"
PAIRS_JSON="/data/swarnim/DATA/swarnim/blip_caption/coco_5k.json"
TEMP_MASK_DIR="${TEMP_MASK_DIR:-sam_mask/batch_masks}"
OUTPUT_DIR="/data/swarnim/L-CAD_updated/results/coco"
BATCH_SIZE="${BATCH_SIZE:-500}"
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-0}"
DDIM_STEPS="${DDIM_STEPS:-50}"
DDIM_ETA="${DDIM_ETA:-0.0}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-5.0}"

echo "Configuration:"
echo "  SAM Checkpoint: $SAM_CHECKPOINT"
echo "  Model Checkpoint: $MODEL_CHECKPOINT"
echo "  Image Directory: $IMG_DIR"
echo "  Pairs JSON: $PAIRS_JSON"
echo "  Temporary Mask Directory: $TEMP_MASK_DIR"
echo "  Output Directory: $OUTPUT_DIR"
echo "  Batch Size: $BATCH_SIZE images"
echo "  Inference Batch Size: $INFERENCE_BATCH_SIZE"
echo "  DDIM Steps: $DDIM_STEPS"
echo ""

# Check if SAM model exists
if [ ! -f "$SAM_CHECKPOINT" ]; then
    echo "Error: SAM model checkpoint not found at $SAM_CHECKPOINT"
    echo "Please download the SAM model checkpoint."
    echo "Download link: https://github.com/facebookresearch/segment-anything"
    exit 1
fi

# Check if colorization model exists
if [ ! -f "$MODEL_CHECKPOINT" ]; then
    echo "Error: Model checkpoint not found at $MODEL_CHECKPOINT"
    echo "Please set MODEL_CHECKPOINT to a full L-CAD checkpoint, e.g. models/auto_weight.ckpt"
    exit 1
fi

# Check if pairs JSON exists
if [ ! -f "$PAIRS_JSON" ]; then
    echo "Error: Pairs JSON not found at $PAIRS_JSON"
    echo "Please create a pairs JSON file with image-caption pairs."
    echo "Format: [[\"image.jpg\", \"description\"], ...]"
    exit 1
fi

# Check if image directory exists
if [ ! -d "$IMG_DIR" ]; then
    echo "Error: Image directory not found at $IMG_DIR"
    exit 1
fi

# Create temporary mask directory if it doesn't exist
mkdir -p "$TEMP_MASK_DIR"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "Starting batch-based inference pipeline..."
echo ""

# Run the batch inference pipeline
CUDA_VISIBLE_DEVICES=1 python batch_inference.py \
    --sam_checkpoint "$SAM_CHECKPOINT" \
    --model_checkpoint "$MODEL_CHECKPOINT" \
    --img_dir "$IMG_DIR" \
    --pairs_json "$PAIRS_JSON" \
    --temp_mask_dir "$TEMP_MASK_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --inference_batch_size "$INFERENCE_BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --ddim_steps "$DDIM_STEPS" \
    --ddim_eta "$DDIM_ETA" \
    --unconditional_guidance_scale "$GUIDANCE_SCALE" \
    --use_attn_guidance

echo ""
echo "=========================================="
echo "Batch-based inference completed!"
echo "=========================================="
echo ""
echo "Output images saved to: $OUTPUT_DIR"
echo "Temporary masks have been cleaned up."
echo ""
