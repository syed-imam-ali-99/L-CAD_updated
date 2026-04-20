# Batch Inference Quick Start Guide

A concise guide to get started with batch-based SAM mask generation and colorization.

## Prerequisites

- SAM model checkpoint: `models/sam_vit_h_4b8939.pth`
- L-CAD colorization checkpoint: configured in `config.yaml`
- Input images in a directory
- Image-caption pairs JSON file

## 5-Minute Setup

### 1. Prepare Data

```bash
# Create pairs JSON file
cat > sam_mask/pairs.json << 'EOF'
[
  ["image1.jpg", "object1, color1, object2"],
  ["image2.jpg", "object3, color2, background"]
]
EOF

# Ensure images are in the directory
ls example/*.jpg
```

### 2. Run Batch Inference

**Option A: Using Shell Script (Recommended)**

```bash
./scripts/run_batch_inference.sh
```

**Option B: Using Python Directly**

```bash
python batch_inference.py \
    --sam_checkpoint models/sam_vit_h_4b8939.pth \
    --img_dir example \
    --pairs_json sam_mask/pairs.json \
    --batch_size 100
```

### 3. Check Results

```bash
# View colorized images
ls image_log/batch_inference/

# Verify masks were cleaned up
ls sam_mask/batch_masks/  # Should be empty or not exist
```

## Configuration Cheat Sheet

### Change Batch Size

```bash
# Process 50 images per batch instead of 100
export BATCH_SIZE=50
./scripts/run_batch_inference.sh
```

Or with Python:

```bash
python batch_inference.py --batch_size 50
```

### Change Output Directory

```bash
export OUTPUT_DIR="my_results"
./scripts/run_batch_inference.sh
```

### Custom Image Directory

```bash
export IMG_DIR="path/to/my/images"
./scripts/run_batch_inference.sh
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 100 | Images per batch (for mask generation) |
| `inference_batch_size` | 1 | Images per inference batch (GPU memory) |
| `ddim_steps` | 50 | DDIM sampling steps (quality vs speed) |
| `guidance_scale` | 5.0 | Unconditional guidance scale |

## Storage Impact

| Scenario | Storage Required |
|----------|------------------|
| 1000 images, batch_size=100 | ~250 MB (temporary) |
| 1000 images, batch_size=200 | ~500 MB (temporary) |
| 1000 images, batch_size=50 | ~125 MB (temporary) |
| 1000 images, no batching | ~2.5 GB (permanent) |

## Pipeline Flow

```
For each batch of 100 images:
  1. Generate SAM masks ──> temp storage
  2. Run colorization   ──> save results
  3. Delete SAM masks   ──> free storage
  4. Repeat for next batch
```

## Common Issues

### "SAM checkpoint not found"
```bash
# Download SAM model
cd models/
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

### "CUDA out of memory"
```bash
# Use smaller inference batch size
python batch_inference.py --inference_batch_size 1 --ddim_steps 30
```

### "Pairs JSON not found"
```bash
# Check file exists
cat sam_mask/pairs.json

# Format should be:
# [["img1.jpg", "desc1"], ["img2.jpg", "desc2"]]
```

## Performance Tips

1. **Faster Processing**: Increase `batch_size` to 200 (requires more storage)
2. **Less Storage**: Decrease `batch_size` to 50
3. **Faster Inference**: Decrease `ddim_steps` to 30 (may reduce quality)
4. **Multi-GPU**: Not currently supported (process batches sequentially)

## Verify Installation

```bash
# Test imports
python -c "from batch_inference import BatchInferencePipeline; print('✓ OK')"
python -c "from sam_mask.batch_mask_generator import BatchSamMaskGenerator; print('✓ OK')"
python -c "from colorization_dataset_batch import BatchMyDataset; print('✓ OK')"
```

## Next Steps

- Read full documentation: [`BATCH_INFERENCE_README.md`](BATCH_INFERENCE_README.md)
- Adjust parameters in `scripts/run_batch_inference.sh`
- Monitor progress in console output
- Review results in output directory

## Example Output

```
==============================================================
BATCH 1/5
==============================================================

[Step 1/3] Generating SAM masks for batch 1...
[1/450] Processing image1.jpg... Generated 15 masks

[Step 2/3] Running colorization inference on batch 1...
Processed 100/100 images in this batch

[Step 3/3] Cleaning up masks for batch 1...
Cleanup complete.

Batch 1/5 completed in 245.32 seconds
Progress: 100/450 images (22.2%)
```

## Support

For detailed information, see [`BATCH_INFERENCE_README.md`](BATCH_INFERENCE_README.md).
