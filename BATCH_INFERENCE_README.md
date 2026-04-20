# Batch-Based SAM Mask Generation and Inference Pipeline

This document describes the batch-based pipeline for generating SAM masks and running colorization inference. This approach processes images in batches to minimize storage requirements by only keeping masks for the current batch on disk.

## Overview

The batch-based pipeline addresses the storage limitations of precomputing all SAM masks by:

1. **Processing images in batches** (default: 100 images per batch)
2. **Generating SAM masks** only for the current batch
3. **Running colorization inference** using the generated masks
4. **Cleaning up masks** after each batch to free storage
5. **Repeating** for subsequent batches until all images are processed

At any point in time, only the masks for the current batch (≤100 images) exist on disk.

## Key Features

- ✅ **Storage Efficient**: Only stores masks for current batch (~100 images)
- ✅ **Automatic Cleanup**: Deletes masks after each batch completes
- ✅ **Synchronized Pipeline**: Ensures masks are generated before inference
- ✅ **Edge Case Handling**: Properly handles final batch with fewer than 100 images
- ✅ **Progress Tracking**: Shows detailed progress for each batch
- ✅ **Configurable**: Flexible batch sizes and parameters

## Architecture

### New Files Created

1. **`batch_inference.py`** - Main orchestrator script
   - Coordinates mask generation and inference
   - Manages batch processing workflow
   - Handles cleanup after each batch

2. **`sam_mask/batch_mask_generator.py`** - SAM mask generator
   - Generates masks for specific batches
   - Saves masks to temporary directory
   - Provides batch information utilities

3. **`colorization_dataset_batch.py`** - Batch-aware dataset
   - Loads images and masks from batch-specific directories
   - Supports filtering by batch indices
   - Handles missing masks gracefully

4. **`scripts/run_batch_inference.sh`** - Convenience script
   - Easy-to-use shell wrapper
   - Environment variable configuration
   - Input validation

## Usage

### Quick Start

```bash
# Run with default settings (100 images per batch)
./scripts/run_batch_inference.sh
```

### Python Direct Usage

```bash
python batch_inference.py \
    --sam_checkpoint models/sam_vit_h_4b8939.pth \
    --img_dir example \
    --pairs_json sam_mask/pairs.json \
    --batch_size 100 \
    --output_dir image_log/batch_inference
```

### Configuration Options

#### Environment Variables (for shell script)

```bash
# SAM model checkpoint path
export SAM_CHECKPOINT="models/sam_vit_h_4b8939.pth"

# Image directory
export IMG_DIR="example"

# Image-caption pairs JSON file
export PAIRS_JSON="sam_mask/pairs.json"

# Temporary directory for batch masks (will be cleaned up)
export TEMP_MASK_DIR="sam_mask/batch_masks"

# Output directory for colorized images
export OUTPUT_DIR="image_log/batch_inference"

# Number of images per batch
export BATCH_SIZE=100

# Batch size for inference (typically 1 for memory)
export INFERENCE_BATCH_SIZE=1

# Number of data loading workers
export NUM_WORKERS=0

# DDIM sampling steps
export DDIM_STEPS=50

# DDIM eta parameter
export DDIM_ETA=0.0

# Unconditional guidance scale
export GUIDANCE_SCALE=5.0

# Then run
./scripts/run_batch_inference.sh
```

#### Command-Line Arguments (for Python script)

```
Required Arguments:
  --sam_checkpoint STR     Path to SAM model checkpoint
  --img_dir STR           Directory containing input images
  --pairs_json STR        JSON file with image-caption pairs

Optional Arguments:
  --temp_mask_dir STR     Temporary directory for batch masks
                          (default: sam_mask/batch_masks)
  --output_dir STR        Output directory for colorized images
                          (default: image_log/batch_inference)
  --batch_size INT        Images per batch for mask generation
                          (default: 100)
  --inference_batch_size INT  Batch size for inference
                          (default: 1)
  --num_workers INT       Number of data loading workers
                          (default: 0)
  --ddim_steps INT        Number of DDIM sampling steps
                          (default: 50)
  --ddim_eta FLOAT        DDIM eta parameter
                          (default: 0.0)
  --unconditional_guidance_scale FLOAT
                          Unconditional guidance scale
                          (default: 5.0)
  --use_attn_guidance     Use attention guidance
                          (default: True)
```

## Pipeline Workflow

For each batch of images, the pipeline executes the following steps:

### Step 1: Generate SAM Masks

```
Batch 1: Generate masks for images 0-99
├── Load SAM model
├── For each image in batch:
│   ├── Load image
│   ├── Generate SAM masks
│   └── Save masks to temp_mask_dir/{image_name}/
└── Report: N masks generated
```

### Step 2: Run Colorization Inference

```
Batch 1: Run inference on images 0-99
├── Load colorization model
├── Create dataset with batch indices
├── For each mini-batch:
│   ├── Load image and masks
│   ├── Run DDIM sampling with SAM guidance
│   ├── Decode and save colorized image
│   └── Report progress
└── All images in batch processed
```

### Step 3: Cleanup Masks

```
Batch 1: Delete masks for images 0-99
├── For each image in batch:
│   └── Delete temp_mask_dir/{image_name}/
└── Storage freed for next batch
```

### Repeat for All Batches

```
Total: 450 images, Batch size: 100
├── Batch 1: Images 0-99    (100 images) ✓
├── Batch 2: Images 100-199 (100 images) ✓
├── Batch 3: Images 200-299 (100 images) ✓
├── Batch 4: Images 300-399 (100 images) ✓
└── Batch 5: Images 400-449 (50 images)  ✓  [Edge case handled]
```

## Input Format

### Image-Caption Pairs JSON

Create a JSON file (`sam_mask/pairs.json`) with the following format:

```json
[
  ["image1.jpg", "a red car, blue sky, green grass"],
  ["image2.jpg", "a person, yellow shirt, white background"],
  ["image3.jpg", "a cat, brown fur, sitting"]
]
```

Each entry is a list with:
- **First element**: Image filename (must exist in `--img_dir`)
- **Second element**: Caption with comma-separated descriptions for SAM regions

## Output

### Colorized Images

Saved to `--output_dir` (default: `image_log/batch_inference/`):

```
image_log/batch_inference/
├── colorized_image1.png
├── colorized_image2.png
└── colorized_image3.png
```

### Temporary Masks

During processing, masks are temporarily stored in `--temp_mask_dir`:

```
sam_mask/batch_masks/
├── image1/
│   ├── 0.npy
│   ├── 1.npy
│   └── 2.npy
└── image2/
    ├── 0.npy
    └── 1.npy

[These are automatically deleted after each batch]
```

## Storage Requirements

### Without Batch Processing (Original)
- All masks stored simultaneously
- **Storage**: ~N × M × mask_size
  - N = total images
  - M = average masks per image
  - Example: 1000 images × 10 masks × 256KB = **~2.5 GB**

### With Batch Processing (New)
- Only current batch masks stored
- **Storage**: ~B × M × mask_size
  - B = batch size (default 100)
  - M = average masks per image
  - Example: 100 images × 10 masks × 256KB = **~250 MB**
- **Storage Reduction**: ~90% for 1000 images with batch_size=100

## Performance Considerations

### Batch Size Selection

**Larger batches (e.g., 200)**:
- ✅ Fewer SAM model loads (if model is reloaded per batch)
- ✅ Fewer context switches
- ❌ More storage required
- ❌ Longer before seeing first results

**Smaller batches (e.g., 50)**:
- ✅ Lower storage requirements
- ✅ Faster iteration for debugging
- ❌ More overhead from batch switching
- ❌ More SAM model operations

**Recommended**: 100 images (default) provides a good balance.

### Inference Batch Size

- **Default**: 1 (processes one image at a time during inference)
- **Reason**: Colorization model typically requires significant GPU memory
- **Can increase** if you have sufficient GPU memory

## Troubleshooting

### Issue: "SAM model checkpoint not found"

**Solution**: Download the SAM model checkpoint:

```bash
# Download SAM ViT-H checkpoint
cd models/
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

### Issue: "Pairs JSON not found"

**Solution**: Create a pairs JSON file:

```bash
# Create pairs.json
cat > sam_mask/pairs.json << 'EOF'
[
  ["image1.jpg", "description 1"],
  ["image2.jpg", "description 2"]
]
EOF
```

### Issue: "CUDA out of memory"

**Solutions**:
1. Reduce `--inference_batch_size` to 1
2. Reduce `--ddim_steps`
3. Use a smaller model checkpoint
4. Process fewer images

### Issue: "Masks not found for image"

**Cause**: SAM mask generation failed or was skipped

**Solution**:
1. Check that images exist in `--img_dir`
2. Verify SAM model is loaded correctly
3. Check console output for SAM generation errors

### Issue: "Slow processing"

**Solutions**:
1. Increase `--num_workers` for faster data loading (if CPU allows)
2. Decrease `--ddim_steps` for faster inference (may reduce quality)
3. Use GPU for SAM if available
4. Increase `--batch_size` to reduce overhead

## Advanced Usage

### Process a Specific Batch Only

For testing or debugging, you can generate masks for a specific batch:

```bash
# Generate masks for batch 0 (first 100 images)
python sam_mask/batch_mask_generator.py \
    --batch_idx 0 \
    --batch_size 100
```

### View Batch Information

See how images will be divided into batches:

```bash
# Show batch information without processing
python sam_mask/batch_mask_generator.py
```

Output:
```
==============================================================
BATCH INFORMATION
==============================================================
Total images: 450
Batch size: 100
Number of batches: 5

Batch 1: images 0 to 99 (100 images)
Batch 2: images 100 to 199 (100 images)
Batch 3: images 200 to 299 (100 images)
Batch 4: images 300 to 399 (100 images)
Batch 5: images 400 to 449 (50 images)
==============================================================
```

### Custom Batch Size

Process images in smaller batches (e.g., 50 images):

```bash
python batch_inference.py \
    --batch_size 50 \
    --output_dir image_log/batch50_inference
```

## Comparison with Original Pipeline

### Original Pipeline (`inference.py`)

```python
# 1. Pre-generate ALL masks (stored permanently)
python sam_mask/make_mask.py  # Generates masks for all images

# 2. Run inference (loads pre-generated masks)
python inference.py
```

**Issues**:
- ❌ Requires storage for all masks simultaneously
- ❌ Masks must be manually managed
- ❌ No automatic cleanup

### New Batch Pipeline (`batch_inference.py`)

```python
# Single command handles everything
python batch_inference.py
```

**Advantages**:
- ✅ Automatic mask generation per batch
- ✅ Automatic cleanup after each batch
- ✅ Minimal storage footprint
- ✅ Self-contained workflow

## Integration with Existing Code

The batch pipeline is designed to work alongside the existing pipeline:

- **Original files unchanged**: `inference.py`, `colorization_dataset.py` remain intact
- **New files added**: `batch_inference.py`, `colorization_dataset_batch.py`
- **Can use both**: Choose based on your storage constraints

### When to Use Batch Pipeline

- ✅ Processing many images (>100)
- ✅ Limited storage available
- ✅ SAM masks don't need to be kept
- ✅ Automated end-to-end processing

### When to Use Original Pipeline

- ✅ Processing few images (<50)
- ✅ Masks will be reused multiple times
- ✅ Need to manually inspect/select masks
- ✅ Debugging individual components

## Example: Complete Workflow

```bash
# 1. Prepare your data
cd /data/swarnim/L-CAD_updated

# Create pairs JSON
cat > sam_mask/pairs.json << 'EOF'
[
  ["09eacd04461a94ac.jpg", "a car, red color, parking lot"],
  ["5b68696f625d1572.jpg", "a building, blue sky, green trees"],
  ["vg_2377448.jpg", "a person, yellow shirt, smiling"]
]
EOF

# Place your images in example/
cp /path/to/your/images/*.jpg example/

# 2. Run batch inference
./scripts/run_batch_inference.sh

# 3. View results
ls image_log/batch_inference/
```

## Monitoring Progress

The pipeline provides detailed progress information:

```
==========================================================
BATCH 1/5
==========================================================

[Step 1/3] Generating SAM masks for batch 1...
[1/450] Processing 09eacd04461a94ac.jpg...
  Generated 15 masks
[2/450] Processing 5b68696f625d1572.jpg...
  Generated 12 masks
...

[Step 2/3] Running colorization inference on batch 1...
  Processed 50/100 images in this batch
  Processed 100/100 images in this batch

[Step 3/3] Cleaning up masks for batch 1...
Cleanup complete.

Batch 1/5 completed in 245.32 seconds
Progress: 100/450 images (22.2%)
```

## License and Credits

This batch-based pipeline extends the original L-CAD framework with efficient batch processing capabilities.

- Original L-CAD paper and code: [Link to original repository]
- SAM (Segment Anything Model): Meta AI Research

## Support

For issues or questions:
1. Check this README for troubleshooting tips
2. Verify your environment and dependencies
3. Check console output for error messages
4. Review the original L-CAD documentation
