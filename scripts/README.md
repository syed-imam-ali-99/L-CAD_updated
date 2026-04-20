# L-CAD Scripts

This directory contains convenient bash scripts for running various L-CAD tasks.

## Setup

First, run the setup script to prepare your environment:

```bash
./scripts/setup.sh
```

This will:
- Install dependencies from requirements.txt
- Create config.yaml from config.yaml.example
- Create necessary directories
- Make all scripts executable

**Important:** After setup, edit `config.yaml` and update the paths to match your environment.

## Available Scripts

### Training

- **`train.sh`** - Start training from scratch
  ```bash
  ./scripts/train.sh
  ```
  Requires: `init_model.ckpt` in models directory

- **`train_resume.sh`** - Resume training from checkpoint
  ```bash
  ./scripts/train_resume.sh
  ```

### Inference

- **`inference_basic.sh`** - Basic colorization with language prompts (no SAM masks)
  ```bash
  ./scripts/inference_basic.sh
  ```
  Uses model from `cfg.resume_checkpoint`

- **`inference.sh`** - Instance-aware colorization with SAM masks
  ```bash
  ./scripts/inference.sh
  ```
  Requires: Pre-generated SAM masks in `sam_mask/select_masks/`

### Testing

- **`validate.sh`** - Run validation on COCO validation set
  ```bash
  ./scripts/validate.sh
  ```
  Requires: COCO dataset configured in config.yaml

- **`test_multicolor_sam.sh`** - Multi-instance colorization with SAM masks
  ```bash
  ./scripts/test_multicolor_sam.sh
  ```

### SAM Mask Generation

- **`generate_sam_masks.sh`** - Generate SAM masks for images
  ```bash
  ./scripts/generate_sam_masks.sh
  ```

  **Before running:**
  1. Edit `sam_mask/make_mask.py` and add your image filenames to `img_list`
  2. Ensure SAM model checkpoint exists at `models/sam_vit_h_4b8939.pth`

  **After running:**
  1. Review generated masks in `sam_mask/seg_img/`
  2. Manually select relevant masks
  3. Copy to `sam_mask/select_masks/{image_name}/`
  4. Create `sam_mask/pairs.json` with test pairs

## Script Features

All scripts include:
- ✓ Configuration validation (checks if config.yaml exists)
- ✓ Path validation (checks if required files/directories exist)
- ✓ Clear error messages with suggestions
- ✓ Progress indicators
- ✓ Exit on error (`set -e`)

## Troubleshooting

If you get "Permission denied" errors:
```bash
chmod +x scripts/*.sh
```

If config.yaml is missing:
```bash
cp config.yaml.example config.yaml
# Then edit config.yaml with your paths
```

If model checkpoints are missing, see README.md for download links.
