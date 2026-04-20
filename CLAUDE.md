# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

L-CAD (Language-based Colorization with Any-level Descriptions using Diffusion Priors) is a research implementation for automatic and language-guided image colorization using diffusion models. The system accepts grayscale images and natural language descriptions to generate plausible colorizations.

## Environment Setup

- Python 3.9
- PyTorch 1.12
- NVIDIA GPU + CUDA cuDNN required
- Install dependencies: `pip install -r requirements.txt`

**Quick Setup:**
```bash
./scripts/setup.sh
```
This automated setup script will install dependencies, create config.yaml, and prepare necessary directories. See `scripts/README.md` for details.

## Configuration System

**IMPORTANT**: The codebase has been refactored to use a centralized configuration system.

All file paths and hyperparameters are now managed through `config.yaml`:
1. **First-time setup**: Copy `config.yaml.example` to `config.yaml`
2. **Update paths**: Edit `config.yaml` to match your environment (dataset locations, model checkpoints)
3. **Access in code**: Import `cfg` from `config.py` to access configuration values

Key configuration sections:
- `models`: Model checkpoint paths (init_model, resume_checkpoint, largedecoder_checkpoint)
- `datasets`: Dataset directories (COCO, example images, SAM masks)
- `training`: Training hyperparameters (n_gpu, batch_size, learning_rate)
- `testing`: Test/inference settings
- `inference`: DDIM sampling parameters
- `output`: Output directory templates

The `config.yaml` file is gitignored to prevent committing environment-specific paths.

## Architecture

### Core Components

**CLDM (ControlLDM)** - `cldm/cldm.py`
- `ControlLDM_cat`: Main model class extending LatentDiffusion
- Integrates grayscale image control with text conditioning
- Uses cross-attention for text prompts and concatenation for grayscale hints

**LDM (Latent Diffusion Model)** - `ldm/` directory
- `ldm.models.autoencoder.AutoencoderKL_enhanceD`: VAE with enhanced decoder for grayscale encoding
- `ldm.modules.diffusionmodules.openaimodel.CatUNetModel`: Custom UNet with concatenation-based control
- `ldm.models.diffusion.ddim.DDIMSampler_withsam`: DDIM sampler with SAM (Segment Anything Model) mask support

**Dataset & Preprocessing** - `colorization_dataset.py`
- `MyDataset`: Handles train/val/test splits
- RGB to LAB color space conversion (grayscale L channel separated from color AB channels)
- Optional SAM mask integration for instance-aware colorization

### Model Configuration

Two primary configs in `configs/`:
- `cldm_v15_ehdecoder.yaml`: Training configuration (use_checkpoint=True)
- `cldm_sample.yaml`: Inference configuration (use_checkpoint=False for speed)

Key architecture parameters:
- Image size: 256x256 (64x64 in latent space with scale_factor=0.18215)
- UNet channels: 320, attention resolutions: [4, 2, 1]
- Gray encoder output: 512 channels
- CLIP text embeddings: 768-dim context

### Color Space Processing

The system operates in LAB color space:
- L channel (lightness): Used as grayscale input
- AB channels (color): Predicted by the model
- Conversion functions: `rgb2lab()`, `lab2rgb()` in `colorization_dataset.py`

## Common Commands

**NOTE:** All commands can be run using convenient bash scripts in the `scripts/` directory. See `scripts/README.md` for full documentation.

### Inference

**Basic inference with language prompts:**
```bash
./scripts/inference_basic.sh
# or: python colorization_main.py -m
```

**Instance-aware inference with SAM masks:**
```bash
./scripts/inference.sh
# or: python inference.py
```

The inference script:
- Loads model checkpoint from `cfg.largedecoder_checkpoint` (configured in `config.yaml`)
- Processes images from `cfg.example_img_dir` directory
- Uses test pairs from `cfg.sam_pairs_json` or `cfg.example_test_pairs`
- Outputs to directory specified by `cfg.test_output_template` with timestamp

### Training

**Start training:**
```bash
./scripts/train.sh
# or: python colorization_main.py -t
```

**Resume training:**
```bash
./scripts/train_resume.sh
# or: python colorization_main.py -t -r
```

Training configuration (loaded from `config.yaml`):
- Init model path: `cfg.init_model_path`
- GPU count: `cfg.n_gpu` (default: 2)
- Batch size: `cfg.batch_size` (default: 16)
- Learning rate: `cfg.learning_rate_multiplier * n_gpu` (default: 1e-5 per GPU)
- Dataset: Extended COCO-Stuff from `cfg.coco_img_dir`
- Annotations: `cfg.coco_caption_dir/caption_train.json`

### Testing Modes

**Validation on COCO val set:**
```bash
./scripts/validate.sh
# or: python colorization_main.py
```

**Multi-instance with SAM masks:**
```bash
./scripts/test_multicolor_sam.sh
# or: python colorization_main.py -m -s
```

## Model Weights

Model checkpoint paths are configured in `config.yaml` under the `models` section:
- `init_model`: Pre-trained initialization for training (default: `models/init_model.ckpt`)
- `resume_checkpoint`: Default test model (default: `models/auto_weight.ckpt`)
- `largedecoder_checkpoint`: Used by `inference.py` (must be set to your environment)

Download links in README.md (Baidu Pan and Google Drive)

## Key Implementation Details

### SAM Mask Generation

#### Traditional Approach (Pre-processing Stage)

**Important**: SAM masks must be generated BEFORE running inference. They are NOT created on-the-fly.

To generate SAM masks for new images:
```bash
./scripts/generate_sam_masks.sh
```

**Manual steps:**
1. Edit `sam_mask/make_mask.py` to add your image filenames to the `img_list` (line 7-14)
2. Ensure SAM model checkpoint exists: `models/sam_vit_h_4b8939.pth`
3. Run the generation script (shown above)
4. Review masks in `sam_mask/seg_img/` and select relevant ones
5. Copy selected masks to `sam_mask/select_masks/{image_name}/`
6. Create corresponding test pairs in `sam_mask/pairs.json` matching format: `[["image.jpg", "desc1, desc2, desc3"], ...]`

**Workflow**: The script loads SAM model once per image and generates ALL masks at once using `SamAutomaticMaskGenerator`. Each image gets multiple instance masks saved as individual `.npy` files.

#### Batch-Based Approach (NEW - Recommended for Large Datasets)

**NEW**: For processing many images with limited storage, use the batch-based pipeline that generates masks on-the-fly and automatically cleans them up.

**Quick Start:**
```bash
# Process images in batches of 100 (default)
./scripts/run_batch_inference.sh

# Or with custom batch size
python batch_inference.py --batch_size 100
```

**Key Features:**
- Processes images in batches (default: 100 images per batch)
- Generates SAM masks automatically for each batch
- Runs colorization inference immediately after mask generation
- Automatically deletes masks after each batch to free storage
- At any time, only masks for current batch exist on disk (~90% storage reduction)

**When to Use:**
- ✅ Processing many images (>100)
- ✅ Limited storage available
- ✅ Automated end-to-end processing
- ✅ Masks don't need to be reused

**Documentation:**
- Full guide: `BATCH_INFERENCE_README.md`
- Quick start: `BATCH_QUICKSTART.md`

**Pipeline Flow:**
```
For each batch:
  1. Generate SAM masks for 100 images → temp storage
  2. Run colorization inference → save results
  3. Delete SAM masks → free storage
  4. Repeat for next batch
```

**Storage Comparison:**
- Traditional: ~2.5 GB for 1000 images (all masks stored)
- Batch-based: ~250 MB peak usage (only current batch)

### Instance-Aware Sampling

When `use_sam=True`:
- SAM masks must be pre-generated and stored in `sam_mask/select_masks/{image_name}/`
- Multiple masks per image (one per instance), named as `{image_name}_{mask_id}.npy`
- Text prompts split by commas to match instances (one description per mask)
- `DDIMSampler_withsam` applies attention guidance per instance region during the 50 DDIM denoising steps
- Each mask corresponds to one comma-separated phrase in the text prompt
- Processing is sequential: batch_size=1, one image at a time with all its masks loaded together

### Text Tokenization

Text processing in `inference.py:65-75`:
- CLIP tokenizer splits text on comma delimiters
- `split_idx` tracks prompt boundaries for multi-instance scenarios
- Max token length: 77 (CLIP limit)

### Model Loading

Use helper functions from `cldm.model`:
- `create_model(config_path)`: Instantiates model from YAML
- `load_state_dict(ckpt_path, location='cpu')`: Loads checkpoints or safetensors

### Gray Encoder Integration

The grayscale encoder (`g_encoder`) in AutoencoderKL:
- Processes 3-channel grayscale input (L channel repeated)
- Output features concatenated to UNet at multiple resolutions
- Provides structural guidance to colorization process

## Dataset Structure

**Training/Validation:**
- Images: `{img_dir}/{split}2017/*.jpg`
- Captions: `{caption_dir}/caption_{split}.json`
- Format: `{"image_name.jpg": ["caption1", "caption2", ...]}`

**Testing:**
- Images: `example/*.jpg`
- Pairs: `example/test-pair.json` format: `[["image.jpg", "caption"], ...]`
- SAM masks: `sam_mask/select_masks/{image_name}/*.npy`

## Modifying for New Tasks

**To change image resolution:**
1. Update `img_size` in dataset initialization
2. Adjust `image_size` in model config (latent space = img_size/8)

**To use different text encoders:**
1. Modify `cond_stage_config` in YAML configs
2. Current: `ldm.modules.encoders.modules.FrozenCLIPEmbedder`

**To adjust sampling parameters:**
Edit in `config.yaml` under the `inference` section:
- `ddim_steps`: Number of denoising steps (50 default)
- `ddim_eta`: Stochasticity (0.0 = deterministic)
- `unconditional_guidance_scale`: Classifier-free guidance strength (5.0 default)
- `use_attn_guidance`: Enable/disable attention guidance (true default)

## Common Issues and Troubleshooting

**Configuration issues:**
- If `config.yaml` is missing, copy from `config.yaml.example`
- Update all paths in `config.yaml` to match your environment before running
- Model checkpoint paths under `models` section
- Dataset paths under `datasets` section

**SAM mask errors:**
- Ensure masks are pre-generated before running inference with `-s` flag
- Verify mask file naming: `{image_name_without_extension}_{mask_id}.npy`
- Check that number of comma-separated phrases in text matches number of mask files
- Masks must be in `sam_mask/select_masks/{image_name}/` directory structure

**Memory issues:**
- Reduce `batch_size` in `config.yaml` under `training` section (default: 16)
- Use `use_checkpoint=True` in model config for gradient checkpointing
- Adjust `n_gpu` in `config.yaml` under `training` section (default: 2)
- Enable memory saving: set `memory.save_memory: true` in `config.yaml`
