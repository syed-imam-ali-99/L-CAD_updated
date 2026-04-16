# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

L-CAD (Language-based Colorization with Any-level Descriptions using Diffusion Priors) is a research implementation for automatic and language-guided image colorization using diffusion models. The system accepts grayscale images and natural language descriptions to generate plausible colorizations.

## Environment Setup

- Python 3.9
- PyTorch 1.12
- NVIDIA GPU + CUDA cuDNN required
- Install dependencies: `pip install -r requirements.txt`

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

### Inference

**Basic inference with language prompts:**
```bash
python colorization_main.py
```

**Instance-aware inference with SAM masks:**
```bash
python inference.py
```

The inference script:
- **Note**: Line 35 has hardcoded path `/data/swarnim/L-CAD/models/largedecoder-checkpoint.pth` - update this to match your setup
- Processes images from `example/` directory
- Uses test pairs from `example/test-pair.json` or `sam_mask/pairs.json`
- Outputs to `./image_log/test_YYYY-MM-DD-HH-MM-SS/`

### Training

**Start training:**
```bash
python colorization_main.py -t
```

**Resume training:**
```bash
python colorization_main.py -t -r
```

Training configuration (hardcoded in `colorization_main.py:66-74`):
- Requires init model: `models/init_model.ckpt`
- Default: 2 GPUs, batch_size=16, lr=1e-5 per GPU
- Dataset: Extended COCO-Stuff from `/data/cz-data/coco/`
- Annotations: `resources/coco/caption_train.json`

### Testing Modes

**Multi-instance colorization without SAM:**
```bash
python colorization_main.py -m
```

**Multi-instance with SAM masks:**
```bash
python colorization_main.py -m -s
```

**Validation:**
```bash
python colorization_main.py
```
(No flags runs validation on COCO val set)

## Model Weights

Models should be placed in `./models/`:
- `init_model.ckpt`: Pre-trained initialization for training
- `auto_weight.ckpt`: Default test model
- `largedecoder-checkpoint.pth`: Used by `inference.py`

Download links in README.md (Baidu Pan and Google Drive)

## Key Implementation Details

### SAM Mask Generation (Pre-processing Stage)

**Important**: SAM masks must be generated BEFORE running inference. They are NOT created on-the-fly.

To generate SAM masks for new images:
1. Edit `sam_mask/make_mask.py` to add your image filenames to the `img_list` (line 7-14)
2. Ensure SAM model checkpoint exists: `models/sam_vit_h_4b8939.pth`
3. Run the mask generation script:
```bash
cd sam_mask
python make_mask.py
```
4. Masks will be saved to `sam_mask/masks/` and visualizations to `sam_mask/seg_img/`
5. Manually select relevant masks and copy them to `sam_mask/select_masks/{image_name}/`
6. Create corresponding test pairs in `sam_mask/pairs.json` matching format: `[["image.jpg", "desc1, desc2, desc3"], ...]`

**Workflow**: The script loads SAM model once per image and generates ALL masks at once using `SamAutomaticMaskGenerator`. Each image gets multiple instance masks saved as individual `.npy` files.

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
Edit in `inference.py:49-51` or `colorization_main.py`:
- `ddim_steps`: Number of denoising steps (50 default)
- `ddim_eta`: Stochasticity (0.0 = deterministic)
- `unconditional_guidance_scale`: Classifier-free guidance strength (5.0 default)

## Common Issues and Troubleshooting

**Hardcoded paths in inference.py:**
- Line 35: Update `resume_path` to point to your model checkpoint location
- The script references `/data/swarnim/L-CAD/models/` which may not exist on your system

**SAM mask errors:**
- Ensure masks are pre-generated before running inference with `-s` flag
- Verify mask file naming: `{image_name_without_extension}_{mask_id}.npy`
- Check that number of comma-separated phrases in text matches number of mask files
- Masks must be in `sam_mask/select_masks/{image_name}/` directory structure

**Memory issues:**
- Reduce `batch_size` in training (currently 16 on line 70 of colorization_main.py)
- Use `use_checkpoint=True` in model config for gradient checkpointing
- Training requires 2 GPUs by default; adjust `n_gpu` on line 67 if needed

**Dataset path issues:**
- Training expects COCO dataset at `/data/cz-data/coco/` (line 83)
- Update hardcoded paths in `colorization_main.py` to match your data location
