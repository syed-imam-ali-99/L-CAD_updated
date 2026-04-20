"""
Batch-based Inference Pipeline
Generates SAM masks and performs colorization inference in batches
to avoid storing all masks at once
"""

import os
import sys
import argparse
import shutil
import time
from pathlib import Path

import torch
import einops
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader

from share import *
from config import cfg
from cldm.model import create_model, load_state_dict
from ldm.models.diffusion.ddim import DDIMSampler_withsam
from colorization_dataset_batch import BatchMyDataset
from pair_utils import load_image_caption_pairs
from sam_mask.batch_mask_generator import BatchSamMaskGenerator


class BatchInferencePipeline:
    """
    Orchestrates batch-based SAM mask generation and colorization inference
    """

    def __init__(self, args):
        """
        Initialize the pipeline

        Args:
            args: Parsed command-line arguments
        """
        self.args = args
        self.batch_size = args.batch_size
        self.inference_batch_size = args.inference_batch_size

        # Paths
        self.sam_checkpoint = args.sam_checkpoint
        self.model_checkpoint = args.model_checkpoint
        self.img_dir = args.img_dir
        self.pairs_json = args.pairs_json
        self.temp_mask_dir = Path(args.temp_mask_dir)
        self.output_dir = Path(args.output_dir)

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load pairs to determine total number of images
        self.pairs = load_image_caption_pairs(self.pairs_json)

        self.total_images = len(self.pairs)
        print(f"Total images to process: {self.total_images}")
        print(f"Batch size: {self.batch_size}")
        print(f"Inference batch size: {self.inference_batch_size}")

        # Initialize SAM mask generator
        print("\nInitializing SAM mask generator...")
        self.mask_generator = BatchSamMaskGenerator(
            sam_checkpoint=self.sam_checkpoint,
            img_dir=self.img_dir,
            pairs_json=self.pairs_json,
            output_dir=str(self.temp_mask_dir),
            batch_size=self.batch_size
        )

        # Initialize colorization model
        print("\nInitializing colorization model...")
        self.model = self._load_model()

        print("\nPipeline initialized successfully!")

    def _load_model(self):
        """Load the colorization model"""
        model = create_model(cfg.cldm_sample_config).cpu()
        model.load_state_dict(load_state_dict(self.model_checkpoint, location='cpu'))
        model = model.cuda()
        model.usesam = True
        return model

    def save_images(self, samples, batch, prefix=''):
        """
        Save colorized images

        Args:
            samples: Tensor of colorized images
            batch: Batch dictionary containing metadata
            prefix: Prefix for output filenames
        """
        for i in range(samples.shape[0]):
            img_name = batch['name'][i]
            grid = samples[i].transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.cpu().numpy()
            grid = (grid * 255).astype(np.uint8)

            filename = prefix + '_' + img_name.replace("jpg", "png")
            path = self.output_dir / filename
            os.makedirs(path.parent, exist_ok=True)
            Image.fromarray(grid).save(path)

    def run_inference_on_batch(self, batch_start, batch_end):
        """
        Run colorization inference on a batch of images

        Args:
            batch_start: Start index of the batch
            batch_end: End index of the batch
        """
        print(f"\nRunning inference on batch: images {batch_start} to {batch_end - 1}")
        print("=" * 60)

        # Create batch indices
        batch_indices = list(range(batch_start, min(batch_end, self.total_images)))

        # Create dataset for this batch
        dataset = BatchMyDataset(
            img_dir=self.img_dir,
            caption_dir=None,
            split='test',
            use_sam=True,
            pairs_json=self.pairs_json,
            sam_mask_dir=str(self.temp_mask_dir),
            batch_indices=batch_indices
        )

        # Create dataloader
        dataloader = DataLoader(
            dataset,
            num_workers=self.args.num_workers,
            batch_size=self.inference_batch_size,
            shuffle=False
        )

        # Process each mini-batch
        processed_count = 0
        for batch_idx, batch in enumerate(dataloader):
            # Prepare control input
            control = batch[self.model.control_key]
            control = control.to(self.model.device)
            control = einops.rearrange(control, 'b h w c -> b c h w')
            N = control.shape[0]
            c_cat = control.to(memory_format=torch.contiguous_format).float()
            gray_z = self.model.first_stage_model.g_encoder(c_cat)

            # Prepare text conditioning
            xc = batch['txt']
            c = self.model.get_learned_conditioning(xc)

            # Get tokens and split indices
            tokens = self.model.cond_stage_model.tokenizer.tokenize(xc[0])
            split_idx = []
            for idx, token in enumerate(tokens):
                if token == ',</w>':
                    split_idx.append(idx + 1)
            if len(tokens) > 0:
                split_idx.append(len(tokens))

            # Get SAM masks
            sam_mask = batch['mask']

            # Prepare unconditional conditioning
            uc_cross = self.model.get_unconditional_conditioning(N)
            uc_cat = c_cat
            uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}

            # Run DDIM sampling
            ddim_sampler = DDIMSampler_withsam(self.model)
            cond = {"c_concat": [c_cat], "c_crossattn": [c]}
            b, c_dim, h, w = cond["c_concat"][0].shape
            shape = (self.model.channels, h // 8, w // 8)

            samples_cfg, intermediates = ddim_sampler.sample(
                self.args.ddim_steps,
                b,
                shape,
                cond,
                eta=self.args.ddim_eta,
                unconditional_guidance_scale=self.args.unconditional_guidance_scale,
                unconditional_conditioning=uc_full,
                verbose=False,
                use_attn_guidance=self.args.use_attn_guidance,
                sam_mask=sam_mask,
                split_id=split_idx,
                tokens=tokens
            )

            # Decode and save images
            x_samples = self.model.decode_first_stage(samples_cfg, gray_z)
            x_samples = torch.clamp(x_samples, -1., 1.)
            x_samples = (x_samples + 1.0) / 2.0

            self.save_images(x_samples, batch=batch, prefix='colorized')

            processed_count += len(batch['name'])
            print(f"  Processed {processed_count}/{len(batch_indices)} images in this batch")

        print("=" * 60)
        print(f"Inference complete for batch: {processed_count} images processed\n")

    def cleanup_batch_masks(self, batch_start, batch_end):
        """
        Delete masks for the current batch to free up storage

        Args:
            batch_start: Start index of the batch
            batch_end: End index of the batch
        """
        print(f"Cleaning up masks for batch: images {batch_start} to {batch_end - 1}")

        batch_pairs = self.pairs[batch_start:min(batch_end, self.total_images)]
        for img_name, _ in batch_pairs:
            img_mask_dir = self.temp_mask_dir / img_name.split('.')[0]
            if img_mask_dir.exists():
                shutil.rmtree(img_mask_dir)

        print("Cleanup complete.\n")

    def run(self):
        """
        Run the complete batch-based pipeline
        """
        print("\n" + "=" * 60)
        print("STARTING BATCH-BASED INFERENCE PIPELINE")
        print("=" * 60)

        start_time = time.time()

        # Get batch information
        batches = self.mask_generator.get_batch_info()
        num_batches = len(batches)

        print(f"\nTotal batches: {num_batches}")
        print(f"Processing {self.total_images} images in batches of {self.batch_size}\n")

        # Process each batch
        for batch_idx, (batch_start, batch_end, batch_num) in enumerate(batches):
            batch_time_start = time.time()

            print("\n" + "=" * 60)
            print(f"BATCH {batch_num}/{num_batches}")
            print("=" * 60)

            # Step 1: Generate SAM masks for this batch
            print(f"\n[Step 1/3] Generating SAM masks for batch {batch_num}...")
            batch_img_names = self.mask_generator.generate_batch(batch_start, batch_end)

            # Step 2: Run inference on this batch
            print(f"\n[Step 2/3] Running colorization inference on batch {batch_num}...")
            self.run_inference_on_batch(batch_start, batch_end)

            # Step 3: Cleanup masks for this batch
            print(f"\n[Step 3/3] Cleaning up masks for batch {batch_num}...")
            self.cleanup_batch_masks(batch_start, batch_end)

            batch_time_end = time.time()
            batch_duration = batch_time_end - batch_time_start

            print(f"\nBatch {batch_num}/{num_batches} completed in {batch_duration:.2f} seconds")
            print(f"Progress: {min(batch_end, self.total_images)}/{self.total_images} images "
                  f"({100 * min(batch_end, self.total_images) / self.total_images:.1f}%)")

        end_time = time.time()
        total_duration = end_time - start_time

        print("\n" + "=" * 60)
        print("BATCH-BASED INFERENCE PIPELINE COMPLETED")
        print("=" * 60)
        print(f"Total images processed: {self.total_images}")
        print(f"Total time: {total_duration:.2f} seconds")
        print(f"Average time per image: {total_duration / self.total_images:.2f} seconds")
        print(f"Output directory: {self.output_dir}")
        print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Batch-based SAM mask generation and colorization inference'
    )

    # Input/Output paths
    parser.add_argument('--sam_checkpoint', type=str,
                        default='models/sam_vit_h_4b8939.pth',
                        help='Path to SAM model checkpoint')
    parser.add_argument('--img_dir', type=str,
                        default='example',
                        help='Directory containing input images')
    parser.add_argument('--pairs_json', type=str,
                        default='sam_mask/pairs.json',
                        help='JSON file with image-caption pairs')
    parser.add_argument('--temp_mask_dir', type=str,
                        default='sam_mask/batch_masks',
                        help='Temporary directory for batch masks')
    parser.add_argument('--output_dir', type=str,
                        default='image_log/batch_inference',
                        help='Output directory for colorized images')
    parser.add_argument('--model_checkpoint', type=str,
                        default=cfg.resume_checkpoint,
                        help='Full L-CAD model checkpoint')

    # Batch settings
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Number of images per batch for mask generation')
    parser.add_argument('--inference_batch_size', type=int, default=1,
                        help='Batch size for inference (usually 1)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loading workers')

    # Inference hyperparameters
    parser.add_argument('--ddim_steps', type=int, default=50,
                        help='Number of DDIM sampling steps')
    parser.add_argument('--ddim_eta', type=float, default=0.0,
                        help='DDIM eta parameter')
    parser.add_argument('--unconditional_guidance_scale', type=float, default=5.0,
                        help='Unconditional guidance scale')
    parser.add_argument('--use_attn_guidance', action='store_true', default=True,
                        help='Use attention guidance')

    args = parser.parse_args()

    # Create and run pipeline
    pipeline = BatchInferencePipeline(args)
    pipeline.run()


if __name__ == "__main__":
    main()
