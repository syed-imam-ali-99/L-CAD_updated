"""
Batch-based SAM Mask Generator
Generates SAM masks for batches of images to avoid storing all masks at once
"""

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from PIL import Image
import numpy as np
import os
import argparse
from pathlib import Path
import shutil

from pair_utils import load_image_caption_pairs


class BatchSamMaskGenerator:
    """Generate SAM masks in batches to optimize storage"""

    def __init__(self, sam_checkpoint, img_dir, pairs_json, output_dir, batch_size=100):
        """
        Initialize the batch SAM mask generator

        Args:
            sam_checkpoint: Path to SAM model checkpoint
            img_dir: Directory containing input images
            pairs_json: JSON file containing image-caption pairs
            output_dir: Directory to store temporary mask files
            batch_size: Number of images to process in each batch
        """
        self.sam_checkpoint = sam_checkpoint
        self.img_dir = img_dir
        self.pairs_json = pairs_json
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load SAM model
        print(f"Loading SAM model from {sam_checkpoint}...")
        self.sam = sam_model_registry["default"](checkpoint=sam_checkpoint).cuda()
        self.mask_generator = SamAutomaticMaskGenerator(self.sam)

        # Load image-caption pairs
        self.pairs = load_image_caption_pairs(pairs_json)

        print(f"Loaded {len(self.pairs)} image-caption pairs")
        print(f"Batch size: {batch_size}")

    def generate_masks_for_image(self, img_name):
        """
        Generate SAM masks for a single image

        Args:
            img_name: Name of the image file

        Returns:
            List of mask dictionaries
        """
        img_path = os.path.join(self.img_dir, img_name)

        # Check if image exists
        if not os.path.exists(img_path):
            print(f"Warning: Image not found: {img_path}")
            return []

        # Load image
        img = np.array(Image.open(img_path).convert('RGB'))

        # Generate masks
        masks = self.mask_generator.generate(img)

        return masks

    def save_masks_for_image(self, img_name, masks):
        """
        Save masks for a single image

        Args:
            img_name: Name of the image file
            masks: List of mask dictionaries from SAM
        """
        # Create directory for this image's masks
        img_mask_dir = self.output_dir / img_name.split('.')[0]
        img_mask_dir.mkdir(parents=True, exist_ok=True)

        # Save each mask
        for i, mask in enumerate(masks):
            mask_path = img_mask_dir / f"{i}.npy"
            np.save(mask_path, mask['segmentation'])

    def generate_batch(self, batch_start, batch_end):
        """
        Generate masks for a batch of images

        Args:
            batch_start: Start index of the batch
            batch_end: End index of the batch (exclusive)

        Returns:
            List of image names processed in this batch
        """
        batch_pairs = self.pairs[batch_start:batch_end]
        batch_img_names = []

        print(f"\nProcessing batch {batch_start//self.batch_size + 1}: "
              f"images {batch_start} to {min(batch_end, len(self.pairs))-1}")
        print("=" * 60)

        for idx, (img_name, caption) in enumerate(batch_pairs):
            print(f"[{batch_start + idx + 1}/{len(self.pairs)}] Processing {img_name}...")

            # Generate masks for this image
            masks = self.generate_masks_for_image(img_name)

            if masks:
                # Save masks
                self.save_masks_for_image(img_name, masks)
                batch_img_names.append(img_name)
                print(f"  Generated {len(masks)} masks")
            else:
                print(f"  Warning: No masks generated")

        print("=" * 60)
        print(f"Batch complete: {len(batch_img_names)} images processed\n")

        return batch_img_names

    def cleanup_batch(self, batch_img_names):
        """
        Delete masks for a batch of images to free up storage

        Args:
            batch_img_names: List of image names whose masks should be deleted
        """
        print(f"Cleaning up masks for {len(batch_img_names)} images...")

        for img_name in batch_img_names:
            img_mask_dir = self.output_dir / img_name.split('.')[0]
            if img_mask_dir.exists():
                shutil.rmtree(img_mask_dir)

        print("Cleanup complete.\n")

    def get_batch_info(self):
        """
        Get information about batches

        Returns:
            List of tuples (batch_start, batch_end, batch_number)
        """
        total_images = len(self.pairs)
        batches = []

        for batch_num, batch_start in enumerate(range(0, total_images, self.batch_size)):
            batch_end = min(batch_start + self.batch_size, total_images)
            batches.append((batch_start, batch_end, batch_num + 1))

        return batches


def main():
    parser = argparse.ArgumentParser(description='Generate SAM masks in batches')
    parser.add_argument('--sam_checkpoint', type=str,
                        default='models/sam_vit_h_4b8939.pth',
                        help='Path to SAM model checkpoint')
    parser.add_argument('--img_dir', type=str,
                        default='example',
                        help='Directory containing input images')
    parser.add_argument('--pairs_json', type=str,
                        default='sam_mask/pairs.json',
                        help='JSON file with image-caption pairs')
    parser.add_argument('--output_dir', type=str,
                        default='sam_mask/batch_masks',
                        help='Directory to store temporary masks')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Number of images per batch')
    parser.add_argument('--batch_idx', type=int, default=None,
                        help='Specific batch index to process (0-based). If None, shows batch info.')

    args = parser.parse_args()

    # Create generator
    generator = BatchSamMaskGenerator(
        sam_checkpoint=args.sam_checkpoint,
        img_dir=args.img_dir,
        pairs_json=args.pairs_json,
        output_dir=args.output_dir,
        batch_size=args.batch_size
    )

    # Get batch information
    batches = generator.get_batch_info()

    if args.batch_idx is None:
        # Display batch information
        print("\n" + "=" * 60)
        print("BATCH INFORMATION")
        print("=" * 60)
        print(f"Total images: {len(generator.pairs)}")
        print(f"Batch size: {args.batch_size}")
        print(f"Number of batches: {len(batches)}\n")

        for batch_start, batch_end, batch_num in batches:
            print(f"Batch {batch_num}: images {batch_start} to {batch_end-1} "
                  f"({batch_end - batch_start} images)")

        print("\nTo generate masks for a specific batch, run:")
        print(f"python {__file__} --batch_idx <batch_number>")
        print("=" * 60 + "\n")
    else:
        # Process specific batch
        if args.batch_idx < 0 or args.batch_idx >= len(batches):
            print(f"Error: Invalid batch index {args.batch_idx}. "
                  f"Must be between 0 and {len(batches)-1}")
            return

        batch_start, batch_end, batch_num = batches[args.batch_idx]
        print(f"\nGenerating masks for batch {batch_num}")
        batch_img_names = generator.generate_batch(batch_start, batch_end)
        print(f"Successfully generated masks for {len(batch_img_names)} images")


if __name__ == "__main__":
    main()
