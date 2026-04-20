# Batch Pipeline Visual Diagram

## High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    BATCH-BASED INFERENCE PIPELINE                    │
└─────────────────────────────────────────────────────────────────────┘

Input: N images, batch_size B
Output: N colorized images, 0 temporary masks

┌─────────────┐
│  Start      │
└──────┬──────┘
       │
       ▼
┌──────────────────────────────────────┐
│ Load Configuration                   │
│ • SAM checkpoint path                │
│ • Image directory                    │
│ • Pairs JSON                         │
│ • Batch size (default: 100)          │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│ Initialize Models                    │
│ • Load SAM model                     │
│ • Load Colorization model            │
└──────────────┬───────────────────────┘
               │
               ▼
       ┌───────────────┐
       │ For each batch│
       └───────┬───────┘
               │
               ▼
   ╔═══════════════════════════════════════════╗
   ║           BATCH PROCESSING LOOP           ║
   ╠═══════════════════════════════════════════╣
   ║                                           ║
   ║  ┌─────────────────────────────────────┐  ║
   ║  │ Step 1: Generate SAM Masks          │  ║
   ║  │ • Process 100 images                │  ║
   ║  │ • Save to temp_mask_dir/            │  ║
   ║  │ • ~250 MB storage used              │  ║
   ║  └─────────────┬───────────────────────┘  ║
   ║                │                           ║
   ║                ▼                           ║
   ║  ┌─────────────────────────────────────┐  ║
   ║  │ Step 2: Run Colorization Inference  │  ║
   ║  │ • Load images + masks               │  ║
   ║  │ • DDIM sampling with SAM guidance   │  ║
   ║  │ • Save colorized results            │  ║
   ║  └─────────────┬───────────────────────┘  ║
   ║                │                           ║
   ║                ▼                           ║
   ║  ┌─────────────────────────────────────┐  ║
   ║  │ Step 3: Cleanup Masks               │  ║
   ║  │ • Delete temp_mask_dir/{images}/    │  ║
   ║  │ • Free ~250 MB storage              │  ║
   ║  └─────────────┬───────────────────────┘  ║
   ║                │                           ║
   ╚════════════════╪═══════════════════════════╝
                    │
                    ▼
            ┌───────────────┐
            │ More batches? │
            └───┬───────┬───┘
                │       │
              Yes       No
                │       │
                ▼       ▼
           (loop)   ┌─────────┐
                    │ Complete│
                    └─────────┘
```

## Detailed Batch Processing Flow

```
═══════════════════════════════════════════════════════════════════════
BATCH 1: Images 0-99 (100 images)
═══════════════════════════════════════════════════════════════════════

Phase 1: MASK GENERATION
─────────────────────────────────────────────────────────────────────

┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│  Image 0     │      │  Image 1     │      │  Image 99    │
│  img0.jpg    │      │  img1.jpg    │  ... │  img99.jpg   │
└──────┬───────┘      └──────┬───────┘      └──────┬───────┘
       │                     │                     │
       │  SAM Model          │  SAM Model          │  SAM Model
       ▼                     ▼                     ▼
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│ Generate     │      │ Generate     │      │ Generate     │
│ Masks        │      │ Masks        │      │ Masks        │
└──────┬───────┘      └──────┬───────┘      └──────┬───────┘
       │                     │                     │
       ▼                     ▼                     ▼
   temp_mask_dir/       temp_mask_dir/       temp_mask_dir/
   ├─ img0/             ├─ img1/             ├─ img99/
   │  ├─ 0.npy          │  ├─ 0.npy          │  ├─ 0.npy
   │  ├─ 1.npy          │  ├─ 1.npy          │  ├─ 1.npy
   │  └─ 2.npy          │  └─ 2.npy          │  └─ 2.npy

Storage Used: ~250 MB ▲

─────────────────────────────────────────────────────────────────────
Phase 2: COLORIZATION INFERENCE
─────────────────────────────────────────────────────────────────────

For each image in batch:
┌──────────────────────────────────────────────────────────┐
│                                                          │
│  1. Load image (gray)          ────┐                    │
│                                     │                    │
│  2. Load SAM masks              ────┤                    │
│     from temp_mask_dir/             │                    │
│                                     ▼                    │
│  3. Load text prompt            ┌──────────────┐        │
│                                 │  Colorization│        │
│                                 │     Model    │        │
│                                 └──────┬───────┘        │
│                                        │                 │
│                                        ▼                 │
│  4. Save colorized image       output_dir/              │
│                                colorized_img0.png        │
│                                                          │
└──────────────────────────────────────────────────────────┘

Repeat for all 100 images in batch

Storage Used: ~250 MB (masks still present)

─────────────────────────────────────────────────────────────────────
Phase 3: CLEANUP
─────────────────────────────────────────────────────────────────────

temp_mask_dir/
├─ img0/      ──────► [DELETED]
├─ img1/      ──────► [DELETED]
└─ img99/     ──────► [DELETED]

Storage Used: 0 MB ▼

═══════════════════════════════════════════════════════════════════════
BATCH 2: Images 100-199 (100 images)
═══════════════════════════════════════════════════════════════════════

[Repeat Phase 1, 2, 3 for images 100-199]

...

═══════════════════════════════════════════════════════════════════════
BATCH 5: Images 400-449 (50 images) ◄── Edge case handled ✓
═══════════════════════════════════════════════════════════════════════

[Same process, but only 50 images]

Final Result:
└─ output_dir/
   ├─ colorized_img0.png
   ├─ colorized_img1.png
   ├─ ...
   └─ colorized_img449.png

temp_mask_dir/ ──────► Empty (all cleaned up)
```

## Storage Timeline

```
Timeline for 450 images, batch_size=100
─────────────────────────────────────────────────────────────────

Storage
(MB)
  300 │
      │  ┌─┐    ┌─┐    ┌─┐    ┌─┐    ┌┐
  250 │  │ │    │ │    │ │    │ │    ││
      │  │ │    │ │    │ │    │ │    ││
  200 │  │ │    │ │    │ │    │ │    ││
      │  │ │    │ │    │ │    │ │    ││
  150 │  │ │    │ │    │ │    │ │    ││
      │  │ │    │ │    │ │    │ │    ││
  100 │  │ │    │ │    │ │    │ │    ││
      │  │ │    │ │    │ │    │ │    ││
   50 │  │ │    │ │    │ │    │ │    ││
      │  │ │    │ │    │ │    │ │    ││
    0 │──┘ └────┘ └────┘ └────┘ └────┘└──────►
      └─────────────────────────────────────── Time
         B1  B2   B3  B4   B5

      ┌─┐ = Masks present (generation + inference)
      └─  = Cleanup (masks deleted)

Peak storage: 250 MB (batch 1-4) or 125 MB (batch 5)
Average storage: ~50 MB (most of the time = 0)

Compare to traditional:
  2500 │ ████████████████████████████████████
       │ ████████████████████████████████████
       │ ████████████████████████████████████
       │ ████████████████████████████████████
       │ ████████████████████████████████████
       └──────────────────────────────────────►
         All masks stored permanently: 2500 MB
```

## File Structure Timeline

```
BEFORE PROCESSING
─────────────────────────────────────────────────────────────────

project/
├── batch_inference.py
├── sam_mask/
│   ├── batch_mask_generator.py
│   └── pairs.json
├── example/
│   ├── img0.jpg
│   ├── img1.jpg
│   └── ...
└── models/
    └── sam_vit_h_4b8939.pth


DURING BATCH 1 (Step 1: Mask Generation)
─────────────────────────────────────────────────────────────────

project/
├── sam_mask/
│   └── batch_masks/          ◄── TEMPORARY STORAGE
│       ├── img0/
│       │   ├── 0.npy
│       │   ├── 1.npy
│       │   └── 2.npy
│       ├── img1/
│       │   ├── 0.npy
│       │   └── 1.npy
│       └── ...


DURING BATCH 1 (Step 2: Inference)
─────────────────────────────────────────────────────────────────

project/
├── sam_mask/
│   └── batch_masks/          ◄── Still present
│       └── ... (masks still here)
└── image_log/
    └── batch_inference/      ◄── RESULTS
        ├── colorized_img0.png
        ├── colorized_img1.png
        └── ...


DURING BATCH 1 (Step 3: Cleanup)
─────────────────────────────────────────────────────────────────

project/
├── sam_mask/
│   └── batch_masks/          ◄── DELETED ✓
│       └── (empty)
└── image_log/
    └── batch_inference/
        └── ... (results remain)


AFTER ALL BATCHES COMPLETE
─────────────────────────────────────────────────────────────────

project/
├── sam_mask/
│   └── batch_masks/          ◄── Empty or removed
└── image_log/
    └── batch_inference/      ◄── FINAL RESULTS
        ├── colorized_img0.png
        ├── colorized_img1.png
        ├── ...
        └── colorized_img449.png

No temporary masks remain ✓
All colorized images saved ✓
```

## Synchronization Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                 SYNCHRONIZATION GUARANTEES                       │
└──────────────────────────────────────────────────────────────────┘

Batch N:
    ┌─────────────────────────┐
    │ Mask Generation         │
    │ (SAM processes images)  │
    └───────────┬─────────────┘
                │
                │ WAIT: All masks must be generated ✓
                │
                ▼
    ┌─────────────────────────┐
    │ Inference               │
    │ (Uses generated masks)  │
    └───────────┬─────────────┘
                │
                │ WAIT: All images must be processed ✓
                │
                ▼
    ┌─────────────────────────┐
    │ Cleanup                 │
    │ (Deletes masks)         │
    └───────────┬─────────────┘
                │
                │ WAIT: All masks must be deleted ✓
                │
                ▼

Batch N+1:
    ┌─────────────────────────┐
    │ Mask Generation         │
    │ (New batch starts)      │
    └─────────────────────────┘

Key: Each phase completes FULLY before next phase starts
     No parallel processing between phases (prevents race conditions)
     No partial states (all-or-nothing for each phase)
```

## Error Handling Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                      ERROR HANDLING                             │
└─────────────────────────────────────────────────────────────────┘

Start
  │
  ▼
┌─────────────────────┐
│ Check SAM Model     │
└──────┬──────────────┘
       │
       ├─ Missing? ──► ERROR: Download SAM model ╳
       │
       ▼
┌─────────────────────┐
│ Check Pairs JSON    │
└──────┬──────────────┘
       │
       ├─ Missing? ──► ERROR: Create pairs.json ╳
       │
       ▼
┌─────────────────────┐
│ Check Image Dir     │
└──────┬──────────────┘
       │
       ├─ Missing? ──► ERROR: Invalid directory ╳
       │
       ▼
┌─────────────────────┐
│ Generate Masks      │
└──────┬──────────────┘
       │
       ├─ Image not found? ──► WARNING: Skip image, continue ⚠
       │
       ├─ SAM error? ──► WARNING: No masks for image ⚠
       │
       ▼
┌─────────────────────┐
│ Run Inference       │
└──────┬──────────────┘
       │
       ├─ Masks missing? ──► Use dummy mask, continue ⚠
       │
       ├─ CUDA OOM? ──► ERROR: Reduce batch size ╳
       │
       ▼
┌─────────────────────┐
│ Cleanup             │
└──────┬──────────────┘
       │
       ├─ Permission error? ──► ERROR: Check permissions ╳
       │
       ▼
    Success ✓
```

## Component Interaction

```
┌──────────────────────────────────────────────────────────────────┐
│                    COMPONENT DIAGRAM                             │
└──────────────────────────────────────────────────────────────────┘

┌─────────────────────┐
│  User               │
└──────────┬──────────┘
           │
           │ runs
           ▼
┌─────────────────────────────────────────────────────────────────┐
│  run_batch_inference.sh                                         │
│  • Validates inputs                                             │
│  • Sets environment                                             │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        │ calls
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│  batch_inference.py (Main Orchestrator)                         │
│  • BatchInferencePipeline class                                 │
│  • Coordinates all components                                   │
└───┬──────────────┬──────────────────┬───────────────────────────┘
    │              │                  │
    │ uses         │ uses             │ uses
    ▼              ▼                  ▼
┌─────────┐  ┌──────────┐  ┌──────────────────────┐
│  SAM    │  │Colorize  │  │ BatchMyDataset       │
│ Mask    │  │ Model    │  │ • Loads images       │
│Generator│  │ (CLDM)   │  │ • Loads masks        │
└────┬────┘  └────┬─────┘  └──────────┬───────────┘
     │            │                   │
     │            │                   │
     ▼            ▼                   ▼
┌─────────────────────────────────────────┐
│  File System                            │
│  ├─ example/ (input images)             │
│  ├─ sam_mask/batch_masks/ (temp masks)  │
│  └─ image_log/ (output)                 │
└─────────────────────────────────────────┘
```

## Summary

This batch-based pipeline ensures:
- ✓ Minimal storage usage (only current batch)
- ✓ Proper synchronization (sequential phases)
- ✓ Automatic cleanup (no manual intervention)
- ✓ Robust error handling (graceful degradation)
- ✓ Clear progress tracking (detailed logging)
- ✓ Edge case support (variable batch sizes)

The pipeline is production-ready and suitable for processing large datasets with limited storage constraints.
