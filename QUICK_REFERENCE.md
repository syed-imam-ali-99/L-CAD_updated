# Batch-Based Pipeline - Quick Reference Card

## 🚀 Quick Start (3 Commands)

```bash
# 1. Create pairs JSON
cat > sam_mask/pairs.json << 'EOF'
[["image1.jpg", "object1, color1"], ["image2.jpg", "object2, color2"]]
EOF

# 2. Ensure SAM model exists
ls models/sam_vit_h_4b8939.pth

# 3. Run batch inference
./scripts/run_batch_inference.sh
```

## 📁 Files Created

| File | Size | Purpose |
|------|------|---------|
| `batch_inference.py` | 12K | Main orchestrator |
| `sam_mask/batch_mask_generator.py` | — | SAM mask generator |
| `colorization_dataset_batch.py` | 9.4K | Batch-aware dataset |
| `scripts/run_batch_inference.sh` | — | Shell wrapper |
| `BATCH_INFERENCE_README.md` | 14K | Full documentation |
| `BATCH_QUICKSTART.md` | 4.3K | Quick start guide |
| `BATCH_PIPELINE_DIAGRAM.md` | 24K | Visual diagrams |
| `IMPLEMENTATION_COMPLETE.md` | — | Completion checklist |

## ⚙️ Key Parameters

```bash
# Batch size (images per batch)
--batch_size 100

# Output directory
--output_dir image_log/batch_inference

# SAM checkpoint
--sam_checkpoint models/sam_vit_h_4b8939.pth

# Image directory
--img_dir example

# Pairs JSON
--pairs_json sam_mask/pairs.json
```

## 💾 Storage Savings

| Images | Traditional | Batch (100) | Savings |
|--------|-------------|-------------|---------|
| 100 | ~250 MB | ~250 MB | 0% |
| 1,000 | ~2.5 GB | ~250 MB | 90% |
| 10,000 | ~25 GB | ~250 MB | 99% |

## 🔄 Pipeline Flow

```
For each batch:
  1. Generate SAM masks (100 images) → temp storage
  2. Run colorization inference → save results
  3. Delete SAM masks → free storage
  4. Repeat
```

## 📊 Progress Output

```
BATCH 1/5
[Step 1/3] Generating SAM masks...
[Step 2/3] Running inference...
[Step 3/3] Cleaning up...
Progress: 100/450 (22.2%)
```

## ⚠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| SAM checkpoint not found | Download from Segment Anything |
| Pairs JSON not found | Create with format: `[["img.jpg", "desc"]]` |
| CUDA out of memory | Use `--inference_batch_size 1 --ddim_steps 30` |
| Permission denied | Run `chmod +x scripts/run_batch_inference.sh` |

## 📚 Documentation

- **Full Guide**: `BATCH_INFERENCE_README.md`
- **Quick Start**: `BATCH_QUICKSTART.md`
- **Diagrams**: `BATCH_PIPELINE_DIAGRAM.md`
- **Details**: `BATCH_IMPLEMENTATION_SUMMARY.md`

## ✅ Requirements Met

- ✓ Process in batches of 100 images
- ✓ Generate SAM masks per batch
- ✓ Run inference immediately
- ✓ Auto-delete masks after batch
- ✓ No precomputation
- ✓ Only current batch on disk
- ✓ Proper synchronization
- ✓ Handle edge cases (<100 images)

## 🎯 Use Cases

**Use Batch Pipeline:**
- Many images (>100)
- Limited storage
- Automated processing
- Masks not reused

**Use Traditional:**
- Few images (<50)
- Ample storage
- Manual mask selection
- Debugging

## 🔍 Verification

```bash
# Test installation
python3 test_batch_pipeline.py

# View batch information
python sam_mask/batch_mask_generator.py

# Run with custom settings
python batch_inference.py --batch_size 50
```

---

**Status**: ✅ COMPLETE & PRODUCTION READY

All requirements implemented, documented, and tested.
Storage reduction: ~90% | Backwards compatible: Yes
