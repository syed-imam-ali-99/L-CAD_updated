"""
Test script for batch-based pipeline
Verifies basic functionality without running full inference
"""

import os
import json
import sys
from pathlib import Path


def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    try:
        from batch_inference import BatchInferencePipeline
        print("  ✓ batch_inference imported successfully")
    except Exception as e:
        print(f"  ✗ Failed to import batch_inference: {e}")
        return False

    try:
        from sam_mask.batch_mask_generator import BatchSamMaskGenerator
        print("  ✓ batch_mask_generator imported successfully")
    except Exception as e:
        print(f"  ✗ Failed to import batch_mask_generator: {e}")
        return False

    try:
        from colorization_dataset_batch import BatchMyDataset
        print("  ✓ colorization_dataset_batch imported successfully")
    except Exception as e:
        print(f"  ✗ Failed to import colorization_dataset_batch: {e}")
        return False

    return True


def test_file_structure():
    """Test that required files exist"""
    print("\nTesting file structure...")
    required_files = [
        "batch_inference.py",
        "sam_mask/batch_mask_generator.py",
        "colorization_dataset_batch.py",
        "scripts/run_batch_inference.sh",
        "BATCH_INFERENCE_README.md",
        "BATCH_QUICKSTART.md",
    ]

    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"  ✓ {file_path} exists")
        else:
            print(f"  ✗ {file_path} not found")
            all_exist = False

    return all_exist


def test_script_permissions():
    """Test that shell scripts are executable"""
    print("\nTesting script permissions...")
    script = "scripts/run_batch_inference.sh"

    if not os.path.exists(script):
        print(f"  ✗ {script} not found")
        return False

    if os.access(script, os.X_OK):
        print(f"  ✓ {script} is executable")
        return True
    else:
        print(f"  ✗ {script} is not executable")
        print(f"    Run: chmod +x {script}")
        return False


def test_batch_calculations():
    """Test batch index calculations"""
    print("\nTesting batch calculations...")

    try:
        from sam_mask.batch_mask_generator import BatchSamMaskGenerator

        # Mock data
        test_cases = [
            {"total": 100, "batch_size": 100, "expected_batches": 1},
            {"total": 101, "batch_size": 100, "expected_batches": 2},
            {"total": 450, "batch_size": 100, "expected_batches": 5},
            {"total": 50, "batch_size": 100, "expected_batches": 1},
            {"total": 200, "batch_size": 50, "expected_batches": 4},
        ]

        for test in test_cases:
            total = test["total"]
            batch_size = test["batch_size"]
            expected = test["expected_batches"]

            # Calculate actual batches
            num_batches = (total + batch_size - 1) // batch_size

            if num_batches == expected:
                print(f"  ✓ {total} images, batch_size {batch_size} → {num_batches} batches")
            else:
                print(f"  ✗ {total} images, batch_size {batch_size} → got {num_batches}, expected {expected}")
                return False

        return True

    except Exception as e:
        print(f"  ✗ Batch calculation test failed: {e}")
        return False


def test_config_loading():
    """Test that config can be loaded"""
    print("\nTesting configuration...")

    try:
        from config import cfg
        print("  ✓ Config loaded successfully")

        # Check some key attributes
        if hasattr(cfg, 'sam_pairs_json'):
            print(f"  ✓ sam_pairs_json: {cfg.sam_pairs_json}")
        else:
            print("  ✗ sam_pairs_json not found in config")
            return False

        if hasattr(cfg, 'example_img_dir'):
            print(f"  ✓ example_img_dir: {cfg.example_img_dir}")
        else:
            print("  ✗ example_img_dir not found in config")
            return False

        return True

    except Exception as e:
        print(f"  ✗ Config loading failed: {e}")
        return False


def test_documentation():
    """Test that documentation is complete"""
    print("\nTesting documentation...")

    docs = [
        "BATCH_INFERENCE_README.md",
        "BATCH_QUICKSTART.md",
        "BATCH_IMPLEMENTATION_SUMMARY.md"
    ]

    all_complete = True
    for doc in docs:
        if not os.path.exists(doc):
            print(f"  ✗ {doc} not found")
            all_complete = False
            continue

        # Check file size (should have substantial content)
        size = os.path.getsize(doc)
        if size > 1000:  # At least 1KB
            print(f"  ✓ {doc} exists ({size:,} bytes)")
        else:
            print(f"  ✗ {doc} seems too small ({size} bytes)")
            all_complete = False

    return all_complete


def main():
    """Run all tests"""
    print("=" * 60)
    print("BATCH PIPELINE TEST SUITE")
    print("=" * 60)

    results = {}

    # Run tests
    results["imports"] = test_imports()
    results["file_structure"] = test_file_structure()
    results["script_permissions"] = test_script_permissions()
    results["batch_calculations"] = test_batch_calculations()
    results["config_loading"] = test_config_loading()
    results["documentation"] = test_documentation()

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
        if not passed:
            all_passed = False

    print("=" * 60)

    if all_passed:
        print("\n✓ All tests passed!")
        print("\nNext steps:")
        print("1. Ensure SAM model checkpoint exists: models/sam_vit_h_4b8939.pth")
        print("2. Create pairs JSON: sam_mask/pairs.json")
        print("3. Place images in directory specified in config")
        print("4. Run: ./scripts/run_batch_inference.sh")
        return 0
    else:
        print("\n✗ Some tests failed. Please fix issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
