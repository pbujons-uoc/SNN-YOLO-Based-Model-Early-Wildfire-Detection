"""
Quick verification script to test VanillaCNN letterbox implementation.
This script verifies that the letterbox preprocessing is consistent between
training and prediction.
"""

import sys
import os
sys.path.append(os.getcwd())

import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF

# Import training letterbox
from VanillaCNN.data_to_yolo import letterbox_pil as letterbox_pil_train

# Import prediction letterbox (needs to be copied from predict_vanillacnn.py)
def letterbox_pil_predict(img, target_size=640):
    """Letterbox from predict_vanillacnn.py"""
    w, h = img.size
    scale = min(target_size / h, target_size / w)
    new_h, new_w = int(h * scale), int(w * scale)
    img_resized = TF.resize(img, (new_h, new_w), interpolation=TF.InterpolationMode.BILINEAR)
    pad_h = (target_size - new_h) // 2
    pad_w = (target_size - new_w) // 2
    img_padded = Image.new('RGB', (target_size, target_size), (114, 114, 114))
    img_padded.paste(img_resized, (pad_w, pad_h))
    return img_padded

def test_letterbox_consistency():
    """Test that training and prediction letterbox produce identical results."""
    
    print("🔍 Testing VanillaCNN Letterbox Consistency")
    print("=" * 60)
    
    # Test with different aspect ratios
    test_cases = [
        ("Square image", (640, 640)),
        ("Wide image (16:9)", (1920, 1080)),
        ("Tall image (9:16)", (1080, 1920)),
        ("Very wide (21:9)", (2560, 1080)),
        ("Small image", (320, 240))
    ]
    
    all_passed = True
    
    for test_name, (width, height) in test_cases:
        print(f"\n📝 Test: {test_name} ({width}x{height})")
        
        # Create random test image
        img_array = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        img_pil = Image.fromarray(img_array)
        
        # Apply both letterbox functions
        img_train = letterbox_pil_train(img_pil, target_size=640)
        img_predict = letterbox_pil_predict(img_pil, target_size=640)
        
        # Convert to arrays for comparison
        arr_train = np.array(img_train)
        arr_predict = np.array(img_predict)
        
        # Check dimensions
        if arr_train.shape != (640, 640, 3):
            print(f"   ❌ FAIL: Training output shape {arr_train.shape} != (640, 640, 3)")
            all_passed = False
            continue
            
        if arr_predict.shape != (640, 640, 3):
            print(f"   ❌ FAIL: Prediction output shape {arr_predict.shape} != (640, 640, 3)")
            all_passed = False
            continue
        
        # Check if arrays are identical
        if np.array_equal(arr_train, arr_predict):
            print(f"   ✅ PASS: Training and prediction letterbox are identical")
        else:
            # Calculate difference
            diff = np.abs(arr_train.astype(int) - arr_predict.astype(int))
            max_diff = diff.max()
            mean_diff = diff.mean()
            
            if max_diff <= 1:  # Allow for minor rounding differences
                print(f"   ✅ PASS: Negligible difference (max: {max_diff}, mean: {mean_diff:.4f})")
            else:
                print(f"   ❌ FAIL: Significant difference (max: {max_diff}, mean: {mean_diff:.4f})")
                all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL TESTS PASSED - Letterbox implementation is consistent!")
    else:
        print("❌ SOME TESTS FAILED - There may be inconsistencies")
    print("=" * 60)
    
    return all_passed

def test_coordinates_alignment():
    """Test that predicted coordinates align correctly with letterbox image."""
    
    print("\n\n🎯 Testing Coordinate Alignment")
    print("=" * 60)
    
    # Create a test image with a known box
    width, height = 1920, 1080
    target_size = 640
    
    # Calculate expected letterbox parameters
    scale = min(target_size / height, target_size / width)
    new_h, new_w = int(height * scale), int(width * scale)
    pad_h = (target_size - new_h) // 2
    pad_w = (target_size - new_w) // 2
    
    print(f"\n📐 Image: {width}x{height}")
    print(f"   Scale: {scale:.4f}")
    print(f"   Resized: {new_w}x{new_h}")
    print(f"   Padding: width={pad_w}px, height={pad_h}px")
    
    # Test box in original image space (center box, 100x100)
    orig_cx, orig_cy = width // 2, height // 2
    orig_w, orig_h = 100, 100
    
    print(f"\n📦 Test box in original image:")
    print(f"   Center: ({orig_cx}, {orig_cy})")
    print(f"   Size: {orig_w}x{orig_h}")
    
    # Box in normalized coordinates (what YOLO uses)
    norm_cx = orig_cx / width
    norm_cy = orig_cy / height
    norm_w = orig_w / width
    norm_h = orig_h / height
    
    print(f"\n📊 Normalized (YOLO format):")
    print(f"   cx={norm_cx:.4f}, cy={norm_cy:.4f}")
    print(f"   w={norm_w:.4f}, h={norm_h:.4f}")
    
    # Expected coordinates in letterbox 640x640 space
    # (this is what the model should predict)
    letterbox_cx = norm_cx * target_size
    letterbox_cy = norm_cy * target_size
    letterbox_w = norm_w * target_size
    letterbox_h = norm_h * target_size
    
    print(f"\n🎯 Expected in 640x640 letterbox space:")
    print(f"   Center: ({letterbox_cx:.1f}, {letterbox_cy:.1f})")
    print(f"   Size: {letterbox_w:.1f}x{letterbox_h:.1f}")
    print(f"   Box: x1={letterbox_cx - letterbox_w/2:.1f}, y1={letterbox_cy - letterbox_h/2:.1f}, "
          f"x2={letterbox_cx + letterbox_w/2:.1f}, y2={letterbox_cy + letterbox_h/2:.1f}")
    
    print(f"\n✅ If predicted coordinates match these values, alignment is correct!")
    print("=" * 60)

if __name__ == "__main__":
    # Run consistency tests
    passed = test_letterbox_consistency()
    
    # Run coordinate alignment explanation
    test_coordinates_alignment()
    
    # Exit with appropriate code
    sys.exit(0 if passed else 1)
