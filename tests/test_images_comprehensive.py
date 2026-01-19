"""
Comprehensive image testing script for TFM.

Two test modes:
1. Small test (--test-size small): 12 carefully selected images from test dataset
   - 3 images with no objects (empty)
   - 3 images with only smoke
   - 3 images with only fire
   - 3 images with both fire and smoke
   - Outputs: metrics CSV + visualization images with bounding boxes

2. Full test (--test-size full): All images in test dataset
   - Outputs: metrics CSV + energy CSV only (no visualization images)

Outputs per model:
- results_test_images.csv: Metrics (mAP, precision, recall, etc.)
- energy_results.csv: Energy consumption and timing
- [small mode only] Predicted and ground truth images with bounding boxes
- [small mode only] For encoded models: spike encoding visualizations

Usage:
    python tests/test_images_comprehensive.py --dataset D-Fire --test-size small
    python tests/test_images_comprehensive.py --dataset D-Fire --test-size full
    python tests/test_images_comprehensive.py --dataset D-Fire --test-size full --models spikeyolo spikeyolo_latency
"""

import argparse
import os
import cv2
import numpy as np
import torch
import json
import csv
from pathlib import Path
from tqdm import tqdm
from codecarbon import EmissionsTracker
import time
from collections import defaultdict
import random

from model_utils import ModelWrapper, project_root, DEFAULT_WEIGHTS


# ==============================
# ADAPTIVE CONFIDENCE THRESHOLDS
# ==============================
# Optimized confidence thresholds per model based on training results
# - YOLOv8 & SpikeYOLO: High performance, use standard threshold
# - Encoded variants: Lower threshold to compensate for poor recall
ADAPTIVE_CONFIDENCE_THRESHOLDS = {
    'yolov8': 0.25,
    'spikeyolo': 0.25,
    'spikeyolo_latency': 0.12,   # Lower to improve recall
    'spikeyolo_poisson': 0.12,   # Lower to improve recall
    'vanillacnn': 0.15,
}

# ==============================
# MODEL TO DATASET MAPPING
# ==============================
# Maps each model to its corresponding pre-encoded test dataset
# RGB models use original D-Fire, encoded models use pre-encoded datasets
MODEL_DATASET_MAPPING = {
    'yolov8': 'D-Fire',
    'spikeyolo': 'D-Fire',
    'vanillacnn': 'D-Fire',
    'spikeyolo_latency': 'latency_4',
    'spikeyolo_poisson': 'poisson_4',
}


# ==============================
# HELPER FUNCTIONS
# ==============================

def letterbox_resize_tensor(tensor, target_size=640):
    """
    Resize spike tensor using letterbox (maintain aspect ratio, add padding).
    Same method used by YOLO during training.
    
    Args:
        tensor: [T, C, H, W] spike tensor
        target_size: target size (default 640)
    
    Returns:
        resized_tensor: [T, C, target_size, target_size]
        scale_ratio: scaling factor applied
        (pad_w, pad_h): padding added (left/right, top/bottom)
    """
    T, C, h, w = tensor.shape
    
    # Calculate scale ratio to fit image in target_size maintaining aspect ratio
    scale = min(target_size / h, target_size / w)
    
    # New dimensions after scaling
    new_h = int(h * scale)
    new_w = int(w * scale)
    
    # Resize each channel independently
    # Reshape to [T*C, 1, H, W] for interpolation
    tensor_reshaped = tensor.reshape(T * C, 1, h, w).float()
    
    # Resize maintaining aspect ratio
    tensor_scaled = torch.nn.functional.interpolate(
        tensor_reshaped,
        size=(new_h, new_w),
        mode='nearest'
    )
    
    # Calculate padding to center the image
    pad_h = (target_size - new_h) // 2
    pad_w = (target_size - new_w) // 2
    pad_h_extra = target_size - new_h - pad_h
    pad_w_extra = target_size - new_w - pad_w
    
    # Apply padding (left, right, top, bottom)
    tensor_padded = torch.nn.functional.pad(
        tensor_scaled,
        (pad_w, pad_w_extra, pad_h, pad_h_extra),
        mode='constant',
        value=0
    )
    
    # Reshape back to [T, C, target_size, target_size]
    tensor_final = tensor_padded.reshape(T, C, target_size, target_size)
    
    return tensor_final.to(tensor.dtype), scale, (pad_w, pad_h)


def decode_spikes_to_image(spikes_tensor, encoding_type='poisson'):
    """
    Reconstructs an RGB image from a spike tensor [T, C, H, W].
    Same as verification.py for consistency.
    """
    T, C, H, W = spikes_tensor.shape
    spikes_tensor = spikes_tensor.float()
    
    if encoding_type in ['poisson', 'rate']:
        img_float = spikes_tensor.mean(dim=0)  # [C, H, W]
        img_float = img_float.permute(1, 2, 0)  # [H, W, C]
        img_np = img_float.cpu().numpy()
        img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        img_np = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        return img_np
    
    elif encoding_type == 'latency':
        first_spike_times = torch.argmax(spikes_tensor.float(), dim=0)  # [C, H, W]
        any_spike = torch.any(spikes_tensor, dim=0).float()
        if T > 1:
            img_float = 1.0 - (first_spike_times.float() / (T - 1))
        else:
            img_float = torch.ones_like(first_spike_times).float()
        img_float = img_float * any_spike
        img_float = img_float.permute(1, 2, 0)
        img_np = img_float.cpu().numpy()
        img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        img_np = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        return img_np
    
    else:
        raise ValueError(f"Unknown encoding type: {encoding_type}")


def parse_yolo_label(label_path, img_h, img_w):
    """
    Parses YOLO format label file.
    Returns: list of (cls, x1, y1, x2, y2) in pixel coordinates.
    """
    if not os.path.exists(label_path):
        return []
    
    boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            x_c, y_c, w, h = map(float, parts[1:5])
            
            x_c *= img_w
            y_c *= img_h
            w *= img_w
            h *= img_h
            
            x1 = int(x_c - w/2)
            y1 = int(y_c - h/2)
            x2 = int(x_c + w/2)
            y2 = int(y_c + h/2)
            
            boxes.append((cls, x1, y1, x2, y2))
    
    return boxes


def draw_boxes_on_image(img, boxes, class_names={0: 'smoke', 1: 'fire'}, show_conf=False):
    """
    Draws bounding boxes on image.
    boxes: list of (cls, x1, y1, x2, y2) or (cls, x1, y1, x2, y2, conf)
    """
    img_copy = img.copy()
    
    for box in boxes:
        if len(box) == 5:
            cls, x1, y1, x2, y2 = box
            conf = None
        else:
            cls, x1, y1, x2, y2, conf = box
        
        color = (0, 0, 255) if cls == 1 else (255, 0, 0)  # fire=red, smoke=blue
        label = class_names.get(cls, str(cls))
        
        if conf is not None and show_conf:
            label = f"{label} {conf:.2f}"
        
        cv2.rectangle(img_copy, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        
        # Add label background
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        c2 = (int(x1) + t_size[0], int(y1) - t_size[1] - 3)
        cv2.rectangle(img_copy, (int(x1), int(y1)), c2, color, -1)
        cv2.putText(img_copy, label, (int(x1), int(y1) - 2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return img_copy


def analyze_label_file(label_path):
    """
    Returns category: 'empty', 'smoke_only', 'fire_only', 'both'
    """
    if not os.path.exists(label_path):
        return 'empty'
    
    has_smoke = False
    has_fire = False
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            if cls == 0:
                has_smoke = True
            elif cls == 1:
                has_fire = True
    
    if not has_smoke and not has_fire:
        return 'empty'
    elif has_smoke and not has_fire:
        return 'smoke_only'
    elif has_fire and not has_smoke:
        return 'fire_only'
    else:
        return 'both'


def get_all_test_images(dataset_path):
    """
    Gets all images from test dataset.
    Returns: list of image_filenames (stems only, no extension)
    """
    test_images_dir = Path(dataset_path) / "test" / "images"
    
    if not test_images_dir.exists():
        raise FileNotFoundError(f"Test images directory not found: {test_images_dir}")
    
    all_images = []
    for img_path in sorted(test_images_dir.glob("*")):
        if img_path.suffix.lower() not in ['.jpg', '.png', '.jpeg', '.pt']:
            continue
        all_images.append(img_path.stem)  # Store only stem
    
    print(f"Found {len(all_images)} test images")
    return all_images


def select_test_images(dataset_path, images_needed_per_category=3):
    """
    Selects 12 images: 3 per category (empty, smoke_only, fire_only, both).
    Returns: dict with category -> list of image_filenames (stems only, no extension)
    """
    test_images_dir = Path(dataset_path) / "test" / "images"
    test_labels_dir = Path(dataset_path) / "test" / "labels"
    
    if not test_images_dir.exists():
        raise FileNotFoundError(f"Test images directory not found: {test_images_dir}")
    
    # Categorize all images by filename stem
    categories = defaultdict(list)
    
    for img_path in sorted(test_images_dir.glob("*")):
        if img_path.suffix.lower() not in ['.jpg', '.png', '.jpeg']:
            continue
        
        label_path = test_labels_dir / (img_path.stem + '.txt')
        category = analyze_label_file(label_path)
        categories[category].append(img_path.stem)  # Store only stem
    
    # Report available images
    print("\nAvailable images per category:")
    for cat in ['empty', 'smoke_only', 'fire_only', 'both']:
        print(f"  {cat:15s}: {len(categories[cat])} images")
    
    # Select N images per category - randomly selected
    selected = {}
    for cat in ['empty', 'smoke_only', 'fire_only', 'both']:
        available = categories[cat]
        if len(available) < images_needed_per_category:
            print(f"  WARNING: Only {len(available)} images available for '{cat}', requested {images_needed_per_category}")
            selected[cat] = available
        else:
            # Randomly select images from the available set
            selected[cat] = random.sample(available, images_needed_per_category)
            # Show which images were selected (by filename)
            print(f"  {cat}: Randomly selected {selected[cat]}")
    
    return selected


def iou(boxA, boxB):
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    denom = float(boxAArea + boxBArea - interArea)
    if denom == 0:
        return 0.0
    return interArea / denom


def compute_ap_per_class(predictions, ground_truths, cls, iou_threshold=0.5):
    """
    Compute Average Precision for a single class at a given IoU threshold.
    Uses 101-point interpolation (COCO-style).
    
    Args:
        predictions: dict {img_name: [(cls, x1, y1, x2, y2, conf), ...]}
        ground_truths: dict {img_name: [(cls, x1, y1, x2, y2), ...]}
        cls: class ID to compute AP for
        iou_threshold: IoU threshold for positive detection
    
    Returns:
        ap: Average Precision value
    """
    # Collect all predictions and ground truths for this class
    all_preds = []  # [(conf, img_name, pred_idx, pred_box)]
    gt_count = 0
    gt_per_image = {}  # {img_name: [gt_boxes]}
    
    for img_name in ground_truths.keys():
        gt_boxes = [gt for gt in ground_truths.get(img_name, []) if gt[0] == cls]
        gt_per_image[img_name] = gt_boxes
        gt_count += len(gt_boxes)
        
        pred_boxes = predictions.get(img_name, [])
        for i, pred in enumerate(pred_boxes):
            if pred[0] == cls:
                all_preds.append((pred[5], img_name, i, pred[1:5]))
    
    if gt_count == 0:
        return 0.0
    
    if len(all_preds) == 0:
        return 0.0
    
    # Sort predictions by confidence (descending)
    all_preds.sort(key=lambda x: x[0], reverse=True)
    
    # Track which ground truths have been matched
    matched = {img_name: [False] * len(gt_per_image[img_name]) for img_name in gt_per_image.keys()}
    
    tp = []
    fp = []
    
    for conf, img_name, pred_idx, pred_box in all_preds:
        gt_boxes = gt_per_image.get(img_name, [])
        
        best_iou = 0
        best_idx = -1
        
        for j, gt in enumerate(gt_boxes):
            if matched[img_name][j]:
                continue
            
            gt_box = gt[1:5]
            iou_val = iou(pred_box, gt_box)
            
            if iou_val > best_iou:
                best_iou = iou_val
                best_idx = j
        
        if best_iou >= iou_threshold and best_idx >= 0:
            if not matched[img_name][best_idx]:
                tp.append(1)
                fp.append(0)
                matched[img_name][best_idx] = True
            else:
                tp.append(0)
                fp.append(1)
        else:
            tp.append(0)
            fp.append(1)
    
    # Cumulative sums
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    recalls = tp_cumsum / gt_count
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
    
    # Add sentinel values at beginning and end
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([1.0], precisions, [0.0]))
    
    # Make precision monotonically decreasing
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    
    # Compute AP using 101-point interpolation
    recall_thresholds = np.linspace(0, 1, 101)
    ap = 0.0
    for t in recall_thresholds:
        # Find precision at this recall level
        idx = np.searchsorted(recalls, t, side='left')
        if idx < len(precisions):
            ap += precisions[idx]
    ap /= 101
    
    return ap


def compute_map(predictions, ground_truths, num_classes=2):
    """
    Compute mAP50 and mAP50-95.
    
    Args:
        predictions: dict {img_name: [(cls, x1, y1, x2, y2, conf), ...]}
        ground_truths: dict {img_name: [(cls, x1, y1, x2, y2), ...]}
        num_classes: number of classes (default 2: smoke and fire)
    
    Returns:
        dict with mAP50, mAP50-95, per-class APs, and mean IoU
    """
    # Compute mAP@0.5
    ap50_per_class = []
    for cls in range(num_classes):
        ap = compute_ap_per_class(predictions, ground_truths, cls, iou_threshold=0.5)
        ap50_per_class.append(ap)
    mAP50 = np.mean(ap50_per_class)
    
    # Compute mAP@[0.5:0.95] (average over IoU thresholds 0.5, 0.55, ..., 0.95)
    iou_thresholds = np.linspace(0.5, 0.95, 10)
    ap_per_threshold = []
    
    for iou_thresh in iou_thresholds:
        ap_per_class = []
        for cls in range(num_classes):
            ap = compute_ap_per_class(predictions, ground_truths, cls, iou_threshold=iou_thresh)
            ap_per_class.append(ap)
        ap_per_threshold.append(np.mean(ap_per_class))
    
    mAP50_95 = np.mean(ap_per_threshold)
    
    # Compute mean IoU for all matched predictions
    all_ious = []
    for img_name in ground_truths.keys():
        gt_boxes = ground_truths.get(img_name, [])
        pred_boxes = predictions.get(img_name, [])
        
        for pred in pred_boxes:
            pred_cls = pred[0]
            pred_box = pred[1:5]
            
            best_iou = 0
            for gt in gt_boxes:
                if gt[0] == pred_cls:
                    gt_box = gt[1:5]
                    iou_val = iou(pred_box, gt_box)
                    best_iou = max(best_iou, iou_val)
            
            if best_iou > 0:
                all_ious.append(best_iou)
    
    mean_iou = np.mean(all_ious) if len(all_ious) > 0 else 0.0
    
    # Calculate AP50-95 per class (average across IoU thresholds)
    ap50_95_per_class = []
    for cls in range(num_classes):
        class_aps = []
        for threshold in iou_thresholds:
            ap = compute_ap_per_class(predictions, ground_truths, cls, threshold)
            class_aps.append(ap)
        ap50_95_per_class.append(np.mean(class_aps))
    
    return {
        'mAP50': mAP50,
        'mAP50_95': mAP50_95,
        'AP50_smoke': ap50_per_class[0] if len(ap50_per_class) > 0 else 0.0,
        'AP50_fire': ap50_per_class[1] if len(ap50_per_class) > 1 else 0.0,
        'AP50_95_smoke': ap50_95_per_class[0] if len(ap50_95_per_class) > 0 else 0.0,
        'AP50_95_fire': ap50_95_per_class[1] if len(ap50_95_per_class) > 1 else 0.0,
        'mean_IoU': mean_iou,
    }


def compute_metrics(predictions, ground_truths, iou_threshold=0.5):
    """
    Compute TP, FP, FN for a set of images.
    predictions: dict {img_name: [(cls, x1, y1, x2, y2, conf), ...]}
    ground_truths: dict {img_name: [(cls, x1, y1, x2, y2), ...]}
    """
    tp = 0
    fp = 0
    fn = 0
    
    for img_name in ground_truths.keys():
        gt_boxes = ground_truths.get(img_name, [])
        pred_boxes = predictions.get(img_name, [])
        
        matched_gt = [False] * len(gt_boxes)
        matched_pred = [False] * len(pred_boxes)
        
        # Match predictions to ground truths
        for i, pred in enumerate(pred_boxes):
            pred_cls = pred[0]
            pred_box = pred[1:5]
            
            best_iou = 0
            best_idx = -1
            
            for j, gt in enumerate(gt_boxes):
                if matched_gt[j]:
                    continue
                if gt[0] != pred_cls:
                    continue
                
                gt_box = gt[1:5]
                iou_val = iou(pred_box, gt_box)
                
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_idx = j
            
            if best_iou >= iou_threshold and best_idx >= 0:
                tp += 1
                matched_gt[best_idx] = True
                matched_pred[i] = True
            else:
                fp += 1
        
        # Count unmatched ground truths as false negatives
        fn += sum(1 for m in matched_gt if not m)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'TP': tp,
        'FP': fp,
        'FN': fn,
        'precision': precision,
        'recall': recall,
        'F1': f1
    }


# ==============================
# MAIN TESTING FUNCTION
# ==============================

def test_model(model_name, weights_path, selected_images, dataset_path, output_dir, gt_output_dir, time_steps=4, conf_threshold=0.25, save_images=True, imgsz=640):
    """
    Test a single model on selected images.
    
    Args:
        selected_images: dict of category -> list of image stems (no extension)
        dataset_path: Path to model-specific dataset (e.g., D-Fire, latency_4, poisson_4)
        gt_output_dir: Shared ground truth output directory (only first model writes here)
        save_images: If True, save visualization images with bounding boxes (small test mode).
                     If False, only compute metrics (full test mode).
        imgsz: Target image size for inference (default 640)
    """
    print(f"\n{'='*80}")
    print(f"Testing Model: {model_name}")
    print(f"{'='*80}\n")
    
    # Create output directories
    model_output = Path(output_dir) / model_name
    model_output.mkdir(parents=True, exist_ok=True)
    
    # Only create predicted images directory if saving visualizations
    if save_images:
        pred_imgs_dir = model_output / "predicted"
        pred_imgs_dir.mkdir(exist_ok=True)
    
    # Load model
    try:
        wrapper = ModelWrapper(model_name, weights_path, time_steps=time_steps)
    except Exception as e:
        print(f"Error loading model: {e}")
        return False
    
    class_names = {0: 'smoke', 1: 'fire'}
    is_encoded = model_name.lower() in ['spikeyolo_latency', 'spikeyolo_poisson']
    encoding_type = 'latency' if 'latency' in model_name.lower() else 'poisson' if 'poisson' in model_name.lower() else None
    
    # Energy tracking
    energy_dir = model_output / "energy"
    energy_dir.mkdir(exist_ok=True)
    
    # Accumulators (no encoding phase - using pre-encoded datasets)
    total_inference_time = 0.0
    total_inference_energy = 0.0  # kg CO2 emissions
    total_inference_energy_kwh = 0.0  # kWh energy consumed
    
    predictions = {}
    ground_truths = {}
    
    # Flatten selected images (these are stems only)
    all_image_stems = []
    for category, img_list in selected_images.items():
        all_image_stems.extend(img_list)
    
    print(f"Processing {len(all_image_stems)} test images...")
    
    # Get model-specific paths
    test_images_dir = Path(dataset_path) / "test" / "images"
    test_labels_dir = Path(dataset_path) / "test" / "labels"
    
    # For ground truth visualization (always use original D-Fire dataset)
    dfire_images_dir = Path(project_root) / "datasets" / "D-Fire" / "test" / "images"
    
    for img_stem in tqdm(all_image_stems):
        # Determine file extension based on model type
        if is_encoded:
            img_extension = '.pt'
        else:
            # Find the actual extension in directory
            possible_exts = ['.jpg', '.jpeg', '.png']
            img_path = None
            for ext in possible_exts:
                candidate = test_images_dir / (img_stem + ext)
                if candidate.exists():
                    img_path = str(candidate)
                    break
            if img_path is None:
                print(f"  Warning: Could not find image for {img_stem}")
                continue
        
        if is_encoded:
            img_path = str(test_images_dir / (img_stem + '.pt'))
        
        label_path = str(test_labels_dir / (img_stem + '.txt'))
        
        # Load original RGB image for ground truth and dimensions
        # Always load from D-Fire for consistent ground truth
        original_img_path = None
        for ext in ['.jpg', '.jpeg', '.png']:
            candidate = dfire_images_dir / (img_stem + ext)
            if candidate.exists():
                original_img_path = str(candidate)
                break
        
        if original_img_path is None:
            print(f"  Warning: Could not find original image for {img_stem}")
            continue
        
        img_bgr = cv2.imread(original_img_path)
        if img_bgr is None:
            print(f"  Warning: Could not load {original_img_path}")
            continue
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_bgr.shape[:2]
        img_name = img_stem + ('.pt' if is_encoded else Path(original_img_path).suffix)
        
        # Parse ground truth
        gt_boxes = parse_yolo_label(label_path, h, w)
        ground_truths[img_name] = gt_boxes
        
        # === PREDICTION ===
        if is_encoded:
            # Load pre-encoded spike tensor from .pt file
            # img_path already points to .pt file in pre-encoded dataset
            encoded_tensor = torch.load(img_path)  # [T, C, H, W]
            
            # Store original size for bbox scaling
            T, C, original_h, original_w = encoded_tensor.shape
            
            # Resize tensor to model input size using letterbox
            # This matches YOLO's training preprocessing (maintains aspect ratio + padding)
            if (original_h, original_w) != (imgsz, imgsz):
                encoded_tensor, scale_ratio, (pad_w, pad_h) = letterbox_resize_tensor(
                    encoded_tensor, target_size=imgsz
                )
            else:
                scale_ratio = 1.0
                pad_w, pad_h = 0, 0
            
            # Inference phase only (no encoding needed)
            tracker_infer = EmissionsTracker(
                project_name=f"{model_name}_inference",
                measure_power_secs=1,
                save_to_file=False,
                log_level='error'
            )
            tracker_infer.start()
            t_inf_start = time.time()
            
            # Pass tensor with original shape for proper bbox scaling
            dets = wrapper.predict_tensor(encoded_tensor, conf_thres=conf_threshold, original_shape=(original_h, original_w))
            
            t_inf_end = time.time()
            inf_emissions = tracker_infer.stop()
            
            inference_time = t_inf_end - t_inf_start
            total_inference_time += inference_time
            if inf_emissions:
                total_inference_energy += float(inf_emissions)
            # Also capture energy in kWh
            if hasattr(tracker_infer, '_total_energy') and tracker_infer._total_energy:
                total_inference_energy_kwh += tracker_infer._total_energy.kWh
            
            # Visualize encoded representation (decode for visualization only)
            reconstructed = decode_spikes_to_image(encoded_tensor, encoding_type=encoding_type)
            if reconstructed.shape[:2] != (h, w):
                reconstructed = cv2.resize(reconstructed, (w, h))
            vis_base = reconstructed
            
        else:
            # RGB models: load original image and predict
            tracker = EmissionsTracker(
                project_name=f"{model_name}_inference",
                measure_power_secs=1,
                save_to_file=False,
                log_level='error'
            )
            tracker.start()
            t_start = time.time()
            
            dets = wrapper.predict(img_path, conf_thres=conf_threshold)
            
            t_end = time.time()
            emissions = tracker.stop()
            
            inference_time = t_end - t_start
            total_inference_time += inference_time
            if emissions:
                total_inference_energy += float(emissions)
            # Also capture energy in kWh
            if hasattr(tracker, '_total_energy') and tracker._total_energy:
                total_inference_energy_kwh += tracker._total_energy.kWh
            
            vis_base = img_bgr.copy()
        
        # Convert detections to format: (cls, x1, y1, x2, y2, conf)
        pred_boxes = []
        for det in (dets or []):
            pred_boxes.append((int(det[5]), det[0], det[1], det[2], det[3], det[4]))
        predictions[img_name] = pred_boxes
        
        # === SAVE VISUALIZATIONS (only in small test mode) ===
        if save_images:
            # 1. Predicted bounding boxes (on encoded representation for encoded models)
            pred_vis = draw_boxes_on_image(vis_base, pred_boxes, class_names, show_conf=True)
            # Save with original image extension for consistency
            pred_out_name = img_stem + Path(original_img_path).suffix
            pred_out_path = pred_imgs_dir / pred_out_name
            cv2.imwrite(str(pred_out_path), pred_vis)
            
            # 2. Ground truth bounding boxes (shared folder, only write once)
            gt_out_path = gt_output_dir / (img_stem + Path(original_img_path).suffix)
            if not gt_out_path.exists():
                gt_vis = draw_boxes_on_image(img_bgr, gt_boxes, class_names, show_conf=False)
                cv2.imwrite(str(gt_out_path), gt_vis)
    
    # === COMPUTE METRICS ===
    print("\nComputing metrics...")
    metrics = compute_metrics(predictions, ground_truths, iou_threshold=0.5)
    
    # Compute mAP metrics
    print("Computing mAP metrics...")
    map_metrics = compute_map(predictions, ground_truths, num_classes=2)
    
    # === SAVE METRICS CSV ===
    metrics_csv = model_output / "results_test_images.csv"
    with open(metrics_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        writer.writerow(['confidence_threshold', conf_threshold])
        writer.writerow(['precision', metrics['precision']])
        writer.writerow(['recall', metrics['recall']])
        writer.writerow(['F1', metrics['F1']])
        writer.writerow(['TP', metrics['TP']])
        writer.writerow(['FP', metrics['FP']])
        writer.writerow(['FN', metrics['FN']])
        writer.writerow(['mAP50', map_metrics['mAP50']])
        writer.writerow(['mAP50_95', map_metrics['mAP50_95']])
        writer.writerow(['AP50_smoke', map_metrics['AP50_smoke']])
        writer.writerow(['AP50_fire', map_metrics['AP50_fire']])
        writer.writerow(['AP50_95_smoke', map_metrics['AP50_95_smoke']])
        writer.writerow(['AP50_95_fire', map_metrics['AP50_95_fire']])
        writer.writerow(['mean_IoU', map_metrics['mean_IoU']])
        writer.writerow(['num_images', len(all_image_stems)])
    
    print(f"Metrics saved: {metrics_csv}")
    
    # === SAVE ENERGY CSV ===
    energy_csv = model_output / "energy_results.csv"
    
    # Calculate energy per image (inference only - using pre-encoded datasets)
    energy_per_image = total_inference_energy / len(all_image_stems) if len(all_image_stems) > 0 else 0
    
    # For encoded models: adjust inference energy by dividing by time_steps
    # This approximates the cost as if processing a single-timestep equivalent
    if is_encoded:
        adjusted_inference_energy = total_inference_energy / time_steps
        adjusted_energy_per_image = adjusted_inference_energy / len(all_image_stems) if len(all_image_stems) > 0 else 0
    else:
        adjusted_inference_energy = total_inference_energy
        adjusted_energy_per_image = energy_per_image
    
    with open(energy_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['phase', 'total_time_s', 'total_energy_kwh', 'total_emissions_kg_co2', 'avg_time_per_image_s', 'avg_energy_per_image_kwh', 'avg_emissions_per_image_kg_co2'])
        
        writer.writerow([
            'inference',
            total_inference_time,
            total_inference_energy_kwh,
            total_inference_energy,
            total_inference_time / len(all_image_stems),
            total_inference_energy_kwh / len(all_image_stems),
            total_inference_energy / len(all_image_stems)
        ])
        
        # Add adjusted inference for encoded models
        if is_encoded:
            adjusted_inference_energy_kwh = total_inference_energy_kwh / time_steps
            writer.writerow([
                f'inference_adjusted_div_{time_steps}',
                total_inference_time / time_steps,
                adjusted_inference_energy_kwh,
                adjusted_inference_energy,
                total_inference_time / time_steps / len(all_image_stems),
                adjusted_inference_energy_kwh / len(all_image_stems),
                adjusted_inference_energy / len(all_image_stems)
            ])
    
    print(f"Energy results saved: {energy_csv}")
    
    # === SUMMARY ===
    print(f"\n{'='*80}")
    print(f"Results Summary for {model_name}")
    print(f"Confidence:  {conf_threshold}")
    print(f"Precision:   {metrics['precision']:.4f}")
    print(f"Recall:      {metrics['recall']:.4f}")
    print(f"F1 Score:    {metrics['F1']:.4f}")
    print(f"TP: {metrics['TP']}, FP: {metrics['FP']}, FN: {metrics['FN']}")
    print(f"mAP50:       {map_metrics['mAP50']:.4f}")
    print(f"mAP50-95:    {map_metrics['mAP50_95']:.4f}")
    print(f"Mean IoU:    {map_metrics['mean_IoU']:.4f}")
    
    print(f"\nInference - Total: {total_inference_time:.2f}s, Energy: {total_inference_energy_kwh:.6f} kWh, Emissions: {total_inference_energy:.6f} kg CO2")
    print(f"            Avg/img: {total_inference_time/len(all_image_stems):.3f}s, {total_inference_energy/len(all_image_stems):.6f} kg CO2")
    
    if is_encoded:
        print(f"Inference (adjusted ÷{time_steps}): {adjusted_inference_energy:.6f} kg CO2")
        print(f"            Avg/img: {adjusted_energy_per_image:.6f} kg CO2")
    
    print(f"{'='*80}\n")
    
    return True


# ==============================
# MAIN
# ==============================

def main():
    parser = argparse.ArgumentParser(description="Comprehensive image testing for TFM")
    parser.add_argument('--dataset', type=str, default='D-Fire', help='Dataset name (subfolder in datasets/)')
    parser.add_argument('--models', type=str, nargs='+', default=None, 
                        help='Models to test. Default: all available models')
    parser.add_argument('--output-dir', type=str, default='Results_test_images', 
                        help='Output directory for results')
    parser.add_argument('--time-steps', type=int, default=4, help='Time steps for spike encoding')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size for inference (default: 640)')
    parser.add_argument('--conf', type=float, default=None, 
                        help='Override confidence threshold for all models (default: use adaptive thresholds per model)')
    parser.add_argument('--test-size', type=str, choices=['full', 'small'], default='small',
                        help='Test size: "full" = all test images (metrics only), "small" = 12 selected images (with visualizations)')
    parser.add_argument('--images-per-category', type=int, default=3, 
                        help='Number of images per category for small test (empty/smoke/fire/both)')
    parser.add_argument('--num-runs', type=int, default=1,
                        help='Number of independent test runs to execute (only for small test mode)')
    
    args = parser.parse_args()
    
    # Enforce num-runs=1 for full test mode
    if args.test_size == 'full' and args.num_runs > 1:
        print("Warning: --num-runs > 1 is only supported for small test mode. Setting num_runs=1 for full mode.")
        args.num_runs = 1
    
    # Print confidence strategy
    if args.conf is not None:
        print(f"\nUsing uniform confidence threshold: {args.conf}")
    else:
        print(f"\nUsing adaptive confidence thresholds per model:")
        for model, conf in ADAPTIVE_CONFIDENCE_THRESHOLDS.items():
            print(f"  {model:25s} -> conf={conf}")
    
    # Note: Dataset path will be determined per model based on MODEL_DATASET_MAPPING
    print(f"\nTest mode: {args.test_size}")
    if args.num_runs > 1:
        print(f"Number of independent runs: {args.num_runs}")
    print(f"Using pre-encoded datasets (no runtime encoding):")
    for model, dataset in MODEL_DATASET_MAPPING.items():
        print(f"  {model:25s} -> datasets/{dataset}/test")
    
    save_images = (args.test_size == 'small')
    
    # Reference dataset for image selection
    reference_dataset = Path(project_root) / "datasets" / "D-Fire"
    if not reference_dataset.exists():
        print(f"Error: Reference dataset not found at {reference_dataset}")
        return
    
    # Determine models to test
    if args.models:
        models_to_test = args.models
    else:
        models_to_test = list(DEFAULT_WEIGHTS.keys())
    
    print(f"\nModels to test: {', '.join(models_to_test)}")
    
    # ==============================
    # MULTIPLE RUNS LOOP
    # ==============================
    for run_idx in range(1, args.num_runs + 1):
        print(f"\n{'#'*80}")
        print(f"# RUN {run_idx}/{args.num_runs}")
        print(f"{'#'*80}\n")
        
        # Select images for this run (randomly selected each time)
        if args.test_size == 'full':
            all_test_images = get_all_test_images(reference_dataset)
            selected_images = {'all': all_test_images}
            print(f"\nTesting on {len(all_test_images)} images (all models use same images)")
        else:
            print(f"\nSelecting test images from reference dataset (D-Fire) - Run {run_idx}...")
            selected_images = select_test_images(reference_dataset, args.images_per_category)
            total_selected = sum(len(imgs) for imgs in selected_images.values())
            print(f"Selected {total_selected} images (all models will use these)")
        
        # Create output directory with run number
        if args.num_runs > 1:
            test_mode_dir = f'small_test{run_idx}' if args.test_size == 'small' else f'full_test{run_idx}'
        else:
            test_mode_dir = 'full_test' if args.test_size == 'full' else 'small_test'
        
        output_dir = Path(args.output_dir) / test_mode_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create shared ground truth directory (only for small test)
        if save_images:
            gt_output_dir = output_dir / "ground_truth"
            gt_output_dir.mkdir(exist_ok=True)
            print(f"\nShared ground truth will be saved to: {gt_output_dir}")
        else:
            gt_output_dir = None
        
        print(f"Results will be saved to: {output_dir}")
        
        # Test each model
        results = {}
        for model_name in models_to_test:
            if model_name not in DEFAULT_WEIGHTS:
                print(f"  Warning: Unknown model '{model_name}', skipping")
                continue
            
            weights_path = DEFAULT_WEIGHTS[model_name]
            
            if not Path(weights_path).exists():
                print(f"  Warning: Weights not found for {model_name} at {weights_path}, skipping")
                continue
            
            # Get model-specific dataset path (normalize model name to lowercase for lookup)
            model_dataset = MODEL_DATASET_MAPPING.get(model_name.lower(), args.dataset)
            dataset_path = Path(project_root) / "datasets" / model_dataset
            
            if not dataset_path.exists():
                print(f"  Warning: Dataset not found for {model_name} at {dataset_path}, skipping")
                continue
            
            print(f"\n{model_name}: Using dataset {model_dataset}")
            
            # Determine confidence threshold: use override if provided, else adaptive
            if args.conf is not None:
                conf_threshold = args.conf
            else:
                # Normalize model name for lookup (handle case variations)
                model_key = model_name.lower().replace('_', '').replace('-', '')
                conf_threshold = ADAPTIVE_CONFIDENCE_THRESHOLDS.get(
                    model_name.lower(), 
                    ADAPTIVE_CONFIDENCE_THRESHOLDS.get(model_key, 0.25)
                )
            
            print(f"Using confidence threshold: {conf_threshold}")
            
            success = test_model(
                model_name=model_name,
                weights_path=weights_path,
                selected_images=selected_images,
                dataset_path=dataset_path,
                output_dir=output_dir,
                gt_output_dir=gt_output_dir,
                time_steps=args.time_steps,
                conf_threshold=conf_threshold,
                save_images=save_images,
                imgsz=args.imgsz
            )
            
            results[model_name] = success
        
        # Summary for this run
        print(f"\n{'='*80}")
        print(f"RUN {run_idx}/{args.num_runs} COMPLETED")
        print(f"{'='*80}")
        for model_name, success in results.items():
            status = "Success" if success else "Failed"
            print(f"{model_name:25s} {status}")
        
        print(f"\nResults saved in: {output_dir}")
        print(f"{'='*80}\n")
    
    # Final summary for all runs
    print("\n" + "="*80)
    print("ALL TESTING COMPLETED")
    print("="*80)
    if args.num_runs > 1:
        print(f"Executed {args.num_runs} independent test runs")
        print(f"Results folders created in: {Path(args.output_dir)}")
        for run_idx in range(1, args.num_runs + 1):
            test_mode_dir = f'small_test{run_idx}' if args.test_size == 'small' else f'full_test{run_idx}'
            print(f"  - {test_mode_dir}")
    else:
        test_mode_dir = 'full_test' if args.test_size == 'full' else 'small_test'
        print(f"Results saved in: {Path(args.output_dir) / test_mode_dir}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
