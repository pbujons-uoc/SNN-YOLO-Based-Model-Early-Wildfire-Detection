"""
Calculate IoU, mAP50, and mAP50-95 metrics for existing test results.

This script reads the results_test_images.csv files in Results_image_small subdirectories
and adds the mAP metrics columns that were missing from previous runs.

Usage:
    python calculate_map_metrics.py
"""

import os
import csv
import json
import numpy as np
from pathlib import Path
from collections import defaultdict


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
    """
    all_preds = []
    gt_count = 0
    gt_per_image = {}
    
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
    
    # Add sentinel values
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([1.0], precisions, [0.0]))
    
    # Make precision monotonically decreasing
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    
    # Compute AP using 101-point interpolation
    recall_thresholds = np.linspace(0, 1, 101)
    ap = 0.0
    for t in recall_thresholds:
        idx = np.searchsorted(recalls, t, side='left')
        if idx < len(precisions):
            ap += precisions[idx]
    ap /= 101
    
    return ap


def compute_map(predictions, ground_truths, num_classes=2):
    """
    Compute mAP50 and mAP50-95.
    """
    # Compute mAP@0.5
    ap50_per_class = []
    for cls in range(num_classes):
        ap = compute_ap_per_class(predictions, ground_truths, cls, iou_threshold=0.5)
        ap50_per_class.append(ap)
    mAP50 = np.mean(ap50_per_class)
    
    # Compute mAP@[0.5:0.95]
    iou_thresholds = np.linspace(0.5, 0.95, 10)
    ap_per_threshold = []
    
    for iou_thresh in iou_thresholds:
        ap_per_class = []
        for cls in range(num_classes):
            ap = compute_ap_per_class(predictions, ground_truths, cls, iou_threshold=iou_thresh)
            ap_per_class.append(ap)
        ap_per_threshold.append(np.mean(ap_per_class))
    
    mAP50_95 = np.mean(ap_per_threshold)
    
    # Compute mean IoU
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
    
    return {
        'mAP50': mAP50,
        'mAP50_95': mAP50_95,
        'AP50_smoke': ap50_per_class[0] if len(ap50_per_class) > 0 else 0.0,
        'AP50_fire': ap50_per_class[1] if len(ap50_per_class) > 1 else 0.0,
        'mean_IoU': mean_iou,
    }


def parse_yolo_label(label_path, img_h, img_w):
    """Parse YOLO format label file."""
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


def load_predictions_from_json(json_path):
    """Load predictions from a JSON file (if saved by test script)."""
    if not os.path.exists(json_path):
        return None
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    predictions = {}
    for img_name, pred_list in data.items():
        predictions[img_name] = [
            (pred['cls'], pred['x1'], pred['y1'], pred['x2'], pred['y2'], pred['conf'])
            for pred in pred_list
        ]
    
    return predictions


def process_test_folder(test_folder_path):
    """
    Process a single test folder (e.g., small_test1).
    
    Since predictions were not saved, we cannot calculate mAP metrics.
    This function will print a warning and skip the folder.
    """
    test_folder = Path(test_folder_path)
    print(f"\nProcessing: {test_folder.name}")
    
    # Look for model subfolders
    model_folders = [d for d in test_folder.iterdir() if d.is_dir()]
    
    if not model_folders:
        print(f"  No model folders found in {test_folder}")
        return
    
    print(f"  WARNING: Predictions were not saved during testing.")
    print(f"  Cannot calculate mAP metrics without prediction data.")
    print(f"  Please re-run the test with the updated test_images_comprehensive.py script.")
    
    return


def main():
    """
    Main function to process all test results in Results_image_small.
    """
    results_dir = Path("Results_image_small")
    
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return
    
    # Find all test folders (small_test1, small_test2, etc.)
    test_folders = sorted([d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith('small_test')])
    
    if not test_folders:
        print(f"No test folders found in {results_dir}")
        return
    
    print(f"Found {len(test_folders)} test folders")
    
    for test_folder in test_folders:
        process_test_folder(test_folder)
    
    print("\n" + "="*80)
    print("IMPORTANT: Predictions were not saved in previous test runs.")
    print("To calculate mAP metrics, you need to re-run the tests using:")
    print("  python tests/test_images_comprehensive.py --dataset D-Fire --test-size small")
    print("\nThe updated script will:")
    print("  1. Save predictions with bounding boxes and confidence scores")
    print("  2. Calculate IoU, mAP50, and mAP50-95 metrics")
    print("  3. Save all metrics to results_test_images.csv")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
