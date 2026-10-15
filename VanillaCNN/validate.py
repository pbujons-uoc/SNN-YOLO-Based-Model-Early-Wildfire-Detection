"""
Validation script for VanillaCNN model.
Evaluates trained model on validation set and computes metrics.
"""

import os
import argparse
from pathlib import Path
import yaml
import csv
import torch
from torch.utils.data import DataLoader

from model import SimpleYoloCNN
from data_to_yolo import YoloGridDataset
from metrics import evaluate_map


def parse_args():
    parser = argparse.ArgumentParser(description="Validate SimpleYoloCNN (VanillaCNN) model")
    
    parser.add_argument("--model", type=str, required=True, help="Path to trained model weights (.pth)")
    parser.add_argument("--data", type=str, required=True, help="Dataset YAML path")
    parser.add_argument("--batch", type=int, default=16, help="Batch size for validation")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", type=int, default=0, help="GPU device index (or -1 for CPU)")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument("--conf-thresh", type=float, default=0.25, help="Confidence threshold for predictions")
    parser.add_argument("--iou-thresh", type=float, default=0.5, help="IoU threshold for mAP calculation")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save results")
    
    return parser.parse_args()


def load_dataset_yaml(yaml_path: str):
    """Load dataset configuration from YAML file."""
    with open(yaml_path, "r") as f:
        data_cfg = yaml.safe_load(f)
    
    # Get the base path
    base_path = data_cfg.get("path", "")
    yaml_dir = Path(yaml_path).parent
    
    if base_path:
        base_path = Path(base_path)
        if not base_path.is_absolute():
            base_path = yaml_dir / base_path
    else:
        base_path = yaml_dir
    
    # Resolve val path
    val_rel = data_cfg["val"]
    val_base = base_path / val_rel
    
    # Check if images subdirectory exists
    val_images_dir = str(val_base / "images") if (val_base / "images").exists() else str(val_base)
    
    # Infer labels directory
    def infer_labels_dir(images_dir: str):
        p = Path(images_dir)
        parts = list(p.parts)
        if "images" in parts:
            i = parts.index("images")
            parts[i] = "labels"
            return str(Path(*parts))
        else:
            # Assume labels in same directory as images
            return images_dir
    
    val_labels_dir = infer_labels_dir(val_images_dir)
    
    num_classes = data_cfg.get("nc", 1)
    class_names = data_cfg.get("names", [f"class_{i}" for i in range(num_classes)])
    
    return val_images_dir, val_labels_dir, num_classes, class_names


def main():
    args = parse_args()
    
    # Device setup
    if args.device >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device}")
    else:
        device = torch.device("cpu")
    
    print(f"\n{'='*80}")
    print(f"🔍 VanillaCNN VALIDATION")
    print(f"{'='*80}\n")
    print(f"Model weights: {args.model}")
    print(f"Dataset: {args.data}")
    print(f"Device: {device}")
    print(f"Batch size: {args.batch}")
    print(f"Image size: {args.imgsz}")
    print(f"Confidence threshold: {args.conf_thresh}")
    print(f"IoU threshold: {args.iou_thresh}\n")
    
    # Load dataset configuration
    val_images_dir, val_labels_dir, num_classes, class_names = load_dataset_yaml(args.data)
    
    print(f"Val images: {val_images_dir}")
    print(f"Val labels: {val_labels_dir}")
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {class_names}\n")
    
    # Create validation dataset
    val_dataset = YoloGridDataset(
        image_dir=val_images_dir,
        label_dir=val_labels_dir,
        img_size=args.imgsz,
        grid_size=7,
        num_classes=num_classes
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"Validation set: {len(val_dataset)} images\n")
    
    # Load model
    print("Loading model...")
    model = SimpleYoloCNN(
        grid_size=7,
        num_classes=num_classes,
        num_boxes_per_cell=2
    )
    
    # Load weights
    if not os.path.exists(args.model):
        print(f"❌ ERROR: Model weights not found at {args.model}")
        return
    
    checkpoint = torch.load(args.model, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    print("✅ Model loaded successfully\n")
    
    # Run validation
    print("Running validation...")
    print(f"{'='*80}\n")
    
    with torch.no_grad():
        map50, map50_95 = evaluate_map(
            model=model,
            dataloader=val_loader,
            device=device,
            conf_thresh=args.conf_thresh,
            iou_thresh=args.iou_thresh,
            num_classes=num_classes,
            verbose=True
        )
    
    # Print results
    print(f"\n{'='*80}")
    print(f"📊 VALIDATION RESULTS")
    print(f"{'='*80}\n")
    print(f"mAP@0.5:      {map50:.4f}")
    print(f"mAP@0.5:0.95: {map50_95:.4f}")
    print(f"\n{'='*80}\n")
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.model).parent.parent
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results to CSV file (compatible format with other models)
    csv_file = output_dir / "results_val.csv"
    with open(csv_file, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["metrics/mAP50(B)", map50])
        writer.writerow(["metrics/mAP50-95(B)", map50_95])
    
    print(f"✅ CSV results saved to: {csv_file}")
    
    # Save detailed results to text file
    txt_file = output_dir / "validation_results.txt"
    with open(txt_file, "w") as f:
        f.write(f"VanillaCNN Validation Results\n")
        f.write(f"{'='*50}\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Dataset: {args.data}\n")
        f.write(f"Confidence threshold: {args.conf_thresh}\n")
        f.write(f"IoU threshold: {args.iou_thresh}\n")
        f.write(f"\nMetrics:\n")
        f.write(f"mAP@0.5:      {map50:.4f}\n")
        f.write(f"mAP@0.5:0.95: {map50_95:.4f}\n")
    
    print(f"✅ Text results saved to: {txt_file}\n")


if __name__ == "__main__":
    main()
