import os
import argparse
from pathlib import Path
import csv

import yaml
import torch
from torch.utils.data import DataLoader

from model import SimpleYoloCNN
from loss import SimpleYoloLoss
from data_to_yolo import YoloGridDataset
from metrics import evaluate_map
from codecarbon import EmissionsTracker


# ==============================
# DEFAULT PATHS
# ==============================
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_SMALL_DATA_YAML = os.path.join(ROOT_DIR, "../datasets/Small_D-Fire/data.yaml")
DEFAULT_D_FIRE_YAML     = os.path.join(ROOT_DIR, "../datasets/D-Fire/data.yaml")
DEFAULT_TRAIN_CFG_YAML  = os.path.join(ROOT_DIR, "train_config.yaml")
DEFAULT_RESULTS_ROOT    = os.path.join(ROOT_DIR, "runs")


# ==============================
# ARGUMENT PARSER
# ==============================
def parse_args():
    parser = argparse.ArgumentParser(description="Train SimpleYoloCNN (VanillaCNN detector baseline)")

    parser.add_argument("--small", action="store_true", help="Use the Small_D-Fire dataset")
    parser.add_argument("--data", type=str, default=None, help="Dataset YAML path (overrides --small)")
    parser.add_argument("--config", type=str, default=None, help="Training config YAML (overrides default)")

    # Device selection
    parser.add_argument("--cpu", action="store_true", help="Force CPU only")
    parser.add_argument("--gpu", type=int, default=None, help="Use GPU index (e.g., --gpu 0)")

    # Training hyperparameters (CLI overrides)
    parser.add_argument("--epochs", type=int, help="Override number of epochs")
    parser.add_argument("--batch", type=int, help="Override batch size")
    parser.add_argument("--imgsz", type=int, help="Override image size")
    parser.add_argument("--num-workers", type=int, help="Override number of dataloader workers")
    parser.add_argument("--lr", type=float, help="Override learning rate")  # IMPORTANT: launcher passes --lr

    # Output directory control
    parser.add_argument("--project", type=str, default=DEFAULT_RESULTS_ROOT, help="Project/Root results directory")
    parser.add_argument("--name", type=str, default=None, help="Experiment name (subdirectory)")

    return parser.parse_args()


# ==============================
# LOAD DATASET YAML
# ==============================
def load_dataset_yaml(yaml_path: str):
    with open(yaml_path, "r") as f:
        data_cfg = yaml.safe_load(f)

    # Ultralytics-style fields commonly found in data.yaml:
    # train: path/to/images/train
    # val:   path/to/images/val
    # nc: number of classes
    # names: [class names]
    
    # Get the base path (can be absolute or relative to yaml location)
    base_path = data_cfg.get("path", "")
    yaml_dir = Path(yaml_path).parent
    
    # Resolve base path: if relative, make it relative to yaml location
    if base_path:
        base_path = Path(base_path)
        if not base_path.is_absolute():
            base_path = yaml_dir / base_path
    else:
        base_path = yaml_dir
    
    # Resolve train and val paths relative to base_path
    train_rel = data_cfg["train"]
    val_rel = data_cfg["val"]
    
    train_base = base_path / train_rel
    val_base = base_path / val_rel
    
    # Check if images subdirectory exists (standard YOLO structure)
    # If train/images exists, use it; otherwise use train/ directly
    train_images_dir = str(train_base / "images") if (train_base / "images").exists() else str(train_base)
    val_images_dir = str(val_base / "images") if (val_base / "images").exists() else str(val_base)

    # labels are usually inferred from images path by replacing "/images/" with "/labels/"
    # but your existing project might store explicit label dirs; keep robust:
    def infer_labels_dir(images_dir: str):
        # common YOLO layout
        p = Path(images_dir)
        parts = list(p.parts)
        if "images" in parts:
            i = parts.index("images")
            parts[i] = "labels"
            return str(Path(*parts))
        # fallback: sibling folder "labels"
        return str(p.parent / "labels") if (p.parent / "labels").exists() else str(p / "labels")

    train_labels_dir = data_cfg.get("train_labels", infer_labels_dir(train_images_dir))
    val_labels_dir = data_cfg.get("val_labels", infer_labels_dir(val_images_dir))

    nc = int(data_cfg["nc"])
    class_names = data_cfg.get("names", [str(i) for i in range(nc)])

    return train_images_dir, train_labels_dir, val_images_dir, val_labels_dir, nc, class_names


# ==============================
# LOAD TRAIN CONFIG YAML
# ==============================
def load_train_config(cfg_path: str):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    cfg.setdefault("img_size", 640)
    cfg.setdefault("S", 20)
    cfg.setdefault("num_epochs", 30)
    cfg.setdefault("learning_rate", 1e-4)
    cfg.setdefault("batch_size", 16)
    cfg.setdefault("num_workers", 4)

    return cfg


# ==============================
# MAIN TRAIN FUNCTION
# ==============================
def train():
    args = parse_args()

    # SAVE DIRECTORY SETUP (launcher-compatible: project/name -> project/name)
    results_root = Path(args.project) if args.project else Path(DEFAULT_RESULTS_ROOT)

    if args.name:
        run_name = args.name
    else:
        if args.data:
            run_name = Path(args.data).parent.name
        elif args.small:
            run_name = "Small_D-Fire"
        else:
            run_name = "D-Fire"

    save_dir = results_root / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------
    # CODECARBON SETUP
    # -------------------------------
    tracker = EmissionsTracker(
        project_name="VanillaCNN",
        output_dir=str(save_dir),
        output_file="vanilla_cnn_emissions.csv",
        save_to_file=True,
        log_level="error",
        logging_logger=None,
    )
    tracker.start()

    try:
        # DATASET SELECTION
        if args.data:
            dataset_yaml = args.data
        elif args.small:
            dataset_yaml = DEFAULT_SMALL_DATA_YAML
        else:
            dataset_yaml = DEFAULT_D_FIRE_YAML

        print(f"\nUsing dataset config: {dataset_yaml}")

        (train_images_dir, train_labels_dir,
         val_images_dir, val_labels_dir,
         num_classes, class_names) = load_dataset_yaml(dataset_yaml)

        # TRAIN CONFIG
        train_cfg_yaml = args.config if args.config else DEFAULT_TRAIN_CFG_YAML
        print(f"Using training config: {train_cfg_yaml}\n")
        cfg = load_train_config(train_cfg_yaml)

        # APPLY OVERRIDES FROM CLI
        img_size      = args.imgsz if args.imgsz else cfg["img_size"]
        S             = cfg["S"]
        num_epochs    = args.epochs if args.epochs else cfg["num_epochs"]
        learning_rate = args.lr if args.lr is not None else cfg["learning_rate"]
        batch_size    = args.batch if args.batch else cfg["batch_size"]
        num_workers   = args.num_workers if args.num_workers else cfg["num_workers"]

        # DEVICE SELECTION (robust)
        if args.cpu or not torch.cuda.is_available():
            device = torch.device("cpu")
        else:
            gpu_idx = args.gpu if args.gpu is not None else 0
            device = torch.device(f"cuda:{gpu_idx}")

        print(f"Using device: {device}")

        # DATA LOADING
        train_ds = YoloGridDataset(train_images_dir, train_labels_dir, S, num_classes, img_size)
        val_ds   = YoloGridDataset(val_images_dir, val_labels_dir, S, num_classes, img_size)

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=(device.type == "cuda")
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=(device.type == "cuda")
        )

        print(f"Train images: {len(train_ds)} | Val images: {len(val_ds)}")

        # MODEL + LOSS + OPTIMIZER
        model = SimpleYoloCNN(num_classes=num_classes, S=S).to(device)
        criterion = SimpleYoloLoss()  # assumes your updated, stable YOLO-like loss
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        # Results CSV
        results_csv_path = save_dir / "vanilla_cnn_results.csv"
        if not results_csv_path.exists():
            with open(results_csv_path, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "epoch",
                    "train/box_loss",
                    "train/cls_loss",
                    "metrics/mAP50(B)",
                    "metrics/mAP50-95(B)",
                    "val/box_loss",
                    "val/cls_loss",
                    "lr",
                ])

        best_map50 = 0.0

        # COCO thresholds for mAP50-95
        iou_list = [x / 100 for x in range(50, 100, 5)]  # 0.50..0.95

        # ==============================
        # TRAIN LOOP
        # ==============================
        for epoch in range(num_epochs):
            model.train()

            running_box = 0.0
            running_cls = 0.0

            for images, targets in train_loader:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                preds = model(images)
                loss, box_loss, cls_loss = criterion(preds, targets)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                running_box += float(box_loss.item())
                running_cls += float(cls_loss.item())

            scheduler.step()

            avg_box = running_box / max(len(train_loader), 1)
            avg_cls = running_cls / max(len(train_loader), 1)

            # VALIDATION LOSS
            model.eval()
            val_running_box = 0.0
            val_running_cls = 0.0
            with torch.no_grad():
                for v_imgs, v_targs in val_loader:
                    v_imgs = v_imgs.to(device, non_blocking=True)
                    v_targs = v_targs.to(device, non_blocking=True)
                    _, v_box, v_cls = criterion(model(v_imgs), v_targs)
                    val_running_box += float(v_box.item())
                    val_running_cls += float(v_cls.item())

            val_avg_box = val_running_box / max(len(val_loader), 1)
            val_avg_cls = val_running_cls / max(len(val_loader), 1)

            # EVALUATION: mAP50 and mAP50-95 (YOLO comparable)
            mAP50, mAP50_95, per_class = evaluate_map(
                model=model,
                val_ds=val_ds,
                val_loader=val_loader,
                labels_dir=val_labels_dir,
                class_names=class_names,
                S=S,
                img_size=img_size,
                iou_thres=iou_list,          # triggers COCO-style averaging
                conf_thres_eval=0.001,
                nms_iou_thres=0.5
            )

            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"  Train box/cls: {avg_box:.4f} / {avg_cls:.4f}")
            print(f"  Val   box/cls: {val_avg_box:.4f} / {val_avg_cls:.4f}")
            print(f"  mAP@0.5:       {mAP50:.4f}")
            print(f"  mAP@0.5:0.95:  {mAP50_95:.4f}")

            for cname, stats in per_class.items():
                print(f"    {cname:>6}: AP50={stats['AP']:.4f}, P={stats['P']:.4f}, R={stats['R']:.4f}")

            # Save best by mAP50 (common in YOLO training logs); you can swap to mAP50_95 if preferred
            if mAP50 > best_map50:
                best_map50 = mAP50
                torch.save(model.state_dict(), save_dir / "best_simple_yolo_cnn.pt")
                print(f"  New best mAP@0.5: {best_map50:.4f} (saved)")

            # Append to CSV
            with open(results_csv_path, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch + 1,
                    avg_box,
                    avg_cls,
                    mAP50,
                    mAP50_95,
                    val_avg_box,
                    val_avg_cls,
                    optimizer.param_groups[0]["lr"],
                ])

        # FINAL SAVE
        torch.save(model.state_dict(), save_dir / "last_simple_yolo_cnn.pt")
        print(f"\nTraining finished. Last model saved.")
        print(f"Best mAP@0.5: {best_map50:.4f}")
        print(f"Results CSV saved at: {results_csv_path}")

    finally:
        emissions = tracker.stop()
        print(f"\n[CodeCarbon] Estimated emissions: {emissions:.6f} kg CO₂eq")


if __name__ == "__main__":
    train()
