import os
import argparse
import torch
import cv2
import sys
import csv
import time
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import numpy as np

# CodeCarbon
#from codecarbon import EmissionsTracker
# If you are offline, use instead:
# from codecarbon import OfflineEmissionsTracker

# Ensure we can import VanillaCNN modules
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(SCRIPT_DIR))
try:
    from VanillaCNN.model import SimpleYoloCNN
    from VanillaCNN.yolo_decoder import decode_predictions
    from VanillaCNN.nms import nms
except ImportError:
    try:
        # Fallback when executed with cwd inside VanillaCNN folder
        from model import SimpleYoloCNN
        from yolo_decoder import decode_predictions
        from nms import nms
    except ImportError:
        print("Error: Could not import VanillaCNN modules. Checked package and local imports.")
        print(f"Debug paths: PROJECT_ROOT={PROJECT_ROOT}, SCRIPT_DIR={SCRIPT_DIR}, CWD={os.getcwd()}")
        sys.exit(1)

# Class names from D-Fire
CLASSES = {0: 'smoke', 1: 'fire'}
COLORS = {0: (255, 0, 0), 1: (0, 0, 255)}  # BGR: Smoke=Blue-ish, Fire=Red
PROJECT_ROOT = Path(__file__).resolve().parents[1]
#CODECARBON_ROOT = PROJECT_ROOT / "codecarbon_results"
#EMISSIONS_LOG_DELAY_S = float(os.getenv("EMISSIONS_LOG_DELAY_S", "0.20"))


def letterbox_cv2(img, target_size=640):
    h, w = img.shape[:2]

    scale = min(target_size / h, target_size / w)
    new_h, new_w = int(h * scale), int(w * scale)

    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_h = (target_size - new_h) // 2
    pad_w = (target_size - new_w) // 2

    img_padded = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    img_padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = img_resized

    return img_padded


def letterbox_pil(img, target_size=640):
    w, h = img.size

    scale = min(target_size / h, target_size / w)
    new_h, new_w = int(h * scale), int(w * scale)

    img_resized = TF.resize(img, (new_h, new_w), interpolation=TF.InterpolationMode.BILINEAR)

    pad_h = (target_size - new_h) // 2
    pad_w = (target_size - new_w) // 2

    img_padded = Image.new('RGB', (target_size, target_size), (114, 114, 114))
    img_padded.paste(img_resized, (pad_w, pad_h))

    return img_padded


def plot_vanilla_boxes(image_path, detections, output_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image {image_path}")
        return

    img_letterboxed = letterbox_cv2(img, target_size=640)

    for (x1, y1, x2, y2, score, cls) in detections:
        color = COLORS.get(cls, (0, 255, 0))
        label = f"{CLASSES.get(cls, cls)} {score:.2f}"

        cv2.rectangle(img_letterboxed, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        c2 = int(x1) + t_size[0], int(y1) - t_size[1] - 3
        cv2.rectangle(img_letterboxed, (int(x1), int(y1)), c2, color, -1)
        cv2.putText(
            img_letterboxed,
            label,
            (int(x1), int(y1) - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

    cv2.imwrite(output_path, img_letterboxed)
    print(f"Saved result to {output_path}")


def save_vanilla_predictions_txt(detections, txt_path, img_size=640):
    with open(txt_path, "w", encoding="utf-8") as f:
        for (x1, y1, x2, y2, score, cls) in detections:
            x1 = max(0.0, min(float(x1), float(img_size)))
            y1 = max(0.0, min(float(y1), float(img_size)))
            x2 = max(0.0, min(float(x2), float(img_size)))
            y2 = max(0.0, min(float(y2), float(img_size)))

            xc = ((x1 + x2) / 2.0) / float(img_size)
            yc = ((y1 + y2) / 2.0) / float(img_size)
            bw = max(0.0, (x2 - x1) / float(img_size))
            bh = max(0.0, (y2 - y1) / float(img_size))

            f.write(f"{int(cls)} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f} {float(score):.6f}\n")


def _infer_dataset_name_from_path(image_path):
    p = Path(image_path)
    split_names = {"train", "val", "test"}
    if p.parent.name.lower() in {"images", "imgs"} and p.parent.parent.name.lower() in split_names:
        return p.parent.parent.parent.name or "dataset"
    if p.parent.name.lower() in split_names:
        return p.parent.parent.name or "dataset"
    return p.parent.name or "dataset"


def init_emissions_csv(csv_path, model_name="unknown", dataset_name="dataset"):
    file_exists = os.path.exists(csv_path)
    if not file_exists:
        with open(csv_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "image_name",
                "task_name",
                "emissions_kg_co2eq",
                "energy_consumed_kwh",
                "duration_s",
                "timestamp"
            ])
        print(f"[CodeCarbon][INIT] model={model_name} dataset={dataset_name} csv={csv_path}")


def append_emission_row(csv_path, image_name, task_name, emissions_data, duration_s, model_name="unknown", dataset_name="dataset"):
    emissions_value = getattr(emissions_data, "emissions", None)
    energy_consumed = getattr(emissions_data, "energy_consumed", None)
    '''
    with open(csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            image_name,
            task_name,
            emissions_value,
            energy_consumed,
            duration_s,
            time.strftime("%Y-%m-%d %H:%M:%S")
        ])
    print(
        f"[CodeCarbon][ROW] model={model_name} dataset={dataset_name} image={image_name} "
        f"emissions_kg={emissions_value} energy_kwh={energy_consumed} duration_s={duration_s:.4f}"
    )
    
    if EMISSIONS_LOG_DELAY_S > 0:
        print(f"[CodeCarbon][SLEEP] waiting {EMISSIONS_LOG_DELAY_S:.2f}s to reduce RAM pressure...")
        time.sleep(EMISSIONS_LOG_DELAY_S)
        '''


def main():
    parser = argparse.ArgumentParser(description="Run VanillaCNN Inference with CodeCarbon")
    parser.add_argument('--source', type=str, required=True, help='Path to image or directory')
    parser.add_argument(
        '--model',
        type=str,
        default=os.path.join("Results", "VanillaCNN", "D-Fire", "best_simple_yolo_cnn.pt"),
        help='Path to model weights'
    )
    parser.add_argument('--conf-thres', type=float, default=0.15, help='Confidence threshold for decoding predictions')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IoU threshold for NMS')
    args = parser.parse_args()

    weights_path = args.model
    output_dir = os.path.join("Results", "VanillaCNN", "predictions")
    labels_dir = os.path.join("Results", "VanillaCNN", "labels")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    '''
    codecarbon_dir = CODECARBON_ROOT / "vanillacnn"
    codecarbon_dir.mkdir(parents=True, exist_ok=True)
    emissions_csv = codecarbon_dir / "emissions.csv"
    '''
    model_name = Path(weights_path).stem if weights_path else "vanillacnn"
    source_for_dataset = args.source if os.path.isfile(args.source) else os.path.join(args.source, "images")
    dataset_name = _infer_dataset_name_from_path(source_for_dataset)
    '''
    init_emissions_csv(emissions_csv, model_name=model_name, dataset_name=dataset_name)
    '''

    if not os.path.exists(weights_path):
        print(f"Error: Weights not found at {weights_path}")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading VanillaCNN model from {weights_path} on {device}...")

    model = SimpleYoloCNN(num_classes=2, S=20).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    transform = T.Compose([
        T.ToTensor()
    ])

    if os.path.isdir(args.source):
        images = [
            os.path.join(args.source, f)
            for f in os.listdir(args.source)
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))
        ]
    else:
        images = [args.source]

    # One tracker reused for task mode
    # Online mode:
    '''
    tracker = EmissionsTracker(
        project_name="vanillacnn_inference",
        output_dir=str(codecarbon_dir),
        save_to_file=False,   # task mode results will be written by us
        log_level="warning"
    )
    '''

    # Offline mode example:
    # tracker = OfflineEmissionsTracker(
    #     project_name="vanillacnn_inference",
    #     country_iso_code="ESP",   # or FRA, etc.
    #     output_dir=output_dir,
    #     save_to_file=False,
    #     log_level="warning"
    # )

    try:
        for img_path in images:
            print(f"Processing {img_path}...")

            img_pil = Image.open(img_path).convert('RGB')
            img_letterboxed = letterbox_pil(img_pil, target_size=640)
            img_tensor = transform(img_letterboxed).unsqueeze(0).to(device)

            image_name = os.path.basename(img_path)
            task_name = f"inference_{image_name}"

            with torch.no_grad():
                # Track ONLY the forward pass
                #tracker.start_task(task_name)
                t0 = time.perf_counter()

                preds = model(img_tensor)

                duration_s = time.perf_counter() - t0
                #emissions_data = tracker.stop_task()

                # Write one row per image
                '''
                append_emission_row(
                    emissions_csv,
                    image_name=image_name,
                    task_name=task_name,
                    emissions_data=emissions_data,
                    duration_s=duration_s,
                    model_name=model_name,
                    dataset_name=_infer_dataset_name_from_path(img_path),
                )
                '''

                # Rest of pipeline is not tracked here
                batch_dets = decode_predictions(preds, conf_thres=args.conf_thres, S=20, img_size=640)
                dets = batch_dets[0]
                final_dets = nms(dets, iou_thres=args.iou_thres)

            base_name = os.path.splitext(os.path.basename(img_path))[0]
            save_path = os.path.join(output_dir, f"{base_name}_prediction.jpg")
            txt_path = os.path.join(labels_dir, f"{base_name}.txt")
            plot_vanilla_boxes(img_path, final_dets, save_path)
            save_vanilla_predictions_txt(final_dets, txt_path, img_size=640)
            print(f"Saved labels to {txt_path}")
    except Exception as e:
        print(f"An error occurred during inference: {e}")
    #finally:
        # End tracker cleanly
        #tracker.stop()

    #print(f"Per-image emissions saved to: {emissions_csv}")


if __name__ == "__main__":
    main()