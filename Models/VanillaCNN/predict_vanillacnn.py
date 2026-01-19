import os
import argparse
import torch
import cv2
import sys
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import numpy as np

# Ensure we can import VanillaCNN modules
sys.path.append(os.getcwd())
try:
    from VanillaCNN.model import SimpleYoloCNN
    from VanillaCNN.yolo_decoder import decode_predictions
    from VanillaCNN.nms import nms
except ImportError:
    print("Error: Could not import VanillaCNN modules. Make sure you run this script from the 'Code' directory.")
    sys.exit(1)

# Class names from D-Fire
CLASSES = {0: 'smoke', 1: 'fire'}
COLORS = {0: (255, 0, 0), 1: (0, 0, 255)} # BGR: Smoke=Blue-ish, Fire=Red

def letterbox_cv2(img, target_size=640):
    """
    Resize image using letterbox (maintains aspect ratio + padding).
    Same as training to ensure consistency.
    
    Args:
        img: cv2 image (BGR, HxWxC)
        target_size: target size (default 640)
    
    Returns:
        letterboxed cv2 image [target_size, target_size]
    """
    h, w = img.shape[:2]
    
    # Calculate scale to fit within target_size
    scale = min(target_size / h, target_size / w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    # Resize maintaining aspect ratio
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Calculate padding
    pad_h = (target_size - new_h) // 2
    pad_w = (target_size - new_w) // 2
    
    # Create padded image (gray padding = 114)
    img_padded = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    img_padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = img_resized
    
    return img_padded

def letterbox_pil(img, target_size=640):
    """
    Resize PIL image using letterbox (maintains aspect ratio + padding).
    
    Args:
        img: PIL Image
        target_size: target size (default 640)
    
    Returns:
        letterboxed PIL Image [target_size, target_size]
    """
    w, h = img.size
    
    # Calculate scale to fit within target_size
    scale = min(target_size / h, target_size / w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    # Resize maintaining aspect ratio
    img_resized = TF.resize(img, (new_h, new_w), interpolation=TF.InterpolationMode.BILINEAR)
    
    # Calculate padding
    pad_h = (target_size - new_h) // 2
    pad_w = (target_size - new_w) // 2
    
    # Create padded image (gray padding = 114/255 ≈ 0.447)
    img_padded = Image.new('RGB', (target_size, target_size), (114, 114, 114))
    img_padded.paste(img_resized, (pad_w, pad_h))
    
    return img_padded

def plot_vanilla_boxes(image_path, detections, output_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image {image_path}")
        return

    # Use letterbox resize to match training preprocessing
    img_letterboxed = letterbox_cv2(img, target_size=640)
    
    for (x1, y1, x2, y2, score, cls) in detections:
        color = COLORS.get(cls, (0, 255, 0))
        label = f"{CLASSES.get(cls, cls)} {score:.2f}"
        
        # Box
        cv2.rectangle(img_letterboxed, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        
        # Label
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        c2 = int(x1) + t_size[0], int(y1) - t_size[1] - 3
        cv2.rectangle(img_letterboxed, (int(x1), int(y1)), c2, color, -1)
        cv2.putText(img_letterboxed, label, (int(x1), int(y1) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imwrite(output_path, img_letterboxed)
    print(f"Saved result to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Run VanillaCNN Inference")
    parser.add_argument('--source', type=str, required=True, help='Path to image or directory')
    parser.add_argument('--model', type=str, default=os.path.join("Results", "VanillaCNN", "D-Fire", "best_simple_yolo_cnn.pt"), help='Path to model weights')
    args = parser.parse_args()

    # Define paths
    weights_path = args.model
    output_dir = os.path.join("Results", "VanillaCNN", "predictions")
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(weights_path):
        print(f"Error: Weights not found at {weights_path}")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading VanillaCNN model from {weights_path} on {device}...")
    
    # Initialize model
    model = SimpleYoloCNN(num_classes=2, S=20).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # Transform - use ToTensor only (letterbox is applied in preprocessing)
    transform = T.Compose([
        T.ToTensor()
    ])

    # Handle directory or single file
    if os.path.isdir(args.source):
        images = [os.path.join(args.source, f) for f in os.listdir(args.source) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    else:
        images = [args.source]

    for img_path in images:
        print(f"Processing {img_path}...")
        
        # Load and apply letterbox (same as training)
        img_pil = Image.open(img_path).convert('RGB')
        img_letterboxed = letterbox_pil(img_pil, target_size=640)
        img_tensor = transform(img_letterboxed).unsqueeze(0).to(device)

        with torch.no_grad():
            preds = model(img_tensor)
            
            # Decode
            batch_dets = decode_predictions(preds, conf_thres=0.1, S=20, img_size=640)
            dets = batch_dets[0] # first image
            
            # NMS
            final_dets = nms(dets, iou_thres=0.5)

        # Save result
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        save_path = os.path.join(output_dir, f"{base_name}_prediction.jpg")
        
        plot_vanilla_boxes(img_path, final_dets, save_path)

if __name__ == "__main__":
    main()
