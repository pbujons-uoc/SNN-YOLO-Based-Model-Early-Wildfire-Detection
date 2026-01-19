import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF


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


class YoloGridDataset(Dataset):
    """
    Loads images + YOLO labels and converts them into a grid SxS target tensor.
    Uses letterbox resize to maintain aspect ratio.
    """
    def __init__(self, images_dir, labels_dir, S=20, num_classes=2, img_size=640):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.S = S
        self.num_classes = num_classes
        self.img_size = img_size

        self.img_files = sorted([
            f for f in os.listdir(images_dir)
            if f.lower().endswith(("jpg", "jpeg", "png"))
        ])

        # Note: letterbox resize is applied before transform
        self.transform = T.Compose([
            # Augmentation: Color Jitter (changes brightness/contrast/etc)
            # This helps the model generalize to different lighting conditions
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = self.images_dir / img_name
        label_path = self.labels_dir / f"{Path(img_name).stem}.txt"

        # load & preprocess image with letterbox
        img = Image.open(img_path).convert("RGB")
        img = letterbox_pil(img, target_size=self.img_size)
        img = self.transform(img)

        # create target grid
        target = torch.zeros((self.S, self.S, 5 + self.num_classes))

        # load YOLO label
        if label_path.exists():
            with open(label_path, "r") as f:
                for line in f:
                    cls, cx, cy, w, h = map(float, line.split())
                    cls = int(cls)

                    cell_x = int(cx * self.S)
                    cell_y = int(cy * self.S)

                    # relative to cell
                    target[cell_y, cell_x, 0:4] = torch.tensor([
                        cx * self.S - cell_x,   # cx within cell [0,1)
                        cy * self.S - cell_y,   # cy within cell [0,1)
                        w,                      # width relative to image
                        h,                      # height relative to image
                    ])
                    target[cell_y, cell_x, 4] = 1.0
                    target[cell_y, cell_x, 5 + cls] = 1.0

        return img, target
