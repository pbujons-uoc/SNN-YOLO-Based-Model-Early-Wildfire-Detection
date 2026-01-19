# utils_spike_encoding.py

import torch
from PIL import Image
import numpy as np

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def is_image_file(path):
    return any(path.lower().endswith(ext) for ext in IMG_EXTENSIONS)


def load_image_as_tensor(path):
    img = Image.open(path).convert("RGB")
    img = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1)  # [C,H,W]
