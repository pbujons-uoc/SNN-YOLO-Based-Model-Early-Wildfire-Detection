# tests/inspect_encoder.py
import cv2
import torch
from model_utils import ModelWrapper

wrapper = ModelWrapper('SpikeYOLO_latency', 'Results/SpikeYOLO_latency/weights/best.pt')
cap = cv2.VideoCapture('datasets/video_tests/big_fire2.mp4')
ret, frame = cap.read()
cap.release()

img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
enc = wrapper.encode_frame_single(img_rgb)

print(f"encode_frame_single output:")
print(f"  shape: {enc.shape}")
print(f"  dtype: {enc.dtype}")
enc_f = enc.float()
print(f"  min: {enc_f.min()}, max: {enc_f.max()}, mean: {enc_f.mean()}")
print(f"  sum: {enc_f.sum()}, nonzero: {(enc_f != 0).sum()}")
print(f"  first 10 values: {enc_f.flatten()[:10]}")
print(f"  device: {enc.device}")

# Ahora apila 4 veces como en test_video y mira la entrada del modelo
stacked = torch.stack([enc for _ in range(4)], dim=0)
stacked_batch = stacked.unsqueeze(0).to(wrapper.device)
print(f"\nStacked [1,T,C,H,W]:")
print(f"  shape: {stacked_batch.shape}")
print(f"  dtype: {stacked_batch.dtype}")
print(f"  device: {stacked_batch.device}")