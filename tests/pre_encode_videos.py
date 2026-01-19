"""
Pre-encode videos for faster testing (encoding done offline).
This script encodes videos with different encoding types and saves them to disk.

Usage:
    python tests/pre_encode_videos.py
    python tests/pre_encode_videos.py --videos big_fire1.mp4
    python tests/pre_encode_videos.py --encodings latency poisson
"""

import argparse
import cv2
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import sys
import os

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from model_utils import ModelWrapper, DEFAULT_WEIGHTS

VIDEO_DIR = Path(project_root) / "datasets" / "video_tests"
OUTPUT_DIR = Path(project_root) / "datasets" / "video_tests_encoded"


def get_available_videos():
    """Get list of video files in datasets/video_tests."""
    if not VIDEO_DIR.exists():
        raise FileNotFoundError(f"Video directory not found: {VIDEO_DIR}")
    
    videos = list(VIDEO_DIR.glob("*.mp4")) + list(VIDEO_DIR.glob("*.avi"))
    return sorted([v.name for v in videos])


def encode_video(video_path, encoding_type, time_steps=4):
    """
    Encode a video and save to disk.
    
    Args:
        video_path: Path to input video
        encoding_type: 'latency' or 'poisson'
        time_steps: Number of time steps for encoding
    
    Returns:
        Path to encoded video file
    """
    video_name = video_path.stem
    
    # Create output directory
    output_dir = OUTPUT_DIR / encoding_type / f"T{time_steps}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Output path (save as .pt file)
    output_path = output_dir / f"{video_name}.pt"
    
    if output_path.exists():
        print(f"Already encoded: {output_path}")
        return output_path
    
    # Initialize encoder wrapper (using SpikeYOLO_latency or SpikeYOLO_poisson weights)
    if encoding_type == 'latency':
        model_name = 'SpikeYOLO_latency'
    elif encoding_type == 'poisson':
        model_name = 'SpikeYOLO_poisson'
    else:
        raise ValueError(f"Unknown encoding type: {encoding_type}")
    
    weights_path = DEFAULT_WEIGHTS[model_name]
    print(f"\nEncoding {video_path.name} with {encoding_type} encoding (T={time_steps})...")
    print(f"Using weights: {weights_path}")
    
    # Initialize wrapper with correct API
    wrapper = ModelWrapper(
        model_name,
        weights_path,
        time_steps=time_steps,
        encoding=encoding_type
    )
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video info: {total_frames} frames at {fps} FPS")
    
    # Encode all frames
    encoded_frames = []
    original_shapes = []
    
    pbar = tqdm(total=total_frames, desc="Encoding")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        original_shapes.append((h, w))
        
        # Encode frame [1, T, C, H, W]
        enc = wrapper.encode_image(img_rgb)
        # Remove batch dimension and move to CPU
        enc = enc.squeeze(0).cpu()  # [T, C, H, W]
        
        encoded_frames.append(enc)
        pbar.update(1)
    
    pbar.close()
    cap.release()
    
    # Save encoded frames as a single tensor
    # Stack all frames: [num_frames, T, C, H, W]
    encoded_video = torch.stack(encoded_frames, dim=0)
    
    # Save metadata along with encoded video
    save_data = {
        'encoded_frames': encoded_video,
        'original_shapes': original_shapes,
        'encoding_type': encoding_type,
        'time_steps': time_steps,
        'fps': fps,
        'num_frames': total_frames,
        'video_name': video_name
    }
    
    torch.save(save_data, output_path)
    print(f"Saved encoded video to: {output_path}")
    print(f"Tensor shape: {encoded_video.shape}")
    
    return output_path


def encode_video_single_steps(video_path, encoding_type, time_steps=4):
    """
    Encode a video as single-step encodings for sliding window mode.
    
    Args:
        video_path: Path to input video
        encoding_type: 'latency' or 'poisson'
        time_steps: Number of time steps (used for naming, but each frame is single-step)
    
    Returns:
        Path to encoded video file
    """
    video_name = video_path.stem
    
    # Create output directory
    output_dir = OUTPUT_DIR / encoding_type / f"T{time_steps}_single"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Output path (save as .pt file)
    output_path = output_dir / f"{video_name}.pt"
    
    if output_path.exists():
        print(f"Already encoded (single-step): {output_path}")
        return output_path
    
    # Initialize encoder wrapper
    if encoding_type == 'latency':
        model_name = 'SpikeYOLO_latency'
    elif encoding_type == 'poisson':
        model_name = 'SpikeYOLO_poisson'
    else:
        raise ValueError(f"Unknown encoding type: {encoding_type}")
    
    weights_path = DEFAULT_WEIGHTS[model_name]
    print(f"\nEncoding {video_path.name} with {encoding_type} encoding (single-step for sliding window)...")
    print(f"Using weights: {weights_path}")
    
    # Initialize wrapper with correct API
    wrapper = ModelWrapper(
        model_name,
        weights_path,
        time_steps=time_steps,
        encoding=encoding_type
    )
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video info: {total_frames} frames at {fps} FPS")
    
    # Encode all frames as single-step
    encoded_frames = []
    original_shapes = []
    
    pbar = tqdm(total=total_frames, desc="Encoding (single-step)")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        original_shapes.append((h, w))
        
        # Encode frame as single step [C, H, W]
        enc_single = wrapper.encode_frame_single(img_rgb)
        if enc_single is None:
            raise RuntimeError("Single-step encoding not available for this model.")
        
        # Move to CPU
        enc_single = enc_single.cpu()  # [C, H, W]
        
        encoded_frames.append(enc_single)
        pbar.update(1)
    
    pbar.close()
    cap.release()
    
    # Save encoded frames as a list of tensors
    # Stack all frames: [num_frames, C, H, W]
    encoded_video = torch.stack(encoded_frames, dim=0)
    
    # Save metadata along with encoded video
    save_data = {
        'encoded_frames': encoded_video,
        'original_shapes': original_shapes,
        'encoding_type': encoding_type,
        'time_steps': time_steps,
        'fps': fps,
        'num_frames': total_frames,
        'video_name': video_name,
        'single_step': True
    }
    
    torch.save(save_data, output_path)
    print(f"Saved encoded video (single-step) to: {output_path}")
    print(f"Tensor shape: {encoded_video.shape}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Pre-encode videos for testing")
    parser.add_argument('--videos', type=str, nargs='+', default=None,
                        help='Videos to encode (default: all in datasets/video_tests)')
    parser.add_argument('--encodings', type=str, nargs='+', default=['latency', 'poisson'],
                        choices=['latency', 'poisson'],
                        help='Encoding types to use')
    parser.add_argument('--time-steps', type=int, default=4,
                        help='Time steps for spike encoding')
    
    args = parser.parse_args()
    
    # Get videos to encode
    available_videos = get_available_videos()
    if args.videos:
        videos_to_encode = [v for v in args.videos if v in available_videos]
        if len(videos_to_encode) < len(args.videos):
            print(f"Warning: Some videos not found in {VIDEO_DIR}")
    else:
        videos_to_encode = available_videos
    
    print(f"\nVideos to encode: {', '.join(videos_to_encode)}")
    print(f"Encodings: {', '.join(args.encodings)}")
    print(f"Time steps: {args.time_steps}\n")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Encode each video with each encoding type
    for video_name in videos_to_encode:
        video_path = VIDEO_DIR / video_name
        
        for encoding_type in args.encodings:
            # Encode for repeat mode (full time steps)
            encode_video(video_path, encoding_type, args.time_steps)
            
            # Encode for sliding mode (single steps)
            encode_video_single_steps(video_path, encoding_type, args.time_steps)
    
    print("\n" + "="*80)
    print("PRE-ENCODING COMPLETED")
    print("="*80)
    print(f"Encoded videos saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
