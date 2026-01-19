"""
Simplified video testing script - ONLY measures inference time and energy.
NO encoding measurements.

Usage:
    python tests/test_video_simple.py --model SpikeYOLO --video datasets/video_tests/big_fire1.mp4
    python tests/test_video_simple.py --model YOLO --video datasets/video_tests/big_fire1.mp4
"""

import argparse
import os
import cv2
import time
import json
from tqdm import tqdm
from codecarbon import EmissionsTracker
from pathlib import Path
import torch

from model_utils import ModelWrapper, project_root, DEFAULT_WEIGHTS


def main():
    parser = argparse.ArgumentParser(description="Simple video testing - only inference metrics")
    parser.add_argument('--model', type=str, required=True, choices=DEFAULT_WEIGHTS.keys())
    parser.add_argument('--weights', type=str, help='Path to weights file')
    parser.add_argument('--video', type=str, required=True, help='Path to input video')
    parser.add_argument('--output-dir', type=str, default='Results_test_video', help='Base output directory')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--time-steps', type=int, default=4, help='Time steps for spike encoding')
    parser.add_argument('--temporal-mode', type=str, choices=['repeat', 'sliding', 'batched'], default='repeat',
                        help='Temporal mode: repeat (same frame), sliding (overlapping), batched (non-overlapping)')
    
    args = parser.parse_args()
    
    # Setup
    weights = args.weights if args.weights else DEFAULT_WEIGHTS[args.model]
    video_name = Path(args.video).stem
    
    # Determine encoding type
    encoding_type = None
    if 'latency' in args.model.lower():
        encoding_type = 'latency'
    elif 'poisson' in args.model.lower():
        encoding_type = 'poisson'
    
    # Output directory
    output_dir = Path(args.output_dir) / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"Testing: {args.model} on {video_name}")
    print(f"Encoding: {encoding_type if encoding_type else 'RGB'}")
    print(f"Temporal mode: {args.temporal_mode if encoding_type else 'N/A'}")
    print(f"{'='*80}\n")
    
    # Load model
    try:
        wrapper = ModelWrapper(args.model, weights, time_steps=args.time_steps, encoding=encoding_type)
        # Set temporal mode if applicable
        if hasattr(wrapper, 'temporal_mode') and encoding_type:
            wrapper.temporal_mode = args.temporal_mode
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Open video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Could not open video {args.video}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames to process: {total_frames}")
    
    # Initialize tracking variables
    frames_processed = 0
    
    # Start energy tracker
    tracker = EmissionsTracker(
        project_name=f"{args.model}_inference",
        measure_power_secs=1,
        save_to_file=False,
        log_level='error'
    )
    
    # Process video
    pbar = tqdm(total=total_frames, desc="Reading frames")
    
    # ============================================
    # STEP 1: Read and pre-process ALL frames (NOT measured)
    # ============================================
    
    if encoding_type:
        # ENCODED MODELS (SpikeYOLO_latency, SpikeYOLO_poisson) - use temporal mode
        
        if wrapper.temporal_mode == 'batched':
            # BATCHED MODE: non-overlapping temporal windows
            frame_buffer = []
            batched_tensors = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    # Process remaining frames in buffer if any
                    if frame_buffer:
                        # Pad buffer to time_steps if needed
                        while len(frame_buffer) < args.time_steps:
                            frame_buffer.append(frame_buffer[-1])
                        
                        # Pre-encode this batch
                        encoded_frames = []
                        for f in frame_buffer:
                            img_rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                            from model_utils import letterbox_resize
                            img_resized, _, _ = letterbox_resize(img_rgb, target_size=640)
                            
                            if encoding_type == 'latency':
                                from data_encoding.latency_encoding import latency_encode
                                spike = latency_encode(img_resized, args.time_steps)[0]
                                encoded_frames.append(spike.to(torch.uint8).cpu())
                            elif encoding_type == 'poisson':
                                from data_encoding.poisson_encoding import poisson_encode
                                spike = poisson_encode(img_resized, args.time_steps)[0]
                                encoded_frames.append(spike.to(torch.uint8).cpu())
                        
                        input_tensor = torch.stack(encoded_frames, dim=0).unsqueeze(0)
                        batched_tensors.append((input_tensor, frame_buffer[-1].shape[:2]))
                        pbar.update(len(frame_buffer))
                    break
                
                frame_buffer.append(frame)
                
                # When buffer is full, encode it
                if len(frame_buffer) == args.time_steps:
                    encoded_frames = []
                    for f in frame_buffer:
                        img_rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                        from model_utils import letterbox_resize
                        img_resized, _, _ = letterbox_resize(img_rgb, target_size=640)
                        
                        if encoding_type == 'latency':
                            from data_encoding.latency_encoding import latency_encode
                            spike = latency_encode(img_resized, args.time_steps)[0]
                            encoded_frames.append(spike.to(torch.uint8).cpu())
                        elif encoding_type == 'poisson':
                            from data_encoding.poisson_encoding import poisson_encode
                            spike = poisson_encode(img_resized, args.time_steps)[0]
                            encoded_frames.append(spike.to(torch.uint8).cpu())
                    
                    input_tensor = torch.stack(encoded_frames, dim=0).unsqueeze(0)
                    batched_tensors.append((input_tensor, frame_buffer[-1].shape[:2]))
                    pbar.update(len(frame_buffer))
                    frame_buffer = []
            
            pbar.close()
            preprocessed_tensors = batched_tensors
            frames_per_inference = args.time_steps
            
        else:
            # REPEAT/SLIDING MODE: frame-by-frame encoding
            all_frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                all_frames.append(frame)
                pbar.update(1)
            
            pbar.close()
            pbar = tqdm(total=len(all_frames), desc="Encoding")
            
            preprocessed_tensors = []
            for frame in all_frames:
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                input_tensor = wrapper.encode_image(img_rgb)  # Returns [1, T, C, H, W]
                preprocessed_tensors.append((input_tensor, frame.shape[:2]))
                pbar.update(1)
            
            pbar.close()
            frames_per_inference = 1
    
    else:
        # RGB MODELS (YOLO, VanillaCNN, SpikeYOLO) - simple frame-by-frame
        all_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            all_frames.append(frame)
            pbar.update(1)
        
        pbar.close()
        pbar = tqdm(total=len(all_frames), desc="Preprocessing")
        
        preprocessed_tensors = []
        for frame in all_frames:
            tensor_data = wrapper.preprocess_to_tensor(frame)
            preprocessed_tensors.append(tensor_data)
            pbar.update(1)
        
        pbar.close()
        frames_per_inference = 1
    
    # ============================================
    # STEP 2: Run ONLY inference (MEASURED)
    # ============================================
    pbar = tqdm(total=len(preprocessed_tensors), desc="Inference")
    inference_start_time = time.time()
    tracker.start()
    
    if encoding_type:
        # Encoded models
        for input_tensor, orig_shape in preprocessed_tensors:
            _ = wrapper.predict_tensor(input_tensor, conf_thres=args.conf, iou_thres=0.45, original_shape=orig_shape)
            frames_processed += frames_per_inference
            pbar.update(1)
    else:
        # RGB models
        for tensor_data in preprocessed_tensors:
            _ = wrapper.predict_tensor_only(tensor_data, conf_thres=args.conf)
            frames_processed += 1
            pbar.update(1)
    
    pbar.close()
    cap.release()
    
    # Stop tracking
    inference_time = time.time() - inference_start_time
    emissions_kg = tracker.stop()
    
    # Get energy in kWh
    energy_kwh = 0.0
    if hasattr(tracker, '_total_energy') and tracker._total_energy:
        energy_kwh = tracker._total_energy.kWh
    
    # Calculate per-frame metrics
    energy_per_frame = energy_kwh / frames_processed if frames_processed > 0 else 0.0
    time_per_frame = inference_time / frames_processed if frames_processed > 0 else 0.0
    
    # Results
    results = {
        'model': args.model,
        'encoding': encoding_type,
        'temporal_mode': args.temporal_mode if encoding_type else None,
        'video': video_name,
        'frames_processed': frames_processed,
        'inference_time_s': inference_time,
        'inference_energy_kwh': energy_kwh,
        'emissions_kg_co2': float(emissions_kg) if emissions_kg else 0.0,
        'time_per_frame_s': time_per_frame,
        'energy_per_frame_kwh': energy_per_frame,
        'fps': frames_processed / inference_time if inference_time > 0 else 0.0
    }
    
    # Save results
    output_file = output_dir / f"{video_name}_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"RESULTS")
    print(f"{'='*80}")
    print(f"Frames processed:     {frames_processed}")
    print(f"Total time:           {inference_time:.2f} s")
    print(f"Total energy:         {energy_kwh:.6f} kWh")
    print(f"Total emissions:      {results['emissions_kg_co2']:.6f} kg CO2")
    print(f"Time per frame:       {time_per_frame*1000:.2f} ms")
    print(f"Energy per frame:     {energy_per_frame*1000000:.2f} µWh")
    print(f"FPS:                  {results['fps']:.2f}")
    print(f"\nResults saved to: {output_file}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
