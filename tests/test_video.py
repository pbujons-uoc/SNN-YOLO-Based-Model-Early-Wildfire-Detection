import argparse
import os
import cv2
import time
import torch
import numpy as np
from tqdm import tqdm
import json
import logging
from datetime import datetime
from collections import deque
from codecarbon import EmissionsTracker
from model_utils import ModelWrapper, project_root, DEFAULT_WEIGHTS
from pathlib import Path

def load_preencoded_video(video_path, encoding_type, time_steps, temporal_mode):
    """
    Load a pre-encoded video from disk.
    
    Args:
        video_path: Path to original video
        encoding_type: 'latency' or 'poisson'
        time_steps: Number of time steps
        temporal_mode: 'repeat' or 'sliding'
    
    Returns:
        Dictionary with encoded frames and metadata
    """
    video_name = Path(video_path).stem
    encoded_dir = Path(project_root) / "datasets" / "video_tests_encoded" / encoding_type
    
    # Choose the right directory based on temporal mode
    if temporal_mode == 'sliding':
        encoded_dir = encoded_dir / f"T{time_steps}_single"
    else:
        encoded_dir = encoded_dir / f"T{time_steps}"
    
    encoded_file = encoded_dir / f"{video_name}.pt"
    
    if not encoded_file.exists():
        raise FileNotFoundError(
            f"Pre-encoded video not found: {encoded_file}\n"
            f"Please run: python tests/pre_encode_videos.py"
        )
    
    print(f"Loading pre-encoded video from: {encoded_file}")
    data = torch.load(encoded_file, map_location='cpu')
    
    return data


def decode_spikes_to_image(spikes_tensor, encoding_type='poisson'):
    """
    Reconstructs an RGB image from a spike tensor [T, C, H, W].
    For rate/poisson: average spikes over time.
    For latency: find first spike time and map to intensity.
    Returns: numpy array (H, W, 3) uint8 0-255 in grayscale
    """
    T, C, H, W = spikes_tensor.shape
    spikes_tensor = spikes_tensor.float()
    
    if encoding_type in ['poisson', 'rate']:
        img_float = spikes_tensor.mean(dim=0)  # [C, H, W]
        img_float = img_float.permute(1, 2, 0)  # [H, W, C]
        img_np = img_float.cpu().numpy()
        img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
        # Convert to grayscale
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        img_np = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        return img_np
    
    elif encoding_type == 'latency':
        first_spike_times = torch.argmax(spikes_tensor.float(), dim=0)  # [C, H, W]
        any_spike = torch.any(spikes_tensor, dim=0).float()  # [C, H, W]
        if T > 1:
            img_float = 1.0 - (first_spike_times.float() / (T - 1))
        else:
            img_float = torch.ones_like(first_spike_times).float()
        img_float = img_float * any_spike
        img_float = img_float.permute(1, 2, 0)  # [H, W, C]
        img_np = img_float.cpu().numpy()
        img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
        # Convert to grayscale
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        img_np = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        return img_np
    
    else:
        raise ValueError(f"Unknown encoding type: {encoding_type}")


def inverse_letterbox_coords(boxes, letterbox_size=640, original_shape=(1080, 1920)):
    """
    Transform bbox coordinates from letterbox space back to original image space.
    
    Args:
        boxes: [[x1, y1, x2, y2, conf, cls], ...] in letterbox space (e.g., 640x640)
        letterbox_size: size of the letterbox (default 640)
        original_shape: (H, W) of original image
    
    Returns:
        boxes_original: coordinates scaled to original image space
    """
    if not boxes:
        return []
    
    h, w = original_shape
    scale = min(letterbox_size / h, letterbox_size / w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    pad_h = (letterbox_size - new_h) // 2
    pad_w = (letterbox_size - new_w) // 2
    
    boxes_original = []
    for box in boxes:
        x1, y1, x2, y2 = box[:4]
        
        # Remove padding offset and scale back
        x1 = (x1 - pad_w) / scale
        y1 = (y1 - pad_h) / scale
        x2 = (x2 - pad_w) / scale
        y2 = (y2 - pad_h) / scale
        
        # Clip to original bounds
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        
        boxes_original.append([x1, y1, x2, y2, *box[4:]])
    
    return boxes_original


def draw_predictions(img, detections, class_names):
    """
    Draws predictions on the image.
    detections: [[x1, y1, x2, y2, conf, cls], ...]
    """
    if not detections:
        return img

    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        cls_id = int(cls)

        if isinstance(class_names, dict):
            label_text = class_names.get(cls_id, str(cls_id))
        elif isinstance(class_names, list) and 0 <= cls_id < len(class_names):
            label_text = class_names[cls_id]
        else:
            label_text = str(cls_id)

        label = f"{label_text} {conf:.2f}"

        p1 = (int(x1), int(y1))
        p2 = (int(x2), int(y2))

        if isinstance(label_text, str) and 'fire' in label_text.lower():
            color = (0, 0, 255)
        else:
            color = (255, 0, 0)

        cv2.rectangle(img, p1, p2, color, 2)

        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        c2 = (p1[0] + t_size[0], p1[1] - t_size[1] - 3)
        cv2.rectangle(img, p1, c2, color, -1)
        cv2.putText(img, label, (p1[0], p1[1]-2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return img


def iou(boxA, boxB):
    # boxes are [x1,y1,x2,y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    denom = float(boxAArea + boxBArea - interArea)
    if denom == 0:
        return 0.0
    return interArea / denom


def filter_and_limit(dets, min_area=0, max_boxes=None):
    # dets: [[x1,y1,x2,y2,conf,cls], ...]
    filtered = []
    for d in (dets or []):
        w = max(0, d[2] - d[0])
        h = max(0, d[3] - d[1])
        area = w * h
        if area >= min_area:
            filtered.append(d)
    # sort by conf desc
    filtered.sort(key=lambda x: x[4], reverse=True)
    if max_boxes is not None and len(filtered) > max_boxes:
        filtered = filtered[:max_boxes]
    return filtered


class SimpleTracker:
    """Very small IoU-based tracker to require temporal persistence"""
    def __init__(self, iou_thresh=0.5, persist_frames=1):
        self.iou_thresh = iou_thresh
        self.persist_frames = persist_frames
        self.tracks = []  # each: {bbox, cls, conf, count, last_seen, missed}

    def update(self, detections, frame_idx):
        # detections: list of [x1,y1,x2,y2,conf,cls]
        matches = [False] * len(detections)
        # first try to match existing tracks
        for t in self.tracks:
            t['matched'] = False
        for i, d in enumerate(detections):
            best = None
            best_iou = 0.0
            for t in self.tracks:
                if t['cls'] != int(d[5]):
                    continue
                iou_v = iou(t['bbox'], d[:4])
                if iou_v > best_iou:
                    best_iou = iou_v
                    best = t
            if best and best_iou >= self.iou_thresh:
                # update track
                best['bbox'] = d[:4]
                best['conf'] = float(d[4])
                best['count'] += 1
                best['last_seen'] = frame_idx
                best['missed'] = 0
                best['matched'] = True
                matches[i] = True
            # else leave unmatched for now
        # create tracks for unmatched detections
        for i, d in enumerate(detections):
            if not matches[i]:
                self.tracks.append({
                    'bbox': d[:4],
                    'cls': int(d[5]),
                    'conf': float(d[4]),
                    'count': 1,
                    'last_seen': frame_idx,
                    'missed': 0,
                    'matched': True
                })
        # increment missed and remove old tracks
        new_tracks = []
        for t in self.tracks:
            if not t.get('matched', False):
                t['missed'] += 1
            if t['missed'] <= max(1, self.persist_frames):
                new_tracks.append(t)
        self.tracks = new_tracks

    def get_persistent(self):
        # return tracks with count >= persist_frames
        out = []
        for t in self.tracks:
            if t['count'] >= self.persist_frames:
                out.append([float(t['bbox'][0]), float(t['bbox'][1]), float(t['bbox'][2]), float(t['bbox'][3]), float(t['conf']), float(t['cls'])])
        return out

def main():
    parser = argparse.ArgumentParser(description="Test Video with various models and energy tracking")
    parser.add_argument('--model', type=str, required=True, choices=DEFAULT_WEIGHTS.keys())
    parser.add_argument('--weights', type=str, help='Path to weights file.')
    parser.add_argument('--video', type=str, required=True, help='Path to input video')
    parser.add_argument('--output', type=str, default=None, help='Output video path. Defaults to Results_test/{model}/{encoding}_{time-steps}/{video_name}.mp4')
    parser.add_argument('--output-name', type=str, default=None, help='Custom name for output video (without extension). If not provided, uses the video filename.')
    parser.add_argument('--time-steps', type=int, default=4, help='Time steps for spike encoding (default 4)')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size for inference (default: 640)')
    parser.add_argument('--conf', type=float, default=None, help='Confidence threshold (if not set, uses model-specific default)')
    parser.add_argument('--encoding', type=str, choices=['latency', 'poisson'], default=None, help='Force specific encoding type')
    parser.add_argument('--temporal-mode', type=str, choices=['repeat', 'sliding'], default='repeat',
                        help='How to convert a frame into T steps: repeat the same frame (repeat) or use a sliding window of T consecutive frames (sliding)')
    parser.add_argument('--use-preencoded', action='store_true', help='Use pre-encoded videos from datasets/video_tests_encoded (skips encoding time, only measures inference)')
    parser.add_argument('--debug', action='store_true', help='Enable per-frame debug logging of detections and save a debug JSON')
    parser.add_argument('--debug-sample', type=int, default=50, help='Save first N frames detections to debug JSON (default: 50)')

    # Simple filtering and persistence options
    parser.add_argument('--min-area', type=int, default=0, help='Minimum bbox area (px^2) to keep')
    parser.add_argument('--max-boxes', type=int, default=None, help='Maximum boxes to keep per frame (after filtering), by confidence')
    parser.add_argument('--temporal-persist', type=int, default=1, help='Number of consecutive frames a box must appear to be shown')
    parser.add_argument('--persist-iou', type=float, default=0.5, help='IoU threshold to associate boxes across frames for persistence')

    # Confidence grid search (optional automatic threshold selection)
    parser.add_argument('--conf-grid', action='store_true', help='Run a quick grid over confidence values on the first frames to select a conf automatically')
    parser.add_argument('--conf-grid-only', action='store_true', help='Run conf grid and exit')
    parser.add_argument('--conf-grid-values', type=str, default='0.0001,0.001,0.01,0.05,0.1,0.2', help='Comma-separated conf values to evaluate during grid search')
    parser.add_argument('--conf-grid-frames', type=int, default=50, help='Number of initial frames to sample per conf candidate')
    parser.add_argument('--conf-target-dets', type=float, default=1.0, help='Target average detections per frame for automatic conf selection')
    
    args = parser.parse_args()
    
    # Set model-specific default confidence thresholds if not provided
    if args.conf is None:
        model_conf_defaults = {
            'YOLO': 0.25,
            'SpikeYOLO': 0.25,
            'SpikeYOLO_latency': 0.12,
            'SpikeYOLO_poisson': 0.12,
            'VanillaCNN': 0.2
        }
        args.conf = model_conf_defaults.get(args.model, 0.25)
        print(f"Using model-specific confidence threshold: {args.conf}")

    # Suppress verbose CodeCarbon logs unless in debug mode
    if not args.debug:
        logging.getLogger('codecarbon').setLevel(logging.ERROR)
    
    # Set Dynamic Output Paths
    video_basename = os.path.splitext(os.path.basename(args.video))[0]
    
    # Determine custom output name (default to video basename)
    output_name = args.output_name if args.output_name else video_basename
    
    # Build directory structure: Results_test_video/{model}/{encoding}_{time-steps}/{output_name}.mp4
    # For spike models, use encoding info; for RGB models, use "rgb" as placeholder
    if args.encoding:
        subdir = f"{args.encoding}_{args.time_steps}"
    else:
        # For RGB models or when encoding not specified
        subdir = "rgb"
    
    base_out_dir = os.path.join("Results_test_video", args.model, subdir)
    os.makedirs(base_out_dir, exist_ok=True)
    
    if args.output is None:
        args.output = os.path.join(base_out_dir, f"{output_name}.mp4")
        
    energy_dir = os.path.join(base_out_dir, "energy_results")
    os.makedirs(energy_dir, exist_ok=True)

    # Per-video energy CSV filename (e.g., myvideo.csv)
    energy_file = f"{video_basename}.csv"

    # Emissions accumulators (kg CO2)
    infer_emissions_total = 0.0
    prediction_emissions_total = 0.0

    # Per-task records and timing
    infer_records = []
    prediction_records = []

    encode_total_duration = 0.0
    infer_total_duration = 0.0
    prediction_total_duration = 0.0

    tracker_infer_start_time = None
    prediction_start_time = None
    
    weights = args.weights if args.weights else DEFAULT_WEIGHTS[args.model]
    
    # Check video
    if not os.path.exists(args.video):
        print(f"Error: Video file {args.video} not found.")
        return
        
    # Load model
    try:
        # Try with imgsz parameter (new version)
        try:
            wrapper = ModelWrapper(args.model, weights, time_steps=args.time_steps, encoding=args.encoding, imgsz=args.imgsz)
        except TypeError as e:
            # Fallback for old version without imgsz parameter
            if 'imgsz' in str(e):
                print(f"Warning: ModelWrapper doesn't support imgsz parameter, using default. Please update model_utils.py")
                wrapper = ModelWrapper(args.model, weights, time_steps=args.time_steps, encoding=args.encoding)
                # Set imgsz manually after initialization
                wrapper.imgsz = args.imgsz
            else:
                raise
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Video Capture
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Could not open video {args.video}")
        return
        
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Processing {args.video} ({width}x{height} @ {fps} fps). Total frames: {total_frames}")
    
    # Video Writer
    # Fallback FPS if video metadata doesn't provide it
    if fps is None or fps <= 0:
        print(f"Warning: video FPS not detected (fps={fps}). Falling back to 30.0 FPS.")
        fps = 30.0

    # Ensure parent directory for custom output exists
    out_dir = os.path.dirname(args.output)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        print(f"Created output directory: {out_dir}")

    # If conf-grid mode requested, run simple grid on first frames to choose a conf
    if args.conf_grid:
        print("Running quick confidence grid search...")
        vals = [float(x) for x in args.conf_grid_values.split(',')]
        sample_frames = args.conf_grid_frames
        cap_grid = cv2.VideoCapture(args.video)
        frames = []
        for i in range(sample_frames):
            ret, f = cap_grid.read()
            if not ret:
                break
            frames.append(f)
        cap_grid.release()
        if len(frames) == 0:
            print("Warning: couldn't read frames for conf-grid, skipping grid.")
        else:
            results = []
            if args.temporal_mode == 'sliding':
                print("Note: grid search uses per-frame (repeat) predictions even if sliding mode is active — estimates may differ.")
            for conf_cand in vals:
                s = 0
                for f in frames:
                    dets = []
                    try:
                        dets = wrapper.predict(f, conf_thres=conf_cand)
                    except Exception as e:
                        print(f"Warning during grid predict: {e}")
                    dets_f = filter_and_limit(dets, min_area=args.min_area, max_boxes=args.max_boxes)
                    s += len(dets_f)
                avg = s / max(1, len(frames))
                results.append({'conf': conf_cand, 'avg_dets_per_frame': avg})
                print(f"conf={conf_cand} -> avg_dets/frame={avg:.3f}")
            # Choose conf closest to target
            best = min(results, key=lambda r: abs(r['avg_dets_per_frame'] - args.conf_target_dets))
            chosen = best['conf']
            print(f"Selected conf={chosen} (avg_dets/frame={best['avg_dets_per_frame']:.3f})")
            args.conf = chosen
            if args.conf_grid_only:
                print('Grid-only mode, exiting')
                return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # For spike models: create two outputs (_rgb and _encoded)
    # For RGB models: create single output
    out_rgb = None
    out_encoded = None
    
    if wrapper.is_spike:
        # Generate two output paths
        base, ext = os.path.splitext(args.output)
        output_rgb = f"{base}_rgb{ext}"
        output_encoded = f"{base}_encoded{ext}"
        
        out_rgb = cv2.VideoWriter(output_rgb, fourcc, fps, (width, height))
        out_encoded = cv2.VideoWriter(output_encoded, fourcc, fps, (width, height))
        
        if not out_rgb.isOpened() or not out_encoded.isOpened():
            print(f"Error: Failed to open VideoWriters for outputs '{output_rgb}' / '{output_encoded}'. Check codec or permissions.")
            cap.release()
            return
        print(f"Creating two outputs: {output_rgb} and {output_encoded}")
        out = out_rgb  # Keep compatibility with existing code
    else:
        # Standard RGB model: single output
        out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        if not out.isOpened():
            print(f"Error: Failed to open VideoWriter for output '{args.output}'. Check codec or permissions.")
            cap.release()
            return

    class_names = wrapper.get_class_names()
    frames_written = 0
    frame_idx = 0  # Global frame index across the whole video

    # Debugging containers
    debug_frames = []  # list of {'frame': idx, 'detections': [ {bbox, conf, cls}, ... ] }
    frames_with_detections = 0
    frames_without_detections = 0

    pbar = tqdm(total=total_frames)

    # Initialize simple tracker for temporal persistence
    dt_tracker = SimpleTracker(iou_thresh=args.persist_iou, persist_frames=args.temporal_persist)
    # For debug, show chosen conf
    if args.debug:
        print(f"Using conf={args.conf}, min_area={args.min_area}, max_boxes={args.max_boxes}, temporal_persist={args.temporal_persist}")
    
    # CASE 1: Spike Models (Separate Encoding/Prediction Tracking)
    if wrapper.encoding_type:
        print(f"Mode: Spike Encoding ({wrapper.encoding_type}) -> Energy tracking only for inference (mode={args.temporal_mode})")

        # Use output file for inference only
        energy_file_infer = f"{video_basename}_inference.csv"

        tracker_infer = EmissionsTracker(project_name=f"{args.model}_inference", measure_power_secs=1, save_to_file=True, output_file=energy_file_infer, output_dir=energy_dir)

        # Local running flag
        tracker_infer_running = False

        batch_size = 4  # Adjust based on VRAM/RAM

        # Mode: REPEAT (default) -> same as before: each frame encoded with T steps
        if args.temporal_mode == 'repeat':
            # Check if using pre-encoded videos
            if args.use_preencoded:
                print("Using Pre-Encoded Video (repeat mode) - loading from disk...")
                
                # Load pre-encoded video
                encode_start_time = time.time()
                try:
                    preencoded_data = load_preencoded_video(args.video, wrapper.encoding_type, args.time_steps, args.temporal_mode)
                except Exception as e:
                    print(f"Error loading pre-encoded video: {e}")
                    print("Falling back to on-the-fly encoding...")
                    args.use_preencoded = False
                
                if args.use_preencoded:  # Check again in case fallback occurred
                    encode_total_duration = time.time() - encode_start_time
                    
                    # Extract data
                    encoded_video_tensor = preencoded_data['encoded_frames']  # [num_frames, T, C, H, W]
                    all_original_shapes = preencoded_data['original_shapes']
                    
                    print(f"Loaded pre-encoded video: {encoded_video_tensor.shape} in {encode_total_duration:.2f}s")
                    print(f"Expected frames in video: {total_frames}, Encoded frames: {encoded_video_tensor.shape[0]}")
                    
                    # Load all original frames for visualization
                    all_original_frames = []
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_count = 0
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        all_original_frames.append(frame)
                        frame_count += 1
                        pbar.update(1)
                    
                    # Validate frame counts match
                    if len(all_original_frames) != encoded_video_tensor.shape[0]:
                        print(f"⚠️  Warning: Frame count mismatch!")
                        print(f"   Original frames: {len(all_original_frames)}, Encoded frames: {encoded_video_tensor.shape[0]}")
                        print(f"   Using minimum of both: {min(len(all_original_frames), encoded_video_tensor.shape[0])}")
                        # Trim to matching size
                        min_frames = min(len(all_original_frames), encoded_video_tensor.shape[0])
                        all_original_frames = all_original_frames[:min_frames]
                        encoded_video_tensor = encoded_video_tensor[:min_frames]
                        all_original_shapes = all_original_shapes[:min_frames]
                    
                    # Convert encoded video to list of tensors (add batch dimension)
                    all_encoded_frames = [encoded_video_tensor[i:i+1] for i in range(encoded_video_tensor.shape[0])]
                    
                    print(f"✅ Pre-encoded video ready: {len(all_encoded_frames)} frames, {len(all_original_frames)} original frames")
                    print(f"   Now proceeding to Phase 2: Inference...")
                
            if not args.use_preencoded:
                print("Using Batch Processing (repeat mode) - encoding all frames first, then inference...")
                
                # PHASE 1: Encode all frames (without measuring emissions)
                print("Phase 1: Encoding all frames...")
                all_encoded_frames = []
                all_original_frames = []
                all_original_shapes = []
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
                
                encode_start_time = time.time()
                while True:
                    frames_buffer = []
                    for _ in range(batch_size):
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frames_buffer.append(frame)

                    if not frames_buffer:
                        break

                    # Encoding (without energy tracking)
                    for frame in frames_buffer:
                        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        h, w = frame.shape[:2]
                        all_original_shapes.append((h, w))
                        all_original_frames.append(frame)

                        enc = wrapper.encode_image(img_rgb)  # [1, T, C, H, W]
                        all_encoded_frames.append(enc)
                        
                        pbar.update(1)
                        
                encode_total_duration = time.time() - encode_start_time
                print(f"Phase 1 complete: {len(all_encoded_frames)} frames encoded in {encode_total_duration:.2f}s (not counted in emissions)")
            
            # Reset progress bar for inference
            pbar.close()
            pbar = tqdm(total=len(all_encoded_frames), desc="Inference")
            
            # PHASE 2: Inference on encoded frames (with energy tracking)
            print("Phase 2: Running inference on encoded frames...")
            tracker_infer.start()
            tracker_infer_start_time = time.time()
            tracker_infer_running = True
            
            for i in range(len(all_encoded_frames)):
                # Inference Phase
                enc_tensor = all_encoded_frames[i]
                original_shape = all_original_shapes[i]
                
                dets = wrapper.predict_tensor(enc_tensor, conf_thres=args.conf, original_shape=original_shape)
                
                # Render
                frame = all_original_frames[i]
                filtered = filter_and_limit(dets, min_area=args.min_area, max_boxes=args.max_boxes)
                dt_tracker.update(filtered, frame_idx)
                draw_dets = dt_tracker.get_persistent()

                # Debugging: record final drawn detections stats and sample
                num = len(draw_dets) if draw_dets else 0
                if args.debug:
                    if num > 0:
                        frames_with_detections += 1
                    else:
                        frames_without_detections += 1
                    if len(debug_frames) < args.debug_sample:
                        sample = []
                        for det in (draw_dets or []):
                            sample.append({
                                'bbox': [float(det[0]), float(det[1]), float(det[2]), float(det[3])],
                                'conf': float(det[4]),
                                'cls': float(det[5])
                            })
                        debug_frames.append({'frame': frame_idx, 'detections': sample})
                        print(f"[DEBUG] frame {frame_idx}: {num} drawn detections (raw {len(dets) if dets else 0})")

                frame_vis = draw_predictions(frame, draw_dets, class_names)
                out_rgb.write(frame_vis)
                
                # Write encoded version
                if out_encoded is not None:
                    # Reconstruct from encoded tensor
                    enc_tensor_no_batch = enc_tensor.squeeze(0)  # [T, C, H, W]
                    reconstructed = decode_spikes_to_image(enc_tensor_no_batch, encoding_type=args.encoding)
                    # Resize reconstructed to match original frame size
                    if reconstructed.shape[:2] != (height, width):
                        reconstructed = cv2.resize(reconstructed, (width, height))
                    # Transform bbox coords from letterbox space to original image space
                    draw_dets_original = inverse_letterbox_coords(
                        draw_dets, 
                        letterbox_size=args.imgsz, 
                        original_shape=(height, width)
                    )
                    # Draw predictions with corrected coordinates on reconstructed
                    reconstructed_vis = draw_predictions(reconstructed, draw_dets_original, class_names)
                    out_encoded.write(reconstructed_vis)
                
                frames_written += 1
                pbar.update(1)
                frame_idx += 1

        # Mode: SLIDING -> build a rolling window of single-step encodings and slide it
        elif args.temporal_mode == 'sliding':
            T = args.time_steps
            
            # Check if using pre-encoded videos
            if args.use_preencoded:
                print("Using Pre-Encoded Video (sliding mode) - loading from disk...")
                
                # Load pre-encoded video (single-step encodings)
                encode_start_time = time.time()
                try:
                    preencoded_data = load_preencoded_video(args.video, wrapper.encoding_type, args.time_steps, args.temporal_mode)
                except Exception as e:
                    print(f"Error loading pre-encoded video: {e}")
                    print("Falling back to on-the-fly encoding...")
                    args.use_preencoded = False
                
                if args.use_preencoded:  # Check again in case fallback occurred
                    encode_total_duration = time.time() - encode_start_time
                    
                    # Extract data
                    encoded_video_tensor = preencoded_data['encoded_frames']  # [num_frames, C, H, W]
                    all_original_shapes = preencoded_data['original_shapes']
                    
                    print(f"Loaded pre-encoded video (single-step): {encoded_video_tensor.shape} in {encode_total_duration:.2f}s")
                    print(f"Expected frames in video: {total_frames}, Encoded frames: {encoded_video_tensor.shape[0]}")
                    
                    # Load all original frames for visualization
                    all_original_frames = []
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        all_original_frames.append(frame)
                        pbar.update(1)
                    
                    # Validate frame counts match
                    if len(all_original_frames) != encoded_video_tensor.shape[0]:
                        print(f"⚠️  Warning: Frame count mismatch!")
                        print(f"   Original frames: {len(all_original_frames)}, Encoded frames: {encoded_video_tensor.shape[0]}")
                        print(f"   Using minimum of both: {min(len(all_original_frames), encoded_video_tensor.shape[0])}")
                        # Trim to matching size
                        min_frames = min(len(all_original_frames), encoded_video_tensor.shape[0])
                        all_original_frames = all_original_frames[:min_frames]
                        encoded_video_tensor = encoded_video_tensor[:min_frames]
                        all_original_shapes = all_original_shapes[:min_frames]
                    
                    # Convert encoded video to list of tensors
                    all_single_encodings = [encoded_video_tensor[i] for i in range(encoded_video_tensor.shape[0])]
                    
                    print(f"✅ Pre-encoded video ready: {len(all_single_encodings)} frames (single-step)")
                    print(f"   Now proceeding to Phase 2: Sliding window inference...")
                
            if not args.use_preencoded:
                print("Using Sliding-Window Processing (sliding mode) - encoding all frames first, then inference...")
                
                # PHASE 1: Encode all frames as single-step encodings (without measuring emissions)
                print("Phase 1: Encoding all frames as single-step...")
                all_single_encodings = []
                all_original_frames = []
                all_original_shapes = []
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
                
                encode_start_time = time.time()
                while True:
                    frames_buffer = []
                    for _ in range(batch_size):
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frames_buffer.append(frame)

                    if not frames_buffer:
                        break

                    # Encoding single-step for each frame (without energy tracking)
                    for frame in frames_buffer:
                        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        h, w = frame.shape[:2]
                        all_original_shapes.append((h, w))
                        all_original_frames.append(frame)
                        
                        enc_single = wrapper.encode_frame_single(img_rgb)  # [C,H,W] on CPU
                        if enc_single is None:
                            raise RuntimeError("Sliding mode requested but encoder not available for this model.")
                        all_single_encodings.append(enc_single)
                        
                        pbar.update(1)
                        
                encode_total_duration = time.time() - encode_start_time
                print(f"Phase 1 complete: {len(all_single_encodings)} frames encoded in {encode_total_duration:.2f}s (not counted in emissions)")
            
            # Reset progress bar for inference
            pbar.close()
            pbar = tqdm(total=len(all_single_encodings), desc="Inference")
            
            # PHASE 2: Initialize sliding window and run inference (with energy tracking)
            print("Phase 2: Running inference with sliding window...")
            enc_window = deque(maxlen=T)  # holds [C,H,W] tensors (cpu)
            
            # Initialize window with first frame repeated T times
            if len(all_single_encodings) > 0:
                first_enc = all_single_encodings[0]
                for _ in range(T):
                    enc_window.append(first_enc)
            
            # Start inference tracking
            tracker_infer.start()
            tracker_infer_start_time = time.time()
            tracker_infer_running = True
            
            for i in range(len(all_single_encodings)):
                # Slide the window with new encoding
                enc_single = all_single_encodings[i]
                enc_window.append(enc_single)
                
                # Stack window and run inference
                stacked = torch.stack(list(enc_window), dim=0)  # [T, C, H, W]
                stacked = stacked.unsqueeze(0).to(wrapper.device)  # [1, T, C, H, W]
                
                original_shape = all_original_shapes[i]
                dets = wrapper.predict_tensor(stacked, conf_thres=args.conf, original_shape=original_shape)
                
                # Render
                frame = all_original_frames[i]
                filtered = filter_and_limit(dets, min_area=args.min_area, max_boxes=args.max_boxes)
                dt_tracker.update(filtered, frame_idx)
                draw_dets = dt_tracker.get_persistent()

                # Debugging: record final drawn detections stats and sample
                num = len(draw_dets) if draw_dets else 0
                if args.debug:
                    if num > 0:
                        frames_with_detections += 1
                    else:
                        frames_without_detections += 1
                    if len(debug_frames) < args.debug_sample:
                        sample = []
                        for det in (draw_dets or []):
                            sample.append({
                                'bbox': [float(det[0]), float(det[1]), float(det[2]), float(det[3])],
                                'conf': float(det[4]),
                                'cls': float(det[5])
                            })
                        debug_frames.append({'frame': frame_idx, 'detections': sample})
                        print(f"[DEBUG] frame {frame_idx}: {num} drawn detections (raw {len(dets) if dets else 0})")

                frame_vis = draw_predictions(frame, draw_dets, class_names)
                out_rgb.write(frame_vis)
                
                # Write encoded version
                if out_encoded is not None:
                    # Get current sliding window tensor
                    current_window = torch.stack(list(enc_window), dim=0)  # [T, C, H, W]
                    reconstructed = decode_spikes_to_image(current_window, encoding_type=args.encoding)
                    # Resize reconstructed to match original frame size
                    if reconstructed.shape[:2] != (height, width):
                        reconstructed = cv2.resize(reconstructed, (width, height))
                    # Transform bbox coords from letterbox space to original image space
                    draw_dets_original = inverse_letterbox_coords(
                        draw_dets, 
                        letterbox_size=args.imgsz, 
                        original_shape=(height, width)
                    )
                    # Draw predictions with corrected coordinates on reconstructed
                    reconstructed_vis = draw_predictions(reconstructed, draw_dets_original, class_names)
                    out_encoded.write(reconstructed_vis)
                
                frames_written += 1
                pbar.update(1)
                frame_idx += 1

        else:
            raise ValueError(f"Unknown temporal mode: {args.temporal_mode}")
                
    else:
        # CASE 2: RGB Models (Combined/Direct)
        # "calculate with codecarbon the expense of the prediction"
        print(f"Mode: Standard RGB ({args.model}) -> Single Energy Tracking")
        
        tracker = EmissionsTracker(project_name=f"{args.model}_prediction", measure_power_secs=1, save_to_file=True, output_file=energy_file, output_dir=energy_dir)
        tracker_running = False
        prediction_emissions_total = 0.0
        if not tracker_running:
            tracker.start()
            prediction_start_time = time.time()
            tracker_running = True
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Check for VanillaCNN specific preprocessing if not handled inside
            # predict() handles it.
            
            dets = wrapper.predict(frame, conf_thres=args.conf)
            filtered = filter_and_limit(dets, min_area=args.min_area, max_boxes=args.max_boxes)
            dt_tracker.update(filtered, frame_idx)
            draw_dets = dt_tracker.get_persistent()

            # Debugging
            num = len(draw_dets) if draw_dets else 0
            if args.debug:
                if num > 0:
                    frames_with_detections += 1
                else:
                    frames_without_detections += 1
                if len(debug_frames) < args.debug_sample:
                    sample = []
                    for det in (draw_dets or []):
                        sample.append({
                            'bbox': [float(det[0]), float(det[1]), float(det[2]), float(det[3])],
                            'conf': float(det[4]),
                            'cls': float(det[5])
                        })
                    debug_frames.append({'frame': frame_idx, 'detections': sample})
                    print(f"[DEBUG] frame {frame_idx}: {num} drawn detections (raw {len(dets) if dets else 0})")

            frame_vis = draw_predictions(frame, draw_dets, class_names)
            out.write(frame_vis)
            frames_written += 1
            pbar.update(1)
            frame_idx += 1
            
        if tracker_running:
            end = time.time()
            ret = tracker.stop()
            tracker_running = False
            duration = None
            if prediction_start_time:
                duration = end - prediction_start_time
                prediction_total_duration += duration
            val = float(ret) if ret is not None else None
            if val is not None:
                prediction_emissions_total += val
            prediction_records.append({
                'start': datetime.fromtimestamp(prediction_start_time).isoformat() if prediction_start_time else None,
                'end': datetime.fromtimestamp(end).isoformat(),
                'duration_s': duration,
                'emissions_kg': val
            })
            prediction_start_time = None
            
    cap.release()
    # Release all video writers
    if out is not None:
        out.release()
    if out_rgb is not None:
        out_rgb.release()
    if out_encoded is not None:
        out_encoded.release()
    pbar.close()

    # Ensure any trackers still marked as running are stopped cleanly
    try:
        if 'tracker_infer_running' in locals() and tracker_infer_running:
            end_time = time.time()
            ret = tracker_infer.stop()
            tracker_infer_running = False
            duration = None
            if tracker_infer_start_time:
                duration = end_time - tracker_infer_start_time
                infer_total_duration += duration
            val = float(ret) if ret is not None else None
            if val is not None:
                infer_emissions_total += val
            infer_records.append({
                'start': datetime.fromtimestamp(tracker_infer_start_time).isoformat() if tracker_infer_start_time else None,
                'end': datetime.fromtimestamp(end_time).isoformat(),
                'duration_s': duration,
                'emissions_kg': val
            })
            tracker_infer_start_time = None
        if 'tracker_running' in locals() and tracker_running:
            end = time.time()
            ret = tracker.stop()
            tracker_running = False
            duration = None
            if prediction_start_time:
                duration = end - prediction_start_time
                prediction_total_duration += duration
            val = float(ret) if ret is not None else None
            if val is not None:
                prediction_emissions_total += val
            prediction_records.append({
                'start': datetime.fromtimestamp(prediction_start_time).isoformat() if prediction_start_time else None,
                'end': datetime.fromtimestamp(end).isoformat(),
                'duration_s': duration,
                'emissions_kg': val
            })
            prediction_start_time = None
    except Exception as e:
        print(f"Warning: error while stopping trackers: {e}")
    
    # Print output information
    if wrapper.is_spike:
        base, ext = os.path.splitext(args.output)
        output_rgb = f"{base}_rgb{ext}"
        output_encoded = f"{base}_encoded{ext}"
        print(f"RGB output saved to {output_rgb} ({frames_written} frames written)")
        print(f"Encoded output saved to {output_encoded} ({frames_written} frames written)")
    else:
        print(f"Output saved to {args.output} ({frames_written} frames written)")
    
    # Print energy file locations
    if wrapper.encoding_type:
        print(f"Inference energy results saved to {os.path.join(energy_dir, energy_file_infer)}")
    else:
        print(f"Energy results saved to {os.path.join(energy_dir, energy_file)}")

    # Write per-video emissions summary JSON
    # Note: For spike models, encoding time/emissions are NO LONGER tracked - only inference
    # Compute average emissions per minute for each phase when duration available
    def mean_per_min(emissions, duration_s):
        if duration_s and duration_s > 0:
            return float(emissions / (duration_s / 60.0))
        return None

    total_duration = encode_total_duration + infer_total_duration + prediction_total_duration
    
    # Calculate per-frame metrics for fair comparison
    # frames_written = number of unique frames processed
    # For spike models: only inference emissions are counted (encoding is done upfront without tracking)
    total_emissions = infer_emissions_total + prediction_emissions_total
    
    emissions_per_frame = float(total_emissions / frames_written) if frames_written > 0 else 0.0
    inference_emissions_per_frame = float((infer_emissions_total + prediction_emissions_total) / frames_written) if frames_written > 0 else 0.0
    
    # Energy consumed in kWh (approximate from emissions, assuming ~0.267 kg CO2/kWh for Spain grid)
    # More accurate: use energy_consumed from codecarbon if available
    # For now, calculate as emissions / carbon_intensity
    energy_per_frame_kwh = emissions_per_frame / 0.267 if emissions_per_frame > 0 else 0.0
    
    # For encoded models: no adjustment needed now since we're only measuring inference
    # The time_steps adjustment is no longer relevant as encoding is not measured
    is_encoded = wrapper.encoding_type is not None
    
    summary = {
        'video': video_basename,
        'model': args.model,
        'encoding': wrapper.encoding_type,
        'temporal_mode': args.temporal_mode if wrapper.encoding_type else None,
        'frames_processed': int(frames_written),
        'encode_duration_s': float(encode_total_duration) if is_encoded else 0.0,  # Duration tracked but emissions not measured
        'inference_emissions_kg': float(infer_emissions_total + prediction_emissions_total),
        'total_emissions_kg': float(total_emissions),
        'inference_duration_s': float(infer_total_duration + prediction_total_duration),
        'total_duration_s': float(total_duration),
        'emissions_per_frame_kg': emissions_per_frame,
        'inference_emissions_per_frame_kg': inference_emissions_per_frame,
        'energy_per_frame_kwh': energy_per_frame_kwh,
        'inference_mean_kg_per_min': mean_per_min(infer_emissions_total + prediction_emissions_total, infer_total_duration + prediction_total_duration),
        'inference_records': infer_records if is_encoded else [],
        'prediction_records': prediction_records if not is_encoded else []
    }
    summary_path = os.path.join(energy_dir, f"{video_basename}_summary.json")
    try:
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Summary JSON saved to {summary_path}")
    except Exception as e:
        print(f"Warning: could not write summary JSON: {e}")

    # Save debug JSON if requested
    if args.debug:
        debug_path = os.path.join(energy_dir, f"{video_basename}_debug.json")
        try:
            with open(debug_path, 'w') as f:
                json.dump({
                    'video': video_basename,
                    'model': args.model,
                    'frames_with_detections': frames_with_detections,
                    'frames_without_detections': frames_without_detections,
                    'samples_saved': len(debug_frames),
                    'samples': debug_frames
                }, f, indent=2)
            print(f"Debug JSON saved to {debug_path}")
            print(f"Frames with detections: {frames_with_detections}, without detections: {frames_without_detections}")
        except Exception as e:
            print(f"Warning: could not write debug JSON: {e}")

if __name__ == '__main__':
    main()
