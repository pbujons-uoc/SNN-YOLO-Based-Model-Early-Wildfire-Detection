"""Debug helper: sample raw detections from a model on the first N frames of a video.

Usage examples:
python debug/check_model_detections.py \
  --model SpikeYOLO_latency \
  --weights Results/SpikeYOLO_latency/weights/best.pt \
  --video datasets/video_tests/big_fire2.mp4 \
  --frames 10 --conf 0.0001 --time-steps 4 --out debug/big_fire2_detections.json

This script prints a short summary to stdout and writes a JSON file with per-frame raw detections
for both direct predict() and (for spiking models) a repeated T-step predict_tensor() call.
"""
import os
import argparse
import json
import cv2
import torch
from model_utils import ModelWrapper, project_root


def serialize_dets(dets):
    if dets is None:
        return []
    return [[float(x) for x in d] for d in dets]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--video', type=str, required=True)
    parser.add_argument('--frames', type=int, default=10, help='Number of frames to sample')
    parser.add_argument('--conf', type=float, default=0.0001, help='Confidence threshold (very permissive by default)')
    parser.add_argument('--time-steps', type=int, default=4, help='Time steps to use when constructing a stacked tensor for spiking models')
    parser.add_argument('--out', type=str, default=None, help='Path to save debug JSON (optional)')
    args = parser.parse_args()

    # Resolve default output path: prefer project_root/debug/<video>_detections.json if not specified.
    video_basename = os.path.splitext(os.path.basename(args.video))[0]
    if args.out is None:
        args.out = os.path.join(project_root, 'debug', f"{video_basename}_detections.json")
    else:
        # If user passed a relative path that begins with 'debug/', make it relative to project_root
        if not os.path.isabs(args.out) and args.out.replace('\\', '/').startswith('debug/'):
            args.out = os.path.join(project_root, args.out)

    print(f"Loading model {args.model} (weights={args.weights})")
    # ModelWrapper signature: ModelWrapper(model_type, weights_path, time_steps=4, device='cuda', encoding=None)
    wrapper = ModelWrapper(args.model, args.weights, time_steps=args.time_steps)
    cls = wrapper.get_class_names()
    print("Class names:", cls)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    results = {'video': args.video, 'model': args.model, 'frames': []}

    for i in range(args.frames):
        ret, frame = cap.read()
        if not ret:
            print(f"Reached end of video at frame {i}")
            break

        h, w = frame.shape[:2]
        # 1) direct predict (if available)
        try:
            dets_direct = wrapper.predict(frame, conf_thres=args.conf)
        except Exception as e:
            dets_direct = None
            print(f"Frame {i}: wrapper.predict() raised: {e}")

        # 2) for spiking models, try single-step encode + repeat T steps -> predict_tensor
        dets_tensor = None
        if getattr(wrapper, 'encoding_type', None):
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            enc_single = wrapper.encode_frame_single(img_rgb)
            if enc_single is None:
                print(f"Frame {i}: encode_frame_single() returned None")
            else:
                # Build repeated stack [T, C, H, W]
                try:
                    stacked = torch.stack([enc_single for _ in range(args.time_steps)], dim=0)
                    stacked = stacked.unsqueeze(0).to(wrapper.device)  # [1,T,C,H,W] - BATCH-FIRST
                    dets_tensor = wrapper.predict_tensor(stacked, conf_thres=args.conf, original_shape=(h, w))
                except Exception as e:
                    dets_tensor = None
                    print(f"Frame {i}: predict_tensor() raised: {e}")

        print(f"Frame {i}: direct_count={len(dets_direct) if dets_direct else 0}, tensor_count={len(dets_tensor) if dets_tensor else 0}")
        if dets_direct:
            print("  direct sample:", serialize_dets(dets_direct)[:3])
        if dets_tensor:
            print("  tensor sample:", serialize_dets(dets_tensor)[:3])

        results['frames'].append({
            'frame_idx': i,
            'direct': serialize_dets(dets_direct),
            'tensor': serialize_dets(dets_tensor)
        })

    cap.release()

    if args.out:
        import os as _os
        _out_dir = _os.path.dirname(args.out)
        if _out_dir:
            _os.makedirs(_out_dir, exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved debug JSON to {args.out}")
    else:
        print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
