"""
Test all models on all videos in datasets/video_tests.
Outputs to Results_test_video/{model}/ with:
- Predicted video with bounding boxes
- Encoded video visualization (for encoded models)
- Emissions CSV with encoding/inference separation

Usage:
    python tests/test_all_videos.py
    python tests/test_all_videos.py --models SpikeYOLO YOLO
    python tests/test_all_videos.py --videos big_fire1.mp4
"""

import argparse
import subprocess
import os
from pathlib import Path
import csv
import json

from model_utils import DEFAULT_WEIGHTS, project_root

# Confidence thresholds per model (from test_images_comprehensive.py)
ADAPTIVE_CONFIDENCE_THRESHOLDS = {
    'SpikeYOLO': 0.25,
    'SpikeYOLO_latency': 0.15,
    'SpikeYOLO_poisson': 0.15,
    'YOLO': 0.25,
    'VanillaCNN': 0.30,
}

VIDEO_DIR = Path(project_root) / "datasets" / "video_tests"
OUTPUT_BASE = Path(project_root) / "Results_test_video"


def get_available_videos():
    """Get list of video files in datasets/video_tests."""
    if not VIDEO_DIR.exists():
        raise FileNotFoundError(f"Video directory not found: {VIDEO_DIR}")
    
    videos = list(VIDEO_DIR.glob("*.mp4")) + list(VIDEO_DIR.glob("*.avi"))
    return sorted([v.name for v in videos])


def test_model_on_video(model_name, video_name, weights_path, time_steps=4, temporal_mode='repeat', use_preencoded=False):
    """Test a single model on a single video."""
    video_path = VIDEO_DIR / video_name
    
    # For encoded models, organize by temporal mode
    is_encoded = model_name in ['SpikeYOLO_latency', 'SpikeYOLO_poisson']
    if is_encoded:
        model_output_dir = OUTPUT_BASE / model_name / temporal_mode
    else:
        model_output_dir = OUTPUT_BASE / model_name
    
    model_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get confidence threshold
    conf_threshold = ADAPTIVE_CONFIDENCE_THRESHOLDS.get(model_name, 0.25)
    
    # Determine encoding type
    encoding_type = 'latency' if 'latency' in model_name else 'poisson' if 'poisson' in model_name else None
    
    # Output video path
    video_stem = Path(video_name).stem
    output_video = model_output_dir / f"{video_stem}_predicted.mp4"
    
    # Build command
    cmd = [
        "python", "tests/test_video.py",
        "--model", model_name,
        "--weights", weights_path,
        "--video", str(video_path),
        "--output", str(output_video),
        "--conf", str(conf_threshold),
        "--time-steps", str(time_steps),
        "--temporal-mode", temporal_mode,
    ]
    
    if encoding_type:
        cmd.extend(["--encoding", encoding_type])
    
    if use_preencoded:
        cmd.append("--use-preencoded")
    
    print(f"\n{'='*80}")
    print(f"Testing: {model_name} on {video_name} (mode: {temporal_mode})")
    if use_preencoded:
        print(f"Using PRE-ENCODED videos (only measuring inference time)")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        
        # Move energy results to model folder
        # test_video.py saves to Results_test/{model}/{encoding}_{time-steps}/energy_results/
        if is_encoded:
            energy_source = Path("Results_test") / model_name / f"{encoding_type}_{time_steps}" / "energy_results"
        else:
            energy_source = Path("Results_test") / model_name / "rgb" / "energy_results"
        
        if energy_source.exists():
            # Copy energy results to Results_test_video/{model}/{mode}/
            import shutil
            energy_dest = model_output_dir / "energy_results"
            energy_dest.mkdir(exist_ok=True)
            
            for file in energy_source.glob(f"{video_stem}*"):
                shutil.copy(file, energy_dest / file.name)
                print(f"Copied energy results: {file.name} -> {energy_dest}")
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error testing {model_name} on {video_name} (mode: {temporal_mode}): {e}")
        return False


def consolidate_emissions(model_name, temporal_mode=None):
    """Consolidate all video emissions into a single CSV per model (or per model/mode)."""
    if temporal_mode:
        model_dir = OUTPUT_BASE / model_name / temporal_mode
        csv_name = f"emissions_summary_{temporal_mode}.csv"
    else:
        model_dir = OUTPUT_BASE / model_name
        csv_name = "emissions_summary.csv"
    
    energy_dir = model_dir / "energy_results"
    
    if not energy_dir.exists():
        return
    
    # Gather all summary JSONs
    summaries = []
    for json_file in energy_dir.glob("*_summary.json"):
        try:
            with open(json_file, 'r') as f:
                summary = json.load(f)
                summaries.append(summary)
        except Exception as e:
            print(f"Warning: could not read {json_file}: {e}")
    
    if not summaries:
        return
    
    # Write consolidated CSV
    csv_path = model_dir / csv_name
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header with per-frame metrics and adjusted metrics
        writer.writerow([
            'video',
            'model',
            'encoding',
            'temporal_mode',
            'frames_processed',
            'encode_emissions_kg',
            'inference_emissions_kg',
            'total_emissions_kg',
            'encode_duration_s',
            'inference_duration_s',
            'total_duration_s',
            'emissions_per_frame_kg',
            'encode_emissions_per_frame_kg',
            'inference_emissions_per_frame_kg',
            'energy_per_frame_kwh',
            'adjusted_inference_emissions_kg',
            'adjusted_total_emissions_kg',
            'adjusted_emissions_per_frame_kg',
            'adjusted_inference_per_frame_kg',
            'adjusted_energy_per_frame_kwh',
            'encode_kg_per_min',
            'inference_kg_per_min',
            'total_kg_per_min'
        ])
        
        # Data rows
        for s in summaries:
            total_duration = s.get('encode_total_duration_s', 0) + s.get('inference_total_duration_s', 0) + s.get('prediction_total_duration_s', 0)
            writer.writerow([
                s.get('video', ''),
                s.get('model', ''),
                s.get('encoding', 'rgb'),
                s.get('temporal_mode', 'repeat'),
                s.get('frames_processed', 0),
                s.get('encode_emissions_kg', 0),
                s.get('inference_emissions_kg', 0) + s.get('prediction_emissions_kg', 0),
                s.get('total_emissions_kg', 0),
                s.get('encode_total_duration_s', 0),
                s.get('inference_total_duration_s', 0) + s.get('prediction_total_duration_s', 0),
                total_duration,
                s.get('emissions_per_frame_kg', 0),
                s.get('encode_emissions_per_frame_kg', 0),
                s.get('inference_emissions_per_frame_kg', 0),
                s.get('energy_per_frame_kwh', 0),
                s.get('adjusted_inference_emissions_kg', 0),
                s.get('adjusted_total_emissions_kg', 0),
                s.get('adjusted_emissions_per_frame_kg', 0),
                s.get('adjusted_inference_per_frame_kg', 0),
                s.get('adjusted_energy_per_frame_kwh', 0),
                s.get('encode_mean_kg_per_min', 0),
                s.get('inference_mean_kg_per_min', 0),
                s.get('total_mean_kg_per_min', 0)
            ])
    
    print(f"Consolidated emissions saved: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Test all models on video dataset")
    parser.add_argument('--models', type=str, nargs='+', default=None,
                        help='Models to test (default: all available)')
    parser.add_argument('--videos', type=str, nargs='+', default=None,
                        help='Videos to test (default: all in datasets/video_tests)')
    parser.add_argument('--time-steps', type=int, default=4,
                        help='Time steps for spike encoding')
    parser.add_argument('--use-preencoded', action='store_true',
                        help='Use pre-encoded videos (only measure inference time, not encoding time)')
    
    args = parser.parse_args()
    
    # Get videos to test
    available_videos = get_available_videos()
    if args.videos:
        videos_to_test = [v for v in args.videos if v in available_videos]
        if len(videos_to_test) < len(args.videos):
            print(f"Warning: Some videos not found in {VIDEO_DIR}")
    else:
        videos_to_test = available_videos
    
    print(f"\nVideos to test: {', '.join(videos_to_test)}")
    
    # Get models to test
    if args.models:
        models_to_test = [m for m in args.models if m in DEFAULT_WEIGHTS]
        if len(models_to_test) < len(args.models):
            print(f"Warning: Some models not recognized")
    else:
        models_to_test = list(DEFAULT_WEIGHTS.keys())
    
    print(f"Models to test: {', '.join(models_to_test)}\n")
    
    # Create output directory
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    
    # Test each model on each video
    results = {}
    for model_name in models_to_test:
        if model_name not in DEFAULT_WEIGHTS:
            print(f"Warning: Unknown model '{model_name}', skipping")
            continue
        
        weights_path = DEFAULT_WEIGHTS[model_name]
        if not Path(weights_path).exists():
            print(f"Warning: Weights not found for {model_name} at {weights_path}, skipping")
            continue
        
        # Determine if encoded model (needs both temporal modes)
        is_encoded = model_name in ['SpikeYOLO_latency', 'SpikeYOLO_poisson']
        
        if is_encoded:
            # Test with both repeat and sliding modes
            temporal_modes = ['repeat', 'sliding']
        else:
            # RGB models only use repeat (no temporal dimension)
            temporal_modes = ['repeat']
        
        results[model_name] = {}
        
        for temporal_mode in temporal_modes:
            if is_encoded and temporal_mode not in results[model_name]:
                results[model_name][temporal_mode] = {}
            
            for video_name in videos_to_test:
                success = test_model_on_video(
                    model_name=model_name,
                    video_name=video_name,
                    weights_path=weights_path,
                    time_steps=args.time_steps,
                    temporal_mode=temporal_mode,
                    use_preencoded=args.use_preencoded
                )
                
                if is_encoded:
                    results[model_name][temporal_mode][video_name] = success
                else:
                    # For RGB models, store directly without mode nesting
                    results[model_name][video_name] = success
            
            # Consolidate emissions for this model/mode
            if is_encoded:
                consolidate_emissions(model_name, temporal_mode)
        
        # For RGB models, consolidate without mode
        if not is_encoded:
            consolidate_emissions(model_name)
    
    # Final summary
    print("\n" + "="*80)
    print("VIDEO TESTING COMPLETED")
    print("="*80)
    
    for model_name, model_results in results.items():
        print(f"\n{model_name}:")
        
        # Check if encoded model (has temporal mode nesting)
        if isinstance(next(iter(model_results.values()), None), dict):
            # Encoded model: iterate through modes then videos
            for temporal_mode, video_results in model_results.items():
                print(f"  Mode: {temporal_mode}")
                for video_name, success in video_results.items():
                    status = "Success" if success else "Failed"
                    print(f"    {video_name:30s} {status}")
        else:
            # RGB model: iterate directly through videos
            for video_name, success in model_results.items():
                status = "Success" if success else "Failed"
                print(f"  {video_name:30s} {status}")
    
    print(f"\nResults saved in: {OUTPUT_BASE}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
