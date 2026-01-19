"""
Test all models on all videos using simplified metrics.
Generates a consolidated CSV with ONLY:
- Model name
- Encoding/temporal mode
- Frames processed  
- Inference time
- Energy consumed (kWh)
- Energy per frame

Usage:
    python tests/test_all_videos_simple.py
    python tests/test_all_videos_simple.py --models SpikeYOLO YOLO
"""

import argparse
import subprocess
from pathlib import Path
import csv
import json

from model_utils import DEFAULT_WEIGHTS, project_root

VIDEO_DIR = Path(project_root) / "datasets" / "video_tests"
OUTPUT_BASE = Path(project_root) / "Results_test_video"


def get_available_videos():
    """Get list of video files."""
    if not VIDEO_DIR.exists():
        raise FileNotFoundError(f"Video directory not found: {VIDEO_DIR}")
    
    videos = list(VIDEO_DIR.glob("*.mp4")) + list(VIDEO_DIR.glob("*.avi"))
    return sorted([v.name for v in videos])


def test_model_on_video(model_name, video_path, weights_path, temporal_mode='repeat'):
    """Test a single model on a single video."""
    cmd = [
        "python", "tests/test_video_simple.py",
        "--model", model_name,
        "--weights", weights_path,
        "--video", str(video_path),
        "--temporal-mode", temporal_mode
    ]
    
    print(f"\nTesting {model_name} on {video_path.name}...")
    
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        return False


def consolidate_results(models):
    """Consolidate all results into a single CSV."""
    all_results = []
    
    for model in models:
        model_dir = OUTPUT_BASE / model
        if not model_dir.exists():
            continue
        
        # Read all JSON results for this model
        for json_file in model_dir.glob("*_results.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    all_results.append(data)
            except Exception as e:
                print(f"Warning: Could not read {json_file}: {e}")
    
    if not all_results:
        print("No results to consolidate")
        return
    
    # Write consolidated CSV
    csv_path = OUTPUT_BASE / "summary_simple.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            'model',
            'encoding',
            'temporal_mode',
            'video',
            'frames_processed',
            'inference_time_s',
            'total_energy_kwh',
            'emissions_kg_co2',
            'energy_per_frame_kwh',
            'time_per_frame_s',
            'fps'
        ])
        
        # Sort by model, then video
        all_results.sort(key=lambda x: (x['model'], x['video']))
        
        # Data rows
        for r in all_results:
            writer.writerow([
                r['model'],
                r.get('encoding', 'rgb'),
                r.get('temporal_mode', ''),
                r['video'],
                r['frames_processed'],
                r['inference_time_s'],
                r['inference_energy_kwh'],
                r['emissions_kg_co2'],
                r['energy_per_frame_kwh'],
                r['time_per_frame_s'],
                r['fps']
            ])
    
    print(f"\n{'='*80}")
    print(f"Consolidated results saved to: {csv_path}")
    print(f"Total results: {len(all_results)}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Test all models on all videos (simplified)")
    parser.add_argument('--models', nargs='+', choices=list(DEFAULT_WEIGHTS.keys()),
                        help='Models to test (default: all)')
    parser.add_argument('--videos', nargs='+', help='Videos to test (default: all)')
    parser.add_argument('--temporal-mode', type=str, choices=['repeat', 'sliding', 'batched'], default='repeat',
                        help='Temporal mode for encoded models: repeat (same frame), sliding (overlapping windows), batched (non-overlapping windows)')
    
    args = parser.parse_args()
    
    # Get models to test
    models = args.models if args.models else list(DEFAULT_WEIGHTS.keys())
    
    # Get videos to test
    available_videos = get_available_videos()
    if args.videos:
        videos = [v for v in args.videos if v in [av for av in available_videos]]
    else:
        videos = [v for v in available_videos]
    
    print(f"\n{'='*80}")
    print(f"SIMPLIFIED VIDEO TESTING")
    print(f"{'='*80}")
    print(f"Models: {', '.join(models)}")
    print(f"Videos: {', '.join(videos)}")
    print(f"{'='*80}\n")
    
    # Test all combinations
    total = len(models) * len(videos)
    completed = 0
    
    for model in models:
        weights = DEFAULT_WEIGHTS[model]
        
        for video_name in videos:
            video_path = VIDEO_DIR / video_name
            
            success = test_model_on_video(model, video_path, weights, temporal_mode=args.temporal_mode)
            completed += 1
            
            print(f"\nProgress: {completed}/{total} completed")
    
    # Consolidate results
    print(f"\n{'='*80}")
    print("Consolidating results...")
    consolidate_results(models)


if __name__ == '__main__':
    main()
