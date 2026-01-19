"""
Debug script to check if energy files are being created during video testing.
Run this AFTER running test_video.py to diagnose missing energy files.
"""

import os
import json
from pathlib import Path
import argparse

def check_energy_files(base_dir="Results_test_video"):
    """Check for energy files in the results directory."""
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"❌ Results directory not found: {base_path}")
        return False
    
    print(f"Checking for energy files in: {base_path}")
    print("="*80)
    
    # Find all energy_results directories
    energy_dirs = list(base_path.rglob("energy_results"))
    
    if not energy_dirs:
        print("❌ No energy_results directories found!")
        print("\nThis could mean:")
        print("1. The test hasn't been run yet")
        print("2. The output directory is different")
        print("3. There was an error during execution")
        return False
    
    print(f"Found {len(energy_dirs)} energy_results directories:\n")
    
    total_files = 0
    for energy_dir in energy_dirs:
        print(f"📁 {energy_dir.relative_to(base_path)}/")
        
        # List all files in this directory
        files = list(energy_dir.glob("*"))
        if not files:
            print("   ⚠️  EMPTY - No energy files found!")
        else:
            for f in files:
                total_files += 1
                size_kb = f.stat().st_size / 1024
                print(f"   ✅ {f.name} ({size_kb:.1f} KB)")
                
                # If it's a summary JSON, show some details
                if f.suffix == '.json' and 'summary' in f.name:
                    try:
                        with open(f, 'r') as jf:
                            data = json.load(jf)
                            print(f"      → Frames: {data.get('frames_processed', 'N/A')}")
                            print(f"      → Inference emissions: {data.get('inference_emissions_kg', 'N/A'):.6f} kg")
                            print(f"      → Duration: {data.get('inference_duration_s', 'N/A'):.2f} s")
                    except Exception as e:
                        print(f"      ⚠️  Could not read JSON: {e}")
        print()
    
    print("="*80)
    print(f"Summary: Found {total_files} energy files in {len(energy_dirs)} directories")
    
    if total_files == 0:
        print("\n⚠️  NO ENERGY FILES FOUND!")
        print("\nPossible causes:")
        print("1. EmissionsTracker failed to initialize")
        print("2. Script crashed before saving files")
        print("3. Wrong output directory specified")
        print("4. Permissions issue preventing file creation")
        print("\nTry running test_video.py with --debug flag for more info")
        return False
    
    return True


def check_specific_model(model_name, video_name, temporal_mode=None, base_dir="Results_test_video"):
    """Check energy files for a specific model and video."""
    base_path = Path(base_dir) / model_name
    
    if temporal_mode:
        base_path = base_path / temporal_mode
    
    if not base_path.exists():
        print(f"❌ Model directory not found: {base_path}")
        return False
    
    energy_dir = base_path / "energy_results"
    if not energy_dir.exists():
        print(f"❌ Energy results directory not found: {energy_dir}")
        return False
    
    video_stem = Path(video_name).stem
    summary_file = energy_dir / f"{video_stem}_summary.json"
    
    print(f"Checking: {model_name} / {video_name}" + (f" / {temporal_mode}" if temporal_mode else ""))
    print("="*80)
    
    if summary_file.exists():
        print(f"✅ Summary file found: {summary_file}")
        try:
            with open(summary_file, 'r') as f:
                data = json.load(f)
            
            print("\nSummary contents:")
            print(f"  Video: {data.get('video')}")
            print(f"  Model: {data.get('model')}")
            print(f"  Encoding: {data.get('encoding', 'rgb')}")
            print(f"  Temporal mode: {data.get('temporal_mode', 'N/A')}")
            print(f"  Frames processed: {data.get('frames_processed')}")
            print(f"  Encode duration: {data.get('encode_duration_s', 0):.2f} s")
            print(f"  Inference duration: {data.get('inference_duration_s', 0):.2f} s")
            print(f"  Inference emissions: {data.get('inference_emissions_kg', 0):.6f} kg CO2")
            print(f"  Emissions per frame: {data.get('inference_emissions_per_frame_kg', 0):.9f} kg CO2")
            return True
        except Exception as e:
            print(f"⚠️  Could not read summary file: {e}")
            return False
    else:
        print(f"❌ Summary file NOT found: {summary_file}")
        
        # Check if directory has any files
        files = list(energy_dir.glob(f"{video_stem}*"))
        if files:
            print(f"\nBut found {len(files)} related files:")
            for f in files:
                print(f"  - {f.name}")
        else:
            print(f"\nNo files found for video: {video_stem}")
        
        return False


def main():
    parser = argparse.ArgumentParser(description="Check for energy output files")
    parser.add_argument('--base-dir', type=str, default='Results_test_video',
                        help='Base results directory')
    parser.add_argument('--model', type=str, default=None,
                        help='Check specific model')
    parser.add_argument('--video', type=str, default=None,
                        help='Check specific video')
    parser.add_argument('--mode', type=str, default=None,
                        choices=['repeat', 'sliding'],
                        help='Temporal mode for encoded models')
    
    args = parser.parse_args()
    
    if args.model and args.video:
        # Check specific model/video
        success = check_specific_model(args.model, args.video, args.mode, args.base_dir)
    else:
        # Check all energy files
        success = check_energy_files(args.base_dir)
    
    if not success:
        print("\n" + "="*80)
        print("DEBUGGING TIPS:")
        print("="*80)
        print("1. Check if test_video.py completed successfully")
        print("2. Look for error messages in the terminal output")
        print("3. Verify EmissionsTracker is working:")
        print("   python -c \"from codecarbon import EmissionsTracker; print('OK')\"")
        print("4. Run with --debug flag:")
        print("   python tests/test_video.py --model YOLO --video test.mp4 --debug")
        print("5. Check disk space and permissions")
        print("6. Verify output directory exists and is writable")


if __name__ == "__main__":
    main()
