"""
Quick test to verify that pre-encoded video testing works correctly.
This script does a minimal test with a small portion of a video.
"""

import subprocess
import sys
from pathlib import Path
import argparse

def run_command(cmd, description):
    """Run a command and report success/failure."""
    print(f"\n{'='*80}")
    print(f"TEST: {description}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        print(f"✅ {description} - PASSED")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - FAILED")
        print(f"Error code: {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False
    except Exception as e:
        print(f"❌ {description} - FAILED with exception")
        print(f"Exception: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Quick test of pre-encoded video functionality")
    parser.add_argument('--video', type=str, default=None,
                        help='Path to test video (default: first video in datasets/video_tests)')
    parser.add_argument('--skip-encoding', action='store_true',
                        help='Skip pre-encoding step (assumes videos are already encoded)')
    
    args = parser.parse_args()
    
    # Find a test video
    if args.video:
        test_video = Path(args.video)
    else:
        video_dir = Path("datasets") / "video_tests"
        videos = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.avi"))
        if not videos:
            print(f"❌ No videos found in {video_dir}")
            return False
        test_video = videos[0]
    
    if not test_video.exists():
        print(f"❌ Test video not found: {test_video}")
        return False
    
    print("="*80)
    print("QUICK TEST: PRE-ENCODED VIDEO FUNCTIONALITY")
    print("="*80)
    print(f"Test video: {test_video}")
    print()
    
    results = []
    
    # Test 1: Check if files are updated
    results.append(run_command(
        ["python", "tests/check_preencoded_support.py"],
        "1. Verify files are updated"
    ))
    
    if not results[-1]:
        print("\n❌ Files are not updated. Please update test_video.py and related files.")
        return False
    
    # Test 2: Pre-encode a single video (latency, repeat mode, 4 steps)
    if not args.skip_encoding:
        results.append(run_command(
            ["python", "tests/pre_encode_videos.py", 
             "--videos", test_video.name,
             "--encodings", "latency",
             "--time-steps", "4"],
            "2. Pre-encode test video (latency, T=4)"
        ))
        
        if not results[-1]:
            print("\n⚠️  Pre-encoding failed. Subsequent tests will fail.")
    else:
        print("\n⏭️  Skipping pre-encoding step (--skip-encoding)")
    
    # Test 3: Run test_video.py with --use-preencoded
    results.append(run_command(
        ["python", "tests/test_video.py",
         "--model", "SpikeYOLO_latency",
         "--video", str(test_video),
         "--encoding", "latency",
         "--time-steps", "4",
         "--temporal-mode", "repeat",
         "--use-preencoded"],
        "3. Test video with pre-encoded (repeat mode)"
    ))
    
    # Test 4: Check if energy files were created
    results.append(run_command(
        ["python", "tests/check_energy_outputs.py",
         "--model", "SpikeYOLO_latency",
         "--video", test_video.name,
         "--mode", "repeat"],
        "4. Verify energy output files were created"
    ))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(results)
    total = len(results)
    
    for i, result in enumerate(results, 1):
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"Test {i}: {status}")
    
    print()
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Pre-encoded video functionality is working correctly.")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
