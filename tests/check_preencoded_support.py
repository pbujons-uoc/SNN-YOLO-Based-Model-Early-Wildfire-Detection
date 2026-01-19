#!/usr/bin/env python3
"""
Script to verify that test_video.py has the --use-preencoded argument.
Run this to check if your files are up to date.
"""

import sys
import argparse
from pathlib import Path

# Try to import the test_video module and check if it has the argument
def check_test_video_args():
    """Check if test_video.py has the --use-preencoded argument."""
    try:
        # Read the test_video.py file
        test_video_path = Path(__file__).parent / "test_video.py"
        
        if not test_video_path.exists():
            print(f"❌ ERROR: test_video.py not found at {test_video_path}")
            return False
        
        with open(test_video_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for the --use-preencoded argument
        if "--use-preencoded" in content:
            print("✅ SUCCESS: test_video.py has the --use-preencoded argument")
            
            # Check if the function is also there
            if "def load_preencoded_video" in content:
                print("✅ SUCCESS: load_preencoded_video function is present")
            else:
                print("❌ ERROR: load_preencoded_video function is missing")
                return False
            
            # Check if Path is imported
            if "from pathlib import Path" in content:
                print("✅ SUCCESS: Path import is present")
            else:
                print("❌ ERROR: Path import is missing")
                return False
            
            return True
        else:
            print("❌ ERROR: test_video.py does NOT have the --use-preencoded argument")
            print("\nYour test_video.py file is outdated!")
            print("Please update it with the latest version.")
            return False
            
    except Exception as e:
        print(f"❌ ERROR checking file: {e}")
        return False


def check_test_all_videos_args():
    """Check if test_all_videos.py has the --use-preencoded argument."""
    try:
        test_all_path = Path(__file__).parent / "test_all_videos.py"
        
        if not test_all_path.exists():
            print(f"❌ ERROR: test_all_videos.py not found at {test_all_path}")
            return False
        
        with open(test_all_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if "--use-preencoded" in content and "use_preencoded=args.use_preencoded" in content:
            print("✅ SUCCESS: test_all_videos.py has the --use-preencoded argument and passes it correctly")
            return True
        else:
            print("❌ ERROR: test_all_videos.py is missing --use-preencoded support")
            return False
            
    except Exception as e:
        print(f"❌ ERROR checking file: {e}")
        return False


def check_pre_encode_script():
    """Check if pre_encode_videos.py exists and has correct API."""
    try:
        pre_encode_path = Path(__file__).parent / "pre_encode_videos.py"
        
        if not pre_encode_path.exists():
            print(f"❌ ERROR: pre_encode_videos.py not found at {pre_encode_path}")
            return False
        
        with open(pre_encode_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for correct API usage
        errors = []
        
        # Check that it doesn't use wrong kwargs
        if "model_name=model_name" in content:
            errors.append("Uses incorrect 'model_name=model_name' (should be positional)")
        
        if "weights_path=weights_path" in content:
            errors.append("Uses incorrect 'weights_path=weights_path' (should be positional)")
        
        if "encoding_type=encoding_type" in content:
            errors.append("Uses incorrect 'encoding_type=' (should be 'encoding=')")
        
        if "device='cuda'" in content and "ModelWrapper(" in content:
            errors.append("Uses unsupported 'device=' parameter")
        
        if errors:
            print("❌ ERROR: pre_encode_videos.py has incorrect ModelWrapper API usage:")
            for error in errors:
                print(f"   - {error}")
            print("\n   Correct usage should be:")
            print("   wrapper = ModelWrapper(model_name, weights_path, time_steps=..., encoding=...)")
            return False
        
        print(f"✅ SUCCESS: pre_encode_videos.py exists and uses correct API at {pre_encode_path}")
        return True
            
    except Exception as e:
        print(f"❌ ERROR checking file: {e}")
        return False


if __name__ == "__main__":
    print("="*80)
    print("CHECKING VIDEO TEST FILES FOR PRE-ENCODED SUPPORT")
    print("="*80)
    print()
    
    results = []
    
    print("1. Checking test_video.py...")
    results.append(check_test_video_args())
    print()
    
    print("2. Checking test_all_videos.py...")
    results.append(check_test_all_videos_args())
    print()
    
    print("3. Checking pre_encode_videos.py...")
    results.append(check_pre_encode_script())
    print()
    
    print("="*80)
    if all(results):
        print("✅ ALL CHECKS PASSED - Your files are up to date!")
        print()
        print("You can now use:")
        print("  python tests/pre_encode_videos.py")
        print("  python tests/test_all_videos.py --use-preencoded")
    else:
        print("❌ SOME CHECKS FAILED - Please update your files!")
        print()
        print("If running on a remote server, make sure to:")
        print("  1. Copy the updated files to the server")
        print("  2. Or pull the latest changes from git")
    print("="*80)
    
    sys.exit(0 if all(results) else 1)
