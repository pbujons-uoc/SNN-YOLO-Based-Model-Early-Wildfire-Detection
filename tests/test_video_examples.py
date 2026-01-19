"""
Example script to run video tests with the updated emission tracking.
This script demonstrates how to test spike models with only inference emission tracking.

The encoding phase is done upfront without emission tracking,
then inference is measured separately.
"""

# Example 1: Test SpikeYOLO_latency with repeat mode
# python tests/test_video.py --model SpikeYOLO_latency --video datasets/video_tests/big_fire1.mp4 --encoding latency --time-steps 4 --temporal-mode repeat --imgsz 640 --conf 0.12

# Example 2: Test SpikeYOLO_latency with sliding mode
# python tests/test_video.py --model SpikeYOLO_latency --video datasets/video_tests/big_fire2.mp4 --encoding latency --time-steps 4 --temporal-mode sliding --imgsz 640 --conf 0.12

# Example 3: Test SpikeYOLO_poisson with repeat mode
# python tests/test_video.py --model SpikeYOLO_poisson --video datasets/video_tests/small_smoke_fires.mp4 --encoding poisson --time-steps 4 --temporal-mode repeat --imgsz 640 --conf 0.12

# Example 4: Test standard YOLO (RGB, no encoding)
# python tests/test_video.py --model YOLO --video datasets/video_tests/big_fire1.mp4 --imgsz 640 --conf 0.25

# Example 5: Generate emissions summary CSV from all JSON results
# python tests/generate_emissions_summary.py --input Results_test_video --output emissions_summary_repeat.csv --temporal-mode repeat

print("""
Updated test_video.py behavior:
================================

For Spike Models (SpikeYOLO_latency, SpikeYOLO_poisson):
---------------------------------------------------------
1. Phase 1: Encodes all video frames (encoding time tracked, emissions NOT tracked)
2. Phase 2: Runs inference on encoded frames (emissions and time tracked)

This separates encoding from inference and ensures accurate inference-only emission measurements.

Both temporal modes supported:
- repeat: Each frame encoded with T timesteps (same frame repeated)
- sliding: Single-step encoding per frame, inference uses sliding window of T frames

For RGB Models (YOLO, VanillaCNN):
-----------------------------------
- Standard single-pass processing with emission tracking

Outputs:
--------
- <video>_rgb.mp4: Original frames with detections
- <video>_encoded.mp4: Reconstructed encoded frames with detections (spike models only)
- <video>_inference.csv: CodeCarbon emission data (inference only for spike models)
- <video>_summary.json: Complete metrics summary

Generate consolidated CSV:
---------------------------
python tests/generate_emissions_summary.py --input Results_test_video --output emissions_summary.csv
""")
