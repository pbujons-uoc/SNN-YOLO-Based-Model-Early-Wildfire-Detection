#!/bin/bash
# Bash script to run all video tests for all models
# Updated for inference-only emission tracking (January 2026)
#
# This script tests:
# - 6 models: YOLO, SpikeYOLO, VanillaCNN, SpikeYOLO_latency, SpikeYOLO_poisson
# - 3 videos: big_fire1.mp4, big_fire2.mp4, small_smoke_fires.mp4
# - 2 temporal modes for spike encoded models: repeat, sliding
#
# Total tests: 9 (RGB models) + 12 (spike encoded models) = 21 tests

set +e  # Continue on errors

videos=("big_fire1.mp4" "big_fire2.mp4" "small_smoke_fires.mp4")
video_dir="datasets/video_tests"

echo "========================================"
echo "Starting All Video Tests"
echo "========================================"
echo ""

test_num=0
total_tests=21

# ============================================================================
# RGB MODELS (No encoding, single temporal mode)
# ============================================================================

echo "==== RGB MODELS ===="
echo ""

## YOLO (Standard RGB)
for video in "${videos[@]}"; do
    test_num=$((test_num + 1))
    echo "[$test_num/$total_tests] Testing YOLO with $video"
    python tests/test_video.py \
        --model YOLO \
        --video "$video_dir/$video" \
        --imgsz 640 \
        --conf 0.25
    echo ""
done

## SpikeYOLO (RGB - no encoding)
for video in "${videos[@]}"; do
    test_num=$((test_num + 1))
    echo "[$test_num/$total_tests] Testing SpikeYOLO with $video"
    python tests/test_video.py \
        --model SpikeYOLO \
        --video "$video_dir/$video" \
        --imgsz 640 \
        --conf 0.25
    echo ""
done

## VanillaCNN (Custom RGB)
for video in "${videos[@]}"; do
    test_num=$((test_num + 1))
    echo "[$test_num/$total_tests] Testing VanillaCNN with $video"
    python tests/test_video.py \
        --model VanillaCNN \
        --video "$video_dir/$video" \
        --imgsz 640 \
        --conf 0.2
    echo ""
done

# ============================================================================
# SPIKE MODELS - LATENCY ENCODING
# ============================================================================

echo "==== SPIKE MODELS - LATENCY ENCODING ===="
echo ""

## SpikeYOLO_latency - REPEAT mode
for video in "${videos[@]}"; do
    test_num=$((test_num + 1))
    echo "[$test_num/$total_tests] Testing SpikeYOLO_latency (REPEAT) with $video"
    python tests/test_video.py \
        --model SpikeYOLO_latency \
        --video "$video_dir/$video" \
        --encoding latency \
        --time-steps 4 \
        --temporal-mode repeat \
        --imgsz 640 \
        --conf 0.12
    echo ""
done

## SpikeYOLO_latency - SLIDING mode
for video in "${videos[@]}"; do
    test_num=$((test_num + 1))
    echo "[$test_num/$total_tests] Testing SpikeYOLO_latency (SLIDING) with $video"
    python tests/test_video.py \
        --model SpikeYOLO_latency \
        --video "$video_dir/$video" \
        --encoding latency \
        --time-steps 4 \
        --temporal-mode sliding \
        --imgsz 640 \
        --conf 0.12
    echo ""
done

# ============================================================================
# SPIKE MODELS - POISSON ENCODING
# ============================================================================

echo "==== SPIKE MODELS - POISSON ENCODING ===="
echo ""

## SpikeYOLO_poisson - REPEAT mode
for video in "${videos[@]}"; do
    test_num=$((test_num + 1))
    echo "[$test_num/$total_tests] Testing SpikeYOLO_poisson (REPEAT) with $video"
    python tests/test_video.py \
        --model SpikeYOLO_poisson \
        --video "$video_dir/$video" \
        --encoding poisson \
        --time-steps 4 \
        --temporal-mode repeat \
        --imgsz 640 \
        --conf 0.12
    echo ""
done

## SpikeYOLO_poisson - SLIDING mode
for video in "${videos[@]}"; do
    test_num=$((test_num + 1))
    echo "[$test_num/$total_tests] Testing SpikeYOLO_poisson (SLIDING) with $video"
    python tests/test_video.py \
        --model SpikeYOLO_poisson \
        --video "$video_dir/$video" \
        --encoding poisson \
        --time-steps 4 \
        --temporal-mode sliding \
        --imgsz 640 \
        --conf 0.12
    echo ""
done

# ============================================================================
# GENERATE SUMMARY REPORTS
# ============================================================================

echo "========================================"
echo "Generating Summary Reports"
echo "========================================"
echo ""

# Generate summary for repeat mode
echo "Generating emissions summary for REPEAT mode..."
python tests/generate_emissions_summary.py \
    --input Results_test_video \
    --output emissions_summary_repeat.csv \
    --temporal-mode repeat
echo ""

# Generate summary for sliding mode
echo "Generating emissions summary for SLIDING mode..."
python tests/generate_emissions_summary.py \
    --input Results_test_video \
    --output emissions_summary_sliding.csv \
    --temporal-mode sliding
echo ""

# Generate summary for all modes combined
echo "Generating emissions summary for ALL modes..."
python tests/generate_emissions_summary.py \
    --input Results_test_video \
    --output emissions_summary_all.csv
echo ""

echo "========================================"
echo "All Tests Completed!"
echo "========================================"
echo ""
echo "Results saved to:"
echo "  - Results_test_video/<model>/<encoding>_<timesteps>/"
echo ""
echo "Summary CSVs:"
echo "  - emissions_summary_repeat.csv"
echo "  - emissions_summary_sliding.csv"
echo "  - emissions_summary_all.csv"
echo ""
