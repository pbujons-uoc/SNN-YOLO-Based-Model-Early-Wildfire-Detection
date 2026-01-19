# PowerShell script to run all video tests for all models
# Updated for inference-only emission tracking (January 2026)
#
# This script tests:
# - 6 models: YOLO, SpikeYOLO, VanillaCNN, SpikeYOLO_latency, SpikeYOLO_poisson
# - 3 videos: big_fire1.mp4, big_fire2.mp4, small_smoke_fires.mp4
# - 2 temporal modes for spike encoded models: repeat, sliding
#
# Total tests: 9 (RGB models) + 12 (spike encoded models) = 21 tests

$ErrorActionPreference = "Continue"
$videos = @("big_fire1.mp4", "big_fire2.mp4", "small_smoke_fires.mp4")
$video_dir = "datasets/video_tests"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Starting All Video Tests" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Test counter
$test_num = 0
$total_tests = 21

# ============================================================================
# RGB MODELS (No encoding, single temporal mode)
# ============================================================================

Write-Host "==== RGB MODELS ====" -ForegroundColor Yellow
Write-Host ""

## YOLO (Standard RGB)
foreach ($video in $videos) {
    $test_num++
    Write-Host "[$test_num/$total_tests] Testing YOLO with $video" -ForegroundColor Green
    python tests/test_video.py `
        --model YOLO `
        --video "$video_dir/$video" `
        --imgsz 640 `
        --conf 0.25
    Write-Host ""
}

## SpikeYOLO (RGB - no encoding)
foreach ($video in $videos) {
    $test_num++
    Write-Host "[$test_num/$total_tests] Testing SpikeYOLO with $video" -ForegroundColor Green
    python tests/test_video.py `
        --model SpikeYOLO `
        --video "$video_dir/$video" `
        --imgsz 640 `
        --conf 0.25
    Write-Host ""
}

## VanillaCNN (Custom RGB)
foreach ($video in $videos) {
    $test_num++
    Write-Host "[$test_num/$total_tests] Testing VanillaCNN with $video" -ForegroundColor Green
    python tests/test_video.py `
        --model VanillaCNN `
        --video "$video_dir/$video" `
        --imgsz 640 `
        --conf 0.2
    Write-Host ""
}

# ============================================================================
# SPIKE MODELS - LATENCY ENCODING
# ============================================================================

Write-Host "==== SPIKE MODELS - LATENCY ENCODING ====" -ForegroundColor Yellow
Write-Host ""

## SpikeYOLO_latency - REPEAT mode
foreach ($video in $videos) {
    $test_num++
    Write-Host "[$test_num/$total_tests] Testing SpikeYOLO_latency (REPEAT) with $video" -ForegroundColor Green
    python tests/test_video.py `
        --model SpikeYOLO_latency `
        --video "$video_dir/$video" `
        --encoding latency `
        --time-steps 4 `
        --temporal-mode repeat `
        --imgsz 640 `
        --conf 0.12
    Write-Host ""
}

## SpikeYOLO_latency - SLIDING mode
foreach ($video in $videos) {
    $test_num++
    Write-Host "[$test_num/$total_tests] Testing SpikeYOLO_latency (SLIDING) with $video" -ForegroundColor Green
    python tests/test_video.py `
        --model SpikeYOLO_latency `
        --video "$video_dir/$video" `
        --encoding latency `
        --time-steps 4 `
        --temporal-mode sliding `
        --imgsz 640 `
        --conf 0.12
    Write-Host ""
}

# ============================================================================
# SPIKE MODELS - POISSON ENCODING
# ============================================================================

Write-Host "==== SPIKE MODELS - POISSON ENCODING ====" -ForegroundColor Yellow
Write-Host ""

## SpikeYOLO_poisson - REPEAT mode
foreach ($video in $videos) {
    $test_num++
    Write-Host "[$test_num/$total_tests] Testing SpikeYOLO_poisson (REPEAT) with $video" -ForegroundColor Green
    python tests/test_video.py `
        --model SpikeYOLO_poisson `
        --video "$video_dir/$video" `
        --encoding poisson `
        --time-steps 4 `
        --temporal-mode repeat `
        --imgsz 640 `
        --conf 0.12
    Write-Host ""
}

## SpikeYOLO_poisson - SLIDING mode
foreach ($video in $videos) {
    $test_num++
    Write-Host "[$test_num/$total_tests] Testing SpikeYOLO_poisson (SLIDING) with $video" -ForegroundColor Green
    python tests/test_video.py `
        --model SpikeYOLO_poisson `
        --video "$video_dir/$video" `
        --encoding poisson `
        --time-steps 4 `
        --temporal-mode sliding `
        --imgsz 640 `
        --conf 0.12
    Write-Host ""
}

# ============================================================================
# GENERATE SUMMARY REPORTS
# ============================================================================

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Generating Summary Reports" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Generate summary for repeat mode
Write-Host "Generating emissions summary for REPEAT mode..." -ForegroundColor Green
python tests/generate_emissions_summary.py `
    --input Results_test_video `
    --output emissions_summary_repeat.csv `
    --temporal-mode repeat
Write-Host ""

# Generate summary for sliding mode
Write-Host "Generating emissions summary for SLIDING mode..." -ForegroundColor Green
python tests/generate_emissions_summary.py `
    --input Results_test_video `
    --output emissions_summary_sliding.csv `
    --temporal-mode sliding
Write-Host ""

# Generate summary for all modes combined
Write-Host "Generating emissions summary for ALL modes..." -ForegroundColor Green
python tests/generate_emissions_summary.py `
    --input Results_test_video `
    --output emissions_summary_all.csv
Write-Host ""

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "All Tests Completed!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Results saved to:" -ForegroundColor Yellow
Write-Host "  - Results_test_video/<model>/<encoding>_<timesteps>/" -ForegroundColor White
Write-Host ""
Write-Host "Summary CSVs:" -ForegroundColor Yellow
Write-Host "  - emissions_summary_repeat.csv" -ForegroundColor White
Write-Host "  - emissions_summary_sliding.csv" -ForegroundColor White
Write-Host "  - emissions_summary_all.csv" -ForegroundColor White
Write-Host ""
