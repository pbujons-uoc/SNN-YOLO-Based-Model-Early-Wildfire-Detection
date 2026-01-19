"""
Generate emissions summary CSV from individual video JSON summaries.
This script processes the *_summary.json files and creates a consolidated CSV
with per-frame emissions metrics.

Usage:
    python generate_emissions_summary.py --input Results_test_video --output emissions_summary.csv
"""

import argparse
import os
import json
import csv
from pathlib import Path


def find_summary_jsons(base_dir):
    """Find all *_summary.json files in the directory tree."""
    summary_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('_summary.json'):
                summary_files.append(os.path.join(root, file))
    return summary_files


def load_summary(json_path):
    """Load and parse a summary JSON file."""
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load {json_path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Generate emissions summary CSV from JSON files")
    parser.add_argument('--input', type=str, default='Results_test_video',
                        help='Base directory to search for summary JSONs (default: Results_test_video)')
    parser.add_argument('--output', type=str, default='emissions_summary.csv',
                        help='Output CSV file path (default: emissions_summary.csv)')
    parser.add_argument('--temporal-mode', type=str, default=None,
                        help='Filter by temporal mode (repeat/sliding). If not set, includes all.')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input directory {args.input} not found.")
        return
    
    print(f"Searching for summary JSON files in {args.input}...")
    json_files = find_summary_jsons(args.input)
    print(f"Found {len(json_files)} summary files.")
    
    if len(json_files) == 0:
        print("No summary files found. Exiting.")
        return
    
    # Load all summaries
    summaries = []
    for json_file in json_files:
        summary = load_summary(json_file)
        if summary:
            # Filter by temporal mode if specified
            if args.temporal_mode:
                if summary.get('temporal_mode') != args.temporal_mode:
                    continue
            summaries.append(summary)
    
    print(f"Loaded {len(summaries)} valid summaries.")
    
    if len(summaries) == 0:
        print("No valid summaries to process. Exiting.")
        return
    
    # Define CSV headers (simplified - only inference metrics now)
    headers = [
        'video',
        'model',
        'encoding',
        'temporal_mode',
        'frames_processed',
        'inference_emissions_kg',
        'total_emissions_kg',
        'encode_duration_s',
        'inference_duration_s',
        'total_duration_s',
        'emissions_per_frame_kg',
        'inference_emissions_per_frame_kg',
        'energy_per_frame_kwh',
        'inference_kg_per_min',
    ]
    
    # Write CSV
    with open(args.output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        
        for summary in summaries:
            row = {
                'video': summary.get('video', ''),
                'model': summary.get('model', ''),
                'encoding': summary.get('encoding', 'rgb'),
                'temporal_mode': summary.get('temporal_mode', 'N/A'),
                'frames_processed': summary.get('frames_processed', 0),
                'inference_emissions_kg': summary.get('inference_emissions_kg', 0.0),
                'total_emissions_kg': summary.get('total_emissions_kg', 0.0),
                'encode_duration_s': summary.get('encode_duration_s', 0.0),
                'inference_duration_s': summary.get('inference_duration_s', 0.0),
                'total_duration_s': summary.get('total_duration_s', 0.0),
                'emissions_per_frame_kg': summary.get('emissions_per_frame_kg', 0.0),
                'inference_emissions_per_frame_kg': summary.get('inference_emissions_per_frame_kg', 0.0),
                'energy_per_frame_kwh': summary.get('energy_per_frame_kwh', 0.0),
                'inference_kg_per_min': summary.get('inference_mean_kg_per_min', 0.0),
            }
            writer.writerow(row)
    
    print(f"CSV written to {args.output}")
    print(f"Total rows: {len(summaries)}")


if __name__ == '__main__':
    main()
