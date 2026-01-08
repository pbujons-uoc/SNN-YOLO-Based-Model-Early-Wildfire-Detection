"""
Aggregate test results from multiple test runs.

Reads all test results from Results_test_small/ and computes average metrics per model.
Outputs: Results_test_small/average_results.csv
"""

import csv
from pathlib import Path
from collections import defaultdict
import statistics

# Model names
MODELS = ['YOLO', 'SpikeYOLO', 'SpikeYOLO_latency', 'SpikeYOLO_poisson', 'VanillaCNN']

def read_metrics_csv(csv_path):
    """Read results_test_images.csv and return dict of metrics."""
    metrics = {}
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            if len(row) >= 2:
                metrics[row[0]] = float(row[1])
    return metrics

def read_energy_csv(csv_path):
    """Read energy_results.csv and return dict of metrics."""
    energy = {}
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            if len(row) >= 5:
                phase = row[0]
                energy[f'{phase}_total_time_s'] = float(row[1])
                energy[f'{phase}_total_energy_kg_co2'] = float(row[2])
                energy[f'{phase}_avg_time_per_image_s'] = float(row[3])
                energy[f'{phase}_avg_energy_per_image_kg_co2'] = float(row[4])
    return energy

def main():
    base_dir = Path('Results_test_small')
    
    if not base_dir.exists():
        print(f"Error: {base_dir} not found")
        return
    
    # Find all test result directories
    test_dirs = sorted([d for d in base_dir.glob('Results_test_images_*') if d.is_dir()])
    
    if not test_dirs:
        print(f"No test directories found in {base_dir}")
        return
    
    print(f"Found {len(test_dirs)} test runs")
    
    # Collect metrics per model across all test runs
    model_metrics = defaultdict(lambda: defaultdict(list))
    # Store all individual test runs for finding maximum
    model_test_runs = defaultdict(lambda: defaultdict(list))
    
    for test_dir in test_dirs:
        print(f"Processing {test_dir.name}...")
        
        small_test_dir = test_dir / 'small_test'
        if not small_test_dir.exists():
            print(f"  Warning: {small_test_dir} not found, skipping")
            continue
        
        for model in MODELS:
            model_dir = small_test_dir / model
            if not model_dir.exists():
                print(f"  Warning: {model} not found in {test_dir.name}, skipping")
                continue
            
            # Read metrics CSV
            metrics_csv = model_dir / 'results_test_images.csv'
            if metrics_csv.exists():
                metrics = read_metrics_csv(metrics_csv)
                
                # Store individual run
                run_data = {'test_run': test_dir.name}
                run_data.update(metrics)
                
                for key, value in metrics.items():
                    model_metrics[model][key].append(value)
            else:
                print(f"  Warning: {metrics_csv} not found")
                run_data = {'test_run': test_dir.name}
            
            # Read energy CSV
            energy_csv = model_dir / 'energy_results.csv'
            if energy_csv.exists():
                energy = read_energy_csv(energy_csv)
                run_data.update(energy)
                
                for key, value in energy.items():
                    model_metrics[model][key].append(value)
            else:
                print(f"  Warning: {energy_csv} not found")
            
            # Store complete run data
            model_test_runs[model]['runs'].append(run_data)
    
    # Compute averages
    print("\nComputing averages...")
    
    average_results = {}
    for model in MODELS:
        if model not in model_metrics:
            print(f"  Warning: No data for {model}")
            continue
        
        average_results[model] = {}
        for metric, values in model_metrics[model].items():
            if values:
                average_results[model][metric] = statistics.mean(values)
                # Also store std dev for reference
                if len(values) > 1:
                    average_results[model][f'{metric}_std'] = statistics.stdev(values)
    
    # Save aggregated results
    output_csv = base_dir / 'average_results.csv'
    
    # Collect all unique metrics across all models
    all_metrics = set()
    for model_data in average_results.values():
        all_metrics.update(model_data.keys())
    
    # Sort metrics for consistent ordering
    metrics_order = [
        'confidence_threshold',
        'precision',
        'recall',
        'F1',
        'TP',
        'FP',
        'FN',
        'num_images',
        'inference_total_time_s',
        'inference_total_energy_kg_co2',
        'inference_avg_time_per_image_s',
        'inference_avg_energy_per_image_kg_co2',
        'inference_adjusted_div_4_total_time_s',
        'inference_adjusted_div_4_total_energy_kg_co2',
        'inference_adjusted_div_4_avg_time_per_image_s',
        'inference_adjusted_div_4_avg_energy_per_image_kg_co2',
    ]
    
    # Add any metrics not in the predefined order
    remaining_metrics = sorted([m for m in all_metrics if m not in metrics_order and not m.endswith('_std')])
    metrics_order.extend(remaining_metrics)
    
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow(['model'] + metrics_order)
        
        # Data rows
        for model in MODELS:
            if model not in average_results:
                continue
            
            row = [model]
            for metric in metrics_order:
                value = average_results[model].get(metric, '')
                if value != '':
                    # Format numbers
                    if isinstance(value, float):
                        if value < 0.001:
                            row.append(f'{value:.6e}')
                        else:
                            row.append(f'{value:.6f}')
                    else:
                        row.append(value)
                else:
                    row.append('')
            
            writer.writerow(row)
    
    print(f"\nAverage results saved to: {output_csv}")
    
    # Find maximum precision results
    print("\nFinding maximum precision results...")
    
    maximum_results = {}
    for model in MODELS:
        if model not in model_test_runs or not model_test_runs[model]['runs']:
            print(f"  Warning: No data for {model}")
            continue
        
        # Find run with maximum precision
        runs = model_test_runs[model]['runs']
        max_run = max(runs, key=lambda x: x.get('precision', -1))
        maximum_results[model] = max_run
    
    # Save maximum precision results
    output_max_csv = base_dir / 'maximum.csv'
    
    with open(output_max_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header (include test_run column)
        writer.writerow(['model', 'test_run'] + metrics_order)
        
        # Data rows
        for model in MODELS:
            if model not in maximum_results:
                continue
            
            run_data = maximum_results[model]
            row = [model, run_data.get('test_run', '')]
            
            for metric in metrics_order:
                value = run_data.get(metric, '')
                if value != '':
                    # Format numbers
                    if isinstance(value, float):
                        if value < 0.001:
                            row.append(f'{value:.6e}')
                        else:
                            row.append(f'{value:.6f}')
                    else:
                        row.append(value)
                else:
                    row.append('')
            
            writer.writerow(row)
    
    print(f"Maximum precision results saved to: {output_max_csv}")
    
    # Print summary
    print("\n" + "="*80)
    print("AVERAGE RESULTS SUMMARY")
    print("="*80)
    for model in MODELS:
        if model not in average_results:
            continue
        
        data = average_results[model]
        print(f"\n{model}:")
        print(f"  Precision:    {data.get('precision', 'N/A'):.4f}")
        print(f"  Recall:       {data.get('recall', 'N/A'):.4f}")
        print(f"  F1 Score:     {data.get('F1', 'N/A'):.4f}")
        print(f"  Avg Time/img: {data.get('inference_avg_time_per_image_s', 'N/A'):.6f} s")
        print(f"  Avg Energy:   {data.get('inference_avg_energy_per_image_kg_co2', 'N/A'):.6e} kg CO2")
    
    print("\n" + "="*80)
    print("MAXIMUM PRECISION RESULTS SUMMARY")
    print("="*80)
    for model in MODELS:
        if model not in maximum_results:
            continue
        
        data = maximum_results[model]
        print(f"\n{model} (from {data.get('test_run', 'N/A')}):")
        print(f"  Precision:    {data.get('precision', 'N/A'):.4f}")
        print(f"  Recall:       {data.get('recall', 'N/A'):.4f}")
        print(f"  F1 Score:     {data.get('F1', 'N/A'):.4f}")
        print(f"  Avg Time/img: {data.get('inference_avg_time_per_image_s', 'N/A'):.6f} s")
        print(f"  Avg Energy:   {data.get('inference_avg_energy_per_image_kg_co2', 'N/A'):.6e} kg CO2")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
