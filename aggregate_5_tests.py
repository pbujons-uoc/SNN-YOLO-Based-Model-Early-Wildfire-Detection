"""
Aggregate results from 5 test runs in Result_tests_images/Results_test_small.

Generates two CSV files:
1. average_metrics_5_tests.csv: Average of each metric across 5 tests per model
2. best_result_5_tests.csv: Best value of each metric across 5 tests per model
   (Note: For FP and FN, lowest is best)
"""

import pandas as pd
from pathlib import Path
import numpy as np


def load_all_test_results(base_path):
    """
    Load all results_test_images.csv files from small_test1 to small_test5.
    
    Returns:
        dict: {model_name: [df1, df2, df3, df4, df5]}
    """
    base_path = Path(base_path)
    models_data = {}
    
    # Iterate through small_test1 to small_test5
    for test_idx in range(1, 6):
        test_folder = base_path / f"small_test{test_idx}"
        
        if not test_folder.exists():
            print(f"Warning: {test_folder} does not exist, skipping")
            continue
        
        # Find all model folders
        for model_folder in test_folder.iterdir():
            if not model_folder.is_dir():
                continue
            
            model_name = model_folder.name
            csv_path = model_folder / "results_test_images.csv"
            
            if not csv_path.exists():
                print(f"Warning: {csv_path} does not exist, skipping")
                continue
            
            # Read CSV
            df = pd.read_csv(csv_path)
            
            # Initialize model entry if not exists
            if model_name not in models_data:
                models_data[model_name] = []
            
            models_data[model_name].append(df)
    
    return models_data


def compute_average_metrics(models_data):
    """
    Compute average of each metric across all test runs per model.
    
    Returns:
        pd.DataFrame with columns: Model, metric1_avg, metric2_avg, ...
    """
    results = []
    
    for model_name, dfs in models_data.items():
        if len(dfs) == 0:
            continue
        
        row = {'Model': model_name}
        
        # Get all unique metrics (excluding num_images)
        all_metrics = set()
        for df in dfs:
            metrics = df[df['metric'] != 'num_images']['metric'].tolist()
            all_metrics.update(metrics)
        
        # Calculate average for each metric
        for metric in sorted(all_metrics):
            values = []
            for df in dfs:
                metric_row = df[df['metric'] == metric]
                if not metric_row.empty:
                    values.append(float(metric_row['value'].iloc[0]))
            
            if values:
                row[f'{metric}_avg'] = np.mean(values)
                row[f'{metric}_std'] = np.std(values)
        
        results.append(row)
    
    return pd.DataFrame(results)


def compute_best_metrics(models_data):
    """
    Compute best value of each metric across all test runs per model.
    For FP and FN, lowest is best. For others, highest is best.
    
    Returns:
        pd.DataFrame with columns: Model, metric1_best, metric2_best, ...
    """
    results = []
    
    # Metrics where lower is better
    minimize_metrics = ['FP', 'FN']
    
    for model_name, dfs in models_data.items():
        if len(dfs) == 0:
            continue
        
        row = {'Model': model_name}
        
        # Get all unique metrics (excluding num_images)
        all_metrics = set()
        for df in dfs:
            metrics = df[df['metric'] != 'num_images']['metric'].tolist()
            all_metrics.update(metrics)
        
        # Calculate best for each metric
        for metric in sorted(all_metrics):
            values = []
            for df in dfs:
                metric_row = df[df['metric'] == metric]
                if not metric_row.empty:
                    values.append(float(metric_row['value'].iloc[0]))
            
            if values:
                if metric in minimize_metrics:
                    # Lower is better (minimum)
                    row[f'{metric}_best'] = np.min(values)
                else:
                    # Higher is better (maximum)
                    row[f'{metric}_best'] = np.max(values)
        
        results.append(row)
    
    return pd.DataFrame(results)


def main():
    """Main function to aggregate test results."""
    
    base_path = Path("Result_tests_images/Results_test_small")
    
    if not base_path.exists():
        print(f"Error: Base path does not exist: {base_path}")
        return
    
    print("Loading test results from 5 test runs...")
    models_data = load_all_test_results(base_path)
    
    if not models_data:
        print("Error: No test results found")
        return
    
    print(f"\nFound results for {len(models_data)} models:")
    for model_name, dfs in models_data.items():
        print(f"  {model_name:25s} - {len(dfs)} test runs")
    
    # Compute average metrics
    print("\nComputing average metrics...")
    avg_df = compute_average_metrics(models_data)
    
    # Compute best metrics
    print("Computing best metrics...")
    best_df = compute_best_metrics(models_data)
    
    # Save results
    avg_output = "average_metrics_5_tests.csv"
    best_output = "best_result_5_tests.csv"
    
    avg_df.to_csv(avg_output, index=False)
    best_df.to_csv(best_output, index=False)
    
    print(f"\n{'='*80}")
    print("Results saved:")
    print(f"  Average metrics: {avg_output}")
    print(f"  Best metrics:    {best_output}")
    print(f"{'='*80}")
    
    # Display summary
    print("\n" + "="*80)
    print("AVERAGE METRICS SUMMARY")
    print("="*80)
    
    # Display key metrics
    key_metrics = ['precision_avg', 'recall_avg', 'F1_avg', 'mAP50_avg', 'mAP50_95_avg', 'mean_IoU_avg']
    display_cols = ['Model'] + [col for col in key_metrics if col in avg_df.columns]
    
    if display_cols:
        print(avg_df[display_cols].to_string(index=False))
    
    print("\n" + "="*80)
    print("BEST METRICS SUMMARY")
    print("="*80)
    
    # Display key metrics
    key_metrics_best = ['precision_best', 'recall_best', 'F1_best', 'mAP50_best', 'mAP50_95_best', 'mean_IoU_best']
    display_cols_best = ['Model'] + [col for col in key_metrics_best if col in best_df.columns]
    
    if display_cols_best:
        print(best_df[display_cols_best].to_string(index=False))
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
