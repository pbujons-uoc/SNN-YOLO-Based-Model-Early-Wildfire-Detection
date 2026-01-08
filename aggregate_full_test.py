"""
Aggregate energy and metrics results from full_test folder.

Creates two CSV files:
1. full_test_energy_results.csv: Energy consumption per model
2. full_test_metrics_results.csv: Test metrics per model
"""

import pandas as pd
from pathlib import Path


def aggregate_energy_results(base_path):
    """
    Aggregate energy_results.csv from all model folders.
    
    Returns:
        pd.DataFrame with Model column added
    """
    base_path = Path(base_path)
    all_data = []
    
    for model_folder in sorted(base_path.iterdir()):
        if not model_folder.is_dir():
            continue
        
        model_name = model_folder.name
        energy_file = model_folder / "energy_results.csv"
        
        if not energy_file.exists():
            print(f"Warning: {energy_file} not found")
            continue
        
        df = pd.read_csv(energy_file)
        df.insert(0, 'Model', model_name)
        all_data.append(df)
    
    if not all_data:
        return None
    
    return pd.concat(all_data, ignore_index=True)


def aggregate_metrics_results(base_path):
    """
    Aggregate results_test_images.csv from all model folders.
    Converts from long format (metric, value) to wide format with Model column.
    
    Returns:
        pd.DataFrame with Model column and one column per metric
    """
    base_path = Path(base_path)
    all_data = []
    
    for model_folder in sorted(base_path.iterdir()):
        if not model_folder.is_dir():
            continue
        
        model_name = model_folder.name
        metrics_file = model_folder / "results_test_images.csv"
        
        if not metrics_file.exists():
            print(f"Warning: {metrics_file} not found")
            continue
        
        df = pd.read_csv(metrics_file)
        
        # Convert from long to wide format
        row = {'Model': model_name}
        for _, metric_row in df.iterrows():
            metric = metric_row['metric']
            value = metric_row['value']
            row[metric] = value
        
        all_data.append(row)
    
    if not all_data:
        return None
    
    return pd.DataFrame(all_data)


def main():
    """Main function to aggregate full test results."""
    
    base_path = Path("Result_tests_images/full_test")
    
    if not base_path.exists():
        print(f"Error: Base path does not exist: {base_path}")
        return
    
    print("Aggregating energy results...")
    energy_df = aggregate_energy_results(base_path)
    
    if energy_df is not None:
        output_file = "full_test_energy_results.csv"
        energy_df.to_csv(output_file, index=False)
        print(f"  Saved: {output_file}")
        print(f"  Models: {len(energy_df)}")
    
    print("\nAggregating metrics results...")
    metrics_df = aggregate_metrics_results(base_path)
    
    if metrics_df is not None:
        output_file = "full_test_metrics_results.csv"
        metrics_df.to_csv(output_file, index=False)
        print(f"  Saved: {output_file}")
        print(f"  Models: {len(metrics_df)}")
    
    print("\n" + "="*80)
    print("FULL TEST RESULTS AGGREGATED")
    print("="*80)
    
    if energy_df is not None:
        print("\nEnergy Results:")
        print(energy_df.to_string(index=False))
    
    if metrics_df is not None:
        print("\nMetrics Results:")
        # Display key columns
        key_cols = ['Model', 'precision', 'recall', 'F1', 'mAP50', 'mAP50_95', 'AP50_smoke', 'AP50_fire', 'AP50_95_smoke', 'AP50_95_fire', 'mean_IoU']
        display_cols = [col for col in key_cols if col in metrics_df.columns]
        if display_cols:
            print(metrics_df[display_cols].to_string(index=False))
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
