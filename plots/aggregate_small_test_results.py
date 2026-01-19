"""
Aggregate results from Results_image_small folder.
Calculates average and maximum values for each model across all test runs.
"""

import pandas as pd
import os
from pathlib import Path
import numpy as np

def aggregate_results():
    """Aggregate results from all small test runs."""
    
    base_path = Path("Results_image_small")
    test_folders = sorted([f for f in base_path.iterdir() if f.is_dir()])
    
    models = ["SpikeYOLO", "SpikeYOLO_latency", "SpikeYOLO_poisson", "VanillaCNN", "YOLO"]
    
    # Dictionaries to store all values
    metrics_data = {model: [] for model in models}
    energy_data = {model: [] for model in models}
    
    print(f"Found {len(test_folders)} test folders: {[f.name for f in test_folders]}\n")
    
    # Collect all data
    for test_folder in test_folders:
        print(f"Processing {test_folder.name}...")
        
        for model in models:
            model_path = test_folder / model
            
            if not model_path.exists():
                print(f"  {model}: Not found, skipping")
                continue
            
            # Read metrics
            metrics_file = model_path / "results_test_images.csv"
            if metrics_file.exists():
                df_metrics = pd.read_csv(metrics_file)
                metrics_dict = dict(zip(df_metrics['metric'], df_metrics['value']))
                metrics_dict['test_folder'] = test_folder.name
                metrics_data[model].append(metrics_dict)
            else:
                print(f"  {model}: metrics file not found")
            
            # Read energy
            energy_file = model_path / "energy_results.csv"
            if energy_file.exists():
                df_energy = pd.read_csv(energy_file)
                energy_dict = df_energy.iloc[0].to_dict()
                energy_dict['test_folder'] = test_folder.name
                energy_data[model].append(energy_dict)
            else:
                print(f"  {model}: energy file not found")
    
    print("\n" + "="*80)
    print("AGGREGATED RESULTS")
    print("="*80)
    
    # Calculate and display aggregated metrics
    print("\n" + "-"*80)
    print("METRICS - AVERAGE AND BEST VALUES")
    print("-"*80)
    
    metrics_summary = []
    
    for model in models:
        if not metrics_data[model]:
            print(f"\n{model}: No data available")
            continue
        
        df = pd.DataFrame(metrics_data[model])
        
        print(f"\n{model}:")
        print(f"  Number of runs: {len(df)}")
        
        # Select numeric columns
        numeric_cols = ['precision', 'recall', 'F1', 'TP', 'FP', 'FN']
        available_cols = [col for col in numeric_cols if col in df.columns]
        
        if not available_cols:
            print("  No numeric metrics found")
            continue
        
        summary_row = {'Model': model}
        
        # Metrics where lower is better
        minimize_metrics = ['FP', 'FN']
        
        for col in available_cols:
            avg_val = df[col].mean()
            std_val = df[col].std()
            
            # For FP and FN, we want minimum (lower is better)
            # For others, we want maximum (higher is better)
            if col in minimize_metrics:
                best_val = df[col].min()
                print(f"  {col}:")
                print(f"    Average: {avg_val:.4f}")
                print(f"    Best (min): {best_val:.4f}")
                print(f"    Std Dev: {std_val:.4f}")
            else:
                best_val = df[col].max()
                print(f"  {col}:")
                print(f"    Average: {avg_val:.4f}")
                print(f"    Best (max): {best_val:.4f}")
                print(f"    Std Dev: {std_val:.4f}")
            
            summary_row[f'{col}_avg'] = avg_val
            summary_row[f'{col}_best'] = best_val
            summary_row[f'{col}_std'] = std_val
        
        metrics_summary.append(summary_row)
    
    # Calculate and display aggregated energy
    print("\n" + "-"*80)
    print("ENERGY CONSUMPTION - AVERAGE AND BEST VALUES (MINIMUM)")
    print("-"*80)
    
    energy_summary = []
    
    for model in models:
        if not energy_data[model]:
            print(f"\n{model}: No data available")
            continue
        
        df = pd.DataFrame(energy_data[model])
        
        print(f"\n{model}:")
        print(f"  Number of runs: {len(df)}")
        
        # Select numeric columns
        numeric_cols = ['total_time_s', 'total_energy_kg_co2', 
                       'avg_time_per_image_s', 'avg_energy_per_image_kg_co2']
        available_cols = [col for col in numeric_cols if col in df.columns]
        
        if not available_cols:
            print("  No energy metrics found")
            continue
        
        summary_row = {'Model': model}
        
        # For energy, lower is always better (minimum)
        for col in available_cols:
            avg_val = df[col].mean()
            best_val = df[col].min()  # Minimum is best for energy/time
            std_val = df[col].std()
            
            if 'kg_co2' in col:
                print(f"  {col}:")
                print(f"    Average: {avg_val:.2e} kg CO2")
                print(f"    Best (min): {best_val:.2e} kg CO2")
                print(f"    Std Dev: {std_val:.2e} kg CO2")
            else:
                print(f"  {col}:")
                print(f"    Average: {avg_val:.6f} s")
                print(f"    Best (min): {best_val:.6f} s")
                print(f"    Std Dev: {std_val:.6f} s")
            
            summary_row[f'{col}_avg'] = avg_val
            summary_row[f'{col}_best'] = best_val
            summary_row[f'{col}_std'] = std_val
        
        energy_summary.append(summary_row)
    
    # Save summary to CSV files
    if metrics_summary:
        df_metrics_summary = pd.DataFrame(metrics_summary)
        output_file = "Results_image_small_metrics_summary.csv"
        df_metrics_summary.to_csv(output_file, index=False)
        print(f"\n✓ Metrics summary saved to: {output_file}")
    
    if energy_summary:
        df_energy_summary = pd.DataFrame(energy_summary)
        output_file = "Results_image_small_energy_summary.csv"
        df_energy_summary.to_csv(output_file, index=False)
        print(f"✓ Energy summary saved to: {output_file}")
    
    # Create comparison tables
    print("\n" + "="*80)
    print("COMPARISON TABLE - METRICS")
    print("="*80)
    
    if metrics_summary:
        comparison_metrics = pd.DataFrame([
            {
                'Model': row['Model'],
                'Precision (avg)': f"{row.get('precision_avg', 0):.4f}",
                'Precision (best)': f"{row.get('precision_best', 0):.4f}",
                'Recall (avg)': f"{row.get('recall_avg', 0):.4f}",
                'Recall (best)': f"{row.get('recall_best', 0):.4f}",
                'F1 (avg)': f"{row.get('F1_avg', 0):.4f}",
                'F1 (best)': f"{row.get('F1_best', 0):.4f}",
                'FP (avg)': f"{row.get('FP_avg', 0):.2f}",
                'FP (best/min)': f"{row.get('FP_best', 0):.2f}",
                'FN (avg)': f"{row.get('FN_avg', 0):.2f}",
                'FN (best/min)': f"{row.get('FN_best', 0):.2f}",
            }
            for row in metrics_summary
        ])
        print(comparison_metrics.to_string(index=False))
    
    print("\n" + "="*80)
    print("COMPARISON TABLE - ENERGY")
    print("="*80)
    
    if energy_summary:
        comparison_energy = pd.DataFrame([
            {
                'Model': row['Model'],
                'Avg Time/Image (s)': f"{row.get('avg_time_per_image_s_avg', 0):.6f}",
                'Best Time/Image (min)': f"{row.get('avg_time_per_image_s_best', 0):.6f}",
                'Avg Energy/Image (kg CO2)': f"{row.get('avg_energy_per_image_kg_co2_avg', 0):.2e}",
                'Best Energy/Image (min)': f"{row.get('avg_energy_per_image_kg_co2_best', 0):.2e}",
            }
            for row in energy_summary
        ])
        print(comparison_energy.to_string(index=False))
    
    print("\n" + "="*80)

if __name__ == "__main__":
    aggregate_results()
