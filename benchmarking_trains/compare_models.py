import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob

def clean_data(df):
    """
    Cleans the dataframe to handle concatenated training runs.
    Keeps only the LAST valid training sequence (based on epoch resets).
    """
    # 1. Standardize column names
    df.columns = [c.strip() for c in df.columns]
    
    # 2. Coerce to numeric (handles repeated headers which become non-numeric)
    if 'epoch' not in df.columns:
        return df
        
    df['epoch'] = pd.to_numeric(df['epoch'], errors='coerce')
    df = df.dropna(subset=['epoch'])
    
    # 3. Detect resets in epoch to find the last run
    # If epoch goes 1, 2, 3, 1, 2... we want the last sequence
    df = df.reset_index(drop=True)
    
    # Find indices where current epoch <= previous epoch (indicating a restart)
    # We always include index 0 as a start, so we look for others.
    # shift(1) compares current with previous. 
    # If df['epoch'][i] <= df['epoch'][i-1], it's a reset.
    resets = df[df['epoch'] <= df['epoch'].shift(1, fill_value=-1)].index.tolist()
    
    if resets:
        last_reset = resets[-1]
        # Keep from last reset to end
        df = df.iloc[last_reset:]
    
    return df

def load_results(root_dir):
    """
    Scans the root_dir (and specific known paths) for results.
    Returns a dictionary: {model_name: {'metrics': df, 'energy': df_energy}}
    """
    models_data = {}
    
    # 1. Search immediate subdirectories (Standard YOLO/SpikeYOLO structure)
    subdirs = [d for d in glob(os.path.join(root_dir, '*')) if os.path.isdir(d)]
    
    # 2. Add VanillaCNN paths if they exist elsewhere (e.g. Results/VanillaCNN/D-Fire)
    # Check project root relative to script execution?
    # Assuming script run from 'Code', let's check 'Results/VanillaCNN' if no vanilla found in root_dir
    vanilla_found = False
    for d in subdirs:
        if 'vanilla' in os.path.basename(d).lower():
            vanilla_found = True
            
    if not vanilla_found:
        possible_vanilla = os.path.join("Results", "VanillaCNN", "D-Fire")
        if os.path.exists(possible_vanilla):
            subdirs.append(possible_vanilla)

    def find_file_in_dir(base_dir, patterns):
        """Search recursively for given filename patterns and return the best candidate.

        Preference heuristic: deeper path (more separators) and newer modification time.
        """
        candidates = []
        for pat in patterns:
            candidates += glob(os.path.join(base_dir, '**', pat), recursive=True)
        if not candidates:
            return None
        # Sort by depth then by modification time (both descending)
        candidates = sorted(candidates, key=lambda p: (p.count(os.sep), os.path.getmtime(p)), reverse=True)
        return candidates[0]

    for d in subdirs:
        model_name = os.path.basename(d)
        # Normalize VanillaCNN directory name which sometimes contains dataset subfolder (D-Fire)
        if model_name == "D-Fire" or os.path.basename(os.path.dirname(d)) == "VanillaCNN":
            model_name = "VanillaCNN"

        # 1. Metrics - search recursively for standard files
        df_metrics = None
        metrics_file = find_file_in_dir(d, ['results.csv', 'vanilla_cnn_results.csv'])
        if metrics_file and os.path.exists(metrics_file):
            try:
                df_metrics = pd.read_csv(metrics_file)
                df_metrics = clean_data(df_metrics)
                # If file is a vanilla results file, normalize column names to YOLO convention
                if os.path.basename(metrics_file) == 'vanilla_cnn_results.csv':
                    rename_map = {
                        'mAP50': 'metrics/mAP50(B)',
                        'train_loss': 'train/box_loss'
                    }
                    df_metrics = df_metrics.rename(columns=rename_map)
                    model_name = "VanillaCNN"
            except Exception as e:
                print(f"Error reading metrics for {model_name} (from {metrics_file}): {e}")

        # 2. Energy (emissions.csv or vanilla_cnn_emissions.csv)
        df_energy = None
        emissions_file = find_file_in_dir(d, ['emissions.csv', 'vanilla_cnn_emissions.csv', 'spikeyolo_emissions.csv'])
        if emissions_file and os.path.exists(emissions_file):
            try:
                df_energy = pd.read_csv(emissions_file)
            except Exception as e:
                print(f"Error reading energy for {model_name} (from {emissions_file}): {e}")

        if df_metrics is not None:
            models_data[model_name] = {'metrics': df_metrics, 'energy': df_energy}
            
    return models_data

def plot_metric_comparison(models_data, metric_col, title, output_path, ylabel="Value"):
    plt.figure(figsize=(10, 6))
    
    plotted = False
    for model_name, data in models_data.items():
        df = data['metrics']
        if df is not None and metric_col in df.columns:
            # Check if epoch exists, else use index
            x = df['epoch'] if 'epoch' in df.columns else df.index
            plt.plot(x, df[metric_col], label=model_name, linewidth=2)
            plotted = True
            
    if plotted:
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(output_path)
        print(f"Saved {output_path}")
    else:
        print(f"Skipping {title}: Metric '{metric_col}' not found in any model.")
    
    plt.close()

def plot_energy_comparison(models_data, output_dir):
    # Bar chart for Total Energy / Total Time
    totals = []
    
    for model_name, data in models_data.items():
        df_e = data['energy']
        if df_e is not None:
            # CodeCarbon appends rows. Total is usually sum of 'emissions' if runs are separate tasks,
            # Or if it tracks per epoch, we sum.
            # Assuming standard CodeCarbon CSV: timestamp, project_name, duration, emissions, etc.
            # If multiple runs exist in file, we might need to filter. assuming one training run per CSV file valid.
            total_emissions = df_e['emissions'].sum() # kg NO, usually it's cumulative? No, it's delta.
            # Wait, standard codecarbon.csv lines are sessions.
            # If default usage: one line per session start/stop? Or periodic csv write?
            # Usually periodic. We sum 'emissions' column?
            # Actually CodeCarbon 'emissions' column is "Emissions created *since the last measurement*"? No, it's cumulative for the run usually?
            # Let's assume 'emissions' is cumulative total for the session in the last row, OR rows are periodic updates.
            # Safer: Max emissions - Min emissions? Or Sum of 'emissions' (if delta)?
            # CodeCarbon 2.x standard output: 'emissions': Total emissions generated by the experiment.
            # Usually one row per 'stop()' call or periodic.
            # If periodic, the last row has the total.
            # Let's take the MAX of emissions column as the Total for that run.
            
            total_kwh = df_e['energy_consumed'].max() if 'energy_consumed' in df_e.columns else 0
            total_co2 = df_e['emissions'].max()
            duration = df_e['duration'].max() if 'duration' in df_e.columns else 0
            
            # Get best mAP
            best_map = 0
            if data['metrics'] is not None and 'metrics/mAP50(B)' in data['metrics'].columns:
                best_map = data['metrics']['metrics/mAP50(B)'].max()
            
            totals.append({
                'Model': model_name,
                'Total CO2 (kg)': total_co2,
                'Total Energy (kWh)': total_kwh,
                'Duration (s)': duration,
                'Best mAP50': best_map
            })
    
    if not totals:
        print("No energy data found.")
        return

    df_res = pd.DataFrame(totals)
    
    # Plot CO2
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_res, x='Model', y='Total CO2 (kg)')
    plt.title("Total CO2 Emissions Comparison")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "compare_energy_co2.png"))
    plt.close()
    
    # Plot Efficiency (mAP / kgCO2)
    # Avoid zero div
    df_res['Efficiency (mAP per kgCO2)'] = df_res.apply(lambda row: row['Best mAP50'] / row['Total CO2 (kg)'] if row['Total CO2 (kg)'] > 0 else 0, axis=1)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_res, x='Model', y='Efficiency (mAP per kgCO2)')
    plt.title("Model Efficiency: Accuracy per unit of Carbon")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "compare_efficiency.png"))
    plt.close()
    
    print("Saved energy comparisons.")

def main():
    parser = argparse.ArgumentParser(description="Compare Training Results Only")
    parser.add_argument('--root', type=str, default='Results', help='Root directory containing model subfolders')
    parser.add_argument('--output', type=str, default='benchmarking_trains_results', help='Output directory for plots')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.root):
        print(f"Error: Root directory {args.root} does not exist.")
        return
        
    os.makedirs(args.output, exist_ok=True)
    
    print(f"Loading results from {args.root}...")
    models_data = load_results(args.root)
    
    if not models_data:
        print("No valid model results found.")
        return
        
    print(f"Found models: {list(models_data.keys())}")
    
    # 1. Metrics Plots
    # mAP50
    plot_metric_comparison(models_data, 'metrics/mAP50(B)', "Validation mAP @ 0.5", os.path.join(args.output, "compare_mAP50.png"), "mAP")
    
    # mAP50-95
    plot_metric_comparison(models_data, 'metrics/mAP50-95(B)', "Validation mAP @ 0.5:0.95", os.path.join(args.output, "compare_mAP50-95.png"), "mAP")
    
    # Losses
    plot_metric_comparison(models_data, 'train/box_loss', "Training Box Loss", os.path.join(args.output, "compare_loss_box_train.png"), "Loss")
    plot_metric_comparison(models_data, 'val/box_loss', "Validation Box Loss", os.path.join(args.output, "compare_loss_box_val.png"), "Loss")
    
    # 2. Energy Plots
    plot_energy_comparison(models_data, args.output)
    
    print(f"Done. Results saved to {args.output}")

if __name__ == "__main__":
    main()
