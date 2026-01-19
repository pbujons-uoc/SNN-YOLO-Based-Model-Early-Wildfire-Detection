"""
Script para generar gráficos comparativos de métricas de entrenamiento
de todos los modelos (SpikeYOLO, SpikeYOLO_latency, SpikeYOLO_poisson, YOLOv8, VanillaCNN).
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Configurar estilo de gráficos
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

# Paths base
BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "Results_train_final"
OUTPUT_DIR = Path(__file__).parent / "plots"
OUTPUT_DIR.mkdir(exist_ok=True)

# Configuración de modelos y sus archivos CSV
MODELS = {
    "SpikeYOLO": {
        "csv": RESULTS_DIR / "SpikeYOLO" / "results.csv",
        "color": "#1f77b4",
        "linestyle": "-",
    },
    "SpikeYOLO_latency": {
        "csv": RESULTS_DIR / "SpikeYOLO_latency" / "results.csv",
        "color": "#ff7f0e",
        "linestyle": "--",
    },
    "SpikeYOLO_poisson": {
        "csv": RESULTS_DIR / "SpikeYOLO_poisson" / "results.csv",
        "color": "#2ca02c",
        "linestyle": "-.",
    },
    "YOLOv8": {
        "csv": RESULTS_DIR / "Yolov8" / "results.csv",
        "color": "#d62728",
        "linestyle": "-",
    },
    "VanillaCNN": {
        "csv": RESULTS_DIR / "VanillaCNN" / "vanilla_cnn_results.csv",
        "color": "#9467bd",
        "linestyle": ":",
    },
}


def load_model_data(model_name, csv_path, max_epochs=45):
    """Carga y normaliza datos de CSV de entrenamiento."""
    if not csv_path.exists():
        print(f"WARNING: No se encontró {csv_path}")
        return None
    
    try:
        df = pd.read_csv(csv_path)
        
        # Limpiar nombres de columnas (quitar espacios)
        df.columns = df.columns.str.strip()
        
        # Asegurarse de que exista columna epoch
        if 'epoch' not in df.columns:
            print(f"WARNING: {model_name} no tiene columna 'epoch'")
            return None
        
        # Sample VanillaCNN evenly to match other models' epoch count (45 epochs)
        if model_name == 'VanillaCNN' and len(df) > max_epochs:
            original_len = len(df)
            # Select evenly spaced indices across all epochs
            indices = np.linspace(0, len(df) - 1, max_epochs, dtype=int)
            df = df.iloc[indices].reset_index(drop=True)
            # Re-number epochs to 0-44 for consistency in plots
            df['epoch'] = range(len(df))
            print(f"{model_name}: {original_len} épocas cargadas, muestreadas {len(df)} épocas equiespaciadas para comparación")
        else:
            print(f"{model_name}: {len(df)} épocas cargadas")
        
        print(f"   Columnas: {list(df.columns)}")
        
        return df
    
    except Exception as e:
        print(f"ERROR al cargar {model_name}: {e}")
        return None


def plot_metric_comparison(metric_name, ylabel, title, filename):
    """Genera un gráfico comparativo de una métrica específica."""
    plt.figure(figsize=(14, 8))
    
    has_data = False
    
    for model_name, config in MODELS.items():
        df = load_model_data(model_name, config["csv"])
        
        if df is None:
            continue
        
        # Buscar la columna de la métrica (puede tener espacios o no)
        metric_col = None
        for col in df.columns:
            if metric_name in col:
                metric_col = col
                break
        
        if metric_col is None:
            print(f"{model_name}: no tiene métrica '{metric_name}'")
            continue
        
        # Plot
        plt.plot(
            df['epoch'],
            df[metric_col],
            label=model_name,
            color=config["color"],
            linestyle=config["linestyle"],
            linewidth=2.5,
            marker='o',
            markersize=4,
            alpha=0.8
        )
        has_data = True
    
    if not has_data:
        print(f"No data to plot: {metric_name}")
        return
    
    plt.xlabel("Epoch", fontsize=13, fontweight='bold')
    plt.ylabel(ylabel, fontsize=13, fontweight='bold')
    plt.title(title, fontsize=15, fontweight='bold', pad=20)
    plt.legend(loc='best', fontsize=11, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_loss_comparison():
    """Grafica pérdidas de entrenamiento (box + cls)."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Loss Comparison During Training", fontsize=16, fontweight='bold')
    
    loss_metrics = [
        ("train/box_loss", "Box Loss (Train)", axes[0, 0]),
        ("train/cls_loss", "Class Loss (Train)", axes[0, 1]),
        ("val/box_loss", "Box Loss (Val)", axes[1, 0]),
        ("val/cls_loss", "Class Loss (Val)", axes[1, 1]),
    ]
    
    for metric_name, ylabel, ax in loss_metrics:
        for model_name, config in MODELS.items():
            df = load_model_data(model_name, config["csv"])
            
            if df is None:
                continue
            
            # Buscar columna
            metric_col = None
            for col in df.columns:
                if metric_name in col:
                    metric_col = col
                    break
            
            if metric_col is None:
                continue
            
            ax.plot(
                df['epoch'],
                df[metric_col],
                label=model_name,
                color=config["color"],
                linestyle=config["linestyle"],
                linewidth=2,
                marker='o',
                markersize=3
            )
        
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "loss_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_final_metrics_bar():
    """Gráfico de barras con métricas finales (última época)."""
    metrics_to_compare = [
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
        "metrics/precision(B)",
        "metrics/recall(B)"
    ]
    
    data = {metric: [] for metric in metrics_to_compare}
    model_names = []
    
    for model_name, config in MODELS.items():
        df = load_model_data(model_name, config["csv"])
        
        if df is None:
            continue
        
        model_names.append(model_name)
        
        # Obtener última época
        last_row = df.iloc[-1]
        
        for metric in metrics_to_compare:
            # Buscar columna
            metric_col = None
            for col in df.columns:
                if metric in col:
                    metric_col = col
                    break
            
            if metric_col is not None:
                data[metric].append(float(last_row[metric_col]))
            else:
                data[metric].append(0.0)
    
    # Crear subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Final Metrics Comparison (Last Epoch)", fontsize=16, fontweight='bold')
    
    metric_labels = {
        "metrics/mAP50(B)": "mAP@0.5",
        "metrics/mAP50-95(B)": "mAP@0.5:0.95",
        "metrics/precision(B)": "Precision",
        "metrics/recall(B)": "Recall"
    }
    
    axes_flat = axes.flatten()
    
    for i, metric in enumerate(metrics_to_compare):
        ax = axes_flat[i]
        values = data[metric]
        colors = [MODELS[name]["color"] for name in model_names]
        
        bars = ax.bar(model_names, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        # Añadir valores encima de las barras
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f'{height:.3f}',
                ha='center',
                va='bottom',
                fontsize=10,
                fontweight='bold'
            )
        
        ax.set_ylabel(metric_labels[metric], fontsize=12, fontweight='bold')
        ax.set_ylim(0, max(values) * 1.15 if max(values) > 0 else 1)
        ax.grid(axis='y', alpha=0.3)
        ax.set_xticklabels(model_names, rotation=15, ha='right')
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "final_metrics_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    """Ejecuta generación de todos los gráficos."""
    print("\n" + "="*80)
    print("GENERATING TRAINING COMPARISON PLOTS")
    print("="*80 + "\n")
    
    # Verificar que existan los CSVs
    print("Verifying CSV files...\n")
    for model_name, config in MODELS.items():
        if config["csv"].exists():
            print(f"{model_name}: {config['csv']}")
        else:
            print(f"{model_name}: {config['csv']} (NOT FOUND)")
    
    print("\n" + "="*80)
    print("Generating plots...\n")
    
    # 1. mAP@0.5
    plot_metric_comparison(
        "metrics/mAP50(B)",
        "mAP@0.5",
        "mAP@0.5 Comparison During Training",
        "map50_comparison.png"
    )
    
    # 2. mAP@0.5:0.95
    plot_metric_comparison(
        "metrics/mAP50-95(B)",
        "mAP@0.5:0.95",
        "mAP@0.5:0.95 Comparison During Training",
        "map50_95_comparison.png"
    )
    
    # 3. Precision
    plot_metric_comparison(
        "metrics/precision(B)",
        "Precision",
        "Precision Comparison During Training",
        "precision_comparison.png"
    )
    
    # 4. Recall
    plot_metric_comparison(
        "metrics/recall(B)",
        "Recall",
        "Recall Comparison During Training",
        "recall_comparison.png"
    )
    
    # 5. Pérdidas (4 subplots)
    plot_loss_comparison()
    
    # 6. Gráfico de barras con métricas finales
    plot_final_metrics_bar()
    
    print("\n" + "="*80)
    print("GENERATION COMPLETED")
    print("="*80)
    print(f"\nPlots saved in: {OUTPUT_DIR}\n")


if __name__ == "__main__":
    main()
