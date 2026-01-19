import pandas as pd
from pathlib import Path

# Directorio raíz
root_dir = Path("new_full_test")

# Leer el CSV actual (formato largo)
input_file = root_dir / "combined_results_test_images.csv"
df_long = pd.read_csv(input_file)

# Pivotar: modelos como filas, métricas como columnas
df_wide = df_long.pivot(index='Model', columns='metric', values='value')

# Reset index para que 'Model' sea una columna normal
df_wide = df_wide.reset_index()

# Guardar resultado
output_file = root_dir / "combined_results_test_images.csv"
df_wide.to_csv(output_file, index=False)

print(f"Archivo transformado guardado en: {output_file}")
print(f"\nModelos: {len(df_wide)}")
print(f"Métricas: {list(df_wide.columns[1:])}")
print(f"\nPreview:")
print(df_wide.head())