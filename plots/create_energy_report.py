"""
Create a simplified energy report from Results_image_small_energy_summary.csv
with specific columns requested by user.
"""

import pandas as pd

# Read the energy summary file
df = pd.read_csv('Results_image_small_energy_summary.csv')

# Create the report with requested columns
report = pd.DataFrame({
    'model': df['Model'],
    'time_s': df['total_time_s_avg'],
    'total_energy_kwh': df['total_energy_kg_co2_avg'] / 0.267,  # Convert kg CO2 to kWh (assuming Spain grid)
    'emissions_kg_co2': df['total_energy_kg_co2_avg'],
    'energy_per_image_kwh': df['avg_energy_per_image_kg_co2_avg'] / 0.267,  # Convert to kWh
    'emissions_per_image_kg_co2': df['avg_energy_per_image_kg_co2_avg']
})

# Save to new CSV file
output_file = 'Results_image_small_energy_report.csv'
report.to_csv(output_file, index=False)

print("Energy Report Created")
print("=" * 80)
print(report.to_string(index=False))
print("\n" + "=" * 80)
print(f"Report saved to: {output_file}")
print("\nColumn Descriptions:")
print("  - model: Model name")
print("  - time_s: Average total processing time in seconds")
print("  - total_energy_kwh: Total energy consumed in kWh")
print("  - emissions_kg_co2: Total CO2 emissions in kg")
print("  - energy_per_image_kwh: Energy consumed per image in kWh")
print("  - emissions_per_image_kg_co2: CO2 emissions per image in kg")
