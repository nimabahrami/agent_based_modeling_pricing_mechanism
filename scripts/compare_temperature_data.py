import pandas as pd
import matplotlib.pyplot as plt
import os

# Paths to temperature data
path_2021 = 'data/processed/temperature_data_2021.csv'
path_2024 = 'data/processed/temperature_data_2024.csv'

# Load 2021 data
df_2021 = pd.read_csv(path_2021)
if 'period_end' in df_2021.columns:
    df_2021['period_end'] = pd.to_datetime(df_2021['period_end'])
    df_2021 = df_2021.set_index('period_end')
else:
    raise ValueError('2021 temperature data must have a period_end column')

# Load 2024 data
df_2024 = pd.read_csv(path_2024)
if 'period_end' in df_2024.columns:
    df_2024['period_end'] = pd.to_datetime(df_2024['period_end'])
    df_2024 = df_2024.set_index('period_end')
else:
    raise ValueError('2024 temperature data must have a period_end column')

# Align on common time range
common_index = df_2021.index.intersection(df_2024.index)
df_2021_common = df_2021.loc[common_index]
df_2024_common = df_2024.loc[common_index]

# Compute rolling statistics (24-hour = 96 intervals)
window = 96
ma_2021 = df_2021_common['air_temp'].rolling(window, min_periods=1).mean()
std_2021 = df_2021_common['air_temp'].rolling(window, min_periods=1).std()
ma_2024 = df_2024_common['air_temp'].rolling(window, min_periods=1).mean()
std_2024 = df_2024_common['air_temp'].rolling(window, min_periods=1).std()

# Plot
plt.figure(figsize=(18, 6))
# 2021
plt.plot(df_2021_common.index, ma_2021, color='#1a2238', label='2021 24h Moving Avg')
plt.fill_between(df_2021_common.index, ma_2021-std_2021, ma_2021+std_2021, color='#1a2238', alpha=0.15, label='2021 ±1 Std')
# 2024
plt.plot(df_2024_common.index, ma_2024, color='#E9967A', label='2024 24h Moving Avg')
plt.fill_between(df_2024_common.index, ma_2024-std_2024, ma_2024+std_2024, color='#E9967A', alpha=0.15, label='2024 ±1 Std')

plt.title('Temperature Comparison: 2021 vs 2024 (24h Moving Average ±1 Std)')
plt.xlabel('Date and Time')
plt.ylabel('Air Temperature (°C)')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save plot
os.makedirs('simulation_results', exist_ok=True)
plt.savefig('simulation_results/temperature_comparison_2021_2024.png', dpi=300, bbox_inches='tight')
print('Temperature comparison plot saved to simulation_results/temperature_comparison_2021_2024.png')

# --- Summer Distribution Comparison ---
# Filter for summer months (June, July, August)
summer_months = [6, 7, 8]
summer_2021 = df_2021_common[df_2021_common.index.month.isin(summer_months)]['air_temp']
summer_2024 = df_2024_common[df_2024_common.index.month.isin(summer_months)]['air_temp']

import seaborn as sns
plt.figure(figsize=(10, 6))
sns.histplot(summer_2021, color='#1a2238', label='2021 Summer', kde=True, stat='density', bins=30, alpha=0.6)
sns.histplot(summer_2024, color='#E9967A', label='2024 Summer', kde=True, stat='density', bins=30, alpha=0.6)
plt.title('Distribution of Summer Temperatures (June-August): 2021 vs 2024')
plt.xlabel('Air Temperature (°C)')
plt.ylabel('Density')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('simulation_results/temperature_summer_distribution_2021_2024.png', dpi=300)
print('Summer temperature distribution plot saved to simulation_results/temperature_summer_distribution_2021_2024.png') 