import pandas as pd
import matplotlib.pyplot as plt

csv_path = 'simulation_results/2021_solar/august_15_2021_prices.csv'
df = pd.read_csv(csv_path)

plt.figure(figsize=(16, 8))
plt.plot(df['time'], df['wholesale_prices'], label='Wholesale Price', color='#1a2238', linewidth=2)
plt.plot(df['time'], df['price_signal'], label='Price Signal', color='#E9967A', linewidth=2)
plt.title('Wholesale Price and Price Signal - August 15, 2021', fontsize=24)
plt.xlabel('Time', fontsize=20)
plt.ylabel('Price (EUR/kWh)', fontsize=20)

hour_indices = list(range(0, len(df['time']), 4))
hour_labels = [df['time'][i] for i in hour_indices]
plt.xticks(hour_indices, hour_labels, fontsize=16, rotation=45)
plt.yticks(fontsize=16)
plt.legend(fontsize=18)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('simulation_results/2021_solar/august_15_2021_prices_largefont.png', dpi=300)
print('Plot saved to simulation_results/2021_solar/august_15_2021_prices_largefont.png') 
