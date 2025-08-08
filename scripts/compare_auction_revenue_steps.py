import pandas as pd
import matplotlib.pyplot as plt
import os


files = {
    '2021 Solar': 'simulation_results/2021_solar/simulation_results_2021.csv',
    '2024 Solar': 'simulation_results/2024_solar/simulation_results_2024.csv',
    'Simple Solar': 'simulation_results/simple_solar/simulation_results_simple.csv',
}

auction_col = 'M7: auction_revenue'
colors = {'2021 Solar': '#1a2238', '2024 Solar': '#E9967A', 'Simple Solar': '#0dcaf0'}
auction_data = {}
statistics = {}

for scenario, path in files.items():
    if not os.path.exists(path):
        print(f"File not found: {path}")
        continue
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    if auction_col not in df.columns:
        print(f"Auction revenue column not found in {path}")
        continue
    revenue = pd.to_numeric(df[auction_col], errors='coerce')
    auction_data[scenario] = revenue
    statistics[scenario] = {
        'mean': revenue.mean(),
        'median': revenue.median(),
        'min': revenue.min(),
        'max': revenue.max(),
        'std': revenue.std(),
        'count': revenue.count()
    }

plt.figure(figsize=(16, 7))
for scenario, revenue in auction_data.items():
    plt.plot(revenue.values, label=scenario, color=colors.get(scenario, None), linewidth=2)
plt.title('Auction Revenue Through Steps: Solar Scenarios')
plt.xlabel('Simulation Step')
plt.ylabel('Auction Revenue (EUR)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

stats_text = ''
for scenario, stats in statistics.items():
    stats_text += f"{scenario}:\n  Mean: {stats['mean']:.2f}\n  Median: {stats['median']:.2f}\n  Min: {stats['min']:.2f}\n  Max: {stats['max']:.2f}\n  Std: {stats['std']:.2f}\n  Count: {stats['count']}\n\n"
plt.gca().text(1.01, 0.5, stats_text, transform=plt.gca().transAxes, fontsize=12,
               verticalalignment='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='#1a2238'), family='monospace')

os.makedirs('simulation_results', exist_ok=True)
plt.savefig('simulation_results/auction_revenue_through_steps_comparison.png', dpi=300, bbox_inches='tight')
print('Auction revenue through steps plot saved to simulation_results/auction_revenue_through_steps_comparison.png')

with open('simulation_results/auction_revenue_through_steps_stats.txt', 'w') as f:
    f.write(stats_text)
print('Auction revenue statistics saved to simulation_results/auction_revenue_through_steps_stats.txt') 
