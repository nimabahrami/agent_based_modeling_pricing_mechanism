import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import numpy as np

def extract_total_revenue(revenue_series):
    total_revenue = []
    for rev_data in revenue_series:
        if isinstance(rev_data, str):
            try:
                rev_dict = json.loads(rev_data.replace("'", '"'))
                total_rev = sum([v for v in rev_dict.values() if isinstance(v, (int, float))])
                total_revenue.append(total_rev)
            except:
                total_revenue.append(0)
        else:
            total_revenue.append(0)
    return total_revenue

def extract_total_generation(gen_series):
    total = 0
    for val in gen_series:
        if isinstance(val, str):
            try:
                # If it's a list or dict in string, sum all numbers
                if val.startswith('[') or val.startswith('{'):
                    import ast
                    parsed = ast.literal_eval(val)
                    if isinstance(parsed, (list, tuple)):
                        total += sum(float(x) for x in parsed if isinstance(x, (int, float, str)) and str(x).replace('.','',1).isdigit())
                    elif isinstance(parsed, dict):
                        total += sum(float(x) for x in parsed.values() if isinstance(x, (int, float, str)) and str(x).replace('.','',1).isdigit())
                    else:
                        total += float(parsed)
                else:
                    total += float(val)
            except:
                continue
        elif isinstance(val, (int, float)):
            total += val
    return total

normal_path = 'simulation_results/2024_solar/simulation_results_2024.csv'
simple_path = 'simulation_results/simple_solar/simulation_results_simple.csv'
dynamic_path = 'simulation_results/2021_solar/simulation_results_2021.csv'
normal = pd.read_csv(normal_path)
simple = pd.read_csv(simple_path)
dynamic = pd.read_csv(dynamic_path)

normal_revenue = extract_total_revenue(normal['M7: auction_revenue'])
simple_revenue = extract_total_revenue(simple['M7: auction_revenue'])
dynamic_revenue = extract_total_revenue(dynamic['M7: auction_revenue'])

steps_normal = range(1, len(normal_revenue) + 1)
steps_simple = range(1, len(simple_revenue) + 1)
steps_dynamic = range(1, len(dynamic_revenue) + 1)

plt.figure(figsize=(14, 7))
plt.plot(steps_normal, normal_revenue, label='Normal Solar', color='#A23B72', linewidth=2)
plt.plot(steps_simple, simple_revenue, label='Simple Solar', color='#F18F01', linewidth=2, linestyle='-.')
plt.plot(steps_dynamic, dynamic_revenue, label='Dynamic Solar', color='#2E86AB', linewidth=2, linestyle='--')
plt.xlabel('Simulation Step')
plt.ylabel('Auction Revenue (EUR)')
plt.title('Auction Revenue Through Steps: Normal, Simple, and Dynamic Solar')
plt.legend()
plt.grid(True, alpha=0.4)
plt.tight_layout()
output_steps = 'simulation_results/auction_revenue_steps_comparison_3.png'
plt.savefig(output_steps, dpi=300)
plt.close()
print(f"Stepwise comparison plot saved to: {output_steps}")


normal['total_revenue'] = normal_revenue
simple['total_revenue'] = simple_revenue
dynamic['total_revenue'] = dynamic_revenue
normal['date'] = pd.to_datetime(normal['date'])
simple['date'] = pd.to_datetime(simple['date'])
dynamic['date'] = pd.to_datetime(dynamic['date'])
normal['month'] = normal['date'].dt.to_period('M')
simple['month'] = simple['date'].dt.to_period('M')
dynamic['month'] = dynamic['date'].dt.to_period('M')
monthly_normal = normal.groupby('month')['total_revenue'].mean()
monthly_simple = simple.groupby('month')['total_revenue'].mean()
monthly_dynamic = dynamic.groupby('month')['total_revenue'].mean()
all_months = sorted(set(monthly_normal.index).union(set(monthly_simple.index)).union(set(monthly_dynamic.index)))
monthly_normal = monthly_normal.reindex(all_months, fill_value=0)
monthly_simple = monthly_simple.reindex(all_months, fill_value=0)
monthly_dynamic = monthly_dynamic.reindex(all_months, fill_value=0)

color_map = {'Normal Solar': '#1a2238', 'Simple Solar': '#E9967A', 'Dynamic Solar': '#0dcaf0'}
plt.figure(figsize=(14, 7))
plt.plot([str(m) for m in all_months], monthly_normal, marker='o', label='Normal Solar', color=color_map['Normal Solar'], linewidth=2)
plt.plot([str(m) for m in all_months], monthly_simple, marker='^', label='Simple Solar', color=color_map['Simple Solar'], linewidth=2, linestyle='-.')
plt.plot([str(m) for m in all_months], monthly_dynamic, marker='s', label='Dynamic Solar', color=color_map['Dynamic Solar'], linewidth=2, linestyle='--')
plt.xlabel('Month')
plt.ylabel('Average Auction Revenue (EUR)')
plt.title('Monthly Average Auction Revenue: Normal, Simple, and Dynamic Solar')
plt.legend()
plt.grid(True, alpha=0.4)
plt.tight_layout()
output_monthly = 'simulation_results/auction_revenue_monthly_comparison_3.png'
plt.savefig(output_monthly, dpi=300)
plt.close()
print(f"Monthly average comparison plot saved to: {output_monthly}")

total_normal = sum(normal_revenue)
total_simple = sum(simple_revenue)
total_dynamic = sum(dynamic_revenue)
labels = ['Normal Solar', 'Simple Solar', 'Dynamic Solar']
totals = [total_normal, total_simple, total_dynamic]
plt.figure(figsize=(10, 6))
plt.bar(labels, totals, color=['#A23B72', '#F18F01', '#2E86AB'], alpha=0.8)
plt.ylabel('Total Auction Revenue (EUR)')
plt.title('Total Auction Revenue: Normal, Simple, and Dynamic Solar')
for i, v in enumerate(totals):
    plt.text(i, v + max(totals)*0.01, f'{v:.2f}', ha='center', va='bottom', fontsize=12)
plt.tight_layout()
output_total = 'simulation_results/auction_revenue_total_comparison_3.png'
plt.savefig(output_total, dpi=300)
plt.close()
print(f"Total revenue comparison plot saved to: {output_total}")

print("\nTotal Auction Revenue:")
print(f"Normal Solar: {total_normal:.2f} EUR")
print(f"Simple Solar: {total_simple:.2f} EUR")
print(f"Dynamic Solar: {total_dynamic:.2f} EUR")
with open('simulation_results/auction_revenue_total_comparison_3.txt', 'w') as f:
    f.write('Total Auction Revenue Comparison (3 Simulations)\n')
    f.write(f'Normal Solar: {total_normal:.2f} EUR\n')
    f.write(f'Simple Solar: {total_simple:.2f} EUR\n')
    f.write(f'Dynamic Solar: {total_dynamic:.2f} EUR\n')
print("Total values saved to: simulation_results/auction_revenue_total_comparison_3.txt")

total_gen_normal = extract_total_generation(normal['M4: total_generation'])
total_gen_simple = extract_total_generation(simple['M4: total_generation'])
total_gen_dynamic = extract_total_generation(dynamic['M4: total_generation'])

labels_gen = ['Normal Solar', 'Simple Solar', 'Dynamic Solar']
totals_gen = [total_gen_normal, total_gen_simple, total_gen_dynamic]
plt.figure(figsize=(10, 6))
plt.bar(labels_gen, totals_gen, color=['#A23B72', '#F18F01', '#2E86AB'], alpha=0.8)
plt.ylabel('Total Generation (kWh)')
plt.title('Total Generation: Normal, Simple, and Dynamic Solar')
for i, v in enumerate(totals_gen):
    plt.text(i, v + max(totals_gen)*0.01, f'{v:.2f}', ha='center', va='bottom', fontsize=12)
plt.tight_layout()
output_gen = 'simulation_results/total_generation_comparison_3.png'
plt.savefig(output_gen, dpi=300)
plt.close()
print(f"Total generation comparison plot saved to: {output_gen}")

print("\nTotal Generation:")
print(f"Normal Solar: {total_gen_normal:.2f} kWh")
print(f"Simple Solar: {total_gen_simple:.2f} kWh")
print(f"Dynamic Solar: {total_gen_dynamic:.2f} kWh")
with open('simulation_results/total_generation_comparison_3.txt', 'w') as f:
    f.write('Total Generation Comparison (3 Simulations)\n')
    f.write(f'Normal Solar: {total_gen_normal:.2f} kWh\n')
    f.write(f'Simple Solar: {total_gen_simple:.2f} kWh\n')
    f.write(f'Dynamic Solar: {total_gen_dynamic:.2f} kWh\n')
print("Total generation values saved to: simulation_results/total_generation_comparison_3.txt")

shifted_paths = [
    ('Normal Solar', 'simulation_results/simulation_results.csv'),
    ('Simple Solar', 'simulation_results/simple_solar/simulation_results_simple.csv'),
    ('Dynamic Solar', 'simulation_results/dynamic_solar/simulation_results_dynamic_solar.csv'),
    ('2024 Solar', 'simulation_results/2024_solar/simulation_results_dynamic_solar.csv')
]
shifted_data = {}
for label, path in shifted_paths:
    if os.path.exists(path):
        df = pd.read_csv(path)
        if 'M3: shifted_load' in df.columns and 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df['month'] = df['date'].dt.to_period('M')
            # Aggregate shifted load for each month (sum)
            # Handle possible stringified lists
            def sum_shifted(val):
                if isinstance(val, str):
                    try:
                        import ast
                        parsed = ast.literal_eval(val)
                        if isinstance(parsed, (list, tuple)):
                            return sum(float(x) for x in parsed if isinstance(x, (int, float, str)) and str(x).replace('.','',1).isdigit())
                        else:
                            return float(parsed)
                    except:
                        return 0
                elif isinstance(val, (int, float)):
                    return val
                return 0
            df['shifted'] = df['M3: shifted_load'].apply(sum_shifted)
            monthly = df.groupby('month')['shifted'].sum()
            shifted_data[label] = monthly

all_months = sorted(set(m for series in shifted_data.values() for m in series.index))
bar_width = 0.18
x = np.arange(len(all_months))
plt.figure(figsize=(16, 7))
colors = ['#A23B72', '#F18F01', '#2E86AB', '#4CAF50']
for i, (label, series) in enumerate(shifted_data.items()):
    y = [series.get(m, 0) for m in all_months]
    plt.bar(x + i*bar_width, y, width=bar_width, label=label, color=colors[i % len(colors)], alpha=0.85)
plt.xlabel('Month')
plt.ylabel('Aggregated Shifted Load (kWh)')
plt.title('Monthly Aggregated Shifted Load Comparison')
plt.xticks(x + bar_width * (len(shifted_data)-1)/2, [str(m) for m in all_months], rotation=30)
plt.legend()
plt.tight_layout()
output_shifted = 'simulation_results/monthly_shifted_load_comparison.png'
plt.savefig(output_shifted, dpi=300)
plt.close()
print(f"Monthly shifted load comparison plot saved to: {output_shifted}") 
