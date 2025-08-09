import pandas as pd
import ast
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

def get_monthly_stats(df, scenario_name):
    df['date'] = pd.to_datetime(df['date'])
    df['month_year'] = df['date'].dt.to_period('M')
    monthly_stats = []
    for month_year in df['month_year'].unique():
        month_data = df[df['month_year'] == month_year]
        auction_revenue_col = 'M7: auction_revenue'
        if auction_revenue_col not in month_data.columns:
            continue
        daily_revenues = []
        for idx, row in month_data.iterrows():
            try:
                auction_revenue_dict = ast.literal_eval(row[auction_revenue_col])
                daily_revenue_total = sum(auction_revenue_dict.values())
                if daily_revenue_total > 0:
                    daily_revenues.append(daily_revenue_total)
            except (ValueError, SyntaxError):
                continue
        if daily_revenues:
            monthly_avg = np.mean(daily_revenues)
            monthly_max = np.max(daily_revenues)
        else:
            monthly_avg = monthly_max = 0
        monthly_stats.append({
            'month_year': month_year,
            'average_auction_revenue_eur': monthly_avg,
            'max_auction_revenue_eur': monthly_max,
            'scenario': scenario_name
        })
    return pd.DataFrame(monthly_stats)

def calculate_monthly_auction_revenue():
    print("Loading simulation results...")
    try:
        df_2024 = pd.read_csv('simulation_results/2024_solar/simulation_results_2024.csv')
        df_2021 = pd.read_csv('simulation_results/2021_solar/simulation_results_2021.csv')
        df_simple = pd.read_csv('simulation_results/simple_solar/simulation_results_simple.csv')
        print(f"Loaded {len(df_2024)} (2024), {len(df_2021)} (2021), {len(df_simple)} (simple) simulation days")
    except FileNotFoundError:
        print("Error: One or more simulation results files not found")
        return
    
    stats_2024 = get_monthly_stats(df_2024, '2024 Solar')
    stats_2021 = get_monthly_stats(df_2021, '2021 Solar')
    stats_simple = get_monthly_stats(df_simple, 'Simple Solar')
    
    all_months = sorted(set(stats_2024['month_year']).union(stats_2021['month_year']).union(stats_simple['month_year']))
    
    months_str = [str(m) for m in all_months]
    def get_vals(stats_df, col):
        return [float(stats_df[stats_df['month_year'] == m][col].values[0]) if m in stats_df['month_year'].values else 0 for m in all_months]
    avg_2024 = get_vals(stats_2024, 'average_auction_revenue_eur')
    avg_2021 = get_vals(stats_2021, 'average_auction_revenue_eur')
    avg_simple = get_vals(stats_simple, 'average_auction_revenue_eur')
    max_2024 = get_vals(stats_2024, 'max_auction_revenue_eur')
    max_2021 = get_vals(stats_2021, 'max_auction_revenue_eur')
    max_simple = get_vals(stats_simple, 'max_auction_revenue_eur')
    
    x = np.arange(len(months_str))
    width = 0.22
    plt.figure(figsize=(16, 8))
    
    plt.bar(x - width, avg_2024, width, label='2024 Solar Avg', color='#1a2238', alpha=0.8)
    plt.bar(x, avg_2021, width, label='2021 Solar Avg', color='#E9967A', alpha=0.8)
    plt.bar(x + width, avg_simple, width, label='Simple Solar Avg', color='#0dcaf0', alpha=0.8)
    
    plt.bar(x - width, max_2024, width, fill=False, edgecolor='#1a2238', linewidth=2, linestyle='--', label='2024 Solar Max')
    plt.bar(x, max_2021, width, fill=False, edgecolor='#E9967A', linewidth=2, linestyle='--', label='2021 Solar Max')
    plt.bar(x + width, max_simple, width, fill=False, edgecolor='#0dcaf0', linewidth=2, linestyle='--', label='Simple Solar Max')
    plt.title('Average vs Maximum Daily Auction Revenue by Month (All Scenarios)', fontsize=16)
    plt.xlabel('Month')
    plt.ylabel('Auction Revenue (EUR/day)')
    plt.xticks(x, months_str, rotation=45)
    plt.legend(ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('simulation_results/monthly_auction_revenue_avg_max_all_scenarios.png', dpi=300, bbox_inches='tight')
    print("Average vs Maximum plot for all scenarios saved to: simulation_results/monthly_auction_revenue_avg_max_all_scenarios.png")

if __name__ == "__main__":
    calculate_monthly_auction_revenue() 
