import pandas as pd
import ast
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

def calculate_monthly_shifted_load():

    print("Loading simulation results...")
    
    try:
        df = pd.read_csv('simulation_results/simulation_results.csv')
        print(f"Loaded {len(df)} simulation days")
    except FileNotFoundError:
        print("Error: simulation_results/simulation_results.csv not found")
        return
    
    df['date'] = pd.to_datetime(df['date'])
    
    df['month_year'] = df['date'].dt.to_period('M')
    
    monthly_shifted_loads = []
    
    print("\nProcessing shifted load data for each month...")
    
    for month_year in df['month_year'].unique():
        month_data = df[df['month_year'] == month_year]
        shifted_load_col = 'M3: shifted_load'
        
        if shifted_load_col not in month_data.columns:
            print(f"Warning: {shifted_load_col} column not found for {month_year}")
            continue
        
        daily_shifted_loads = []
        monthly_total_shifted = 0
        days_with_shifted_load = 0
        
        for idx, row in month_data.iterrows():
            try:
                shifted_load_dict = ast.literal_eval(row[shifted_load_col])
                daily_shifted_total = sum(shifted_load_dict.values())
                
                if daily_shifted_total > 0:
                    monthly_total_shifted += daily_shifted_total
                    days_with_shifted_load += 1
                    daily_shifted_loads.append(daily_shifted_total)
                    
            except (ValueError, SyntaxError) as e:
                print(f"Error parsing shifted load data for {row['date']}: {e}")
                continue
    
        if days_with_shifted_load > 0:
            monthly_avg = monthly_total_shifted / days_with_shifted_load
            monthly_max = max(daily_shifted_loads)
            monthly_min = min(daily_shifted_loads)
            monthly_std = np.std(daily_shifted_loads)
            monthly_median = np.median(daily_shifted_loads)
            monthly_25th = np.percentile(daily_shifted_loads, 25)
            monthly_75th = np.percentile(daily_shifted_loads, 75)
            monthly_90th = np.percentile(daily_shifted_loads, 90)
            monthly_95th = np.percentile(daily_shifted_loads, 95)
        else:
            monthly_avg = monthly_max = monthly_min = monthly_std = monthly_median = 0
            monthly_25th = monthly_75th = monthly_90th = monthly_95th = 0
            
        monthly_shifted_loads.append({
            'month_year': month_year,
            'average_shifted_load_kwh': monthly_avg,
            'max_shifted_load_kwh': monthly_max,
            'min_shifted_load_kwh': monthly_min,
            'std_shifted_load_kwh': monthly_std,
            'median_shifted_load_kwh': monthly_median,
            'percentile_25_kwh': monthly_25th,
            'percentile_75_kwh': monthly_75th,
            'percentile_90_kwh': monthly_90th,
            'percentile_95_kwh': monthly_95th,
            'total_days': len(month_data),
            'days_with_shifted_load': days_with_shifted_load,
            'participation_rate': (days_with_shifted_load / len(month_data)) * 100
        })
        
        print(f"{month_year}: Avg={monthly_avg:.2f}, Max={monthly_max:.2f}, "
              f"Std={monthly_std:.2f}, Participation={monthly_shifted_loads[-1]['participation_rate']:.1f}%")
    
    
    results_df = pd.DataFrame(monthly_shifted_loads)
    output_file = 'simulation_results/monthly_shifted_load_averages.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    create_shifted_load_plots(results_df)
    
    return results_df

def create_shifted_load_plots(results_df):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    months = [str(m) for m in results_df['month_year']]
    x = np.arange(len(months))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, results_df['average_shifted_load_kwh'], width, 
                     label='Average', color='#2E86AB', alpha=0.7)
    bars2 = ax1.bar(x + width/2, results_df['max_shifted_load_kwh'], width, 
                     label='Maximum', color='#A23B72', alpha=0.7)
    
    ax1.set_title('Average vs Maximum Daily Shifted Load by Month', fontsize=14)
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Shifted Load (kWh/day)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(months, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'{height:.0f}', ha='center', va='bottom', fontsize=8)
    

    bars = ax2.bar(months, results_df['std_shifted_load_kwh'], color='#F18F01', alpha=0.7)
    ax2.set_title('Standard Deviation of Daily Shifted Load by Month', fontsize=14)
    ax2.set_xlabel('Month')
    ax2.set_ylabel('Standard Deviation (kWh/day)')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    

    ax3.plot(months, results_df['percentile_25_kwh'], 'o-', label='25th Percentile', color='#C73E1D')
    ax3.plot(months, results_df['median_shifted_load_kwh'], 's-', label='Median', color='#2E86AB', linewidth=2)
    ax3.plot(months, results_df['percentile_75_kwh'], '^-', label='75th Percentile', color='#A23B72')
    ax3.plot(months, results_df['percentile_90_kwh'], 'd-', label='90th Percentile', color='#F18F01')
    
    ax3.set_title('Distribution Percentiles by Month', fontsize=14)
    ax3.set_xlabel('Month')
    ax3.set_ylabel('Shifted Load (kWh/day)')
    ax3.tick_params(axis='x', rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    bars = ax4.bar(months, results_df['participation_rate'], color='#2E86AB', alpha=0.7)
    ax4.set_title('Participation Rate by Month', fontsize=14)
    ax4.set_xlabel('Month')
    ax4.set_ylabel('Participation Rate (%)')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 105)
    
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('simulation_results/monthly_shifted_load_comprehensive_chart.png', dpi=300, bbox_inches='tight')
    print("Comprehensive chart saved to: simulation_results/monthly_shifted_load_comprehensive_chart.png")


if __name__ == "__main__":
    results = calculate_monthly_shifted_load() 
