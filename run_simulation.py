
import os
import sys
import pandas as pd
import numpy as np
import shutil
import matplotlib.pyplot as plt
from datetime import datetime
from model.model_code import EnergyCommunity
from model.community_setup import create_community_configuration
from model.enumerations import *
from model.day_ahead_prices import DayAheadPrices


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_example_levers():    
    levers = {
        'L1': 0.7,
        'L2': 0.3,
        'L3': 0.4 
    }
    return levers


def create_example_uncertainties():
    uncertainties = {
        'X1': 0.25,
        'X2': 0.80, 
        'X3': 0.85 
    }
    
    return uncertainties


def display_step_price_metrics(model, step_num):
    latest_data = model.datacollector.get_model_vars_dataframe().iloc[-1]
    
    print(f"\n--- Step {step_num + 1} Price Metrics ---")
    print(f"Date: {latest_data.get('date', 'N/A')}")
    
    if 'M11: clearing_price' in latest_data:
        clearing_price = latest_data['M11: clearing_price']
        print(f"Clearing Price: {clearing_price:.4f} EUR/kWh" if clearing_price is not None else "Clearing Price: N/A")
    
    if 'M12: base_retail_price' in latest_data:
        base_retail_price = latest_data['M12: base_retail_price']
        if base_retail_price is not None:
            if isinstance(base_retail_price, (np.ndarray, list)):
                avg_price = np.mean(base_retail_price)
                print(f"Base Retail Price (avg): {avg_price:.4f} EUR/kWh")
            else:
                print(f"Base Retail Price: {base_retail_price:.4f} EUR/kWh")
        else:
            print("Base Retail Price: N/A")
    
    if 'M10: lcoe_values' in latest_data:
        lcoe_data = latest_data['M10: lcoe_values']
        if isinstance(lcoe_data, str):
            try:
                import json
                lcoe_dict = json.loads(lcoe_data)
                if lcoe_dict:
                    lcoe_values = [v for v in lcoe_dict.values() if v > 0]
                    if lcoe_values:
                        print(f"LCOE Range: {min(lcoe_values):.4f} - {max(lcoe_values):.4f} EUR/kWh (Avg: {sum(lcoe_values)/len(lcoe_values):.4f})")
                    else:
                        print("LCOE: No valid values")
                else:
                    print("LCOE: No data")
            except:
                print("LCOE: Error parsing data")
    
    if 'M9: cove_values' in latest_data:
        cove_data = latest_data['M9: cove_values']
        if isinstance(cove_data, str):
            try:
                import json
                cove_dict = json.loads(cove_data)
                if cove_dict:
                    cove_values = [v for v in cove_dict.values() if v > 0]
                    if cove_values:
                        print(f"COVE Range: {min(cove_values):.4f} - {max(cove_values):.4f} EUR/kWh (Avg: {sum(cove_values)/len(cove_values):.4f})")
                    else:
                        print("COVE: No valid values")
                else:
                    print("COVE: No data")
            except:
                print("COVE: Error parsing data")


def run_basic_simulation(cove_markup):
    print("Setting up Energy Community Model...")
    community_name = 'gridflex_heeten'
    agents_list = create_community_configuration(community_name=community_name)
    levers = create_example_levers()
    uncertainties = create_example_uncertainties()
    start_date = datetime.strptime("2021-02-01", "%Y-%m-%d")
    
    print(f"Community: {community_name}")
    print(f"Number of agents: {len(agents_list)}")
    print(f"Levers: {levers}")
    print(f"Uncertainties: {uncertainties}")
    print(f"Start date: {start_date}")
    
    cove_markup = cove_markup
    print(f"COVE markup factor: {cove_markup}")
    
    print("\nCreating model...")
    model = EnergyCommunity(
        levers=levers,
        uncertainties=uncertainties,
        agents_list=agents_list,
        start_date=start_date,
        cove_markup=cove_markup
    )
    
    steps = 270
    for step in range(steps):
        print(f'\nStep: {step + 1}')
        model.step()
        if (step + 1) % 5 == 0 or step == steps - 1:
            display_step_price_metrics(model, step)
    
    print(f"\nSimulation completed!")
    results = model.datacollector.get_model_vars_dataframe()
    print(f"Results shape: {results.shape}")
    print(f"Date range: {results['date'].min()} to {results['date'].max()}")
    
    return model, results


def display_price_comparison(results, step=None):
    if step is None:
        step = -1  # Last step
    
    print(f"\n{'='*60}")
    print(f"PRICE COMPARISON METRICS (Step {step + 1})")
    print(f"{'='*60}")
    
    if 'M13: price_comparison_summary' in results.columns:
        price_summary = results['M13: price_comparison_summary'].iloc[step]
        if isinstance(price_summary, str):
            try:
                import json
                summary_dict = json.loads(price_summary)
                
                print(f"Date: {results['date'].iloc[step]}")
                print(f"\nClearing Price: {summary_dict.get('clearing_price', 'N/A'):.4f} EUR/kWh")
                print(f"Base Retail Price: {summary_dict.get('base_retail_price', 'N/A'):.4f} EUR/kWh")
                
                print(f"\nLCOE Statistics:")
                print(f"  Average LCOE: {summary_dict.get('average_lcoe', 'N/A'):.4f} EUR/kWh")
                print(f"  Min LCOE: {summary_dict.get('min_lcoe', 'N/A'):.4f} EUR/kWh")
                print(f"  Max LCOE: {summary_dict.get('max_lcoe', 'N/A'):.4f} EUR/kWh")
                
                print(f"\nCOVE Statistics:")
                print(f"  Average COVE: {summary_dict.get('average_cove', 'N/A'):.4f} EUR/kWh")
                print(f"  Min COVE: {summary_dict.get('min_cove', 'N/A'):.4f} EUR/kWh")
                print(f"  Max COVE: {summary_dict.get('max_cove', 'N/A'):.4f} EUR/kWh")
                
                clearing_price = summary_dict.get('clearing_price')
                base_retail_price = summary_dict.get('base_retail_price')
                avg_lcoe = summary_dict.get('average_lcoe')
                avg_cove = summary_dict.get('average_cove')
                
                if all(v is not None for v in [clearing_price, base_retail_price]):
                    price_diff = clearing_price - base_retail_price
                    price_diff_pct = (price_diff / base_retail_price) * 100
                    print(f"\nPrice Analysis:")
                    print(f"  Clearing vs Base Retail: {price_diff:+.4f} EUR/kWh ({price_diff_pct:+.1f}%)")
                
                if all(v is not None for v in [clearing_price, avg_lcoe]):
                    lcoe_diff = clearing_price - avg_lcoe
                    lcoe_diff_pct = (lcoe_diff / avg_lcoe) * 100 if avg_lcoe != 0 else 0
                    print(f"  Clearing vs Avg LCOE: {lcoe_diff:+.4f} EUR/kWh ({lcoe_diff_pct:+.1f}%)")
                
                if all(v is not None for v in [clearing_price, avg_cove]):
                    cove_diff = clearing_price - avg_cove
                    cove_diff_pct = (cove_diff / avg_cove) * 100 if avg_cove != 0 else 0
                    print(f"  Clearing vs Avg COVE: {cove_diff:+.4f} EUR/kWh ({cove_diff_pct:+.1f}%)")
                
            except Exception as e:
                print(f"Error parsing price summary: {e}")
                print(f"Raw data: {price_summary[:200]}...")
        else:
            print(f"Price summary data type: {type(price_summary)}")
    else:
        print("Price comparison summary not available in results")


def analyze_results(results):    
    print("\n" + "="*50)
    print("SIMULATION RESULTS ANALYSIS")
    print("="*50)
    
    print(f"\nTotal simulation days: {len(results)}")
    print(f"Date range: {results['date'].min()} to {results['date'].max()}")
    display_price_comparison(results)
    
    metrics_to_analyze = [
        'M1: realised_demand',
        'M2: scheduled_demand', 
        'M3: shifted_load',
        'M4: total_generation',
        'M5: savings_on_ToD',
        'M6: energy_costs',
        'M7: auction_revenue',
        'M8: dynamic_flexibility',
        'M9: cove_values',
        'M10: lcoe_values'
    ]
    
    for metric in metrics_to_analyze:
        if metric in results.columns:
            print(f"\n{metric}:")
            metric_data = results[metric].iloc[-1]  # Last timestep
            if isinstance(metric_data, str):
                try:
                    import json
                    metric_dict = json.loads(metric_data)
                    for agent, value in metric_dict.items():
                        print(f"  {agent}: {value:.4f}")
                except:
                    print(f"  Raw data: {metric_data[:100]}...")
            else:
                print(f"  Value: {metric_data}")
    
    return results


def plot_results(results):
    os.makedirs('simulation_results', exist_ok=True)
    results_clean = results.iloc[2:].reset_index(drop=True)
    
    plt.style.use('default')
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['axes.edgecolor'] = '#E0E0E0'
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['grid.color'] = '#CCCCCC'  
    plt.rcParams['grid.linestyle'] = '-'
    plt.rcParams['grid.linewidth'] = 0.8 
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    
    actual_color = '#000080'
    predicted_color = '#E9967A' 
    
    plt.figure(figsize=(12, 8))
    if 'M7: auction_revenue' in results_clean.columns:
        revenue_data = results_clean['M7: auction_revenue']
        step_numbers = range(3, len(results) + 1)
        total_revenue = []
        for i, rev_data in enumerate(revenue_data):
            if isinstance(rev_data, str):
                try:
                    import json
                    rev_dict = json.loads(rev_data)
                    total_rev = sum([v for v in rev_dict.values() if isinstance(v, (int, float))])
                    total_revenue.append(total_rev)
                except:
                    total_revenue.append(0)
            else:
                total_revenue.append(0)
        
        plt.plot(step_numbers, total_revenue, color=actual_color, linewidth=2, marker='s', markersize=4, label='Auction Revenue')
        plt.title('Daily Auction Revenue Through Simulation Steps', pad=20)
        plt.ylabel('Daily Revenue (EUR)')
        plt.xlabel('Simulation Step')
        plt.grid(True, alpha=0.6)
        plt.xticks(range(3, len(results) + 1, 10))
        plt.legend()
        plt.tight_layout()
        plt.savefig('simulation_results/auction_revenue_through_steps.png', dpi=300, bbox_inches='tight', facecolor='white')
        print("Auction revenue plot saved to: simulation_results/auction_revenue_through_steps.png")
        plt.close()
    
    plt.figure(figsize=(12, 8))
    if 'M3: shifted_load' in results_clean.columns and len(results_clean) > 0:
        shifted_data = results_clean['M3: shifted_load']
        step_numbers = range(3, len(results) + 1)  # Start from step 3
        
        daily_shifted_load = []
        
        for shifted_step in shifted_data:
            if isinstance(shifted_step, str):
                try:
                    import json
                    shifted_dict = json.loads(shifted_step)
                    daily_total = sum([v for v in shifted_dict.values() if isinstance(v, (int, float))])
                    daily_shifted_load.append(daily_total)
                except:
                    daily_shifted_load.append(0)
            else:
                daily_shifted_load.append(0)
        
        plt.plot(step_numbers, daily_shifted_load, color=predicted_color, linewidth=2, marker='s', markersize=4, label='Shifted Load')
        plt.title('Daily Shifted Load Through Simulation Steps', pad=20)
        plt.ylabel('Daily Shifted Load (kWh)')
        plt.xlabel('Simulation Step')
        plt.grid(True, alpha=0.6)
        plt.xticks(range(3, len(results) + 1, 10))
        plt.legend()
        plt.tight_layout()
        plt.savefig('simulation_results/shifted_load_by_agent.png', dpi=300, bbox_inches='tight', facecolor='white')
        print("Shifted load plot saved to: simulation_results/shifted_load_by_agent.png")
        plt.close()
    
    
    plt.figure(figsize=(15, 8))
    if 'M14: generation_prediction_error' in results_clean.columns and len(results_clean) > 0:
        error_data = results_clean['M14: generation_prediction_error'].iloc[-1]
        
        if isinstance(error_data, str):
            try:
                import json
                error_list = json.loads(error_data)
                if len(error_list) == 96: 
                    time_labels = [f"{i//4:02d}:{(i%4)*15:02d}" for i in range(96)]
                    plt.plot(time_labels, error_list, color=predicted_color, linewidth=2, marker='o', markersize=3, label='Generation Prediction Error')
                    plt.title('Generation Prediction Error - 24-Hour Detailed View (Last Simulation Day)', pad=20)
                    plt.ylabel('Prediction Error (kWh)')
                    plt.xlabel('Time of Day')
                    plt.grid(True, alpha=0.6)
                    plt.xticks(range(0, 96, 8), [time_labels[i] for i in range(0, 96, 8)], rotation=45)
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig('simulation_results/generation_prediction_error.png', dpi=300, bbox_inches='tight', facecolor='white')
                    print("Generation prediction error plot saved to: simulation_results/generation_prediction_error.png")
                    plt.close()
                else:
                    plt.text(0.5, 0.5, 'Invalid data format', ha='center', va='center', transform=plt.gca().transAxes)
                    plt.tight_layout()
                    plt.savefig('simulation_results/generation_prediction_error.png', dpi=300, bbox_inches='tight', facecolor='white')
                    plt.close()
            except:
                plt.text(0.5, 0.5, 'Error parsing generation prediction error data', ha='center', va='center', transform=plt.gca().transAxes)
                plt.tight_layout()
                plt.savefig('simulation_results/generation_prediction_error.png', dpi=300, bbox_inches='tight', facecolor='white')
                plt.close()
        else:
            plt.text(0.5, 0.5, 'No generation prediction error data available', ha='center', va='center', transform=plt.gca().transAxes)
            plt.tight_layout()
            plt.savefig('simulation_results/generation_prediction_error.png', dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
    
    plt.figure(figsize=(15, 8))
    if 'M15: demand_prediction_error' in results_clean.columns and len(results_clean) > 0:
        error_data = results_clean['M15: demand_prediction_error'].iloc[-1]
        
        if isinstance(error_data, str):
            try:
                import json
                error_list = json.loads(error_data)
                if len(error_list) == 96: 
                    time_labels = [f"{i//4:02d}:{(i%4)*15:02d}" for i in range(96)]
                    plt.plot(time_labels, error_list, color=predicted_color, linewidth=2, marker='s', markersize=3, label='Demand Prediction Error')
                    plt.title('Demand Prediction Error - 24-Hour Detailed View (Last Simulation Day)', pad=20)
                    plt.ylabel('Prediction Error (kWh)')
                    plt.xlabel('Time of Day')
                    plt.grid(True, alpha=0.6)
                    plt.xticks(range(0, 96, 8), [time_labels[i] for i in range(0, 96, 8)], rotation=45)
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig('simulation_results/demand_prediction_error.png', dpi=300, bbox_inches='tight', facecolor='white')
                    print("Demand prediction error plot saved to: simulation_results/demand_prediction_error.png")
                    plt.close()
                else:
                    plt.text(0.5, 0.5, 'Invalid data format', ha='center', va='center', transform=plt.gca().transAxes)
                    plt.tight_layout()
                    plt.savefig('simulation_results/demand_prediction_error.png', dpi=300, bbox_inches='tight', facecolor='white')
                    plt.close()
            except:
                plt.text(0.5, 0.5, 'Error parsing demand prediction error data', ha='center', va='center', transform=plt.gca().transAxes)
                plt.tight_layout()
                plt.savefig('simulation_results/demand_prediction_error.png', dpi=300, bbox_inches='tight', facecolor='white')
                plt.close()
        else:
            plt.text(0.5, 0.5, 'No demand prediction error data available', ha='center', va='center', transform=plt.gca().transAxes)
            plt.tight_layout()
            plt.savefig('simulation_results/demand_prediction_error.png', dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
    
    plt.figure(figsize=(12, 8))
    if 'M16: auction_bids_offers' in results_clean.columns and len(results_clean) > 0:
        all_bids = []
        all_offers = []
        
        for step_idx, (date, auction_data) in enumerate(zip(results_clean['date'], results_clean['M16: auction_bids_offers'])):
            if isinstance(auction_data, str):
                try:
                    import json
                    auction_dict = json.loads(auction_data)
                    
                    for agent_id, data in auction_dict.items():
                        if 'bid' in data:
                            all_bids.append((step_idx + 3, data['bid']['price'])) 
                        if 'offer' in data:
                            all_offers.append((step_idx + 3, data['offer']['price']))
                except:
                    continue
        

        if all_bids:
            bid_steps, bid_prices = zip(*all_bids)
            plt.scatter(bid_steps, bid_prices, color=actual_color, s=30, alpha=0.6, label='Bids', marker='o')
        
        if all_offers:
            offer_steps, offer_prices = zip(*all_offers)
            plt.scatter(offer_steps, offer_prices, color=predicted_color, s=30, alpha=0.6, label='Offers', marker='s')
        
        plt.title('Bids and Offers Through Simulation Steps', pad=20)
        plt.ylabel('Price (EUR/kWh)')
        plt.xlabel('Simulation Step')
        plt.legend()
        plt.grid(True, alpha=0.6)
        plt.xticks(range(3, len(results) + 1, 10))
        plt.tight_layout()
        plt.savefig('simulation_results/bids_offers_scatter.png', dpi=300, bbox_inches='tight', facecolor='white')
        print("Bids/Offers scatter plot saved to: simulation_results/bids_offers_scatter.png")
        plt.close()
    else:
        plt.figure(figsize=(12, 8))
        plt.text(0.5, 0.5, 'No auction data available', ha='center', va='center', transform=plt.gca().transAxes)
        plt.tight_layout()
        plt.savefig('simulation_results/bids_offers_scatter.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    plot_price_analysis_for_typical_day(results_clean)
    
    results.to_csv('simulation_results/simulation_results.csv', index=False)
    print("Results saved to: simulation_results/simulation_results.csv")
    print("\nAll standalone plots created successfully!")


def plot_price_analysis_for_typical_day(results):
    
    plt.style.use('default')
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['axes.edgecolor'] = '#E0E0E0'
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['grid.color'] = '#CCCCCC'
    plt.rcParams['grid.linestyle'] = '-'
    plt.rcParams['grid.linewidth'] = 0.8
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    
    fig, ax = plt.subplots(1, 1, figsize=(15, 8))
    typical_day_idx = len(results) // 2
    typical_date = results['date'].iloc[typical_day_idx]
    
    print(f"Analyzing prices for typical day: {typical_date}")
    time_labels = [f"{i//4:02d}:{(i%4)*15:02d}" for i in range(96)]
    
    day_ahead_prices = DayAheadPrices()
    wholesale_prices = day_ahead_prices.get_day_ahead_prices(typical_date, as_series=False)
    
    ax.plot(time_labels, wholesale_prices, color='#2E86AB', linewidth=2, 
            marker='o', markersize=4, label='Day Ahead Price (Wholesale)', alpha=0.8)
    
    signal_prices = None
    if hasattr(results, 'dynamic_prices') and results['dynamic_prices'].iloc[typical_day_idx] is not None:
        signal_prices = results['dynamic_prices'].iloc[typical_day_idx]
    
    if signal_prices is not None:
        ax.plot(time_labels, signal_prices, color='#A23B72', linewidth=2, 
                marker='s', markersize=4, label='Signal Price (Dynamic)', alpha=0.8)
    
    ax.set_title('Price Analysis for Typical Day - Day Ahead vs Signal Prices (96 Quarterly Intervals)', pad=20)
    ax.set_ylabel('Price (EUR/kWh)')
    ax.set_xlabel('Time of Day')
    ax.grid(True, alpha=0.6)
    ax.legend(loc='upper right')
    ax.set_xticks(range(0, 96, 8), [time_labels[i] for i in range(0, 96, 8)], rotation=45)    
    plt.tight_layout()
    plt.savefig('simulation_results/price_analysis_typical_day.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("Price analysis plot saved to: simulation_results/price_analysis_typical_day.png")
    plt.close()


def main():
    
    print("Energy Community Model Simulation - Gridflex Heeten")
    print("="*50)
    
    try:
        cove_markup_factors = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
        for i in cove_markup_factors:
            model, results = run_basic_simulation(i)
        
            analyze_results(results)    
            plot_results(results)        
            
            print("\n" + "="*50)
            print("SIMULATION COMPLETED SUCCESSFULLY!")
            print("="*50)
            output_dir = f'COVE_factor/cove_markup_{i:.1f}'
            os.makedirs(output_dir, exist_ok=True)
            if os.path.exists('simulation_results'):
                for file in os.listdir('simulation_results'):
                    src = os.path.join('simulation_results', file)
                    dst = os.path.join(output_dir, file)
                    if os.path.isfile(src):
                        shutil.copy2(src, dst)
                    elif os.path.isdir(src):
                        shutil.copytree(src, dst, dirs_exist_ok=True)
            
            for file in os.listdir('simulation_results'):
                    src = os.path.join('simulation_results', file)
                    if os.path.isfile(src):
                        os.remove(src)
                    elif os.path.isdir(src):
                        shutil.rmtree(src)
        
        return model, results
        
    except Exception as e:
        print(f"\nError running simulation: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    model, results = main() 
