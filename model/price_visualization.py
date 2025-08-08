import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_august_15_2021_prices():
    """Plot the price data for August 15, 2021 from the saved CSV file."""
    
    print("\nCreating August 15, 2021 price analysis plot...")
    
    # Set modern style for this plot
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
    
    # Check if the CSV file exists
    csv_file = 'simulation_results/august_15_2021_prices.csv'
    
    try:
        # Read the CSV file
        price_data = pd.read_csv(csv_file)
        print(f"Loaded price data with {len(price_data)} records")
        
        # Create the plot
        fig, ax = plt.subplots(1, 1, figsize=(15, 8))
        
        # Time labels for 96 quarterly intervals (15-minute intervals)
        time_labels = [f"{i//4:02d}:{(i%4)*15:02d}" for i in range(96)]
        
        # Plot price_signal vs wholesale_prices
        if 'price_signal' in price_data.columns and 'wholesale_prices' in price_data.columns:
            ax.plot(time_labels, price_data['price_signal'], color='#000080', linewidth=2, 
                    marker='o', markersize=4, label='Price Signal (Dynamic)', alpha=0.8)
            ax.plot(time_labels, price_data['wholesale_prices'], color='#E9967A', linewidth=2, 
                    marker='s', markersize=4, label='Base Retail Prices (Day Ahead)', alpha=0.8)
            
            # Set chart properties
            ax.set_title('August 15, 2021 - Price Signal vs Base Retail Prices (96 Quarterly Intervals)', pad=20)
            ax.set_ylabel('Price (EUR/kWh)')
            ax.set_xlabel('Time of Day')
            ax.grid(True, alpha=0.6)
            ax.legend(loc='upper right')
            ax.set_xticks(range(0, 96, 8), [time_labels[i] for i in range(0, 96, 8)], rotation=45)
            
            # Adjust layout and save
            plt.tight_layout()
            plt.savefig('simulation_results/august_15_2021_prices.png', dpi=300, bbox_inches='tight', facecolor='white')
            print("August 15, 2021 price analysis plot saved to: simulation_results/august_15_2021_prices.png")
            plt.close()
            
        else:
            print("Required columns 'price_signal' and 'wholesale_prices' not found in CSV file")
            print(f"Available columns: {list(price_data.columns)}")
            
    except Exception as e:
        print(f"Error plotting August 15, 2021 prices: {e}")
        import traceback
        traceback.print_exc()