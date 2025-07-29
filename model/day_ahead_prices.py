"""
Process and provide day-ahead electricity prices for the model.
"""

import os
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from functools import lru_cache

class DayAheadPrices:
    """
    A class to process and provide day-ahead electricity prices.
    """
    
    _cache = {}

    def __init__(self, price_file=None):
        
        if price_file is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            price_file = os.path.join(base_dir, "data/processed/Day-ahead Prices_2021_NL.csv")
        
        self.price_data = self.load_and_process_prices(price_file)
        
    @staticmethod
    @lru_cache(maxsize=4)
    def load_and_process_prices(price_file):
        if price_file in DayAheadPrices._cache:
            return DayAheadPrices._cache[price_file]
        df = pd.read_csv(price_file, skiprows=0, encoding='utf-8')
        df.columns = ['MTU', 'Price', 'Currency', 'Zone']
        df['Start'] = df['MTU'].str.split(' - ').str[0]
        df['Start'] = pd.to_datetime(df['Start'], format='%d.%m.%Y %H:%M')
        df['Price'] = df['Price'].astype(float) / 1000
        all_dates = []
        all_prices = []
        for _, row in df.iterrows():
            start_time = row['Start']
            price = row['Price']
            for i in range(4):
                interval_time = start_time + datetime.timedelta(minutes=15*i)
                all_dates.append(interval_time)
                all_prices.append(price)
        price_data = pd.DataFrame({
            'datetime': all_dates,
            'price': all_prices
        })
        price_data.set_index('datetime', inplace=True)
        DayAheadPrices._cache[price_file] = price_data
        return price_data
    
    def get_day_ahead_prices(self, date, as_series=False):
        if isinstance(date, str):
            date = pd.to_datetime(date)
        next_day = date + datetime.timedelta(days=1)
        next_day_str = next_day.strftime('%Y-%m-%d')
        day_prices = self.price_data.loc[next_day_str:next_day_str + ' 23:59:59']
        if as_series:
            return day_prices
        else:
            return day_prices['price'].values
    
    def get_day_ahead_prices_for_range(self, schedule_index):
        """Get day-ahead prices for a specific date range index."""
        # Filter the price data to match the schedule index
        price_series = self.price_data.loc[schedule_index, 'price']
        return price_series
    
    def save_prices_to_csv(self, date, output_file=None):
        if output_file is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            output_file = os.path.join(base_dir, "day_ahead_prices.csv")
        
        prices_series = self.get_day_ahead_prices(date, as_series=True)
        prices_series.to_csv(output_file)
        return output_file
    
    def plot_prices(self, date, save_plot=True, output_file=None):
        if output_file is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            output_file = os.path.join(base_dir, "day_ahead_prices.png")
        
        prices_series = self.get_day_ahead_prices(date, as_series=True)
        
        plt.figure(figsize=(12, 6))
        plt.plot(prices_series.index, prices_series['price'])
        plt.xlabel('Time')
        plt.ylabel('Price (EUR/kWh)')
        plt.title(f'Day-ahead Prices for {date}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_plot:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
        
        plt.show()
        return output_file
    


# Example usage
if __name__ == "__main__":
    # Initialize the DayAheadPrices class
    dap = DayAheadPrices()
    
    # Get day-ahead prices for a specific date
    date = "2021-01-01"
    prices = dap.get_day_ahead_prices(date)
   
    
    print(f"Day-ahead prices for {date} (15-minute intervals):")
    print(prices[:8])
    
    # Save to CSV
    csv_file = dap.save_prices_to_csv(date)
    print(f"Prices saved to: {csv_file}")
    
    # Create and save plot
    plot_file = dap.plot_prices(date)
    print(f"Plot saved to: {plot_file}")
    