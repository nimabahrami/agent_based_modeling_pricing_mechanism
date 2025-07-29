import numpy as np
import pandas as pd

class DynamicPricingEngine:
    def __init__(self, wholesale_prices):

        self.wholesale_prices = wholesale_prices
        
    def calculate_base_retail_price(self, hour, demand, retail_params):
        wholesale_price = self.wholesale_prices
        self.retail = retail_params
        
        fixed_per_kwh = (
            (self.retail['transport_rate'] + self.retail['fixed_delivery'])
            / 9.6 
        )
        
        base_retail_price = (
            wholesale_price
            + self.retail['ode_tax']
            + self.retail['energy_tax']
            + self.retail['variable_delivery']
            + fixed_per_kwh
        )
        
        return base_retail_price
    
    def run_pricing(self, date, supply, demand, retail_params):
       
        base_retail_price = self.calculate_base_retail_price(date, demand, retail_params)
        
        imbalance = (demand - supply) / max(max(supply), 1)
        
        markup_factor = np.clip(
            imbalance * self.retail['max_markup'],
            self.retail['min_markup'],
            self.retail['max_markup']
        )
        
        dynamic_price = base_retail_price * (1 + markup_factor)
        
        return dynamic_price, base_retail_price

