"""
This module contains all agent classes.
"""

import datetime
import random
import math

import numpy as np
import pandas as pd
from model.electricity_auction import ElectricityDoubleAuction
from model.gap_predictor import GapPredictor
from model.day_ahead_prices import DayAheadPrices
from model.price_visualization import plot_august_15_2021_prices
from model.pricing_mechanism import DynamicPricingEngine
from model.COVE_calculator import prosumer_rolling_cove

pd.options.mode.chained_assignment = None  # default='warn'
from mesa import Agent

from model.enumerations import *

# random.seed(123)

# Read the model input data
import os
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(base_dir, "data/processed/model_input_data.csv"), 
                 parse_dates=['Local'], infer_datetime_format=True, index_col=0)
temperature_data = pd.read_csv(
    os.path.join(base_dir, "data/processed/temperature_data_2021.csv"),
    parse_dates=['period_end']
)
# Convert period_end to match the format of Local in cleaned_data.csv
temperature_data['Local'] = temperature_data['period_end'].dt.strftime('%Y-%m-%d %H:%M:%S')
temperature_data.set_index('Local', inplace=True)
temperature_data.index = pd.to_datetime(temperature_data.index, infer_datetime_format=True)
electricity_costs = pd.read_csv(os.path.join(base_dir, "data/processed/electricity_costs.csv"), index_col=1)
electricity_costs = electricity_costs.to_dict(orient='index')


class Coordinator(Agent):
    """This agent manages the energy community."""

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.agent_type = AgentType.COORDINATOR
        self.date = model.date
        self.date_index = pd.date_range(start=self.date, periods=96, freq='15min')
        self.total_energy_export = 0
        self.total_energy_import = 0
        self.predicted_generation_demand = None
        self.auction = ElectricityDoubleAuction()
        self.generation_demand_predictor = GapPredictor(sequence_length=96)
        self.dynamic_prices = None
        self.base_retail_price = None

    def step(self):
        self.update_date()
        self.date_index = pd.date_range(start=self.date, periods=96, freq='15min')
        gap_at_current_step = self.balance_supply_and_demand()
        
        self.predict_gap_for_tomorrow()
        self.calculate_dynamic_prices()
        
        if self.model.participation_in_tod is not None and self.model.participation_in_tod > 0:
            self.release_tod_schedule()
        
        self.auction.reset()
        
 

    def balance_supply_and_demand(self):
        agg_supply = pd.Series(0, index=self.date_index)
        agg_demand = pd.Series(0, index=self.date_index)
        for agent in self.model.schedule.agents:
            if agent.agent_type is AgentType.CONSUMER or agent.agent_type is AgentType.PROSUMER:
                agg_demand += agent.realised_demand
            if agent.agent_type is AgentType.PROSUMER:
                agg_supply += agent.excess_generation
        energy_export = (agg_supply - agg_demand).clip(lower=0)
        energy_import = (agg_demand - agg_supply).clip(lower=0)
        self.total_energy_export = energy_export.sum()
        self.total_energy_import = energy_import.sum()
        
        # Calculate the gap (negative means deficit, positive means surplus)
        gap = agg_supply - agg_demand
        return gap.values


    def predict_gap_for_tomorrow(self):
        
        try:
            previous_day_generation_demand_data = self.get_previous_day_generation_demand_data()
            self.predicted_generation_demand = self.generation_demand_predictor.predict(previous_day_generation_demand_data)
            
            # Calculate detailed prediction errors for each 15-minute interval
            try:
                # Get actual generation and demand for the current day (96 intervals)
                actual_generation = np.zeros(96)
                actual_demand = np.zeros(96)
                
                for agent in self.model.schedule.agents:
                    if agent.agent_type in [AgentType.CONSUMER, AgentType.PROSUMER]:
                        if hasattr(agent, 'realised_demand'):
                            actual_demand += agent.realised_demand.values
                    if agent.agent_type is AgentType.PROSUMER:
                        if hasattr(agent, 'excess_generation'):
                            actual_generation += agent.excess_generation.values
                
                # Calculate detailed prediction errors for each interval
                if self.predicted_generation_demand is not None:
                    predicted_generation = self.predicted_generation_demand[:, 0]  # Shape: (96,)
                    predicted_demand = self.predicted_generation_demand[:, 1]      # Shape: (96,)
                    
                    # Store detailed errors for each 15-minute interval
                    self.generation_prediction_error = np.abs(predicted_generation - actual_generation)  # Shape: (96,)
                    self.demand_prediction_error = np.abs(predicted_demand - actual_demand)              # Shape: (96,)
                else:
                    self.generation_prediction_error = np.zeros(96)
                    self.demand_prediction_error = np.zeros(96)
                    
            except Exception as e:
                print(f"Error calculating detailed prediction errors: {e}")
                self.generation_prediction_error = np.zeros(96)
                self.demand_prediction_error = np.zeros(96)
            
            return self.predicted_generation_demand
            
        except Exception as e:
            print(f"Error predicting gap for tomorrow: {e}")
            self.generation_prediction_error = np.zeros(96)
            self.demand_prediction_error = np.zeros(96)
            return None
        
    def get_previous_day_generation_demand_data(self):
        df = pd.read_csv('data/training/daily_gap_predictor_training_data.csv', parse_dates=['timestamp'])
        df = df.set_index('timestamp')
        prev_day = pd.to_datetime(self.date) - pd.Timedelta(days=1)
        day_data = df.loc[prev_day.strftime('%Y-%m-%d')]
        arr = day_data[['generation', 'demand']].to_numpy()
        return arr[:96] if arr.shape[0] >= 96 else np.pad(arr, ((0, 96 - arr.shape[0]), (0, 0)), mode='constant')


    def calculate_dynamic_prices(self):
        """Calculate dynamic prices based on predicted gap."""
        if self.predicted_generation_demand is None:
            return None
        
        tomorrow = (datetime.datetime.strptime(self.date, '%Y-%m-%d') + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
        tomorrow_index = pd.date_range(start=tomorrow, periods=96, freq='15min')
        
     
        # Get hourly day-ahead prices (already in €/kWh)
        day_ahead_prices = DayAheadPrices()
        wholesale_prices = day_ahead_prices.get_day_ahead_prices(self.date, as_series=False)
        # Create retail parameters
        month = datetime.datetime.strptime(tomorrow, '%Y-%m-%d').strftime('%B')
        retail_params = {
            'transport_rate': electricity_costs[month]['Electricity Transport rate (Euro/day)'],
            'fixed_delivery': electricity_costs[month]['Fixed delivery rate (Euro/day)'],
            'ode_tax': electricity_costs[month]['ODE tax (Environmental Taxes Act) (Euro/kWh)'],
            'energy_tax': electricity_costs[month]['Energy tax (Euro/kWh)'],
            'variable_delivery': electricity_costs[month]['Variable delivery rate (Euro/kWh)'],
            'min_markup': 0.05,  # 5% minimum markup
            'max_markup': 0.25   # 25% maximum markup
        }
        
        # Use the dynamic pricing engine
        dpe = DynamicPricingEngine(wholesale_prices=wholesale_prices)
      
        pricing_results,base_retail_price = dpe.run_pricing(date = self.date,
                                          supply = self.predicted_generation_demand[:,0].clip(min=0),
                                          demand = self.predicted_generation_demand[:,1].clip(min=0),
                                          retail_params=retail_params)
        
        self.dynamic_prices = pd.Series(pricing_results, index=tomorrow_index)
        self.base_retail_price = pd.Series(base_retail_price, index=tomorrow_index)
        
        return self.dynamic_prices

    def release_tod_schedule(self):
        tomorrow = (datetime.datetime.strptime(self.date, '%Y-%m-%d') + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
        index = pd.date_range(start=tomorrow, periods=96, freq='15min')
        
        day_ahead_demand = self.predicted_generation_demand[:,1].clip(min=0)
        day_ahead_supply = self.predicted_generation_demand[:,0].clip(min=0)
        
        # Use predicted gap and dynamic prices if available
        if self.predicted_generation_demand is not None and self.dynamic_prices is not None:
            # Create a price signal for agents based on dynamic prices
            self.model.price_signal = self.dynamic_prices
            self.model.base_retail_price = self.base_retail_price
            index = index.ravel()
            # Determine surplus and deficit timings based on predicted gap
            predicted_gap_series = pd.DataFrame(self.predicted_generation_demand, index=index, columns=["generation", "demand"])
            surplus_times = predicted_gap_series[self.predicted_generation_demand[:,0] - self.predicted_generation_demand[:,1] > 0].index
            deficit_times = predicted_gap_series[self.predicted_generation_demand[:,0] - self.predicted_generation_demand[:,1] < 0].index
            
            # Convert to hourly for ToD schedule
            hourly_index = pd.date_range(start=tomorrow, periods=24, freq='H')
            self.model.tod_surplus_timing = [t for t in hourly_index if any(st.hour == t.hour for st in surplus_times)]
            self.model.tod_deficit_timing = [t for t in hourly_index if any(dt.hour == t.hour for dt in deficit_times)]
        else:
            # Fall back to original method if prediction is not available
            index = pd.date_range(start=tomorrow, periods=24, freq='H')
            day_ahead_df = pd.DataFrame(index=index)
            day_ahead_df['surplus'] = day_ahead_supply.resample('H').sum()
            day_ahead_df['deficit'] = day_ahead_demand.resample('H').sum()
            
            supply_threshold = day_ahead_df['surplus'].quantile(0.7)
            demand_threshold = day_ahead_df['deficit'].quantile(0.7)
            self.model.tod_surplus_timing = day_ahead_df[day_ahead_df['surplus'] > supply_threshold].index.to_list()
            self.model.tod_deficit_timing = day_ahead_df[day_ahead_df['deficit'] > demand_threshold].index.to_list()

    def update_date(self):
        self.date = self.model.date
        return None
        
    def run_auction(self):
        # DEBUG: Print all bids and offers before clearing
        print("\n[DEBUG] --- Auction Bids and Offers ---")
        for agent in self.model.schedule.agents:
            if hasattr(agent, 'bid') and agent.bid is not None:
                print(f"[DEBUG] Agent {agent.unique_id} BID: price={agent.bid[0]}, quantity={agent.bid[1]}")
                self.auction.add_bid(agent.bid[0], agent.bid[1], agent.unique_id)
            if hasattr(agent, 'offer') and agent.offer is not None:
                print(f"[DEBUG] Agent {agent.unique_id} OFFER: price={agent.offer[0]}, quantity={agent.offer[1]}")
                self.auction.add_offer(agent.offer[0], agent.offer[1], agent.unique_id)
        
        # Clear the market
        clearing_price, clearing_quantity, matched_bids, matched_offers, unmatched_bids, unmatched_offers = self.auction.clear_market()
        print(f"[DEBUG] Auction clearing_price: {clearing_price}, clearing_quantity: {clearing_quantity}")
        print(f"[DEBUG] Matched bids: {matched_bids}")
        print(f"[DEBUG] Matched offers: {matched_offers}")
        print(f"[DEBUG] Unmatched bids: {unmatched_bids}")
        print(f"[DEBUG] Unmatched offers: {unmatched_offers}")
        
        # Update agent revenues
        for agent in self.model.schedule.agents:
            if agent.agent_type in [AgentType.CONSUMER, AgentType.PROSUMER]:
                if agent.agent_type == AgentType.CONSUMER:
                    # Consumers only have costs, no revenue from selling
                    agent.auction_revenue = 0  # No revenue from selling
                    agent.auction_cost = self.auction.calculate_cost(agent.unique_id)
                else:  # PROSUMER
                    # Prosumers only have revenue from selling energy
                    agent.auction_revenue = self.auction.calculate_revenue(agent.unique_id)
                    agent.auction_cost = 0  # No cost from buying in auction
                
                agent.matched_quantity = self.auction.get_matched_quantity(agent.unique_id)
                agent.unmatched_quantity = self.auction.get_unmatched_quantity(agent.unique_id)
        
        # DEBUG: Print auction results for each agent
        print("\n[DEBUG] --- Auction Results by Agent ---")
        for agent in self.model.schedule.agents:
            if agent.agent_type in [AgentType.CONSUMER, AgentType.PROSUMER]:
                print(f"[DEBUG] Agent {agent.unique_id} ({agent.agent_type}): "
                      f"Revenue={getattr(agent, 'auction_revenue', 0):.4f}, "
                      f"Cost={getattr(agent, 'auction_cost', 0):.4f}, "
                      f"Matched={getattr(agent, 'matched_quantity', 0):.4f}, "
                      f"Unmatched={getattr(agent, 'unmatched_quantity', 0):.4f}")
        
        return clearing_price, clearing_quantity


class Member(Agent):
    """An agent with fixed initial wealth."""

    def __init__(self, unique_id, model, member_name, agent_type, member_type, demand_flexibility, asset_list):
        """ Initialize agent and variables.
        param unique_id: int: unique identifier for the agent
        :param model: model: model in which the agent lives
        :param member_name: string: name of the member
        :param member_type: MemberType: type of the member (consumer, prosumer, asset)
        :param asset_list: list: list of assets the member owns. Leave empty if the member has no assets.
        """
        super().__init__(unique_id, model)
        self.member_name = member_name
        self.agent_type = agent_type
        self.member_type = member_type
        self.date = self.model.date
        self.date_index = pd.date_range(start=self.date, periods=96, freq='15min')
        self.load = 0
        self.demand_flexibility = demand_flexibility
        self.demand_flexibility_dynamic = 0  # Will be calculated based on COVE
        self.realised_demand = pd.Series(0, index=self.date_index)
        self.excess_generation = pd.Series(0, index=self.date_index)
        self.assets = []
        self.shifted_load = 0  # Load shifted by the member as complaince of ToD schedule
        self.savings_ToD = 0  # Savings in Euros by complying with demand response
        # If agent is a prosumer, initialise assets
        if asset_list is not None:
            self.initialise_asset(asset_list)
        else:
            self.assets = None

        self.scheduled_demand = pd.Series(0, index=self.date_index)
        self.generation_schedule = pd.Series(0, index=self.date_index)
        self.day_ahead_demand = pd.Series(0, index=self.date_index)
        self.day_ahead_supply = pd.Series(0, index=self.date_index)
        self.energy_cost = None
        self.earnings = None
        self.earning_original = None
        self.average_lcoe = self.compute_average_lcoe()
        
        # New attributes for bidding and auction
        self.bid = None  # For consumers
        self.offer = None  # For prosumers
        self.auction_revenue = 0
        self.matched_quantity = 0
        self.unmatched_quantity = 0
        
        # COVE calculation attributes
        self.cove_value = None
        self.cove_windowsize = random.randint(7, 21)  
        self.historical_generation = []
        self.historical_prices = []

    def step(self):
        super().step()
        self.update_date()
        self.get_demand_schedule()
        self.get_generation_schedule()
        self.generate_day_ahead_schedules()
        self.adjust_schedule_for_captive_consumption()
        self.get_previous_days_generation_schedule()
        self.get_previous_days_price_schedule()
        
        # Store historical generation data for COVE calculation
        if self.agent_type is AgentType.PROSUMER:
            self.historical_generation.append(self.generation_history.values)
            if len(self.historical_generation) > self.cove_windowsize:
                self.historical_generation.pop(0)
        
        self.calculate_cove_and_flexibility()
        self.prepare_auction_bid_offer()
        self.adjust_schedule_for_tod()

    def initialise_asset(self, asset_list):
        item = None
        for asset in asset_list:
            if asset['asset_type'] is Solar:
                item = Solar(unique_id=self.model.next_id(), model=self.model,
                             capacity=asset['capacity'], capex=asset['price'], efficiency=asset['efficiency'],
                             owner=self)
            elif asset['asset_type'] is Wind:
                item = Wind(unique_id=self.model.next_id(), model=self.model, capacity=asset['capacity'], owner=self)
            self.assets.append(item)
            self.model.schedule.add(item)
            if asset['asset_type'] in self.model.all_assets:
                self.model.all_assets[asset['asset_type']].append(item)
            else:
                self.model.all_assets[asset['asset_type']] = [item]

    def update_date(self):
        self.date = self.model.date
        return None

    def get_generation_schedule(self):
       
        index = pd.date_range(start=self.date, periods=96, freq='15min')
        generation_schedule = pd.Series(index=index, data=0)
        if self.agent_type is AgentType.PROSUMER:
            for asset in self.assets:
                generation_schedule += asset.generate_supply_schedule()
        else:
            pass
        self.generation_schedule = generation_schedule
        return None
    
    def get_previous_days_generation_schedule(self):
        
        day_window = self.cove_windowsize
        quarters = day_window * 96
        start = (datetime.datetime.strptime(self.date, '%Y-%m-%d') - 
                 datetime.timedelta(days=day_window)).strftime('%Y-%m-%d')
        index = pd.date_range(start=start, periods=quarters, freq='15min')
        generation_for_previous_days = pd.Series(index=index, data=0)
        if self.agent_type is AgentType.PROSUMER:
            for asset in self.assets:
                generation_for_previous_days += asset.generate_supply_schedule_for_range(index)
        else:
            pass
        self.generation_history = generation_for_previous_days
        return None
    
    def get_previous_days_price_schedule(self):
        day_window = self.cove_windowsize
        quarters = day_window * 96
        start = (datetime.datetime.strptime(self.date, '%Y-%m-%d') - 
                 datetime.timedelta(days=day_window)).strftime('%Y-%m-%d')
        index = pd.date_range(start=start, periods=quarters, freq='15min')
        
        # Get day-ahead prices for the date range
        day_ahead_prices = DayAheadPrices()
        price_schedule = day_ahead_prices.get_day_ahead_prices_for_range(index)
        
        # Store the price history for COVE calculation
        self.price_history = price_schedule
        return None

    def get_demand_schedule(self):
        self.scheduled_demand = df.loc[self.date, self.member_name]
        return None

    def generate_day_ahead_schedules(self):
        
        tomorrow = (datetime.datetime.strptime(self.date, '%Y-%m-%d') + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
        demand = df.loc[tomorrow, self.member_name]
        index = pd.date_range(start=tomorrow, periods=96, freq='15min')
        generation = pd.Series(index=index, data=0)
        if self.agent_type is AgentType.PROSUMER:
            for asset in self.assets:
                generation += asset.day_ahead_supply_schedule()

        self.day_ahead_demand = (demand - generation).clip(lower=0)
        self.day_ahead_supply = (generation - demand).clip(lower=0)
        return None

    def adjust_schedule_for_captive_consumption(self):
        """Modifies the demand schedule for an agent based on captive generation"""
        # Adjusting self consumption from the demand schedule and generation schedule
        if self.agent_type is AgentType.PROSUMER:
            self.realised_demand = (self.scheduled_demand - self.generation_schedule).clip(lower=0)
            # Updating the generation schedule based on captive consumption
            self.excess_generation = (self.generation_schedule - self.scheduled_demand).clip(lower=0)
        else:
            self.realised_demand = self.scheduled_demand

    def calculate_cove_and_flexibility(self):
        
        if self.agent_type is AgentType.PROSUMER:
            # Use the most recent data up to the window size
            generation_data = np.array(self.generation_history.values)
            price_data = np.array(self.price_history.values)
            # Flatten the data if needed
            if len(generation_data.shape) > 1:
                generation_data = generation_data.flatten()
            if len(price_data.shape) > 1:
                price_data = price_data.flatten()
            self.cove_value = prosumer_rolling_cove(self.average_lcoe, price_data, generation_data)
            if not isinstance(self.cove_value, (int, float)) or self.cove_value is False:
                self.cove_value = self.average_lcoe * 1.2  # Default to 20% above LCOE
        else:
            self.cove_value = 0
        
        # Calculate dynamic flexibility based on price signal and COVE
        if hasattr(self.model, 'price_signal') and self.model.price_signal is not None and self.cove_value > 0:
            # Calculate the gap between COVE and price signal
            price_signal_avg = self.model.price_signal.mean()
            price_gap = abs(self.cove_value - price_signal_avg)
            
            self.demand_flexibility_dynamic = 1 - np.exp(-price_gap)
            self.demand_flexibility_dynamic = np.clip(self.demand_flexibility_dynamic, 0, 0.5)
        else:
            # Default to static flexibility if no price signal or COVE
            self.demand_flexibility_dynamic = self.demand_flexibility

    def prepare_auction_bid_offer(self):
       
        # Reset previous bid/offer
        self.bid = None
        self.offer = None
        
        day_ahead_prices = DayAheadPrices()
        tomorrow = (datetime.datetime.strptime(self.date, '%Y-%m-%d') + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
        wholesale_prices = day_ahead_prices.get_day_ahead_prices(tomorrow, as_series=False)
        base_retail = self.model.base_retail_price
        average_signal = float(self.model.price_signal.mean())
        
        # Save price_signal and wholesale_prices to CSV if date is August 15, 2021
        if self.date == '2021-08-20':

            time_labels = [f"{i//4:02d}:{(i%4)*15:02d}" for i in range(96)]
                

            price_signal_full = self.model.price_signal.values if hasattr(self.model.price_signal, 'values') else self.model.price_signal
                
                # Create DataFrame with time, wholesale_prices, and price_signal
            price_data = pd.DataFrame({
                    'time': time_labels,
                    'wholesale_prices': base_retail,
                    'price_signal': price_signal_full
                })

            output_file = 'simulation_results/august_15_2021_prices.csv'
            os.makedirs('simulation_results', exist_ok=True)
            price_data.to_csv(output_file, index=False)
            plot_august_15_2021_prices()
                
        
        if self.agent_type is AgentType.PROSUMER and self.excess_generation.sum() > 0:
            # Prosumers offer at a price between LCOE and COVE
            offer_price = random.uniform(self.cove_value, average_signal)
              # 5-20% above LCOE
            
            # Offer the excess generation
            offer_quantity = self.excess_generation.sum()
            self.offer = (float(offer_price), float(offer_quantity))
        
        if self.agent_type in [AgentType.CONSUMER] and self.realised_demand.sum() > 0:
            
            bid_price = average_signal * random.uniform(0.8, 0.95)
            
            # Bid for the demand
            bid_quantity = self.realised_demand.sum()
            self.bid = (float(bid_price), float(bid_quantity))


    def adjust_schedule_for_tod(self):
        # Update the demand schedule based on demand response using dynamic flexibility
        # Only adjust if the agent's bid or offer was matched in the auction
        increased_consumption = 0
        reduced_consumption = 0
        
        # Check if agent had a matched bid or offer
        matched_quantity = getattr(self, 'matched_quantity', 0)
        has_matched_trade = False
        
        if self.agent_type is AgentType.PROSUMER:
            # For prosumers: matched_quantity > 0 means matched offer
            has_matched_trade = matched_quantity > 0
        elif self.agent_type is AgentType.CONSUMER:
            # For consumers: matched_quantity < 0 means matched bid (negative value)
            has_matched_trade = matched_quantity < 0
        
        # Only proceed with ToD adjustment if agent had a matched trade
        if has_matched_trade and random.uniform(0, 1) <= self.model.participation_in_tod:
            if self.model.tod_surplus_timing is not None:
                surplus_times = [t for t in self.model.tod_surplus_timing if t in self.realised_demand.index]
                if surplus_times:
                    updated_schedule = self.realised_demand.loc[surplus_times] * (
                            1 + self.demand_flexibility_dynamic)
                    
                    increased_consumption = min(updated_schedule.sum(), matched_quantity) - self.realised_demand.loc[surplus_times].sum()
                    increased_consumption = abs(increased_consumption)
                    self.realised_demand.loc[surplus_times] = updated_schedule
            
            if self.model.tod_deficit_timing is not None:
                deficit_times = [t for t in self.model.tod_deficit_timing if t in self.realised_demand.index]
                if deficit_times:
                    updated_schedule = self.realised_demand.loc[deficit_times] * (
                            1 - self.demand_flexibility_dynamic)
                           
                    reduced_consumption = self.realised_demand.loc[deficit_times].sum() - min(updated_schedule.sum(), matched_quantity)
                    reduced_consumption = abs(reduced_consumption)
                    self.realised_demand.loc[deficit_times] = updated_schedule
            
            self.shifted_load = max(increased_consumption, reduced_consumption)
        else:
            # No adjustment if no matched trade
            self.shifted_load = 0

    def compute_average_lcoe(self):
        """Computes the average LCOE for a member"""
        if self.agent_type == AgentType.PROSUMER:
            lcoe = 0
            for asset in self.assets:
                lcoe += asset.lcoe
            average_lcoe = lcoe / len(self.assets)
        else:
            average_lcoe = 0
        return average_lcoe

    def get_retail_params(self):
        """Get retail parameters for the current date."""
        month = datetime.datetime.strptime(self.date, '%Y-%m-%d').strftime('%B')
        return {
            'transport_rate': electricity_costs[month]['Electricity Transport rate (Euro/day)'],
            'fixed_delivery': electricity_costs[month]['Fixed delivery rate (Euro/day)'],
            'ode_tax': electricity_costs[month]['ODE tax (Environmental Taxes Act) (Euro/kWh)'],
            'energy_tax': electricity_costs[month]['Energy tax (Euro/kWh)'],
            'variable_delivery': electricity_costs[month]['Variable delivery rate (Euro/kWh)'],
            'min_markup': 0.05,  # 5% minimum markup
            'max_markup': 0.25   # 25% maximum markup
        }

    def compute_savings_ToD(self):
        """Computes savings from shifted load based on auction results."""
        coordinator = self.model.get_coordinator()
        auction = coordinator.auction if coordinator else None
        clearing_price = auction.clearing_price if auction else None
        
        # Get retail params
        retail_params = self.get_retail_params()
        
        # Calculate base retail price
        day_ahead_prices = DayAheadPrices()
        wholesale_prices = day_ahead_prices.get_day_ahead_prices(self.date, as_series=False)
        dpe = DynamicPricingEngine(wholesale_prices=wholesale_prices)
        base_retail_price = dpe.calculate_base_retail_price(0, self.realised_demand.sum(), retail_params)
        
        # Calculate savings based on agent type and auction results
        if self.agent_type is AgentType.PROSUMER:
            # # Prosumers: if offer was matched, use clearing price; else use COVE
            # if clearing_price is not None and hasattr(self, 'offer') and self.offer is not None:
            #     # Check if this agent had a matched offer
            #     matched_quantity = getattr(self, 'matched_quantity', 0)
            #     if matched_quantity > 0:  # Had matched offer
            #         savings_price = base_retail_price - clearing_price - self.cove_value
            #     else:  # No matched offer, use COVE
            #         savings_price =  base_retail_price - self.cove_value
            # else:
            #     savings_price =  base_retail_price - self.cove_value
            pass
                
        elif self.agent_type is AgentType.CONSUMER:
            # Consumers: if bid was matched, use clearing price; else use base retail price
            if clearing_price is not None and hasattr(self, 'bid') and self.bid is not None:
                # Check if this agent had a matched bid
                matched_quantity = getattr(self, 'matched_quantity', 0)
                if matched_quantity < 0:  # Had matched bid (negative for consumers)
                    savings_price = base_retail_price - clearing_price 
                else:  # No matched bid, use base retail price
                    savings_price = 0
            else:
                savings_price = 0
        else:
            savings_price = 0
        
        # Calculate savings from shifted load
        self.savings_ToD = self.shifted_load * savings_price

    def compute_energy_cost(self):
        """Computes the energy cost for a member based on auction results and base retail price."""
        coordinator = self.model.get_coordinator()
        auction = coordinator.auction if coordinator else None
        clearing_price = auction.clearing_price if auction else None
        matched_quantity = getattr(self, 'matched_quantity', 0)
        unmatched_quantity = getattr(self, 'unmatched_quantity', 0)

        # Get retail params
        retail_params = self.get_retail_params()

        # Calculate base retail price
        day_ahead_prices = DayAheadPrices()
        wholesale_prices = day_ahead_prices.get_day_ahead_prices(self.date, as_series=False)
        dpe = DynamicPricingEngine(wholesale_prices=wholesale_prices)
        base_retail_price = dpe.calculate_base_retail_price(0, self.realised_demand.sum(), retail_params)

        if self.agent_type is AgentType.CONSUMER:
            # Consumers: cost = matched_quantity * clearing_price + unmatched_quantity * base_retail price
            # matched_quantity is negative for consumers, unmatched_quantity is negative for consumers
            matched_cost = abs(matched_quantity) * clearing_price if clearing_price is not None else 0
            unmatched_cost = abs(unmatched_quantity) * base_retail_price  # unmatched_quantity is negative, so abs() gives positive cost
            self.energy_cost = matched_cost + unmatched_cost
            
        elif self.agent_type is AgentType.PROSUMER:
            # Prosumers: opportunity cost = unmatched_quantity * COVE (lost revenue from unmatched offers)
            # unmatched_quantity is positive for prosumers (unmatched offers)
            if unmatched_quantity > 0:
                opportunity_cost = unmatched_quantity * (self.cove_value if self.cove_value else base_retail_price)
            else:
                opportunity_cost = 0
            self.energy_cost = opportunity_cost
        else:
            self.energy_cost = 0

        # Calculate savings from shifted load
        self.compute_savings_ToD()

    # def compute_earnings(self):
    #     """Computes the earnings for a member based on auction results and base retail price."""
    #     coordinator = self.model.get_coordinator()
    #     auction = coordinator.auction if coordinator else None
    #     clearing_price = auction.clearing_price if auction else None
    #     matched_quantity = getattr(self, 'matched_quantity', 0)
    #     unmatched_quantity = getattr(self, 'unmatched_quantity', 0)

    #     # Get retail params
    #     retail_params = self.get_retail_params()

    #     # Calculate base retail price
    #     day_ahead_prices = DayAheadPrices()
    #     wholesale_prices = day_ahead_prices.get_day_ahead_prices(self.date, as_series=False)
    #     dpe = DynamicPricingEngine(wholesale_prices=wholesale_prices)
    #     base_retail_price = dpe.calculate_base_retail_price(0, self.excess_generation.sum(), retail_params)

    #     # For prosumers: earnings = matched_quantity * clearing_price + unmatched_quantity * base_retail price
    #     if self.agent_type is AgentType.PROSUMER:
    #         matched_earnings = abs(matched_quantity) * clearing_price if clearing_price is not None else 0
    #         unmatched_earnings = abs(unmatched_quantity) * base_retail_price
    #         self.earnings = matched_earnings + unmatched_earnings
    #     else:
    #         self.earnings = 0

    #     # Calculate savings from shifted load
    #     self.compute_savings_ToD()
        
    def compute_earnings(self):
        self.earnings = 0
        if self.agent_type is AgentType.PROSUMER:
            self.earnings = self.excess_generation.sum() * self.average_lcoe
        else:
            pass
        



class Asset(Agent):
    """An asset of the energy community."""

    def __init__(self, unique_id, model, capacity, efficiency, owner, asset_age,
                 estimated_lifetime_generation, capex, opex, discount_rate=0.055):
        super().__init__(unique_id, model)
        self.date = self.model.date
        self.agent_type = AgentType.ASSET
        self.owner = owner
        self.efficiency = efficiency
        self.capacity = capacity
        self.asset_age = asset_age
        self.estimated_lifetime_generation = estimated_lifetime_generation
        if self.estimated_lifetime_generation == 0:
            self.estimate_lifetime_generation()
        self.supply_schedule = None
        self.day_ahead_schedule = None
        self.discount_rate = discount_rate
        self.capex, self.opex = self.compute_capex_and_opex()
        self.lcoe = None
        self.compute_lcoe()

    def step(self):
        super().step()
        self.update_date()
        self.supply_schedule = self.generate_supply_schedule()
        self.day_ahead_schedule = self.day_ahead_supply_schedule()
        pass
    
    def update_date(self):
        self.date = self.model.date
        return None

    def generate_supply_schedule(self):
        """This function generates a supply schedule for the asset"""
        pass

    def generate_supply_schedule_for_range(self, schedule_index):
        """This function generates a supply schedule for the asset for a given range"""
        return pd.Series(data=0.0, index=schedule_index)

    def day_ahead_supply_schedule(self):
        pass

    def estimate_lifetime_generation(self):
        """Estimates lifetime generation/supply of the asset"""
        lifespan = 25 - self.asset_age
        hours_per_year = 365 * 24
        if self.asset_type is AssetType.SOLAR:
            capacity_factor = 9.138 / 100
        elif self.asset_type is AssetType.WIND:
            capacity_factor = 27.89 / 100

        annual_generation_kWh = hours_per_year * capacity_factor 
        self.estimated_lifetime_generation = lifespan * annual_generation_kWh * self.capacity

    def compute_lcoe(self):
        """Calculates LCOE of the solar or wind plant"""
        lifespan = 25 - self.asset_age  # Typical lifespan of 25 years
        
        # Calculate present value of all costs over the lifespan
        total_costs = self.capex  # Initial investment
        
        # Add present value of annual O&M costs over the lifespan
        for year in range(1, lifespan + 1):
            total_costs += self.opex / ((1 + self.discount_rate) ** year)
        
        # Calculate total lifetime electricity generation
        # For simplicity, we assume constant annual generation
        annual_generation = self.estimated_lifetime_generation / lifespan
        
        # Calculate present value of electricity generation
        total_generation = 0
        for year in range(1, lifespan + 1):
            total_generation += annual_generation / ((1 + self.discount_rate) ** year)
        
        # Calculate LCOE
        if total_generation > 0:
            self.lcoe = total_costs / total_generation
        else:
            self.lcoe = 0.15  # Default value if generation is zero (€0.15/kWh)

    def compute_capex_and_opex(self):
        if self.asset_type is AssetType.SOLAR:
            capex = 1390 * self.capacity # capacity is in kW
            opex = capex * 0.02  # 2% of CAPEX per year for maintenance
        elif self.asset_type is AssetType.WIND:
            capex = 2460000 * self.number_of_turbines
            opex = capex * 0.03  # 3% of CAPEX per year for maintenance
        return capex, opex


class Solar(Asset):
    """A solar asset of the energy community."""

    def __init__(self, unique_id, model, capacity=0, efficiency=0, owner=None, asset_age=0,
                 estimated_lifetime_generation=0, capex=0, opex=0, discount_rate=0.055):
        self.asset_type = AssetType.SOLAR
        super(Solar, self).__init__(unique_id, model, capacity, efficiency, owner, asset_age,
                                    estimated_lifetime_generation, capex, opex, discount_rate)

    def generate_supply_schedule(self):
        """ Generates a schedule for the solar asset based on the capacity and efficiency of the solar panel"""
        super().generate_supply_schedule()
        supply_schedule = self.capacity * self.efficiency * df.loc[self.date, 'Direct [W/m^2]'] / 1000000
        return supply_schedule

    def day_ahead_supply_schedule(self):
        """ Generates a schedule for the solar asset based on the capacity and efficiency of the solar panel"""
        super().day_ahead_supply_schedule()
        tomorrow = (datetime.datetime.strptime(self.date, '%Y-%m-%d') + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
        # supply_schedule = self.capacity * self.efficiency * df.loc[tomorrow, 'Direct [W/m^2]'] / 1000000
        T_cell = temperature_data.loc[tomorrow, 'air_temp']+273  +  df.loc[tomorrow, 'Direct [W/m^2]']/1000 * ((55-20)/0.800)
        supply_schedule =  self.capacity * (df.loc[tomorrow, 'Direct [W/m^2]']/1000)*(1 - 0.005*(T_cell - 293))
        return supply_schedule

    def generate_supply_schedule_for_range(self, schedule_index):
        """ Generates a schedule for the solar asset for a given date range index. """
        supply_schedule = self.capacity * self.efficiency * df.loc[schedule_index, 'Direct [W/m^2]'] / 1000000
        return supply_schedule


class Wind(Asset):
    """A wind asset of the energy community."""

    def __init__(self, unique_id, model, capacity=0, efficiency=0, owner=None, asset_age=1,
                 estimated_lifetime_generation=0, capex=0, opex=0, discount_rate=0.055, number_of_turbines=1,
                 rotor_diameter=125, avg_air_density=1.23):
        self.asset_type = AssetType.WIND
        self.number_of_turbines = number_of_turbines
        self.rotor_diameter = rotor_diameter
        self.avg_air_density = avg_air_density
        self.efficiency = 59.3 / 100  # Betz limit
        self.swept_area = math.pi * (self.rotor_diameter / 2) ** 2
        super(Wind, self).__init__(unique_id, model, capacity, efficiency, owner, asset_age,
                                   estimated_lifetime_generation, capex, opex, discount_rate)

    def generate_supply_schedule(self):
        """ Generates a schedule for the wind asset based on the capacity and efficiency of the wind turbine"""
        super().generate_supply_schedule()
        wind_speed = df.loc[self.date, 'Wind [m/s]']
        wind_speed[wind_speed > 30] = 0  # Wind turbine shuts down if wind speed is greater than 30 m/s
        supply_schedule = 0.5 * self.avg_air_density * self.swept_area * np.power(wind_speed,
                                                                                  3) * self.efficiency * self.number_of_turbines
        return supply_schedule

    def day_ahead_supply_schedule(self):
        """ Generates a schedule for the wind asset based on the capacity and efficiency of the wind turbine"""
        super().day_ahead_supply_schedule()
        tomorrow = (datetime.datetime.strptime(self.date, '%Y-%m-%d') + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
        wind_speed = df.loc[tomorrow, 'Direct [W/m^2]']
        wind_speed[wind_speed > 30] = 0
        supply_schedule = 0.5 * self.avg_air_density * self.swept_area * np.power(wind_speed,
                                                                                  3) * self.efficiency * self.number_of_turbines
        return supply_schedule

    def generate_supply_schedule_for_range(self, schedule_index):
        """ Generates a schedule for the wind asset for a given date range index. """
        supply_schedule = 0.5 * self.avg_air_density * self.swept_area * np.power(df.loc[schedule_index, 'Wind [m/s]'],
                                                                                  3) * self.efficiency * self.number_of_turbines
        return supply_schedule
