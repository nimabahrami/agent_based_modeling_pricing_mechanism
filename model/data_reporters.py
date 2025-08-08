"""This file contains the DataReporter class."""

import json
import numpy as np
import datetime
import os
import pandas as pd

# Custom JSON encoder to handle numpy arrays
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super(NumpyEncoder, self).default(obj)

from model.agents import *
from model.day_ahead_prices import DayAheadPrices
from model.pricing_mechanism import DynamicPricingEngine

# Load electricity costs data
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
electricity_costs = pd.read_csv(os.path.join(base_dir, "data/processed/electricity_costs.csv"), index_col=1)
electricity_costs = electricity_costs.to_dict(orient='index')

def get_realised_demand(self):
    
    members = [AgentType.CONSUMER, AgentType.PROSUMER]
    demand_dict = {}
    for agent in self.schedule.agents:
        if agent.agent_type in members:
            key = str(agent.member_name) + str('_') + str(agent.unique_id)
            demand_dict[key] = agent.realised_demand.sum()
    return demand_dict


def get_scheduled_demand(self):
    
    members = [AgentType.CONSUMER, AgentType.PROSUMER]
    demand_dict = {}
    for agent in self.schedule.agents:
        if agent.agent_type in members:
            key = str(agent.member_name) + str('_') + str(agent.unique_id)
            demand_dict[key] = agent.scheduled_demand.sum(min_count=1)
    return demand_dict


def get_shifted_load(self):
    
    members = [AgentType.CONSUMER, AgentType.PROSUMER]
    shifted_load = {}
    for agent in self.schedule.agents:
        if agent.agent_type in members:
            key = str(agent.member_name) + str('_') + str(agent.unique_id)
            shifted_load[key] = agent.shifted_load
    return json.dumps(shifted_load, cls=NumpyEncoder)


def get_generation(self):
    generation_dict = {}
    for asset_category in self.all_assets.keys():
        supply = 0
        for asset in self.all_assets[asset_category]:
            if getattr(asset, 'supply_schedule', None) is None:
                asset.supply_schedule = asset.generate_supply_schedule()
            supply += asset.supply_schedule.sum()
        generation_dict[str(asset_category)] = supply
    return json.dumps(generation_dict, cls=NumpyEncoder)


def get_savings(self):
    
    savings_dict = {}
    members = [AgentType.CONSUMER, AgentType.PROSUMER]
    for agent in self.schedule.agents:
        if agent.agent_type in members:
            key = str(agent.member_name) + str('_') + str(agent.unique_id)
            savings_dict[key] = agent.savings_ToD
    return json.dumps(savings_dict, cls=NumpyEncoder)


def get_energy_cost(self):
    """
    Returns energy cost for community member for importing electricity from the grid.
    :param self:
    :return:  a dict of energy cost for member per timestep
    """
    costs_dict = {}
    members = [AgentType.CONSUMER, AgentType.PROSUMER]
    for agent in self.schedule.agents:
        if agent.agent_type in members:
            key = str(agent.member_name) + str('_') + str(agent.unique_id)
            costs_dict[key] = agent.energy_cost
    return json.dumps(costs_dict, cls=NumpyEncoder)


def get_date(self):
    """"
    Returns the date for the model simulation
    """
    return self.date


def get_auction_revenue(self):
    """
    Returns the revenue from the electricity auction for each agent.
    For consumers: always 0 (they don't sell energy)
    For prosumers: revenue from selling energy
    :return: a dict of auction revenue per agent
    """
    revenue_dict = {}
    members = [AgentType.CONSUMER, AgentType.PROSUMER]
    for agent in self.schedule.agents:
        if agent.agent_type in members:
            key = str(agent.member_name) + str('_') + str(agent.unique_id)
            if hasattr(agent, 'auction_revenue'):
                revenue_dict[key] = agent.auction_revenue
            else:
                revenue_dict[key] = 0
    return json.dumps(revenue_dict, cls=NumpyEncoder)


def get_earnings(self):
    """
    Returns the revenue from the electricity auction for each agent.
    For consumers: always 0 (they don't sell energy)
    For prosumers: revenue from selling energy
    :return: a dict of auction revenue per agent
    """
    earnings_dict = {}
    members = [AgentType.CONSUMER, AgentType.PROSUMER]
    for agent in self.schedule.agents:
        if agent.agent_type in members:
            key = str(agent.member_name) + str('_') + str(agent.unique_id)
            if hasattr(agent, 'earnings'):
                earnings_dict[key] = agent.earnings
            else:
                earnings_dict[key] = 0
    return json.dumps(earnings_dict, cls=NumpyEncoder)


def get_auction_cost(self):
    """
    Returns the cost from the electricity auction for each agent.
    For consumers: cost of buying energy
    For prosumers: always 0 (they don't buy in auction, only sell)
    :return: a dict of auction cost per agent
    """
    cost_dict = {}
    members = [AgentType.CONSUMER, AgentType.PROSUMER]
    for agent in self.schedule.agents:
        if agent.agent_type in members:
            key = str(agent.member_name) + str('_') + str(agent.unique_id)
            if hasattr(agent, 'auction_cost'):
                cost_dict[key] = agent.auction_cost
            else:
                cost_dict[key] = 0
    return json.dumps(cost_dict, cls=NumpyEncoder)


def get_dynamic_flexibility(self):
    """
    Returns the dynamic flexibility of each agent.
    :return: a dict of dynamic flexibility per agent
    """
    flexibility_dict = {}
    members = [AgentType.CONSUMER, AgentType.PROSUMER]
    for agent in self.schedule.agents:
        if agent.agent_type in members:
            key = str(agent.member_name) + str('_') + str(agent.unique_id)
            if hasattr(agent, 'demand_flexibility_dynamic'):
                flexibility_dict[key] = agent.demand_flexibility_dynamic
            else:
                flexibility_dict[key] = agent.demand_flexibility
    return json.dumps(flexibility_dict, cls=NumpyEncoder)


def get_cove_values(self):
    """
    Returns the COVE values of prosumer agents.
    :return: a dict of COVE values per prosumer
    """
    cove_dict = {}
    for agent in self.schedule.agents:
        if agent.agent_type is AgentType.PROSUMER:
            key = str(agent.member_name) + str('_') + str(agent.unique_id)
            if hasattr(agent, 'cove_value') and agent.cove_value is not None:
                cove_dict[key] = agent.cove_value
            else:
                cove_dict[key] = 0
    return json.dumps(cove_dict, cls=NumpyEncoder)


def get_lcoe_values(self):
    """
    Returns the LCOE values of prosumer agents.
    :return: a dict of LCOE values per prosumer
    """
    lcoe_dict = {}
    for agent in self.schedule.agents:
        if agent.agent_type is AgentType.PROSUMER:
            key = str(agent.member_name) + str('_') + str(agent.unique_id)
            if hasattr(agent, 'average_lcoe') and agent.average_lcoe is not None:
                lcoe_dict[key] = agent.average_lcoe
            else:
                lcoe_dict[key] = 0
    return json.dumps(lcoe_dict, cls=NumpyEncoder)


def get_clearing_price(self):
    """
    Returns the clearing price from the auction.
    :return: clearing price or None if no auction
    """
    coordinator = self.get_coordinator()
    if coordinator and hasattr(coordinator, 'auction'):
        return coordinator.auction.clearing_price
    return None


def get_average_base_retail_price(self):
    """
    Returns the average base retail price for the current day.
    :return: average base retail price
    """
    try:
        # Get the current date
        current_date = self.date
        
        # Get retail params for the day
        month = datetime.datetime.strptime(current_date, '%Y-%m-%d').strftime('%B')
        retail_params = {
            'transport_rate': electricity_costs[month]['Electricity Transport rate (Euro/day)'],
            'fixed_delivery': electricity_costs[month]['Fixed delivery rate (Euro/day)'],
            'ode_tax': electricity_costs[month]['ODE tax (Environmental Taxes Act) (Euro/kWh)'],
            'energy_tax': electricity_costs[month]['Energy tax (Euro/kWh)'],
            'variable_delivery': electricity_costs[month]['Variable delivery rate (Euro/kWh)'],
            'min_markup': 0.05,
            'max_markup': 0.25
        }
        
        # Calculate base retail price for the day
        day_ahead_prices = DayAheadPrices()
        wholesale_prices = day_ahead_prices.get_day_ahead_prices(current_date, as_series=False)
        dpe = DynamicPricingEngine(wholesale_prices=wholesale_prices)
        
        # Use average daily demand for calculation
        total_demand = 0
        for agent in self.schedule.agents:
            if agent.agent_type in [AgentType.CONSUMER, AgentType.PROSUMER]:
                if hasattr(agent, 'realised_demand'):
                    total_demand += agent.realised_demand.sum()
        
        base_retail_price = dpe.calculate_base_retail_price(0, total_demand, retail_params)
        return base_retail_price
        
    except Exception as e:
        print(f"Error calculating average base retail price: {e}")
        return None


def get_price_comparison_summary(self):
    """
    Returns a summary of all price metrics for comparison.
    :return: dict with price metrics
    """
    summary = {}
    
    # Get clearing price
    clearing_price = get_clearing_price(self)
    summary['clearing_price'] = clearing_price
    
    # Get average base retail price
    base_retail_price = get_average_base_retail_price(self)
    summary['base_retail_price'] = base_retail_price
    
    # Get average LCOE across all prosumers
    lcoe_values = []
    for agent in self.schedule.agents:
        if agent.agent_type is AgentType.PROSUMER:
            if hasattr(agent, 'average_lcoe') and agent.average_lcoe is not None:
                lcoe_values.append(agent.average_lcoe)
    
    if lcoe_values:
        summary['average_lcoe'] = np.mean(lcoe_values)
        summary['min_lcoe'] = np.min(lcoe_values)
        summary['max_lcoe'] = np.max(lcoe_values)
    else:
        summary['average_lcoe'] = None
        summary['min_lcoe'] = None
        summary['max_lcoe'] = None
    
    # Get average COVE across all prosumers
    cove_values = []
    for agent in self.schedule.agents:
        if agent.agent_type is AgentType.PROSUMER:
            if hasattr(agent, 'cove_value') and agent.cove_value is not None:
                cove_values.append(agent.cove_value)
    
    if cove_values:
        summary['average_cove'] = np.mean(cove_values)
        summary['min_cove'] = np.min(cove_values)
        summary['max_cove'] = np.max(cove_values)
    else:
        summary['average_cove'] = None
        summary['min_cove'] = None
        summary['max_cove'] = None
    
    return json.dumps(summary, cls=NumpyEncoder)


def get_generation_prediction_error(self):
    """
    Returns the detailed generation prediction error from the coordinator.
    :return: 96-point array of prediction errors or zeros if not available
    """
    coordinator = self.get_coordinator()
    if coordinator and hasattr(coordinator, 'generation_prediction_error'):
        return coordinator.generation_prediction_error.tolist()  # Convert numpy array to list for JSON serialization
    return [0] * 96  # Return 96 zeros if not available


def get_demand_prediction_error(self):
    """
    Returns the detailed demand prediction error from the coordinator.
    :return: 96-point array of prediction errors or zeros if not available
    """
    coordinator = self.get_coordinator()
    if coordinator and hasattr(coordinator, 'demand_prediction_error'):
        return coordinator.demand_prediction_error.tolist()  # Convert numpy array to list for JSON serialization
    return [0] * 96  # Return 96 zeros if not available


def get_auction_bids_offers(self):
    """
    Returns the auction bids and offers data for all agents.
    :return: dict with bids and offers per agent
    """
    auction_data = {}
    members = [AgentType.CONSUMER, AgentType.PROSUMER]
    
    for agent in self.schedule.agents:
        if agent.agent_type in members:
            key = str(agent.member_name) + str('_') + str(agent.unique_id)
            agent_data = {}
            
            # Add bid data if available
            if hasattr(agent, 'bid') and agent.bid is not None:
                agent_data['bid'] = {
                    'price': agent.bid[0],
                    'quantity': agent.bid[1]
                }
            
            # Add offer data if available
            if hasattr(agent, 'offer') and agent.offer is not None:
                agent_data['offer'] = {
                    'price': agent.offer[0],
                    'quantity': agent.offer[1]
                }
            
            if agent_data:  # Only add if there's data
                auction_data[key] = agent_data
    
    return json.dumps(auction_data, cls=NumpyEncoder)
