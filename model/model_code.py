# import datetime
import time

from mesa import Model
from mesa.time import BaseScheduler
from mesa.datacollection import DataCollector

from model.data_reporters import *
from model.agents import Member, Coordinator


class EnergyCommunity(Model):
    """A model with some number of agents."""

    date = None

    def __init__(self,
                 levers=None,
                 uncertainties=None,
                 agents_list=None,
                 start_date=None,
                 cove_markup=1.0):
        super().__init__()

        if levers is None:
            self.levers = {
                "L1": 0.5,
                # Percentage of members participating in the demand response program (Social)
                "L2": 0.2,
                # Percentage of flexible (shift-able)  demand for residential community members (Technical)
                "L3": 0.3,
                # Percentage of flexible (shift-able)  demand for non-residential community members (Technical)
            }
        else:
            self.levers = levers

        if uncertainties is None:
            self.uncertainties = {
                "X1": 0.30,
                # Minimum percentage of flexible demand available for demand response on a single day (Social)
                "X2": 0.75,
                # Maximum percentage of flexible demand available for demand response on a single day (Social)
                "X3": 0.80,
                # Percentage accuracy of day-ahead generation projections from renewable assets (Technical)
            }
        else:
            self.uncertainties = uncertainties

        if start_date is None:
            # Start from the second week of 2021 (January 8th)
            start_date = datetime.datetime(2021, 8, 10)
            self.is_default_start_date = True
        else:
            self.is_default_start_date = False
            
        self.start_date = start_date
        self.date = start_date.strftime('%Y-%m-%d')
        self.date_index = pd.date_range(start=self.date, periods=96, freq='15min')

        self.demand_availability = {'minimum': self.uncertainties['X1'],
                                    'maximum': self.uncertainties['X2']}
        self.participation_in_tod = self.levers['L1']
        self.tick = 0
        self.tod_surplus_timing = None  # Time of day when surplus electricity is available
        self.tod_deficit_timing = None  # Time of day when electricity is needed from the grid
        self.price_signal = None  # Dynamic price signal for the next day
        self.agent_list = agents_list
        self.cove_markup = cove_markup
        self.schedule = BaseScheduler(self)
        self.all_assets = {}
        self.create_agents()
        
        # Add new metrics for auction and dynamic pricing
        self.datacollector = DataCollector(model_reporters={
            "date": get_date,
            # date or the time step for the model simulation
            "M1: realised_demand": get_realised_demand,
            # realised demand after incorporating demand response
            "M2: scheduled_demand": get_scheduled_demand,
            # demand before incorporating demand response
            "M3: shifted_load": get_shifted_load,
            # amount of load moved/shifted because of demand response
            "M4: total_generation": get_generation,
            # generation from the renewable assets in the simulation model
            "M5: savings_on_ToD": get_savings,
            # savings made by avoiding import of electricity from grid by community members
            "M6: energy_costs": get_energy_cost,
            # total expenses made by community members for procuring electricity from the grid
            "M7: auction_revenue": get_auction_revenue,
            # revenue from the electricity auction (prosumers only)
            "M7b: auction_cost": get_auction_cost,
            # cost from the electricity auction (consumers only)
            "M8: dynamic_flexibility": get_dynamic_flexibility,
            # average dynamic flexibility of community members
            "M9: cove_values": get_cove_values,
            # COVE values of prosumers
            "M10: lcoe_values": get_lcoe_values,
            # LCOE values of prosumers
            "M11: clearing_price": get_clearing_price,
            # clearing price from auction
            "M12: base_retail_price": get_average_base_retail_price,
            # average base retail price
            "M13: price_comparison_summary": get_price_comparison_summary,
            # summary of all price metrics for comparison
            "M14: generation_prediction_error": get_generation_prediction_error,
            # generation prediction error from coordinator
            "M15: demand_prediction_error": get_demand_prediction_error,
            # demand prediction error from coordinator
            "M16: auction_bids_offers": get_auction_bids_offers
            # auction bids and offers data
        })

    def step(self):
        """Advance the model by one step."""
        super().step()
        
        coordinator = self.get_coordinator()
        if coordinator:
            # 1. Coordinator sets price signal (and any other pre-auction logic)
            coordinator.predict_gap_for_tomorrow()
            coordinator.calculate_dynamic_prices()
            if self.participation_in_tod is not None and self.participation_in_tod > 0:
                coordinator.release_tod_schedule()
            coordinator.auction.reset()

        # 2. All agents prepare their bids/offers (and update demand/generation, etc.)
        for agent in self.schedule.agents:
            agent.update_date()
            if isinstance(agent, (Member)):
                agent.get_demand_schedule()
                agent.get_generation_schedule()
                agent.generate_day_ahead_schedules()
                agent.adjust_schedule_for_captive_consumption()
                agent.get_previous_days_generation_schedule()
                agent.get_previous_days_price_schedule()
                if hasattr(agent, "calculate_cove_and_flexibility"):
                    agent.calculate_cove_and_flexibility()
                if hasattr(agent, "prepare_auction_bid_offer"):
                    agent.prepare_auction_bid_offer()
                if hasattr(agent, "adjust_schedule_for_tod"):
                    agent.adjust_schedule_for_tod()

        # 3. Coordinator runs the auction and updates agent results
        if coordinator:
            coordinator.run_auction()

        # 4. All agents compute their costs/earnings based on auction results
        for agent in self.schedule.agents:
            if hasattr(agent, "compute_energy_cost"):
                agent.compute_energy_cost()
            if hasattr(agent, "compute_earnings"):
                agent.compute_earnings()
                # agent.compute_earning_original()

        # 5. Collect data
        self.datacollector.collect(self)
        
        # 6. Update date
        self.tick += 1
        self.date = self.tick_to_date(self.tick)

    def create_agents(self):
        """Create agents and add them to the schedule."""
        for agent_details in self.agent_list:
            if agent_details['member_type'] is MemberType.COORDINATOR:
                agent = Coordinator(unique_id=self.next_id(), model=self)
            else:
                agent = Member(unique_id=self.next_id(),
                               member_name=agent_details['member_name'],
                               agent_type=agent_details['agent_type'],
                               member_type=agent_details['member_type'],
                               demand_flexibility=self.select_demand_flexibility(
                                   member_type=agent_details['member_type']),
                               asset_list=agent_details['asset_list'],
                               model=self)
            self.schedule.add(agent)
        return None

    def select_demand_flexibility(self, member_type):
        "Selects demand flexibility from levers for an agent based on member type"
        demand_flexibility = 0.20  # Default member flexibility
        if member_type is MemberType.RESIDENTIAL:
            demand_flexibility = self.levers['L2']
        elif member_type is MemberType.NON_RESIDENTIAL:
            demand_flexibility = self.levers['L3']
        return demand_flexibility
        
    def get_coordinator(self):
        """Get the coordinator agent."""
        for agent in self.schedule.agents:
            if agent.agent_type is AgentType.COORDINATOR:
                return agent
        return None

    def run_simulation(self, steps=300, time_tracking=False, debug=False):

        start_time = time.time()

        for tick in range(steps):
            if debug:
                print(f'Step: {tick}')
            # self.date = self.tick_to_date(tick + 1)
            self.step()

        if time_tracking:
            run_time = round(time.time() - start_time, 2)
            print(f'Run time: {run_time} seconds')

            print('Simulation completed!')

        results = self.datacollector.get_model_vars_dataframe()
        return results

    @staticmethod
    def tick_to_date(tick):
        """
        Converts a tick to a date
        :param tick: int: tick number
        :return:
            date: string: date in format "YYYY-MM-DD"
        """
        year = 2021
        days = tick
        date = datetime.datetime(year, 2, 1) + datetime.timedelta(days - 1)
        date = date.strftime('%Y-%m-%d')
        return date
