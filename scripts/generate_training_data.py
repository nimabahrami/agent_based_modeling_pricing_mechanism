
import os
import sys
import pandas as pd
import numpy as np
import datetime
from pathlib import Path
from model.agents import *
from model.community_setup import create_community_configuration
from model.enumerations import *

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



# Create data directory if it doesn't exist
os.makedirs('data/training', exist_ok=True)

# Load the model input data
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(base_dir, "data/processed/model_input_data.csv"), 
                 parse_dates=['Local'], index_col=0)

# Define the date range for training data
start_date = "2021-02-01"
end_date = "2021-12-30"

# Filter data for the selected period
training_data = df[(df.index >= start_date) & (df.index <= end_date)]

agents_list = create_community_configuration(community_name='gridflex_heeten')

solar_prosumers = 0
wind_prosumers = 0
residential_consumers = 0
non_residential_consumers = 0

for agent in agents_list:
    if agent.get('agent_type') == AgentType.PROSUMER:
        if agent.get('asset_list'):
            for asset in agent.get('asset_list'):
                if asset.get('asset_type') == Solar:
                    solar_prosumers += 1
                elif asset.get('asset_type') == Wind:
                    wind_prosumers += 1
    elif agent.get('agent_type') == AgentType.CONSUMER:
        if agent.get('member_type') == MemberType.RESIDENTIAL:
            residential_consumers += 1
        elif agent.get('member_type') == MemberType.NON_RESIDENTIAL:
            non_residential_consumers += 1

print(f"Community composition:")
print(f"Solar prosumers: {solar_prosumers}")
print(f"Wind prosumers: {wind_prosumers}")
print(f"Residential consumers: {residential_consumers}")
print(f"Non-residential consumers: {non_residential_consumers}")

# Initialize the output dataframe
output_data = pd.DataFrame(index=training_data.index)
output_data['date'] = output_data.index

generation = np.zeros(len(training_data))

for agent in agents_list:
    if agent.get('agent_type') == AgentType.PROSUMER and agent.get('asset_list'):
        for asset in agent.get('asset_list'):
            if asset.get('asset_type') == Solar:
                
                solar_irradiance = training_data['Direct [W/m^2]'].values
                capacity = asset.get('capacity', 20) 
                efficiency = asset.get('efficiency', 0.20) 
                solar_gen = capacity * efficiency * solar_irradiance / 1000000
                generation += solar_gen

            elif asset.get('asset_type') == Wind:
                # Use the wind generation equation
                wind_speed = training_data['Wind [m/s]'].values
                # Wind turbine shuts down if wind speed is greater than 30 m/s
                wind_speed[wind_speed > 30] = 0
                
                capacity = asset.get('capacity', 100)  # Default capacity
                efficiency = asset.get('efficiency', 0.593)  # Betz limit
                rotor_diameter = 125  # Default rotor diameter
                avg_air_density = 1.23  # Default air density
                number_of_turbines = 1  # Default number of turbines
                
                swept_area = np.pi * (rotor_diameter / 2) ** 2
                
                wind_gen = 0.5 * avg_air_density * swept_area * np.power(wind_speed, 3) * efficiency * number_of_turbines
                generation += wind_gen

demand = np.zeros(len(training_data))

if residential_consumers > 0 or solar_prosumers > 0:
    hh_columns = [col for col in training_data.columns if 'hh' in col and 'consumption' in col]
    total_hh_demand = training_data[hh_columns].sum(axis=1).values

    demand += total_hh_demand * (residential_consumers + solar_prosumers)

if non_residential_consumers > 0:
    # Sum all non-residential columns (offices, school, etc.)
    non_res_columns = ['Office 1', 'Office 2', 'Office 3', 'Office 4', 'School', 'EV_charging_station']
    non_res_columns = [col for col in non_res_columns if col in training_data.columns]
    
    if non_res_columns:
        total_non_res_demand = training_data[non_res_columns].sum(axis=1).values
        demand += total_non_res_demand

output_data['generation'] = generation
output_data['demand'] = demand

output_data['gap'] = output_data['generation'] - output_data['demand']
output_path = os.path.join(base_dir, "data/training/gap_predictor_training_data.csv")
output_data.to_csv(output_path)

print(f"Training data saved to {output_path}")
print(f"Data shape: {output_data.shape}")
print(f"Date range: {output_data.index.min()} to {output_data.index.max()}")
print(f"Average generation: {output_data['generation'].mean():.2f} kWh")
print(f"Average demand: {output_data['demand'].mean():.2f} kWh")
print(f"Average gap: {output_data['gap'].mean():.2f} kWh")
