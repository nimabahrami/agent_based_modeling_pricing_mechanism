"""
COVE (Cost of Variable Energy) calculator for prosumers.
"""

import numpy as np
import pandas as pd
from scipy import stats

def prosumer_rolling_cove(lcoe, historical_prices, historical_agent_generation, cove_markup=1.0):
    
    try:
        
        average_generation_estimated = np.mean(historical_agent_generation)
        normalized_price_values = historical_prices / np.mean(historical_prices)
        normalized_price = np.array(normalized_price_values, dtype=float)
        covariance_matrix = np.cov(normalized_price, historical_agent_generation, ddof=0)
        cov_p_g_rw = covariance_matrix[0, 1]
        denominator_cove_rolling = cov_p_g_rw + average_generation_estimated
            

        r_cov_rw = average_generation_estimated / denominator_cove_rolling
        cove = lcoe * r_cov_rw * cove_markup
        return cove
    
    except Exception as e:
        print(f"Error calculating COVE: {e}")
        return lcoe * 1.2 * cove_markup  # Default to 20% above LCOE in case of error