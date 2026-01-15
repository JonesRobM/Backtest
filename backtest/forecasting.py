import pandas as pd
import numpy as np

def run_monte_carlo_simulation(historical_returns: pd.Series, projection_years: int, initial_value: float, num_simulations: int = 500):
    """
    Runs a Monte Carlo simulation to project future portfolio returns.

    Args:
        historical_returns (pd.Series): A series of historical daily returns of the portfolio.
        projection_years (int): The number of years to project into the future.
        initial_value (float): The last known value of the portfolio, to start the simulation from.
        num_simulations (int): The number of simulation paths to generate.

    Returns:
        pd.DataFrame: A DataFrame containing the simulation results, with columns for different quantiles.
    """
    # Calculate historical statistical properties
    mu = historical_returns.mean()
    sigma = historical_returns.std()
    
    num_days = projection_years * 252  # 252 trading days in a year
    
    # Initialize results matrix
    simulation_results = np.zeros((num_days, num_simulations))
    
    # Run simulations
    for i in range(num_simulations):
        next_price = initial_value
        
        # Generate random daily returns for the entire period
        random_shocks = np.random.normal(mu, sigma, num_days)
        
        # Apply the shocks to create the price path
        for j in range(num_days):
            next_price *= (1 + random_shocks[j])
            simulation_results[j, i] = next_price
            
    # --- Process Results ---
    # Create a date index for the projected period
    last_date = historical_returns.index[-1]
    future_dates = pd.to_datetime([last_date + pd.DateOffset(days=x) for x in range(1, num_days + 1)])
    
    # Create a DataFrame from the simulation results
    results_df = pd.DataFrame(simulation_results, index=future_dates)
    
    # Calculate quantiles for uncertainty bounds
    forecast_df = pd.DataFrame(index=future_dates)
    forecast_df['median'] = results_df.quantile(0.5, axis=1)
    forecast_df['upper_bound_75'] = results_df.quantile(0.75, axis=1)
    forecast_df['lower_bound_25'] = results_df.quantile(0.25, axis=1)
    forecast_df['upper_bound_95'] = results_df.quantile(0.95, axis=1)
    forecast_df['lower_bound_05'] = results_df.quantile(0.05, axis=1)
    
    return forecast_df
