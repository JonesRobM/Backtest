import pandas as pd
from .portfolio import Portfolio

def run_backtest(
    data: pd.DataFrame, 
    portfolio: Portfolio, 
    initial_investment: float,
    monthly_topup: float,
    annual_increase: float
):
    """
    Runs an iterative, day-by-day backtest with rebalancing and scheduled investment logic.

    Args:
        data (pd.DataFrame): DataFrame of historical prices for the assets.
        portfolio (Portfolio): A Portfolio object defining target weights and rebalancing strategy.
        initial_investment (float): The starting capital.
        monthly_topup (float): The amount of cash to add each month.
        annual_increase (float): The percentage to increase the monthly top-up each year.

    Returns:
        pd.DataFrame: A DataFrame with the portfolio's value over time.
    """
    holdings = {ticker: 0.0 for ticker in data.columns}
    cash = float(initial_investment)
    current_monthly_topup = float(monthly_topup)
    
    portfolio_history = []
    last_month = None
    last_year = None

    # --- Main Simulation Loop ---
    for current_date, prices in data.iterrows():
        # --- Handle Date Changes ---
        # Yearly increase for the monthly top-up amount
        if last_year is not None and current_date.year > last_year:
            current_monthly_topup *= (1 + annual_increase / 100.0)
        
        # Monthly cash injection
        if last_month is not None and current_date.month != last_month:
            cash += current_monthly_topup
        
        last_month = current_date.month
        last_year = current_date.year
        
        # --- Check for Rebalance or Initial Investment ---
        # A rebalance is triggered by the portfolio's strategy, or on the very first day.
        is_first_day = not portfolio_history
        is_rebalance_day = portfolio.should_rebalance(current_date)
        
        if is_first_day or is_rebalance_day:
            # --- Deploy Cash ---
            # Calculate total value to be invested (current holdings + cash)
            market_value = sum(holdings[ticker] * prices[ticker] for ticker in holdings)
            total_value = market_value + cash
            cash = 0.0

            # Re-calculate holdings based on new total value and target weights
            for ticker, target_weight in portfolio.target_weights.items():
                if prices[ticker] > 0: # Avoid division by zero
                    amount_to_invest = total_value * target_weight
                    holdings[ticker] = amount_to_invest / prices[ticker]
        
        # --- Calculate Portfolio Value for the Day ---
        current_market_value = sum(holdings[ticker] * prices[ticker] for ticker in holdings)
        total_portfolio_value = current_market_value + cash
        portfolio_history.append({'Date': current_date, 'Portfolio Value': total_portfolio_value})

    # --- Format and Return Results ---
    result_df = pd.DataFrame(portfolio_history)
    result_df.set_index('Date', inplace=True)
    
    return result_df, holdings, cash
