import numpy as np
import pandas as pd

def calculate_cagr(series):
    """Calculates the Compound Annual Growth Rate (CAGR) for a given series of portfolio values."""
    start_value = series.iloc[0]
    end_value = series.iloc[-1]
    num_days = (series.index[-1] - series.index[0]).days
    if num_days == 0:
        return 0.0
    num_years = num_days / 365.25
    cagr = (end_value / start_value) ** (1 / num_years) - 1
    return cagr

def calculate_volatility(series):
    """Calculates the annualized volatility of a portfolio's daily returns."""
    daily_returns = series.pct_change().dropna()
    volatility = daily_returns.std() * np.sqrt(252)  # 252 trading days in a year
    return volatility

def calculate_sharpe_ratio(series, risk_free_rate=0.02):
    """Calculates the Sharpe Ratio of a portfolio."""
    cagr = calculate_cagr(series)
    volatility = calculate_volatility(series)
    if volatility == 0:
        return np.nan
    sharpe_ratio = (cagr - risk_free_rate) / volatility
    return sharpe_ratio

def calculate_max_drawdown(series):
    """Calculates the maximum drawdown of a portfolio."""
    cumulative_max = series.cummax()
    drawdown = (series - cumulative_max) / cumulative_max
    max_drawdown = drawdown.min()
    return max_drawdown
