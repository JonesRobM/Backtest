import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def generate_risk_report(price_data: pd.DataFrame, weights: dict):
    """
    Calculates portfolio risk metrics and generates a correlation matrix.

    Args:
        price_data (pd.DataFrame): A DataFrame with dates as the index and prices for each ticker.
        weights (dict): A dictionary mapping tickers to their weights.

    Returns:
        tuple: A tuple containing (dict_of_metrics, correlation_matrix_figure).
    """
    if price_data.shape[0] < 2 or price_data.shape[1] < 1:
        return None, None

    # Ensure weights are in the same order as the price_data columns
    ordered_weights = np.array([weights[ticker] for ticker in price_data.columns])
    
    # Calculate log returns: r_t = ln(P_t / P_{t-1})
    log_returns = np.log(price_data / price_data.shift(1)).dropna()
    
    # Exit if there are not enough returns to calculate covariance
    if len(log_returns) < 2:
        return None, None
        
    # Calculate Annualized Covariance Matrix
    cov_matrix = log_returns.cov() * 252 # 252 trading days
    
    # Calculate Correlation Matrix
    corr_matrix = log_returns.corr()
    
    # --- Calculate Metrics ---
    # 1. Total Portfolio Volatility (Standard Deviation)
    portfolio_variance = np.dot(ordered_weights.T, np.dot(cov_matrix, ordered_weights))
    portfolio_std = np.sqrt(portfolio_variance)
    
    # 2. Eigenvalue Decomposition for Risk Drivers
    try:
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        # Sort eigenvalues and remove potential floating point inaccuracies
        eigenvalues = np.real(sorted(eigenvalues, reverse=True))
        pc1_variance = (eigenvalues[0] / sum(eigenvalues))
    except np.linalg.LinAlgError:
        pc1_variance = np.nan

    metrics = {
        'volatility': portfolio_std,
        'pc1_contribution': pc1_variance
    }

    # --- Generate Correlation Matrix Figure ---
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr_matrix, 
        ax=ax,
        annot=True, 
        cmap='coolwarm', 
        fmt=".2f"
    )
    ax.set_title("Portfolio Asset Correlation Matrix")
    
    return metrics, fig
