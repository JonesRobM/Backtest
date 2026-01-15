import pandas as pd
import yfinance as yf
from streamlit import cache_data

@cache_data
def get_data(tickers, start_date, end_date):
    """
    Downloads historical price data from Yahoo Finance.
    It prioritizes 'Adj Close', but falls back to 'Close' if it's not available.
    """
    # Download the full dataset from yfinance
    data = yf.download(tickers, start=start_date, end=end_date)

    if data.empty:
        # Let the caller handle the empty dataframe.
        return pd.DataFrame()

    # Case 1: Multiple tickers -> data.columns is a MultiIndex
    if isinstance(data.columns, pd.MultiIndex):
        try:
            # Prioritize 'Adj Close'
            price_data = data['Adj Close']
        except KeyError:
            # Fallback to 'Close'
            price_data = data['Close']
    
    # Case 2: Single ticker -> data.columns is a regular Index
    else:
        try:
            price_data = data[['Adj Close']]
        except KeyError:
            price_data = data[['Close']]
        
        # Rename the column to the ticker for consistency
        if len(tickers) == 1:
            price_data.columns = [tickers[0]]

    return price_data.dropna(axis=0, how='all')