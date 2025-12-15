"""
Module for fetching historical stock data.
"""
import yfinance as yf
import pandas as pd
from typing import List


def fetch_stock_data(tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch historical adjusted close prices for given tickers.
    
    Args:
        tickers: List of stock ticker symbols (e.g., ['AAPL', 'MSFT'])
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
    
    Returns:
        DataFrame with adjusted close prices for each ticker
    """
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)
    
    # Handle single vs multiple tickers
    if len(tickers) == 1:
        prices = data[['Adj Close']].copy()
        prices.columns = tickers
    else:
        prices = data['Adj Close'].copy()
    
    # Remove any rows with missing data
    prices = prices.dropna()
    
    return prices


def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate daily returns from price data.
    
    Args:
        prices: DataFrame with price data
    
    Returns:
        DataFrame with daily returns
    """
    returns = prices.pct_change().dropna()
    return returns
