"""
Module for generating sample stock data for testing purposes.
This is useful when real data cannot be fetched from Yahoo Finance.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_sample_stock_data(tickers=['AAPL', 'MSFT'], 
                               start_date='2021-01-01', 
                               end_date='2023-12-31',
                               seed=42):
    """
    Generate synthetic stock price data that mimics real market behavior.
    
    Args:
        tickers: List of ticker symbols
        start_date: Start date as string
        end_date: End date as string
        seed: Random seed for reproducibility
    
    Returns:
        DataFrame with synthetic price data
    """
    np.random.seed(seed)
    
    # Create date range (trading days only)
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    dates = pd.bdate_range(start=start, end=end)
    
    # Realistic parameters for AAPL and MSFT
    # drift represents daily expected return, volatility is daily std dev
    params = {
        'AAPL': {'initial_price': 130, 'drift': 0.0003, 'volatility': 0.015},
        'MSFT': {'initial_price': 250, 'drift': 0.0004, 'volatility': 0.014}
    }
    
    # Generate correlated random returns first
    n_days = len(dates)
    n_assets = len(tickers)
    
    # Generate independent random returns
    independent_returns = np.random.normal(0, 1, (n_days, n_assets))
    
    # Create correlation matrix (0.7 correlation for tech stocks)
    correlation = 0.7
    corr_matrix = np.full((n_assets, n_assets), correlation)
    np.fill_diagonal(corr_matrix, 1.0)
    
    # Apply Cholesky decomposition to get correlated returns
    L = np.linalg.cholesky(corr_matrix)
    correlated_returns = independent_returns @ L.T
    
    prices_dict = {}
    
    for i, ticker in enumerate(tickers):
        if ticker in params:
            p = params[ticker]
        else:
            # Default parameters for other tickers
            p = {
                'initial_price': 100,
                'drift': 0.0003,
                'volatility': 0.015
            }
        
        # Scale returns by drift and volatility
        returns = p['drift'] + p['volatility'] * correlated_returns[:, i]
        
        # Convert returns to prices using cumulative product
        price_ratios = np.exp(returns)
        prices = p['initial_price'] * np.cumprod(price_ratios)
        
        prices_dict[ticker] = prices
    
    # Create DataFrame
    df = pd.DataFrame(prices_dict, index=dates)
    df.index.name = 'Date'
    
    return df
