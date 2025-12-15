"""
Module for calculating portfolio statistics.
"""
import numpy as np
import pandas as pd
from typing import Tuple


def calculate_portfolio_stats(returns: pd.DataFrame) -> dict:
    """
    Calculate basic portfolio statistics.
    
    Args:
        returns: DataFrame with daily returns for each asset
    
    Returns:
        Dictionary containing means, covariance matrix, and correlation matrix
    """
    # Annualize returns (assuming 252 trading days)
    mean_returns = returns.mean() * 252
    
    # Annualize covariance matrix
    cov_matrix = returns.cov() * 252
    
    # Correlation matrix
    corr_matrix = returns.corr()
    
    return {
        'mean_returns': mean_returns,
        'cov_matrix': cov_matrix,
        'corr_matrix': corr_matrix
    }


def portfolio_performance(weights: np.ndarray, mean_returns: pd.Series, 
                         cov_matrix: pd.DataFrame) -> Tuple[float, float]:
    """
    Calculate portfolio expected return and volatility.
    
    Args:
        weights: Array of portfolio weights
        mean_returns: Series of annualized mean returns
        cov_matrix: Covariance matrix of returns
    
    Returns:
        Tuple of (expected_return, volatility)
    """
    portfolio_return = np.sum(mean_returns * weights)
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    return portfolio_return, portfolio_std


def calculate_sharpe_ratio(portfolio_return: float, portfolio_std: float, 
                          risk_free_rate: float = 0.02) -> float:
    """
    Calculate Sharpe ratio for a portfolio.
    
    Args:
        portfolio_return: Expected portfolio return
        portfolio_std: Portfolio standard deviation (volatility)
        risk_free_rate: Risk-free rate (default 2%)
    
    Returns:
        Sharpe ratio
    """
    return (portfolio_return - risk_free_rate) / portfolio_std
