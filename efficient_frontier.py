"""
Module for computing the efficient frontier and optimal portfolios.
"""
import numpy as np
import pandas as pd
from typing import Tuple
from scipy.optimize import minimize
from portfolio_stats import portfolio_performance, calculate_sharpe_ratio


def generate_efficient_frontier(mean_returns: pd.Series, cov_matrix: pd.DataFrame, 
                                num_portfolios: int = 10000) -> pd.DataFrame:
    """
    Generate efficient frontier using random portfolio simulations.
    
    Args:
        mean_returns: Series of annualized mean returns
        cov_matrix: Covariance matrix of returns
        num_portfolios: Number of random portfolios to simulate
    
    Returns:
        DataFrame with portfolio returns, volatilities, Sharpe ratios, and weights
    """
    num_assets = len(mean_returns)
    results = np.zeros((4 + num_assets, num_portfolios))
    
    for i in range(num_portfolios):
        # Generate random weights
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        
        # Calculate portfolio metrics
        portfolio_return, portfolio_std = portfolio_performance(weights, mean_returns, cov_matrix)
        sharpe = calculate_sharpe_ratio(portfolio_return, portfolio_std)
        
        # Store results
        results[0, i] = portfolio_return
        results[1, i] = portfolio_std
        results[2, i] = sharpe
        results[3, i] = portfolio_return / portfolio_std  # Return/risk ratio
        results[4:, i] = weights
    
    # Create DataFrame
    columns = ['Return', 'Volatility', 'Sharpe', 'Return/Risk'] + \
              [f'Weight_{ticker}' for ticker in mean_returns.index]
    
    df = pd.DataFrame(results.T, columns=columns)
    return df


def find_max_sharpe_portfolio(mean_returns: pd.Series, cov_matrix: pd.DataFrame,
                              risk_free_rate: float = 0.02) -> dict:
    """
    Find the portfolio with maximum Sharpe ratio using optimization.
    
    Args:
        mean_returns: Series of annualized mean returns
        cov_matrix: Covariance matrix of returns
        risk_free_rate: Risk-free rate (default 2%)
    
    Returns:
        Dictionary with optimal weights, return, volatility, and Sharpe ratio
    """
    num_assets = len(mean_returns)
    
    # Objective function: negative Sharpe ratio (to minimize)
    def neg_sharpe(weights):
        portfolio_return, portfolio_std = portfolio_performance(weights, mean_returns, cov_matrix)
        return -calculate_sharpe_ratio(portfolio_return, portfolio_std, risk_free_rate)
    
    # Constraints: weights sum to 1
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    
    # Bounds: weights between 0 and 1 (no short selling)
    bounds = tuple((0, 1) for _ in range(num_assets))
    
    # Initial guess: equal weights
    initial_guess = num_assets * [1. / num_assets]
    
    # Optimize
    result = minimize(neg_sharpe, initial_guess, method='SLSQP', 
                     bounds=bounds, constraints=constraints)
    
    optimal_weights = result.x
    portfolio_return, portfolio_std = portfolio_performance(optimal_weights, mean_returns, cov_matrix)
    sharpe = calculate_sharpe_ratio(portfolio_return, portfolio_std, risk_free_rate)
    
    return {
        'weights': optimal_weights,
        'return': portfolio_return,
        'volatility': portfolio_std,
        'sharpe': sharpe
    }


def find_min_variance_portfolio(mean_returns: pd.Series, cov_matrix: pd.DataFrame) -> dict:
    """
    Find the minimum variance portfolio.
    
    Args:
        mean_returns: Series of annualized mean returns
        cov_matrix: Covariance matrix of returns
    
    Returns:
        Dictionary with optimal weights, return, and volatility
    """
    num_assets = len(mean_returns)
    
    # Objective function: portfolio variance
    def portfolio_variance(weights):
        return np.dot(weights.T, np.dot(cov_matrix, weights))
    
    # Constraints: weights sum to 1
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    
    # Bounds: weights between 0 and 1
    bounds = tuple((0, 1) for _ in range(num_assets))
    
    # Initial guess: equal weights
    initial_guess = num_assets * [1. / num_assets]
    
    # Optimize
    result = minimize(portfolio_variance, initial_guess, method='SLSQP',
                     bounds=bounds, constraints=constraints)
    
    optimal_weights = result.x
    portfolio_return, portfolio_std = portfolio_performance(optimal_weights, mean_returns, cov_matrix)
    
    return {
        'weights': optimal_weights,
        'return': portfolio_return,
        'volatility': portfolio_std
    }
