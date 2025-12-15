"""
Markowitz Mean-Variance Portfolio Optimizer Module

This module provides functions to build a Markowitz mean-variance portfolio optimizer
using historical stock price data. It includes utilities for computing returns,
covariance matrices, portfolio performance metrics, efficient frontier generation,
and maximum Sharpe ratio portfolio optimization.

Author: Generated for Mean-Variance Portfolio Construction Project
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from typing import Tuple, Optional


# =============================================================================
# CONSTANTS
# =============================================================================

TRADING_DAYS_PER_YEAR = 252  # Standard number of trading days for annualization


# =============================================================================
# RETURN CALCULATIONS
# =============================================================================

def compute_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily percentage returns from price data.
    
    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame with dates as index and asset prices as columns.
        
    Returns
    -------
    pd.DataFrame
        DataFrame of daily percentage returns (same structure as input).
        First row will be NaN and is dropped.
        
    Examples
    --------
    >>> prices = pd.DataFrame({'AAPL': [100, 102, 101], 'MSFT': [200, 204, 202]})
    >>> returns = compute_daily_returns(prices)
    """
    if prices.empty:
        raise ValueError("Price DataFrame is empty.")
    
    # Calculate percentage returns: (P_t - P_{t-1}) / P_{t-1}
    daily_returns = prices.pct_change().dropna()
    
    return daily_returns


def compute_cumulative_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Compute cumulative returns from daily returns.
    
    Parameters
    ----------
    returns : pd.DataFrame
        DataFrame of daily returns.
        
    Returns
    -------
    pd.DataFrame
        DataFrame of cumulative returns (starting at 1.0).
    """
    if returns.empty:
        raise ValueError("Returns DataFrame is empty.")
    
    # Cumulative return = product of (1 + daily_return)
    cumulative_returns = (1 + returns).cumprod()
    
    return cumulative_returns


# =============================================================================
# EXPECTED RETURNS AND COVARIANCE
# =============================================================================

def compute_expected_returns(
    returns: pd.DataFrame, 
    annualize: bool = True
) -> np.ndarray:
    """
    Compute expected (mean) returns for each asset.
    
    Parameters
    ----------
    returns : pd.DataFrame
        DataFrame of daily returns.
    annualize : bool, default True
        If True, annualize the returns using 252 trading days.
        
    Returns
    -------
    np.ndarray
        1D array of expected returns for each asset.
    """
    if returns.empty:
        raise ValueError("Returns DataFrame is empty.")
    
    # Mean daily returns
    mean_returns = returns.mean().values
    
    if annualize:
        # Annualize by multiplying by trading days
        mean_returns = mean_returns * TRADING_DAYS_PER_YEAR
    
    return mean_returns


def compute_covariance_matrix(
    returns: pd.DataFrame, 
    annualize: bool = True
) -> np.ndarray:
    """
    Compute the covariance matrix of asset returns.
    
    Parameters
    ----------
    returns : pd.DataFrame
        DataFrame of daily returns.
    annualize : bool, default True
        If True, annualize the covariance matrix using 252 trading days.
        
    Returns
    -------
    np.ndarray
        2D covariance matrix.
    """
    if returns.empty:
        raise ValueError("Returns DataFrame is empty.")
    
    # Sample covariance matrix
    cov_matrix = returns.cov().values
    
    if annualize:
        # Annualize by multiplying by trading days
        cov_matrix = cov_matrix * TRADING_DAYS_PER_YEAR
    
    return cov_matrix


# =============================================================================
# PORTFOLIO PERFORMANCE METRICS
# =============================================================================

def portfolio_performance(
    weights: np.ndarray, 
    expected_returns: np.ndarray, 
    cov_matrix: np.ndarray, 
    rf: float = 0.0
) -> Tuple[float, float, float]:
    """
    Compute portfolio return, volatility, and Sharpe ratio.
    
    Parameters
    ----------
    weights : np.ndarray
        1D array of portfolio weights (must sum to 1).
    expected_returns : np.ndarray
        1D array of expected returns for each asset.
    cov_matrix : np.ndarray
        2D covariance matrix.
    rf : float, default 0.0
        Risk-free rate (annualized).
        
    Returns
    -------
    tuple[float, float, float]
        (portfolio_return, portfolio_volatility, sharpe_ratio)
        
    Notes
    -----
    Portfolio return: μ_p = w^T * μ
    Portfolio variance: σ_p^2 = w^T * Σ * w
    Sharpe ratio: S = (μ_p - rf) / σ_p
    """
    weights = np.array(weights)
    
    # Portfolio expected return
    portfolio_return = np.dot(weights, expected_returns)
    
    # Portfolio variance and volatility
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    portfolio_volatility = np.sqrt(portfolio_variance)
    
    # Sharpe ratio
    if portfolio_volatility == 0:
        sharpe_ratio = 0.0
    else:
        sharpe_ratio = (portfolio_return - rf) / portfolio_volatility
    
    return portfolio_return, portfolio_volatility, sharpe_ratio


# =============================================================================
# EFFICIENT FRONTIER (TWO-ASSET CASE)
# =============================================================================

def generate_efficient_frontier(
    expected_returns: np.ndarray, 
    cov_matrix: np.ndarray, 
    n_points: int = 100,
    rf: float = 0.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate the efficient frontier for a two-asset portfolio.
    
    For a two-asset system, we sweep the weight of the first asset from 0 to 1.
    
    Parameters
    ----------
    expected_returns : np.ndarray
        1D array of expected returns for each asset (length 2).
    cov_matrix : np.ndarray
        2x2 covariance matrix.
    n_points : int, default 100
        Number of points to generate on the frontier.
    rf : float, default 0.0
        Risk-free rate for Sharpe ratio calculation.
        
    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        (weights_array, portfolio_returns, portfolio_volatilities, sharpe_ratios)
        - weights_array: (n_points, 2) array of weight combinations
        - portfolio_returns: (n_points,) array of expected returns
        - portfolio_volatilities: (n_points,) array of volatilities
        - sharpe_ratios: (n_points,) array of Sharpe ratios
    """
    if len(expected_returns) != 2:
        raise ValueError("This function is designed for exactly 2 assets.")
    if cov_matrix.shape != (2, 2):
        raise ValueError("Covariance matrix must be 2x2.")
    
    # Generate weight combinations
    w1_values = np.linspace(0, 1, n_points)
    
    weights_array = np.zeros((n_points, 2))
    portfolio_returns = np.zeros(n_points)
    portfolio_volatilities = np.zeros(n_points)
    sharpe_ratios = np.zeros(n_points)
    
    for i, w1 in enumerate(w1_values):
        w2 = 1 - w1
        weights = np.array([w1, w2])
        weights_array[i] = weights
        
        ret, vol, sharpe = portfolio_performance(
            weights, expected_returns, cov_matrix, rf
        )
        
        portfolio_returns[i] = ret
        portfolio_volatilities[i] = vol
        sharpe_ratios[i] = sharpe
    
    return weights_array, portfolio_returns, portfolio_volatilities, sharpe_ratios


# =============================================================================
# MAXIMUM SHARPE RATIO PORTFOLIO
# =============================================================================

def find_max_sharpe_portfolio(
    expected_returns: np.ndarray, 
    cov_matrix: np.ndarray, 
    rf: float = 0.0
) -> Tuple[np.ndarray, float, float, float]:
    """
    Find the portfolio with the maximum Sharpe ratio.
    
    For a two-asset portfolio, this uses numerical optimization to find
    the weight that maximizes the Sharpe ratio.
    
    Parameters
    ----------
    expected_returns : np.ndarray
        1D array of expected returns for each asset.
    cov_matrix : np.ndarray
        2D covariance matrix.
    rf : float, default 0.0
        Risk-free rate (annualized).
        
    Returns
    -------
    tuple[np.ndarray, float, float, float]
        (optimal_weights, optimal_return, optimal_volatility, max_sharpe_ratio)
    """
    if len(expected_returns) != 2:
        raise ValueError("This function is designed for exactly 2 assets.")
    
    def neg_sharpe(w1: float) -> float:
        """Negative Sharpe ratio for minimization."""
        weights = np.array([w1, 1 - w1])
        _, _, sharpe = portfolio_performance(
            weights, expected_returns, cov_matrix, rf
        )
        return -sharpe  # Negative for minimization
    
    # Optimize: find w1 that minimizes negative Sharpe ratio
    result = minimize_scalar(
        neg_sharpe, 
        bounds=(0, 1), 
        method='bounded'
    )
    
    optimal_w1 = result.x
    optimal_weights = np.array([optimal_w1, 1 - optimal_w1])
    
    optimal_return, optimal_volatility, max_sharpe = portfolio_performance(
        optimal_weights, expected_returns, cov_matrix, rf
    )
    
    return optimal_weights, optimal_return, optimal_volatility, max_sharpe


# =============================================================================
# TANGENT LINE (CAPITAL ALLOCATION LINE)
# =============================================================================

def compute_tangent_line(
    optimal_return: float, 
    optimal_volatility: float, 
    rf: float = 0.0, 
    max_vol: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the Capital Allocation Line (tangent line) from the risk-free rate
    through the optimal Sharpe ratio portfolio.
    
    Parameters
    ----------
    optimal_return : float
        Expected return of the optimal portfolio.
    optimal_volatility : float
        Volatility of the optimal portfolio.
    rf : float, default 0.0
        Risk-free rate.
    max_vol : float, optional
        Maximum volatility for the line. If None, extends 20% beyond optimal.
        
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (volatility_values, return_values) for plotting the tangent line.
        
    Notes
    -----
    The Capital Allocation Line equation:
    E[R_p] = rf + (E[R_opt] - rf) / σ_opt * σ_p
    
    Slope = Sharpe ratio of optimal portfolio
    """
    if optimal_volatility <= 0:
        raise ValueError("Optimal volatility must be positive.")
    
    # Sharpe ratio (slope of CAL)
    sharpe_slope = (optimal_return - rf) / optimal_volatility
    
    # Define volatility range
    if max_vol is None:
        max_vol = optimal_volatility * 1.5
    
    volatility_values = np.linspace(0, max_vol, 100)
    
    # CAL equation: E[R] = rf + slope * σ
    return_values = rf + sharpe_slope * volatility_values
    
    return volatility_values, return_values


# =============================================================================
# PORTFOLIO CUMULATIVE RETURNS
# =============================================================================

def compute_portfolio_cumulative_returns(
    returns: pd.DataFrame, 
    weights: np.ndarray
) -> pd.Series:
    """
    Compute cumulative returns for a weighted portfolio.
    
    Parameters
    ----------
    returns : pd.DataFrame
        DataFrame of daily returns for each asset.
    weights : np.ndarray
        1D array of portfolio weights.
        
    Returns
    -------
    pd.Series
        Cumulative returns of the portfolio over time.
    """
    if returns.empty:
        raise ValueError("Returns DataFrame is empty.")
    if len(weights) != returns.shape[1]:
        raise ValueError("Number of weights must match number of assets.")
    
    # Portfolio daily returns (weighted sum)
    portfolio_daily_returns = returns.dot(weights)
    
    # Cumulative returns
    portfolio_cumulative = (1 + portfolio_daily_returns).cumprod()
    
    return portfolio_cumulative


# =============================================================================
# VALIDATION UTILITIES
# =============================================================================

def validate_optimization(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    optimal_weights: np.ndarray,
    rf: float = 0.0,
    n_test_points: int = 1000
) -> dict:
    """
    Validate that the optimal portfolio truly maximizes the Sharpe ratio.
    
    Parameters
    ----------
    expected_returns : np.ndarray
        Expected returns array.
    cov_matrix : np.ndarray
        Covariance matrix.
    optimal_weights : np.ndarray
        Weights of the claimed optimal portfolio.
    rf : float
        Risk-free rate.
    n_test_points : int
        Number of test points to check.
        
    Returns
    -------
    dict
        Validation results including:
        - is_optimal: bool indicating if the solution is optimal
        - optimal_sharpe: Sharpe ratio of optimal portfolio
        - max_sharpe_found: Maximum Sharpe ratio found in test points
        - sharpe_difference: Difference between optimal and max found
    """
    _, _, optimal_sharpe = portfolio_performance(
        optimal_weights, expected_returns, cov_matrix, rf
    )
    
    # Test many weight combinations
    max_sharpe_found = -np.inf
    
    for w1 in np.linspace(0, 1, n_test_points):
        weights = np.array([w1, 1 - w1])
        _, _, sharpe = portfolio_performance(
            weights, expected_returns, cov_matrix, rf
        )
        if sharpe > max_sharpe_found:
            max_sharpe_found = sharpe
    
    sharpe_difference = abs(optimal_sharpe - max_sharpe_found)
    is_optimal = sharpe_difference < 1e-6
    
    return {
        'is_optimal': is_optimal,
        'optimal_sharpe': optimal_sharpe,
        'max_sharpe_found': max_sharpe_found,
        'sharpe_difference': sharpe_difference
    }


# =============================================================================
# DISPLAY UTILITIES
# =============================================================================

def print_portfolio_summary(
    asset_names: list,
    optimal_weights: np.ndarray,
    optimal_return: float,
    optimal_volatility: float,
    max_sharpe: float,
    rf: float = 0.0
) -> None:
    """
    Print a formatted summary of the optimal portfolio.
    
    Parameters
    ----------
    asset_names : list
        Names of the assets.
    optimal_weights : np.ndarray
        Optimal portfolio weights.
    optimal_return : float
        Expected return of optimal portfolio.
    optimal_volatility : float
        Volatility of optimal portfolio.
    max_sharpe : float
        Maximum Sharpe ratio.
    rf : float
        Risk-free rate used.
    """
    print("=" * 60)
    print("OPTIMAL PORTFOLIO SUMMARY (Maximum Sharpe Ratio)")
    print("=" * 60)
    print(f"\nRisk-Free Rate: {rf:.2%}")
    print(f"\nOptimal Weights:")
    for name, weight in zip(asset_names, optimal_weights):
        print(f"  {name}: {weight:.4f} ({weight * 100:.2f}%)")
    print(f"\nPortfolio Metrics:")
    print(f"  Expected Annual Return: {optimal_return:.4f} ({optimal_return * 100:.2f}%)")
    print(f"  Annual Volatility (Std): {optimal_volatility:.4f} ({optimal_volatility * 100:.2f}%)")
    print(f"  Sharpe Ratio: {max_sharpe:.4f}")
    print("=" * 60)
