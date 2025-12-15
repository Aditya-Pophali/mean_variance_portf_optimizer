"""
Module for visualizing portfolio analysis results.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Optional


def plot_efficient_frontier(portfolios: pd.DataFrame, max_sharpe_port: dict,
                            min_var_port: Optional[dict] = None,
                            risk_free_rate: float = 0.02,
                            save_path: Optional[str] = None):
    """
    Plot the efficient frontier with Capital Allocation Line (CAL).
    
    Args:
        portfolios: DataFrame with simulated portfolios
        max_sharpe_port: Dictionary with max Sharpe ratio portfolio info
        min_var_port: Optional dictionary with minimum variance portfolio info
        risk_free_rate: Risk-free rate for CAL
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    # Scatter plot of all portfolios, colored by Sharpe ratio
    scatter = plt.scatter(portfolios['Volatility'], portfolios['Return'],
                         c=portfolios['Sharpe'], cmap='viridis',
                         alpha=0.5, edgecolors='black', linewidth=0.5)
    plt.colorbar(scatter, label='Sharpe Ratio')
    
    # Plot max Sharpe ratio portfolio
    plt.scatter(max_sharpe_port['volatility'], max_sharpe_port['return'],
               marker='*', color='red', s=500, edgecolors='black', linewidth=2,
               label=f"Max Sharpe Ratio\n(SR={max_sharpe_port['sharpe']:.3f})")
    
    # Plot minimum variance portfolio if provided
    if min_var_port:
        plt.scatter(min_var_port['volatility'], min_var_port['return'],
                   marker='P', color='blue', s=300, edgecolors='black', linewidth=2,
                   label='Minimum Variance')
    
    # Plot Capital Allocation Line (tangent line)
    # CAL: E(R_p) = R_f + Sharpe * sigma_p
    max_vol = portfolios['Volatility'].max()
    cal_x = np.linspace(0, max_vol * 1.1, 100)
    cal_y = risk_free_rate + max_sharpe_port['sharpe'] * cal_x
    plt.plot(cal_x, cal_y, 'r--', linewidth=2, label='Capital Allocation Line')
    
    # Plot risk-free rate point
    plt.scatter(0, risk_free_rate, marker='o', color='green', s=200,
               edgecolors='black', linewidth=2, label='Risk-Free Rate')
    
    # Labels and formatting
    plt.title('Efficient Frontier and Capital Allocation Line\n(Markowitz Portfolio Theory)',
             fontsize=16, fontweight='bold')
    plt.xlabel('Volatility (Standard Deviation)', fontsize=12)
    plt.ylabel('Expected Return', fontsize=12)
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Format axes as percentages
    ax = plt.gca()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.1%}'.format(x)))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def print_portfolio_summary(tickers: list, max_sharpe_port: dict,
                           min_var_port: Optional[dict] = None,
                           stats: Optional[dict] = None):
    """
    Print a summary of portfolio analysis results.
    
    Args:
        tickers: List of ticker symbols
        max_sharpe_port: Dictionary with max Sharpe ratio portfolio info
        min_var_port: Optional dictionary with minimum variance portfolio info
        stats: Optional dictionary with portfolio statistics
    """
    print("\n" + "="*70)
    print("PORTFOLIO OPTIMIZATION SUMMARY")
    print("="*70)
    
    if stats:
        print("\nAsset Statistics (Annualized):")
        print("-" * 70)
        for i, ticker in enumerate(tickers):
            mean_ret = stats['mean_returns'].iloc[i]
            print(f"{ticker:>6}: Expected Return = {mean_ret:>7.2%}")
        
        print("\nCorrelation Matrix:")
        print("-" * 70)
        print(stats['corr_matrix'].to_string())
    
    print("\n\nOptimal Portfolio (Maximum Sharpe Ratio):")
    print("-" * 70)
    print(f"Expected Return:  {max_sharpe_port['return']:>7.2%}")
    print(f"Volatility:       {max_sharpe_port['volatility']:>7.2%}")
    print(f"Sharpe Ratio:     {max_sharpe_port['sharpe']:>7.3f}")
    print("\nAsset Allocation:")
    for i, ticker in enumerate(tickers):
        weight = max_sharpe_port['weights'][i]
        print(f"{ticker:>6}: {weight:>7.2%}")
    
    if min_var_port:
        print("\n\nMinimum Variance Portfolio:")
        print("-" * 70)
        print(f"Expected Return:  {min_var_port['return']:>7.2%}")
        print(f"Volatility:       {min_var_port['volatility']:>7.2%}")
        print("\nAsset Allocation:")
        for i, ticker in enumerate(tickers):
            weight = min_var_port['weights'][i]
            print(f"{ticker:>6}: {weight:>7.2%}")
    
    print("\n" + "="*70 + "\n")
