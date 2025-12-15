"""
Main script for Mean-Variance Portfolio Optimization.

This script demonstrates the complete workflow of portfolio optimization using
Markowitz Portfolio Theory for a two-asset portfolio (AAPL and MSFT).
"""
import argparse
from datetime import datetime, timedelta
from data_fetcher import fetch_stock_data, calculate_returns
from portfolio_stats import calculate_portfolio_stats
from efficient_frontier import (
    generate_efficient_frontier,
    find_max_sharpe_portfolio,
    find_min_variance_portfolio
)
from visualizer import plot_efficient_frontier, print_portfolio_summary


def main():
    """
    Main execution function for portfolio optimization.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Mean-Variance Portfolio Optimizer (Markowitz Theory)'
    )
    parser.add_argument(
        '--tickers',
        nargs='+',
        default=['AAPL', 'MSFT'],
        help='Stock ticker symbols (default: AAPL MSFT)'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        default=(datetime.now() - timedelta(days=3*365)).strftime('%Y-%m-%d'),
        help='Start date for historical data (YYYY-MM-DD, default: 3 years ago)'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        default=datetime.now().strftime('%Y-%m-%d'),
        help='End date for historical data (YYYY-MM-DD, default: today)'
    )
    parser.add_argument(
        '--num-portfolios',
        type=int,
        default=10000,
        help='Number of random portfolios to simulate (default: 10000)'
    )
    parser.add_argument(
        '--risk-free-rate',
        type=float,
        default=0.02,
        help='Risk-free rate for Sharpe ratio calculation (default: 0.02)'
    )
    parser.add_argument(
        '--save-plot',
        type=str,
        default=None,
        help='Path to save the plot (optional)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("MEAN-VARIANCE PORTFOLIO OPTIMIZER")
    print("Based on Markowitz Portfolio Theory")
    print("="*70)
    
    # Step 1: Fetch historical data
    print(f"\n[1/6] Fetching historical data for {', '.join(args.tickers)}...")
    print(f"      Period: {args.start_date} to {args.end_date}")
    prices = fetch_stock_data(args.tickers, args.start_date, args.end_date)
    print(f"      Retrieved {len(prices)} days of data")
    
    # Step 2: Calculate returns
    print("\n[2/6] Calculating daily returns...")
    returns = calculate_returns(prices)
    print(f"      Calculated {len(returns)} daily returns")
    
    # Step 3: Calculate portfolio statistics
    print("\n[3/6] Computing portfolio statistics...")
    stats = calculate_portfolio_stats(returns)
    
    # Step 4: Generate efficient frontier
    print(f"\n[4/6] Generating efficient frontier ({args.num_portfolios:,} simulations)...")
    portfolios = generate_efficient_frontier(
        stats['mean_returns'],
        stats['cov_matrix'],
        args.num_portfolios
    )
    
    # Step 5: Find optimal portfolios
    print("\n[5/6] Finding optimal portfolios...")
    max_sharpe_port = find_max_sharpe_portfolio(
        stats['mean_returns'],
        stats['cov_matrix'],
        args.risk_free_rate
    )
    min_var_port = find_min_variance_portfolio(
        stats['mean_returns'],
        stats['cov_matrix']
    )
    print("      ✓ Maximum Sharpe ratio portfolio identified")
    print("      ✓ Minimum variance portfolio identified")
    
    # Step 6: Visualize results
    print("\n[6/6] Generating visualization...")
    
    # Print summary to console
    print_portfolio_summary(args.tickers, max_sharpe_port, min_var_port, stats)
    
    # Create plot
    plot_efficient_frontier(
        portfolios,
        max_sharpe_port,
        min_var_port,
        args.risk_free_rate,
        args.save_plot
    )
    
    print("\n✓ Portfolio optimization complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
