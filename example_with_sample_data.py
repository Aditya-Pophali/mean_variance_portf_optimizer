"""
Example script demonstrating portfolio optimization with synthetic data.
Use this when real data cannot be fetched from Yahoo Finance.
"""
from sample_data_generator import generate_sample_stock_data
from data_fetcher import calculate_returns
from portfolio_stats import calculate_portfolio_stats
from efficient_frontier import (
    generate_efficient_frontier,
    find_max_sharpe_portfolio,
    find_min_variance_portfolio
)
from visualizer import plot_efficient_frontier, print_portfolio_summary


def main():
    """
    Run portfolio optimization with sample data.
    """
    print("\n" + "="*70)
    print("MEAN-VARIANCE PORTFOLIO OPTIMIZER")
    print("Based on Markowitz Portfolio Theory")
    print("Using Synthetic Sample Data")
    print("="*70)
    
    # Configuration
    tickers = ['AAPL', 'MSFT']
    start_date = '2021-01-01'
    end_date = '2023-12-31'
    num_portfolios = 10000
    risk_free_rate = 0.02
    
    # Step 1: Generate sample data
    print(f"\n[1/6] Generating sample data for {', '.join(tickers)}...")
    print(f"      Period: {start_date} to {end_date}")
    prices = generate_sample_stock_data(tickers, start_date, end_date)
    print(f"      Generated {len(prices)} days of data")
    
    # Step 2: Calculate returns
    print("\n[2/6] Calculating daily returns...")
    returns = calculate_returns(prices)
    print(f"      Calculated {len(returns)} daily returns")
    
    # Step 3: Calculate portfolio statistics
    print("\n[3/6] Computing portfolio statistics...")
    stats = calculate_portfolio_stats(returns)
    
    # Step 4: Generate efficient frontier
    print(f"\n[4/6] Generating efficient frontier ({num_portfolios:,} simulations)...")
    portfolios = generate_efficient_frontier(
        stats['mean_returns'],
        stats['cov_matrix'],
        num_portfolios
    )
    
    # Step 5: Find optimal portfolios
    print("\n[5/6] Finding optimal portfolios...")
    max_sharpe_port = find_max_sharpe_portfolio(
        stats['mean_returns'],
        stats['cov_matrix'],
        risk_free_rate
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
    print_portfolio_summary(tickers, max_sharpe_port, min_var_port, stats)
    
    # Create plot
    plot_efficient_frontier(
        portfolios,
        max_sharpe_port,
        min_var_port,
        risk_free_rate,
        save_path='efficient_frontier.png'
    )
    
    print("\n✓ Portfolio optimization complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
