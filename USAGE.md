# Usage Guide

This guide provides detailed instructions on how to use the Mean-Variance Portfolio Optimizer.

## Quick Start

### Option 1: Using Real Market Data

```bash
# Install dependencies
pip install -r requirements.txt

# Run with default settings (AAPL and MSFT, last 3 years)
python main.py

# Customize the analysis
python main.py --tickers AAPL MSFT GOOGL --start-date 2020-01-01 --end-date 2023-12-31
```

**Note**: Requires internet access to download data from Yahoo Finance.

### Option 2: Using Sample Data (No Internet Required)

```bash
# Install dependencies
pip install -r requirements.txt

# Run with synthetic data
python example_with_sample_data.py
```

This is useful for testing, demonstrations, or when you don't have internet access.

## Command-Line Arguments

The `main.py` script accepts the following arguments:

| Argument | Description | Default |
|----------|-------------|---------|
| `--tickers` | Stock ticker symbols (space-separated) | `AAPL MSFT` |
| `--start-date` | Start date (YYYY-MM-DD format) | 3 years ago |
| `--end-date` | End date (YYYY-MM-DD format) | Today |
| `--num-portfolios` | Number of simulated portfolios | `10000` |
| `--risk-free-rate` | Annual risk-free rate (as decimal) | `0.02` (2%) |
| `--save-plot` | Path to save the visualization | None (displays only) |

## Examples

### Example 1: Two-Asset Portfolio (Default)

```bash
python main.py
```

Output:
- Console summary with statistics and optimal allocations
- Visualization of efficient frontier with Capital Allocation Line

### Example 2: Custom Date Range

```bash
python main.py --start-date 2019-01-01 --end-date 2023-12-31
```

### Example 3: Three-Asset Portfolio

```bash
python main.py --tickers AAPL MSFT GOOGL
```

### Example 4: Save Plot to File

```bash
python main.py --save-plot output/portfolio_analysis.png
```

### Example 5: Higher Precision with More Simulations

```bash
python main.py --num-portfolios 50000
```

### Example 6: Different Risk-Free Rate

If Treasury yields are at 4%, adjust accordingly:

```bash
python main.py --risk-free-rate 0.04
```

## Understanding the Output

### Console Output

The program displays:

1. **Asset Statistics**: Annualized expected returns for each asset
2. **Correlation Matrix**: Shows how assets move together
3. **Maximum Sharpe Ratio Portfolio**: The optimal risk-adjusted portfolio
   - Expected Return: Annual return
   - Volatility: Standard deviation (risk)
   - Sharpe Ratio: Risk-adjusted return metric
   - Asset Allocation: Percentage invested in each asset
4. **Minimum Variance Portfolio**: The lowest-risk portfolio

### Visualization

The plot shows:

- **Scatter Points**: Random portfolios colored by Sharpe ratio (yellow = higher)
- **Red Star**: Maximum Sharpe ratio portfolio (optimal)
- **Blue Pentagon**: Minimum variance portfolio
- **Red Dashed Line**: Capital Allocation Line (CAL) - tangent from risk-free rate
- **Green Circle**: Risk-free rate (zero volatility)

## Interpreting Results

### Sharpe Ratio
- **> 1.0**: Good risk-adjusted returns
- **> 2.0**: Excellent risk-adjusted returns
- **< 1.0**: May not be worth the risk

### Asset Allocation
The weights tell you how to distribute your capital:
- If AAPL = 40% and MSFT = 60%, invest $40 in AAPL for every $60 in MSFT

### Capital Allocation Line
Points on the CAL represent portfolios mixing the optimal risky portfolio with the risk-free asset:
- **Left of star**: Mix of risk-free asset + optimal portfolio
- **At star**: 100% in optimal portfolio
- **Right of star**: Leveraged position (borrowing at risk-free rate)

## Using as a Library

You can also use the modules programmatically:

```python
from data_fetcher import fetch_stock_data, calculate_returns
from portfolio_stats import calculate_portfolio_stats
from efficient_frontier import find_max_sharpe_portfolio
from visualizer import plot_efficient_frontier

# Fetch data
prices = fetch_stock_data(['AAPL', 'MSFT'], '2021-01-01', '2023-12-31')
returns = calculate_returns(prices)

# Calculate statistics
stats = calculate_portfolio_stats(returns)

# Find optimal portfolio
optimal = find_max_sharpe_portfolio(
    stats['mean_returns'],
    stats['cov_matrix'],
    risk_free_rate=0.02
)

print(f"Optimal weights: {optimal['weights']}")
print(f"Expected return: {optimal['return']:.2%}")
print(f"Sharpe ratio: {optimal['sharpe']:.3f}")
```

## Troubleshooting

### Issue: Cannot download data from Yahoo Finance

**Solution**: Use the sample data example:
```bash
python example_with_sample_data.py
```

### Issue: Plot doesn't display

**Solution**: Save to file instead:
```bash
python main.py --save-plot output.png
```

### Issue: "RuntimeWarning: invalid value encountered"

This usually means there's insufficient data or issues with data quality. Check:
- Date range is valid
- Ticker symbols are correct
- You have internet access for real data

## Best Practices

1. **Use Sufficient Data**: At least 1-3 years of data for reliable statistics
2. **Diversify**: Use 3+ assets for better diversification
3. **Regular Rebalancing**: Recalculate optimal weights quarterly or semi-annually
4. **Risk-Free Rate**: Update to match current Treasury yields
5. **Consider Transaction Costs**: Real-world trading has costs not reflected here
6. **Backtesting**: Test strategies on historical data before implementation

## Limitations

- **Historical data**: Past performance doesn't guarantee future results
- **Transaction costs**: Not included in calculations
- **Taxes**: Not considered
- **Market impact**: Assumes you can trade without affecting prices
- **Liquidity**: Assumes all assets are equally liquid
- **No short selling**: Current implementation only allows long positions (weights ≥ 0)

## Further Reading

- [Markowitz, H. (1952). Portfolio Selection](https://www.jstor.org/stable/2975974)
- [Modern Portfolio Theory on Wikipedia](https://en.wikipedia.org/wiki/Modern_portfolio_theory)
- [Efficient Frontier Explanation](https://www.investopedia.com/terms/e/efficientfrontier.asp)
