# Mean-Variance Portfolio Optimizer

A clean, modular, and reproducible implementation of Mean-Variance Portfolio Optimization based on **Markowitz Portfolio Theory**. This project uses historical stock data to construct and analyze portfolios, compute the efficient frontier, identify the maximum Sharpe ratio portfolio, and visualize the Capital Allocation Line.

## Features

- **Data Fetching**: Automatically downloads historical stock data using Yahoo Finance
- **Portfolio Statistics**: Computes returns, covariance matrix, and correlation matrix
- **Efficient Frontier**: Generates the efficient frontier through portfolio simulations
- **Optimization**: Identifies the maximum Sharpe ratio and minimum variance portfolios
- **Visualization**: Creates professional plots showing:
  - Efficient frontier
  - Capital Allocation Line (tangent line)
  - Optimal portfolios
  - Risk-free rate point

## Project Structure

```
mean_variance_portf_optimizer/
├── main.py                  # Main execution script
├── data_fetcher.py         # Data downloading and return calculation
├── portfolio_stats.py      # Portfolio statistics calculations
├── efficient_frontier.py   # Efficient frontier and optimization
├── visualizer.py           # Plotting and result display
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Aditya-Pophali/mean_variance_portf_optimizer.git
cd mean_variance_portf_optimizer
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage (AAPL and MSFT)

Run the optimizer with default settings (AAPL and MSFT stocks, 3 years of data):

```bash
python main.py
```

### Custom Usage

Customize the analysis with command-line arguments:

```bash
python main.py --tickers AAPL MSFT GOOGL --start-date 2020-01-01 --end-date 2023-12-31 --num-portfolios 15000 --risk-free-rate 0.03 --save-plot output.png
```

### Available Arguments

- `--tickers`: Stock ticker symbols (default: AAPL MSFT)
- `--start-date`: Start date for historical data in YYYY-MM-DD format (default: 3 years ago)
- `--end-date`: End date for historical data in YYYY-MM-DD format (default: today)
- `--num-portfolios`: Number of random portfolios to simulate (default: 10000)
- `--risk-free-rate`: Risk-free rate for Sharpe ratio calculation (default: 0.02)
- `--save-plot`: Path to save the plot (optional)

## Methodology

### Markowitz Portfolio Theory

The project implements the following key concepts from Markowitz Portfolio Theory:

1. **Expected Return**: Calculated as the weighted average of individual asset returns
   ```
   E(Rp) = Σ wi * E(Ri)
   ```

2. **Portfolio Variance**: Accounts for both individual variances and covariances
   ```
   σp² = Σ Σ wi * wj * σij
   ```

3. **Sharpe Ratio**: Measures risk-adjusted return
   ```
   SR = (E(Rp) - Rf) / σp
   ```

4. **Efficient Frontier**: The set of optimal portfolios offering the highest expected return for a given level of risk

5. **Capital Allocation Line (CAL)**: The tangent line from the risk-free rate to the efficient frontier, representing the best possible combination of risky and risk-free assets

### Optimization Process

1. Fetch historical stock prices using Yahoo Finance API
2. Calculate daily returns and annualize statistics
3. Generate thousands of random portfolios with different weight combinations
4. Use scipy optimization to find:
   - Maximum Sharpe ratio portfolio (optimal risk-adjusted returns)
   - Minimum variance portfolio (lowest risk)
5. Plot the efficient frontier and Capital Allocation Line

## Example Output

The program provides:

1. **Console Summary**: Displays asset statistics, correlation matrix, and optimal portfolio allocations
2. **Visualization**: Shows the efficient frontier with color-coded Sharpe ratios and marked optimal portfolios

### Sample Console Output

```
======================================================================
PORTFOLIO OPTIMIZATION SUMMARY
======================================================================

Asset Statistics (Annualized):
----------------------------------------------------------------------
  AAPL: Expected Return =  25.34%
  MSFT: Expected Return =  28.12%

Correlation Matrix:
----------------------------------------------------------------------
          AAPL      MSFT
AAPL  1.000000  0.734521
MSFT  0.734521  1.000000


Optimal Portfolio (Maximum Sharpe Ratio):
----------------------------------------------------------------------
Expected Return:   27.45%
Volatility:        22.13%
Sharpe Ratio:      1.149

Asset Allocation:
  AAPL:  35.67%
  MSFT:  64.33%
```

## Mathematical Background

### Key Formulas

**Portfolio Return:**
```
Rp = w1*R1 + w2*R2 + ... + wn*Rn
```

**Portfolio Volatility (Two Assets):**
```
σp = √(w1²σ1² + w2²σ2² + 2w1w2σ1σ2ρ12)
```

**Sharpe Ratio:**
```
SR = (Rp - Rf) / σp
```

Where:
- `wi` = weight of asset i
- `Ri` = return of asset i
- `σi` = standard deviation of asset i
- `ρ12` = correlation between assets 1 and 2
- `Rf` = risk-free rate

## Requirements

- Python 3.8+
- numpy >= 1.24.0
- pandas >= 2.0.0
- matplotlib >= 3.7.0
- yfinance >= 0.2.0
- scipy >= 1.10.0

## Module Descriptions

### data_fetcher.py
- `fetch_stock_data()`: Downloads historical price data
- `calculate_returns()`: Computes daily returns from prices

### portfolio_stats.py
- `calculate_portfolio_stats()`: Computes mean returns, covariance, and correlation
- `portfolio_performance()`: Calculates portfolio return and volatility
- `calculate_sharpe_ratio()`: Computes Sharpe ratio

### efficient_frontier.py
- `generate_efficient_frontier()`: Creates random portfolio simulations
- `find_max_sharpe_portfolio()`: Optimizes for maximum Sharpe ratio
- `find_min_variance_portfolio()`: Finds minimum variance portfolio

### visualizer.py
- `plot_efficient_frontier()`: Creates the main visualization
- `print_portfolio_summary()`: Displays results in console

### main.py
- Orchestrates the complete workflow
- Handles command-line arguments
- Coordinates all modules

## License

This project is open source and available for educational purposes.

## Author

Aditya Pophali

## References

- Markowitz, H. (1952). "Portfolio Selection". The Journal of Finance.
- Modern Portfolio Theory and Investment Analysis (textbook references)
