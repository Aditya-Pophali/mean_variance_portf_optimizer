# Mean-Variance Portfolio Optimizer (Markowitz Theory)

This project implements a clean, modular, and reproducible Mean-Variance Portfolio Optimizer based on Markowitz Portfolio Theory. It uses historical stock data (e.g., AAPL and MSFT) to construct and analyze a two-asset portfolio, compute the efficient frontier, identify the maximum Sharpe ratio portfolio, and visualize the Capital Allocation Line (tangent line).

## Key Features
- Reusable Python module `markowitz.py` with:
  - Daily and cumulative returns computation
  - Expected returns and covariance matrix (annualized)
  - Portfolio performance metrics (return, volatility, Sharpe)
  - Two-asset efficient frontier generation
  - Maximum Sharpe ratio optimization
  - Capital Allocation Line computation
  - Validation utilities to confirm optimality
- Jupyter notebook `simple_markowitz_optimizer.ipynb` for:
  - Data loading via yfinance
  - Parameter estimation
  - Efficient frontier computation and visualization
  - Optimal portfolio selection and validation
  - Publication-quality plots and a comprehensive markdown report

## Visualizations
- `daily_returns.png`: Individual daily returns of AAPL and MSFT
- `cumulative_returns.png`: Cumulative returns of AAPL, MSFT, and the optimal portfolio
- `efficient_frontier.png`: Efficient frontier with the optimal Sharpe ratio portfolio and Capital Allocation Line

## How It Works
1. Download adjusted close prices for two assets using yfinance or load from CSV.
2. Compute daily returns; estimate annualized expected returns and covariance matrix.
3. Generate the efficient frontier by sweeping weights between 0 and 1.
4. Optimize to find the portfolio weights that maximize the Sharpe ratio (default risk-free rate 0%).
5. Plot the efficient frontier, mark the optimal portfolio, and draw the tangent line.
6. Validate optimality using a grid search over weights (deterministic historical returns ensure a unique solution).

## Files
- `markowitz.py` — Core reusable module with typed, documented functions
- `simple_markowitz_optimizer.ipynb` — End-to-end analysis and visuals
- `daily_returns.png` — Daily returns plot
- `cumulative_returns.png` — Cumulative returns plot
- `efficient_frontier.png` — Efficient frontier + CAL

## Installation
Use a Python 3.10+ environment with the following packages:

```
pip install numpy pandas yfinance matplotlib seaborn scipy
```

If using the notebook, ensure Jupyter is installed:

```
pip install jupyter
```

## Quick Start
1. Open `simple_markowitz_optimizer.ipynb` and run all cells.
2. Alternatively, import `markowitz.py` in your own scripts and use its functions to compute returns, optimize weights, and generate plots.

## Configuration
- Default tickers: `AAPL`, `MSFT`
- Date range: last 2 years (configurable in the notebook)
- Risk-free rate: `0.0` (configurable)

## Notes & Limitations
- Results depend on historical data; estimates of expected returns and covariance are sample-based and can be noisy.
- The current implementation focuses on a two-asset case for clarity and visualization.
- Extensions can include more assets, constraints, shrinkage estimators, factor models, and robust optimization methods.

## References
- Markowitz, H. (1952). "Portfolio Selection." The Journal of Finance.
- Sharpe, W. F. (1966). "Mutual Fund Performance." The Journal of Business.
- Merton, R. C. (1972). "An Analytic Derivation of the Efficient Portfolio Frontier." Journal of Financial and Quantitative Analysis.
