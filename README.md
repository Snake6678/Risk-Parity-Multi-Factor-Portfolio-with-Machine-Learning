# Risk Parity Multi-Factor Portfolio with Machine Learning

This project demonstrates how to build a multi-factor investment strategy with machine learning forecasts and allocate capital using **risk parity**. It is designed to showcase data processing, predictive modeling, and portfolio optimization in one reproducible workflow.

## Key Components

1. **Data Loading**
   - Loads SPY price data from a CSV file.
   - Two synthetic assets ("BondProxy" and "GoldProxy") are generated from SPY with noise.

2. **Feature Engineering**
   - Computes returns, momentum, volatility, and RSI for each asset.
   - Based on multi-factor models used in portfolio theory.

3. **Machine Learning**
   - Trains a Random Forest Regressor for each asset to predict 5-day returns.
   - Uses an expanding window training set and out-of-sample testing.

4. **Risk Parity Optimization**
   - Allocates capital so each asset contributes equally to total risk.
   - Solved using `scipy.optimize.minimize` under long-only and full-investment constraints.

5. **Demonstration**
   - The main script ties all components together and prints expected returns and portfolio weights.

## Financial Concepts Used

- **Multi-Factor Models**: Explain asset returns using multiple explanatory variables.  
- **Risk Parity**: A strategy that equalizes the contribution of each asset to portfolio volatility.

Sources:
- [Investopedia - Risk Parity](https://www.investopedia.com/terms/r/risk-parity.asp)
- [Investopedia - Multi-Factor Model](https://www.investopedia.com/terms/m/multifactor-model.asp)
- [LuxAlgo Blog on Risk Parity in Python](https://www.luxalgo.com/blog/risk-parity-allocation-with-python/)

## How to Run

1. Clone this repo or download the files.
2. Ensure you have Python 3.8+ and the required libraries (`numpy`, `pandas`, `sklearn`, `scipy`).
3. Run the script:
   ```bash
   python risk_parity_multi_factor.py
   ```

## Files

- `risk_parity_multi_factor.py`: The main script.
- `spy_us_d.csv`: Historical SPY price data used for modeling.
- `README.md`: This file.
