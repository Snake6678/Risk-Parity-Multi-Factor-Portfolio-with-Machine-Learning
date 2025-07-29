"""
risk_parity_multi_factor
========================

This module provides a compact but comprehensive demonstration of how to
construct a simple, multi‑factor portfolio and allocate capital using a
risk parity approach.  The end goal is to showcase a realistic
quantitative workflow for financial markets that emphasises good
software engineering practices.  Recruiters at hedge funds and other
quantitative firms often look for candidates who can combine data
ingestion, feature engineering, machine learning and portfolio
optimisation into a single, reproducible project.  This example is
designed to be self‑contained so that it runs without any external
dependencies or API keys.

Key components
--------------

1. **Data loading and generation** – The `DataLoader` class reads daily
   price data for the SPDR S&P 500 ETF (ticker: SPY) from a
   pre‑downloaded CSV file.  Because access to live market data is
   restricted in this environment, two additional synthetic assets
   (``BondProxy`` and ``GoldProxy``) are generated from the SPY returns
   using simple linear transformations plus small amounts of random
   noise.  These proxies serve as stand‑ins for bonds and gold so that
   the portfolio optimisation stage can allocate risk across multiple
   asset classes.

2. **Feature engineering** – The `FeatureEngineer` class computes a
   handful of well‑known technical indicators (returns, momentum,
   volatility and a simple relative strength index) for each asset.  In
   practice, more sophisticated features could be incorporated,
   including fundamental factors or macroeconomic variables.  The use
   of multiple factors to explain returns is motivated by multi‑factor
   models from finance, which employ more than one factor to explain
   asset prices【380549786142635†L230-L247】.

3. **Machine learning models** – The `ModelTrainer` class builds a
   separate `RandomForestRegressor` for each asset to forecast
   forward‑looking returns (here we predict five‑day ahead returns).
   Random forests are robust to non‑linear relationships and can
   capture interactions between features.  Models are trained on an
   expanding window of historical data up to a configurable cutoff
   date, and then used to generate out‑of‑sample predictions.

4. **Risk parity optimisation** – The `PortfolioOptimizer` class
   implements a risk parity objective using `scipy.optimize.minimize`.
   Risk parity is a portfolio construction technique that equalises the
   contribution of each asset to the overall portfolio risk【391788100410004†L211-L231】【452951232785943†L31-L41】.  In
   contrast to a traditional 60/40 stock‑bond portfolio, risk parity
   allocates capital based on volatility so that no single asset
   dominates the risk budget.  The objective function minimises the
   squared deviations of individual risk contributions from their
   average.  Non‑negativity and full‑investment constraints ensure that
   weights sum to one and are long‑only.

5. **Demonstration** – The `main` function ties everything together.
   It loads the data, constructs features and targets, trains the
   models, generates predictions for the most recent trading day,
   computes a covariance matrix from trailing returns and solves for
   risk parity weights.  The resulting expected returns and portfolio
   weights are printed to the console.  Users can modify the list of
   tickers, the forecast horizon and the train/test split without
   altering the core logic.

Although this example uses synthetic data for two of the assets, the
structure of the code mirrors what one would do with real market data:
download prices, compute factors, fit predictive models and optimise
weights.  For further reading on risk parity and multi‑factor models,
see the references at the end of the accompanying README.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from scipy.optimize import minimize


class DataLoader:
    """Load and prepare price data for multiple assets.

    The SPY price data is read from a CSV file containing daily Open,
    High, Low, Close and Volume columns (in Polish).  Synthetic bond
    and gold proxies are generated from the SPY returns.  The loader
    returns a single DataFrame of price series indexed by date with
    columns for each asset.
    """

    def __init__(self, spy_csv_path: str) -> None:
        self.spy_csv_path = spy_csv_path

    def load_spy(self) -> pd.DataFrame:
        """Load SPY prices from a CSV file.

        Returns
        -------
        pd.DataFrame
            DataFrame indexed by date with a single column ``SPY``
            containing adjusted closing prices.
        """
        df = pd.read_csv(self.spy_csv_path)
        # Rename Polish column names to English for clarity
        df = df.rename(
            columns={
                "Data": "Date",
                "Otwarcie": "Open",
                "Najwyzszy": "High",
                "Najnizszy": "Low",
                "Zamkniecie": "Close",
                "Wolumen": "Volume",
            }
        )
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index()
        # Use close prices as adjusted closes (Stooq data is already adjusted)
        prices = df[["Close"]].rename(columns={"Close": "SPY"})
        return prices

    @staticmethod
    def generate_synthetic_returns(
        base_returns: pd.Series, seed: int = 42
    ) -> Tuple[pd.Series, pd.Series]:
        """Generate synthetic bond and gold returns from SPY returns.

        Parameters
        ----------
        base_returns : pd.Series
            Daily returns of the SPY series.
        seed : int, optional
            Seed for the random number generator, by default 42.

        Returns
        -------
        Tuple[pd.Series, pd.Series]
            Returns for the bond proxy and gold proxy.
        """
        rng = np.random.default_rng(seed)
        # Bond proxy: less volatile and positively correlated with SPY
        bond_noise = rng.normal(0, 0.002, size=len(base_returns))
        bond_returns = 0.3 * base_returns + bond_noise
        # Gold proxy: negatively correlated with SPY
        gold_noise = rng.normal(0, 0.003, size=len(base_returns))
        gold_returns = -0.2 * base_returns + gold_noise
        return bond_returns, gold_returns

    def load_data(self) -> pd.DataFrame:
        """Load price series for SPY and synthetic proxies.

        Returns
        -------
        pd.DataFrame
            Price series for SPY, BondProxy and GoldProxy indexed by date.
        """
        spy_prices = self.load_spy()
        # Compute daily returns for SPY
        spy_returns = spy_prices.pct_change().dropna()
        bond_ret, gold_ret = self.generate_synthetic_returns(spy_returns["SPY"])
        # Construct synthetic price series starting at 1.0
        bond_prices = (1 + pd.Series(bond_ret, index=spy_returns.index)).cumprod()
        gold_prices = (1 + pd.Series(gold_ret, index=spy_returns.index)).cumprod()
        combined = pd.DataFrame(
            {
                "SPY": spy_prices.loc[spy_returns.index, "SPY"],
                "BondProxy": bond_prices,
                "GoldProxy": gold_prices,
            },
            index=spy_returns.index,
        )
        return combined


class FeatureEngineer:
    """Compute technical indicators and assemble feature matrix."""

    @staticmethod
    def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
        return prices.pct_change().dropna()

    @staticmethod
    def compute_momentum(prices: pd.DataFrame, window: int) -> pd.DataFrame:
        return prices / prices.shift(window) - 1

    @staticmethod
    def compute_volatility(returns: pd.DataFrame, window: int) -> pd.DataFrame:
        return returns.rolling(window).std()

    @staticmethod
    def compute_rsi(prices: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """Compute the Relative Strength Index (RSI) for each asset.

        RSI measures the magnitude of recent price changes to evaluate overbought
        or oversold conditions.  The implementation here follows the
        standard definition using exponential moving averages of gains
        and losses.
        """
        delta = prices.diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        gain = up.ewm(alpha=1 / window, adjust=False).mean()
        loss = down.ewm(alpha=1 / window, adjust=False).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def build_features(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Assemble a DataFrame of engineered features.

        Parameters
        ----------
        prices : pd.DataFrame
            Price series for each asset.

        Returns
        -------
        pd.DataFrame
            DataFrame containing features for each asset with multi‑level
            column indexing (asset, feature).
        """
        returns = self.compute_returns(prices)
        momentum_5 = self.compute_momentum(prices, 5)
        momentum_20 = self.compute_momentum(prices, 20)
        vol_20 = self.compute_volatility(returns, 20)
        rsi_14 = self.compute_rsi(prices, 14)
        # Align indices and concatenate features
        features = []
        feature_names = ["return", "momentum_5", "momentum_20", "vol_20", "rsi_14"]
        for asset in prices.columns:
            df = pd.concat(
                [
                    returns[asset],
                    momentum_5[asset],
                    momentum_20[asset],
                    vol_20[asset],
                    rsi_14[asset],
                ],
                axis=1,
            )
            df.columns = pd.MultiIndex.from_product([[asset], feature_names])
            features.append(df)
        feature_df = pd.concat(features, axis=1).dropna()
        return feature_df


class ModelTrainer:
    """Fit and manage machine learning models for return prediction."""

    def __init__(self, horizon: int = 5, random_state: int = 0) -> None:
        self.horizon = horizon
        self.random_state = random_state
        self.models: Dict[str, RandomForestRegressor] = {}

    def prepare_target(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Compute forward returns for each asset over the specified horizon.

        Parameters
        ----------
        prices : pd.DataFrame
            Price series for each asset.

        Returns
        -------
        pd.DataFrame
            Forward returns for each asset.
        """
        returns = prices.pct_change().shift(-self.horizon)
        return returns

    def split_data(
        self, features: pd.DataFrame, targets: pd.DataFrame, split_date: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split features and targets into training and testing sets based on a date.

        Parameters
        ----------
        features : pd.DataFrame
            Feature matrix with multi‑level columns.
        targets : pd.DataFrame
            Forward returns for each asset.
        split_date : str
            Date string (YYYY‑MM‑DD) specifying the end of the training period.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
            X_train, X_test, y_train, y_test dataframes.
        """
        mask = features.index <= pd.to_datetime(split_date)
        X_train = features.loc[mask]
        X_test = features.loc[~mask]
        y_train = targets.loc[mask]
        y_test = targets.loc[~mask]
        return X_train, X_test, y_train, y_test

    def fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame) -> None:
        """Fit a RandomForest model for each asset.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training feature matrix.
        y_train : pd.DataFrame
            Training targets.
        """
        for asset in y_train.columns:
            # Extract features corresponding to this asset and all assets
            # In this simple example we use all features for all assets
            model = RandomForestRegressor(
                n_estimators=200,
                max_depth=6,
                random_state=self.random_state,
            )
            model.fit(X_train, y_train[asset])
            self.models[asset] = model

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict forward returns using the trained models.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix for prediction.

        Returns
        -------
        pd.DataFrame
            DataFrame of predicted returns indexed like ``X``.
        """
        preds = {}
        for asset, model in self.models.items():
            preds[asset] = model.predict(X)
        return pd.DataFrame(preds, index=X.index)


class PortfolioOptimizer:
    """Compute risk parity weights given a covariance matrix."""

    @staticmethod
    def _portfolio_variance(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
        return float(weights.T @ cov_matrix @ weights)

    @staticmethod
    def _risk_contribution(weights: np.ndarray, cov_matrix: np.ndarray) -> np.ndarray:
        port_var = PortfolioOptimizer._portfolio_variance(weights, cov_matrix)
        # Marginal contribution: Σw
        marginal = cov_matrix @ weights
        contrib = weights * marginal / port_var
        return contrib

    @staticmethod
    def _risk_parity_objective(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
        rc = PortfolioOptimizer._risk_contribution(weights, cov_matrix)
        target = np.mean(rc)
        return float(((rc - target) ** 2).sum())

    def solve_risk_parity(self, cov_matrix: np.ndarray) -> np.ndarray:
        """Solve for risk parity weights using SLSQP optimisation.

        Parameters
        ----------
        cov_matrix : np.ndarray
            Covariance matrix of asset returns.

        Returns
        -------
        np.ndarray
            Optimal weights that equalise risk contributions.
        """
        n = cov_matrix.shape[0]
        x0 = np.ones(n) / n
        constraints = (
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "ineq", "fun": lambda w: w},
        )
        bounds = [(0.0, 1.0) for _ in range(n)]
        result = minimize(
            self._risk_parity_objective,
            x0,
            args=(cov_matrix,),
            method="SLSQP",
            constraints=constraints,
            bounds=bounds,
            options={"ftol": 1e-12, "disp": False},
        )
        if not result.success:
            raise RuntimeError(f"Risk parity optimisation failed: {result.message}")
        return result.x


def main() -> None:
    """Run an end‑to‑end demonstration of the algorithm.

    The demo loads price data, builds features, trains predictive models,
    generates expected returns and computes risk parity weights based on
    trailing covariance.  It prints the final expected returns and
    portfolio weights for inspection.
    """
    # ----------------------- Configuration -----------------------
    spy_csv = "spy_us_d.csv"
    train_end = "2023-12-31"  # last date for training
    forecast_horizon = 5       # number of days ahead to forecast

    # ----------------------- Data Loading -----------------------
    loader = DataLoader(spy_csv)
    prices = loader.load_data()

    # ------------------- Feature Engineering --------------------
    fe = FeatureEngineer()
    features = fe.build_features(prices)

    # ----------------------- Target Prep ------------------------
    trainer = ModelTrainer(horizon=forecast_horizon)
    targets = trainer.prepare_target(prices)
    # Align targets with features index
    targets = targets.reindex(features.index)

    # -------------------- Train/Test Split ----------------------
    X_train, X_test, y_train, y_test = trainer.split_data(features, targets, train_end)
    # Remove rows with NaNs in targets (due to horizon shift)
    mask_valid = ~y_train.isna().any(axis=1)
    X_train = X_train.loc[mask_valid]
    y_train = y_train.loc[mask_valid]

    # -------------------------- Training ------------------------
    trainer.fit(X_train, y_train)

    # ---------------------- Prediction --------------------------
    # Use the last row of the test set for forecasting
    if not X_test.empty:
        X_last = X_test.tail(1)
    else:
        X_last = X_train.tail(1)
    predicted_returns = trainer.predict(X_last).iloc[0]

    # ------------------- Covariance Estimation ------------------
    # Use returns from the last 60 observations for covariance
    recent_returns = prices.pct_change().dropna().tail(60)
    cov_matrix = recent_returns.cov().values

    # --------------------- Risk Parity Weights ------------------
    optimizer = PortfolioOptimizer()
    weights = optimizer.solve_risk_parity(cov_matrix)

    # Print results
    assets = list(prices.columns)
    # Print header for predicted returns.  Using f-strings avoids issues
    # with format specifiers when non-ASCII characters are present.
    print(f"Predicted next {forecast_horizon}-day returns (%)")
    for asset, ret in predicted_returns.items():
        print(f"  {asset:10s}: {ret * 100:.3f}")
    print("\nRisk parity weights:")
    for asset, w in zip(assets, weights):
        print(f"  {asset:10s}: {w:.3f}")
    # Expected portfolio return (dot product of predictions and weights)
    exp_portfolio_return = float(predicted_returns @ weights)
    print(f"\nExpected portfolio return over {forecast_horizon} days: {exp_portfolio_return * 100:.3f}%")


if __name__ == "__main__":
    main()
