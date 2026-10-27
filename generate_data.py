"""
Generate a realistic synthetic daily stock-price dataset for 3 fictional
companies across 2 sectors, over 3 years of trading days.
(Synthetic data is used because this sandbox has no internet access;
the code and methodology below apply identically to real market data
pulled from Yahoo Finance / NSE / BSE with yfinance.)
"""
import numpy as np
import pandas as pd

np.random.seed(42)

start_date = "2023-01-02"
n_days = 750  # ~3 trading years
dates = pd.bdate_range(start=start_date, periods=n_days)

companies = {
    "TECHNOVA":  {"start_price": 850.0,  "drift": 0.00055, "vol": 0.018, "sector": "Technology"},
    "GREENFIN":  {"start_price": 420.0,  "drift": 0.00035, "vol": 0.014, "sector": "Financial Services"},
    "AGROMART":  {"start_price": 210.0,  "drift": 0.00020, "vol": 0.022, "sector": "Consumer Retail"},
}

frames = []
for ticker, p in companies.items():
    n = n_days
    # Geometric Brownian motion for the base price path
    daily_returns = np.random.normal(p["drift"], p["vol"], n)

    # add mild seasonality (quarterly earnings bumps) + a mid-series shock
    t = np.arange(n)
    seasonality = 0.004 * np.sin(2 * np.pi * t / 63)
    shock = np.zeros(n)
    shock_day = int(n * 0.55)
    shock[shock_day:shock_day + 10] = -0.02  # a short correction event
    daily_returns = daily_returns + seasonality + shock

    price = p["start_price"] * np.exp(np.cumsum(daily_returns))
    open_p = price * (1 + np.random.normal(0, 0.003, n))
    high_p = np.maximum(open_p, price) * (1 + np.abs(np.random.normal(0, 0.005, n)))
    low_p = np.minimum(open_p, price) * (1 - np.abs(np.random.normal(0, 0.005, n)))
    volume = np.random.lognormal(mean=13.5, sigma=0.35, size=n).astype(int)

    df = pd.DataFrame({
        "Date": dates,
        "Ticker": ticker,
        "Sector": p["sector"],
        "Open": open_p.round(2),
        "High": high_p.round(2),
        "Low": low_p.round(2),
        "Close": price.round(2),
        "Volume": volume,
    })
    frames.append(df)

data = pd.concat(frames, ignore_index=True)

# introduce a few realistic missing values to make cleaning meaningful
mask_idx = np.random.choice(data.index, size=15, replace=False)
data.loc[mask_idx, "Volume"] = np.nan

data.to_csv("/home/claude/finance_project/stock_data.csv", index=False)
print(data.shape)
print(data.head())
print(data.isna().sum())
