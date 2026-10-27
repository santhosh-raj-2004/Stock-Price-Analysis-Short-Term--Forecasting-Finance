"""
Real-World Data Project: Finance
End-to-end analysis of daily stock price data for 3 companies.
Steps: load & clean -> EDA -> feature engineering -> visualization ->
       prediction (next-30-day closing price) -> conclusions.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

plt.rcParams.update({"figure.dpi": 130, "font.size": 9})
OUT = "/home/claude/finance_project/charts"

# ---------- 1. LOAD & CLEAN ----------
df = pd.read_csv("/home/claude/finance_project/stock_data.csv", parse_dates=["Date"])
missing_before = df.isna().sum().sum()

# fill missing Volume per-ticker with the rolling median (robust to outliers)
df["Volume"] = df.groupby("Ticker")["Volume"].transform(
    lambda s: s.fillna(s.median())
)
df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
missing_after = df.isna().sum().sum()

# ---------- 2. FEATURE ENGINEERING ----------
def add_features(g):
    g = g.copy()
    g["Daily_Return_%"] = g["Close"].pct_change() * 100
    g["MA20"] = g["Close"].rolling(20).mean()
    g["MA50"] = g["Close"].rolling(50).mean()
    g["Volatility_20d"] = g["Daily_Return_%"].rolling(20).std()
    return g

df = df.groupby("Ticker", group_keys=False)[df.columns.tolist()].apply(add_features)

# ---------- 3. EDA SUMMARY ----------
summary = df.groupby("Ticker").agg(
    Start_Price=("Close", "first"),
    End_Price=("Close", "last"),
    Mean_Close=("Close", "mean"),
    Max_Close=("Close", "max"),
    Min_Close=("Close", "min"),
    Avg_Daily_Volume=("Volume", "mean"),
    Volatility_Std=("Daily_Return_%", "std"),
).round(2)
summary["Total_Return_%"] = ((summary["End_Price"] / summary["Start_Price"]) - 1) * 100
summary.to_csv("/home/claude/finance_project/summary_stats.csv")
print(summary)

# ---------- 4. VISUALIZATIONS ----------
tickers = df["Ticker"].unique()
colors = {"TECHNOVA": "#2563eb", "GREENFIN": "#16a34a", "AGROMART": "#dc2626"}

# 4a. Price trend with moving averages (one panel per ticker)
fig, axes = plt.subplots(len(tickers), 1, figsize=(9, 9), sharex=True)
for ax, tk in zip(axes, tickers):
    sub = df[df["Ticker"] == tk]
    ax.plot(sub["Date"], sub["Close"], label="Close", color=colors[tk], lw=1)
    ax.plot(sub["Date"], sub["MA20"], label="MA20", color="orange", lw=1)
    ax.plot(sub["Date"], sub["MA50"], label="MA50", color="black", lw=1, ls="--")
    ax.set_title(f"{tk} — Closing Price with Moving Averages")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_ylabel("Price")
axes[-1].set_xlabel("Date")
plt.tight_layout()
plt.savefig(f"{OUT}/01_price_trends.png")
plt.close()

# 4b. Daily returns distribution
fig, ax = plt.subplots(figsize=(8, 4.5))
for tk in tickers:
    sub = df[df["Ticker"] == tk]["Daily_Return_%"].dropna()
    ax.hist(sub, bins=60, alpha=0.5, label=tk, color=colors[tk])
ax.set_title("Distribution of Daily Returns (%)")
ax.set_xlabel("Daily Return (%)")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/02_returns_distribution.png")
plt.close()

# 4c. Rolling volatility over time
fig, ax = plt.subplots(figsize=(8, 4.5))
for tk in tickers:
    sub = df[df["Ticker"] == tk]
    ax.plot(sub["Date"], sub["Volatility_20d"], label=tk, color=colors[tk])
ax.set_title("20-Day Rolling Volatility")
ax.set_ylabel("Std. Dev. of Daily Return (%)")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/03_volatility.png")
plt.close()

# 4d. Correlation heatmap of daily returns across companies
pivot = df.pivot_table(index="Date", columns="Ticker", values="Daily_Return_%")
corr = pivot.corr()
fig, ax = plt.subplots(figsize=(4.5, 4))
im = ax.imshow(corr, cmap="RdYlGn", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
ax.set_yticks(range(len(corr)), corr.columns)
for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center", fontsize=8)
ax.set_title("Return Correlation Across Companies")
fig.colorbar(im, shrink=0.8)
plt.tight_layout()
plt.savefig(f"{OUT}/04_correlation_heatmap.png")
plt.close()

# ---------- 5. PREDICTION: next 30 trading days for TECHNOVA ----------
target = "TECHNOVA"
sub = df[df["Ticker"] == target].reset_index(drop=True)
sub["t"] = np.arange(len(sub))

# Use last 180 days to fit a short-term trend model (linear + recent momentum)
window = sub.tail(180).copy()
X = window[["t"]].values
y = window["Close"].values

split = int(len(window) * 0.85)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = LinearRegression().fit(X_train, y_train)
y_pred_test = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)

# forecast next 30 trading days beyond the dataset
future_t = np.arange(sub["t"].iloc[-1] + 1, sub["t"].iloc[-1] + 31).reshape(-1, 1)
future_pred = model.predict(future_t)
future_dates = pd.bdate_range(sub["Date"].iloc[-1] + pd.Timedelta(days=1), periods=30)

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(sub["Date"].tail(120), sub["Close"].tail(120), label="Historical Close", color=colors[target])
ax.plot(future_dates, future_pred, label="30-Day Forecast", color="purple", ls="--")
ax.axvline(sub["Date"].iloc[-1], color="gray", lw=0.8, ls=":")
ax.set_title(f"{target} — 30-Day Closing Price Forecast (Linear Trend Model)")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/05_forecast.png")
plt.close()

print(f"\nModel test MAE: {mae:.2f} | R^2: {r2:.3f}")
print(f"Forecast range: {future_pred.min():.2f} to {future_pred.max():.2f}")

# save forecast table
forecast_df = pd.DataFrame({"Date": future_dates, "Forecast_Close": future_pred.round(2)})
forecast_df.to_csv("/home/claude/finance_project/forecast_technova.csv", index=False)

print("\nMissing values before/after cleaning:", missing_before, "/", missing_after)
print("\nAll charts saved to", OUT)
