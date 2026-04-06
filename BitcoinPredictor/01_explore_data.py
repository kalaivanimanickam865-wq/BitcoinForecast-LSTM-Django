import yfinance as yf
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

os.makedirs("data", exist_ok=True)

print("Downloading Bitcoin data...")

df = yf.download(
    "BTC-USD",
    start="2018-01-01",
    end="2025-01-01",
    auto_adjust=True,
    progress=False
)

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df.to_csv("data/btc_data.csv")
print("Saved to data/btc_data.csv")

print(f"\nTotal rows     : {len(df)}")
print(f"Columns        : {list(df.columns)}")
print(f"Date from      : {df.index[0].date()}")
print(f"Date to        : {df.index[-1].date()}")

print("\nFirst 5 rows:")
print(df.head())

print("\nBasic statistics:")
print(df.describe().round(2))

print("\nMissing values:")
print(df.isnull().sum())

plt.figure(figsize=(14, 5))
plt.plot(df.index, df["Close"],
         color="orange", linewidth=1.5)
plt.title("Bitcoin (BTC-USD) Closing Price 2018-2025")
plt.xlabel("Date")
plt.ylabel("Price in USD")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("data/btc_price_chart.png")
plt.close()
print("\nChart saved to data/btc_price_chart.png")
print("\n--- Step 1 Complete ---")