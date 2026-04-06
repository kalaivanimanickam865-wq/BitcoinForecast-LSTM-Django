import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
from datetime import datetime

os.makedirs("data", exist_ok=True)

print("=" * 55)
print("  STEP 1 (UPDATED) — FETCH LATEST BTC DATA")
print("=" * 55)

TODAY = datetime.today().strftime("%Y-%m-%d")
print(f"\nFetching BTC data: 2018-01-01 → {TODAY}")

# ── Download latest data ─────────────────────────────
btc = yf.download(
    "BTC-USD",
    start="2018-01-01",
    end=TODAY,
    auto_adjust=True
)

if isinstance(btc.columns, pd.MultiIndex):
    btc.columns = btc.columns.get_level_values(0)

btc.dropna(inplace=True)
btc.to_csv("data/btc_data.csv")

print(f"\nData downloaded!")
print(f"  Rows      : {len(btc)}")
print(f"  From      : {btc.index[0].date()}")
print(f"  To        : {btc.index[-1].date()}")
print(f"  Last price: ${btc['Close'].iloc[-1]:,.2f}")
print(f"\nColumns: {list(btc.columns)}")

# ── Quick price summary ──────────────────────────────
print("\n" + "="*55)
print("PRICE SUMMARY")
print("="*55)
print(f"  All-time high  : ${btc['Close'].max():,.2f}  "
      f"({btc['Close'].idxmax().date()})")
print(f"  All-time low   : ${btc['Close'].min():,.2f}  "
      f"({btc['Close'].idxmin().date()})")
print(f"  Current price  : ${btc['Close'].iloc[-1]:,.2f}")
print(f"  7-day change   : "
      f"{(btc['Close'].iloc[-1] - btc['Close'].iloc[-7]) / btc['Close'].iloc[-7] * 100:+.2f}%")
print(f"  30-day change  : "
      f"{(btc['Close'].iloc[-1] - btc['Close'].iloc[-30]) / btc['Close'].iloc[-30] * 100:+.2f}%")
print(f"  YTD change     : "
      f"{(btc['Close'].iloc[-1] - btc['Close'].iloc[-252]) / btc['Close'].iloc[-252] * 100:+.2f}%")
print("="*55)

print("\nData saved: data/btc_data.csv")
print("Next: run 02b_prepare_updated.py")