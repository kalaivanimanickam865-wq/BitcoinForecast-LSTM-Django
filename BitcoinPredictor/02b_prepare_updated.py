import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

os.makedirs("data", exist_ok=True)

print("=" * 55)
print("  STEP 2 (UPDATED) — PREPARE LATEST DATA")
print("=" * 55)

# ── Load fresh data ──────────────────────────────────
df = pd.read_csv("data/btc_data.csv", index_col=0, parse_dates=True)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

print(f"\nLoaded: {len(df)} rows")
print(f"  From : {df.index[0].date()}")
print(f"  To   : {df.index[-1].date()}")

# ── Scale Close price ────────────────────────────────
close_data = df[["Close"]].values
scaler     = MinMaxScaler(feature_range=(0, 1))
scaled     = scaler.fit_transform(close_data)

joblib.dump(scaler, "data/scaler.pkl")
print(f"\nScaler fitted and saved")
print(f"  Price range: ${close_data.min():,.0f} → ${close_data.max():,.0f}")

# ── Create sequences ─────────────────────────────────
WINDOW_SIZE = 60
X, y = [], []
for i in range(WINDOW_SIZE, len(scaled)):
    X.append(scaled[i - WINDOW_SIZE:i, 0])
    y.append(scaled[i, 0])

X = np.array(X).reshape(-1, WINDOW_SIZE, 1)
y = np.array(y)

print(f"\nSequences created:")
print(f"  X shape: {X.shape}")
print(f"  y shape: {y.shape}")

# ── Train / Test split ───────────────────────────────
split   = int(len(X) * 0.80)
X_train = X[:split]
y_train = y[:split]
X_test  = X[split:]
y_test  = y[split:]

print(f"\nTrain : {X_train.shape[0]} samples")
print(f"Test  : {X_test.shape[0]} samples")
print(f"  Test period: "
      f"{df.index[WINDOW_SIZE + split].date()} → "
      f"{df.index[-1].date()}")

# ── Save ─────────────────────────────────────────────
np.save("data/X_train.npy", X_train)
np.save("data/y_train.npy", y_train)
np.save("data/X_test.npy",  X_test)
np.save("data/y_test.npy",  y_test)

print("\nArrays saved!")
print("Next: run 05b_retrain_hybrid.py")