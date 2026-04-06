import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

os.makedirs("data", exist_ok=True)

# ── Load data ───────────────────────────────────────
print("Loading Bitcoin data...")

df = pd.read_csv(
    "data/btc_data.csv",
    index_col=0,
    parse_dates=True
)

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df.columns = [str(c).strip() for c in df.columns]

close_prices = df[["Close"]].copy()

print(f"Rows loaded    : {len(close_prices)}")
print(f"Date from      : {close_prices.index[0].date()}")
print(f"Date to        : {close_prices.index[-1].date()}")

# ── Scale to 0-1 ────────────────────────────────────
print("\nScaling prices to 0-1 range...")

scaler      = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)

print(f"Original price : ${close_prices['Close'].iloc[100]:,.2f}")
print(f"Scaled price   : {scaled_data[100][0]:.4f}")

# ── Create sliding windows ──────────────────────────
WINDOW_SIZE = 60
print(f"\nCreating sliding windows of {WINDOW_SIZE} days...")

X, y = [], []

for i in range(WINDOW_SIZE, len(scaled_data)):
    X.append(scaled_data[i - WINDOW_SIZE : i, 0])
    y.append(scaled_data[i, 0])

X = np.array(X)
y = np.array(y)

print(f"Total windows  : {len(X)}")
print(f"X shape        : {X.shape}")
print(f"y shape        : {y.shape}")

# ── Reshape for LSTM ────────────────────────────────
X = X.reshape(X.shape[0], X.shape[1], 1)
print(f"\nAfter reshape  : {X.shape}")
print(f"  {X.shape[0]} samples")
print(f"  {X.shape[1]} timesteps")
print(f"  {X.shape[2]} feature")

# ── Train / Test split ──────────────────────────────
split = int(len(X) * 0.80)

X_train = X[:split]
y_train = y[:split]
X_test  = X[split:]
y_test  = y[split:]

print(f"\nTrain samples  : {len(X_train)} (80%)")
print(f"Test samples   : {len(X_test)}  (20%)")

# ── Save everything ─────────────────────────────────
np.save("data/X_train.npy", X_train)
np.save("data/y_train.npy", y_train)
np.save("data/X_test.npy",  X_test)
np.save("data/y_test.npy",  y_test)
joblib.dump(scaler, "data/scaler.pkl")

print("\nSaved:")
print("  data/X_train.npy")
print("  data/y_train.npy")
print("  data/X_test.npy")
print("  data/y_test.npy")
print("  data/scaler.pkl")

# ── Plot train/test split ───────────────────────────
train_dates = close_prices.index[
    WINDOW_SIZE : WINDOW_SIZE + len(X_train)
]
test_dates = close_prices.index[
    WINDOW_SIZE + len(X_train) :
    WINDOW_SIZE + len(X_train) + len(X_test)
]

plt.figure(figsize=(14, 5))
plt.plot(
    train_dates,
    close_prices["Close"].values[
        WINDOW_SIZE : WINDOW_SIZE + len(X_train)
    ],
    color="blue",
    label="Training data",
    linewidth=1.2
)
plt.plot(
    test_dates,
    close_prices["Close"].values[
        WINDOW_SIZE + len(X_train) :
        WINDOW_SIZE + len(X_train) + len(X_test)
    ],
    color="orange",
    label="Test data",
    linewidth=1.2
)
plt.axvline(
    x=train_dates[-1],
    color="red",
    linestyle="--",
    label="Train/Test split"
)
plt.title("Bitcoin Price — Train vs Test Split")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("data/train_test_split.png")
plt.close()
print("\nChart saved to data/train_test_split.png")
print("\n--- Step 2 Complete ---")