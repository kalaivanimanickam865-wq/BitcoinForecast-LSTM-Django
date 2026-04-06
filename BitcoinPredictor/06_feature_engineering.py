import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from sklearn.preprocessing import MinMaxScaler

# ── Load raw BTC data ────────────────────────────────
print("Loading BTC data...")

df = pd.read_csv(
    "data/btc_data.csv",
    index_col=0,
    parse_dates=True
)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# Keep only needed columns
df = df[["Close", "Volume"]].copy()
df.dropna(inplace=True)

print(f"Raw data shape : {df.shape}")
print(f"Date range     : {df.index[0]} → {df.index[-1]}")

# ════════════════════════════════════════════════════
# FEATURE ENGINEERING — 13 Features
# ════════════════════════════════════════════════════

print("\nEngineering 13 features...")

close  = df["Close"]
volume = df["Volume"]

# ── 1. RSI (14) ──────────────────────────────────────
def compute_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs  = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

df["RSI"] = compute_rsi(close, 14)

# ── 2 & 3. MACD + Signal ────────────────────────────
ema12 = close.ewm(span=12, adjust=False).mean()
ema26 = close.ewm(span=26, adjust=False).mean()
df["MACD"]        = ema12 - ema26
df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

# ── 4 & 5. EMA 12 & EMA 26 ──────────────────────────
df["EMA_12"] = ema12
df["EMA_26"] = ema26

# ── 6 & 7. Bollinger Bands (20-day) ─────────────────
sma20  = close.rolling(window=20).mean()
std20  = close.rolling(window=20).std()
df["BB_Upper"] = sma20 + (2 * std20)
df["BB_Lower"] = sma20 - (2 * std20)

# ── 8 & 9. SMA 7 & SMA 30 ───────────────────────────
df["SMA_7"]  = close.rolling(window=7).mean()
df["SMA_30"] = close.rolling(window=30).mean()

# ── 10. Price Rate of Change (10-day) ───────────────
df["ROC"] = close.pct_change(periods=10) * 100

# ── 11. On-Balance Volume (OBV) ──────────────────────
obv = [0]
for i in range(1, len(df)):
    if close.iloc[i] > close.iloc[i - 1]:
        obv.append(obv[-1] + volume.iloc[i])
    elif close.iloc[i] < close.iloc[i - 1]:
        obv.append(obv[-1] - volume.iloc[i])
    else:
        obv.append(obv[-1])
df["OBV"] = obv

# ── Drop NaN rows from rolling calculations ──────────
df.dropna(inplace=True)

# Confirm all 13 features
FEATURES = [
    "Close",       # 1
    "Volume",      # 2
    "RSI",         # 3
    "MACD",        # 4
    "MACD_Signal", # 5
    "EMA_12",      # 6
    "EMA_26",      # 7
    "BB_Upper",    # 8
    "BB_Lower",    # 9
    "SMA_7",       # 10
    "SMA_30",      # 11
    "ROC",         # 12
    "OBV",         # 13
]

df = df[FEATURES]
print(f"\n✓ Features ready : {len(FEATURES)}")
print(f"  Columns        : {list(df.columns)}")
print(f"  Data shape     : {df.shape}")

# ════════════════════════════════════════════════════
# SCALE — separate scaler for Close (for inversion)
# ════════════════════════════════════════════════════

print("\nScaling features...")

# Scale all 13 features together
feature_scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data    = feature_scaler.fit_transform(df.values)

# Separate scaler for Close price only (to invert predictions)
close_scaler = MinMaxScaler(feature_range=(0, 1))
close_scaler.fit(df[["Close"]].values)

# Save both scalers
joblib.dump(feature_scaler, "data/feature_scaler.pkl")
joblib.dump(close_scaler,   "data/close_scaler.pkl")
print("✓ Scalers saved")

# ════════════════════════════════════════════════════
# SEQUENCE CREATION — 60-day windows
# ════════════════════════════════════════════════════

WINDOW_SIZE   = 60
TARGET_COL    = 0          # Close is column 0

print(f"\nCreating sequences (window={WINDOW_SIZE})...")

X, y = [], []
for i in range(WINDOW_SIZE, len(scaled_data)):
    X.append(scaled_data[i - WINDOW_SIZE : i, :])   # all 13 features
    y.append(scaled_data[i, TARGET_COL])             # Close only

X = np.array(X)
y = np.array(y)

print(f"✓ X shape : {X.shape}  →  "
      f"(samples, window, features)")
print(f"  y shape : {y.shape}")

# ── Train / Test split (80 / 20) ─────────────────────
split      = int(len(X) * 0.80)
X_train    = X[:split]
y_train    = y[:split]
X_test     = X[split:]
y_test     = y[split:]

print(f"\nTrain : {X_train.shape}")
print(f"Test  : {X_test.shape}")

# ── Save arrays ───────────────────────────────────────
np.save("data/X_train_v2.npy", X_train)
np.save("data/y_train_v2.npy", y_train)
np.save("data/X_test_v2.npy",  X_test)
np.save("data/y_test_v2.npy",  y_test)
print("\n✓ Arrays saved as *_v2.npy")

# ════════════════════════════════════════════════════
# FEATURE CORRELATION CHART
# ════════════════════════════════════════════════════

print("\nGenerating feature correlation chart...")

corr = df.corr()

fig, ax = plt.subplots(figsize=(11, 9))
im = ax.imshow(corr.values, cmap="coolwarm",
               vmin=-1, vmax=1)
plt.colorbar(im, ax=ax)

ax.set_xticks(range(len(FEATURES)))
ax.set_yticks(range(len(FEATURES)))
ax.set_xticklabels(FEATURES, rotation=45, ha="right")
ax.set_yticklabels(FEATURES)

for i in range(len(FEATURES)):
    for j in range(len(FEATURES)):
        ax.text(j, i, f"{corr.values[i, j]:.2f}",
                ha="center", va="center",
                fontsize=7,
                color="white" if abs(corr.values[i, j]) > 0.6
                              else "black")

ax.set_title("Feature Correlation Matrix — 13 Features",
             fontsize=13, pad=15)
plt.tight_layout()
plt.savefig("data/feature_correlation.png", dpi=150)
plt.close()
print("✓ Correlation chart saved: data/feature_correlation.png")

# ── Feature summary ───────────────────────────────────
print("\n" + "=" * 55)
print("FEATURE SUMMARY")
print("=" * 55)
print(f"{'Feature':<15} {'Min':>12} {'Max':>12} {'Mean':>12}")
print("-" * 55)
for feat in FEATURES:
    mn  = df[feat].min()
    mx  = df[feat].max()
    avg = df[feat].mean()
    print(f"{feat:<15} {mn:>12.2f} {mx:>12.2f} {avg:>12.2f}")
print("=" * 55)

print("\n--- Step 6 Complete ---")
print("Next: Step 7 — Train multi-feature Hybrid LSTM-GRU!")
print(f"\nInput shape for Step 7: "
      f"(batch, {WINDOW_SIZE}, {len(FEATURES)})")