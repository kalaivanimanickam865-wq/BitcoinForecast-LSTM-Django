import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import json
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

tf.random.set_seed(42)
np.random.seed(42)

print("=" * 55)
print("  STEP 5 (RETRAIN) — HYBRID LSTM-GRU on 2018-2025")
print("=" * 55)

# ── Load updated data ────────────────────────────────
X_train = np.load("data/X_train.npy")
y_train = np.load("data/y_train.npy")
X_test  = np.load("data/X_test.npy")
y_test  = np.load("data/y_test.npy")
scaler  = joblib.load("data/scaler.pkl")

df = pd.read_csv("data/btc_data.csv", index_col=0, parse_dates=True)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

print(f"\nX_train : {X_train.shape}")
print(f"X_test  : {X_test.shape}")
print(f"Data up to: {df.index[-1].date()}")

# ── Build model ──────────────────────────────────────
print("\nBuilding Hybrid LSTM-GRU...")

model = Sequential([
    LSTM(units=50, return_sequences=True,
         input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    GRU(units=50, return_sequences=False),
    Dropout(0.2),
    Dense(units=25, activation="relu"),
    Dense(units=1)
])

model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss="mean_squared_error",
    metrics=["mae"]
)

model.summary()
print(f"\nTotal parameters: {model.count_params():,}")

# ── Callbacks ────────────────────────────────────────
callbacks = [
    EarlyStopping(monitor="val_loss", patience=15,
                  restore_best_weights=True, verbose=1),
    ModelCheckpoint(filepath="data/best_hybrid_model.keras",
                    monitor="val_loss", save_best_only=True, verbose=0),
    ReduceLROnPlateau(monitor="val_loss", factor=0.3,
                      patience=7, min_lr=0.000001, verbose=1)
]

# ── Train ────────────────────────────────────────────
print("\nTraining on updated data (2018-2025)...\n")

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=16,
    validation_split=0.1,
    callbacks=callbacks,
    verbose=1
)

print(f"\nTraining complete! Stopped at epoch {len(history.history['loss'])}")

# ── Predictions ──────────────────────────────────────
print("\nEvaluating...")

pred_scaled = model.predict(X_test, verbose=0)
actual_usd  = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
pred_usd    = scaler.inverse_transform(pred_scaled).flatten()

# ── Metrics ──────────────────────────────────────────
rmse = np.sqrt(mean_squared_error(actual_usd, pred_usd))
mae  = mean_absolute_error(actual_usd, pred_usd)
mape = np.mean(np.abs((actual_usd - pred_usd) / actual_usd)) * 100
r2   = r2_score(actual_usd, pred_usd)
da   = np.mean(np.sign(np.diff(actual_usd)) ==
               np.sign(np.diff(pred_usd))) * 100

print("\n" + "="*45)
print("RETRAINED MODEL RESULTS")
print("="*45)
print(f"  RMSE          : ${rmse:,.0f}")
print(f"  MAE           : ${mae:,.0f}")
print(f"  MAPE          : {mape:.2f}%")
print(f"  R²            : {r2:.4f}")
print(f"  Dir. Accuracy : {da:.1f}%")
print("="*45)

# ── Save metrics ─────────────────────────────────────
metrics = {
    "model": "Hybrid LSTM-GRU (Retrained 2025)",
    "data_up_to": str(df.index[-1].date()),
    "rmse": round(rmse, 2),
    "mae":  round(mae, 2),
    "mape": round(mape, 2),
    "r2":   round(r2, 4),
    "da":   round(da, 2)
}
with open("data/hybrid_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("Metrics saved: data/hybrid_metrics.json")

# ── Save predictions ─────────────────────────────────
np.save("data/hybrid_predictions_scaled.npy", pred_scaled)
np.save("data/predictions_scaled.npy",        pred_scaled)

# ── Training chart ───────────────────────────────────
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["loss"],     label="Train", color="blue")
plt.plot(history.history["val_loss"], label="Val",   color="orange")
plt.title("Loss"); plt.legend(); plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history["mae"],     label="Train", color="blue")
plt.plot(history.history["val_mae"], label="Val",   color="orange")
plt.title("MAE"); plt.legend(); plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("data/retrain_history.png")
plt.close()

# ── Prediction chart ─────────────────────────────────
WINDOW_SIZE = 60
train_size  = int((len(df) - WINDOW_SIZE) * 0.80)
test_start  = WINDOW_SIZE + train_size
min_len     = min(len(actual_usd), len(pred_usd))
test_dates  = df.index[test_start : test_start + min_len]

plt.figure(figsize=(14, 6))
plt.plot(test_dates, actual_usd,
         color="blue", linewidth=2, label="Actual Price")
plt.plot(test_dates, pred_usd,
         color="green", linewidth=1.5, linestyle="--",
         label=f"Predicted  RMSE=${rmse:,.0f}")
plt.title(f"BTC Retrained Model — Test Period\n"
          f"Data: 2018 → {df.index[-1].date()}")
plt.xlabel("Date"); plt.ylabel("Price (USD)")
plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("data/retrain_predictions.png", dpi=150)
plt.close()
print("Charts saved!")

# ── Run forecast ─────────────────────────────────────
print("\nGenerating updated 30-day forecast...")
from datetime import timedelta

close_col  = df[["Close"]]
scaled_all = scaler.transform(close_col.values)
current    = float(df["Close"].iloc[-1])
last_date  = df.index[-1]

FORECAST_DAYS = 30
seq = scaled_all[-WINDOW_SIZE:].reshape(1, WINDOW_SIZE, 1)
raw_sc = []
for _ in range(FORECAST_DAYS):
    p = model.predict(seq, verbose=0)[0, 0]
    raw_sc.append(p)
    seq = np.append(seq[:, 1:, :], [[[p]]], axis=1)

raw_usd = scaler.inverse_transform(
    np.array(raw_sc).reshape(-1, 1)).flatten()
raw_chg = np.diff(raw_usd, prepend=current) / \
          np.concatenate([[current], raw_usd[:-1]])

recent     = df["Close"].iloc[-14:].values
ret        = np.diff(recent) / recent[:-1]
momentum   = float(np.mean(ret))
volatility = float(np.std(ret))

prices = [current]
for i in range(FORECAST_DAYS):
    decay   = np.exp(-i / 20)
    blended = 0.40 * raw_chg[i] + 0.60 * momentum * decay
    blended = np.clip(blended, -0.05, 0.05)
    prices.append(prices[-1] * (1 + blended))

forecast_usd = np.array(prices[1:])
widths       = np.array([volatility * np.sqrt(i+1) * forecast_usd[i]
                          for i in range(FORECAST_DAYS)])
upper_band   = forecast_usd + widths
lower_band   = np.maximum(forecast_usd - widths, 1000)

future_dates = pd.date_range(
    start=last_date + timedelta(days=1),
    periods=FORECAST_DAYS, freq="D"
)

change_pct = (forecast_usd[-1] - current) / current * 100

print(f"\n  Current price : ${current:,.0f}")
print(f"  D+7  forecast : ${forecast_usd[6]:,.0f}  "
      f"({(forecast_usd[6]-current)/current*100:+.1f}%)")
print(f"  D+14 forecast : ${forecast_usd[13]:,.0f}  "
      f"({(forecast_usd[13]-current)/current*100:+.1f}%)")
print(f"  D+30 forecast : ${forecast_usd[-1]:,.0f}  "
      f"({change_pct:+.1f}%)")

# Save forecast CSV
forecast_df = pd.DataFrame({
    "Date"       : future_dates.strftime("%Y-%m-%d"),
    "Forecast"   : np.round(forecast_usd, 2),
    "Upper_Band" : np.round(upper_band, 2),
    "Lower_Band" : np.round(lower_band, 2),
    "Change_Pct" : np.round(
        (forecast_usd - current) / current * 100, 2)
})
forecast_df.to_csv("data/forecast_30day.csv", index=False)
print("Forecast saved: data/forecast_30day.csv")

print("\n" + "="*55)
print("  RETRAIN COMPLETE!")
print("="*55)
print(f"  Model trained on data up to: {df.index[-1].date()}")
print(f"  Model saved: data/best_hybrid_model.keras")
print("\nNext: restart Django server to use updated model!")