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

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

tf.random.set_seed(42)
np.random.seed(42)

# ── Load multi-feature data ──────────────────────────
print("Loading multi-feature data (13 features)...")

X_train = np.load("data/X_train_v2.npy")
y_train = np.load("data/y_train_v2.npy")
X_test  = np.load("data/X_test_v2.npy")
y_test  = np.load("data/y_test_v2.npy")

close_scaler = joblib.load("data/close_scaler.pkl")
old_scaler   = joblib.load("data/scaler.pkl")

print(f"X_train : {X_train.shape}")
print(f"X_test  : {X_test.shape}")

# ── Build Fixed Hybrid LSTM-GRU ──────────────────────
# Key fixes vs previous attempt:
#   - Removed BatchNormalization (hurts time-series)
#   - Moderate units — avoids overfitting on high prices
#   - Lower dropout (0.15)
#   - Higher learning rate to escape local minima faster
print("\nBuilding Fixed Hybrid LSTM-GRU (13 features)...")

model = Sequential([

    # LSTM 1 — long-term trend
    LSTM(
        units=100,
        return_sequences=True,
        input_shape=(X_train.shape[1], X_train.shape[2])
    ),
    Dropout(0.15),

    # LSTM 2 — medium-term pattern
    LSTM(
        units=50,
        return_sequences=True
    ),
    Dropout(0.15),

    # GRU — short-term momentum
    GRU(
        units=50,
        return_sequences=False
    ),
    Dropout(0.15),

    # Dense head
    Dense(units=32, activation="relu"),
    Dense(units=1)
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="mean_squared_error",
    metrics=["mae"]
)

model.summary()
print(f"\nTotal parameters: {model.count_params():,}")

# ── Callbacks ────────────────────────────────────────
os.makedirs("data", exist_ok=True)

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=20,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        filepath="data/best_multifeature_model.keras",
        monitor="val_loss",
        save_best_only=True,
        verbose=0
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=8,
        min_lr=0.00001,
        verbose=1
    )
]

# ── Train ────────────────────────────────────────────
print("\nTraining Fixed Hybrid LSTM-GRU (13 features)...")
print("No BatchNorm — cleaner time-series training\n")

history = model.fit(
    X_train, y_train,
    epochs=150,
    batch_size=32,
    validation_split=0.1,
    callbacks=callbacks,
    verbose=1
)

print("\nTraining complete!")
print(f"Stopped at epoch: {len(history.history['loss'])}")

# ── Training chart ───────────────────────────────────
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["loss"],
         label="Train Loss", color="blue")
plt.plot(history.history["val_loss"],
         label="Val Loss", color="orange")
plt.title("Multi-Feature Model — Loss (Fixed)")
plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.legend(); plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history["mae"],
         label="Train MAE", color="blue")
plt.plot(history.history["val_mae"],
         label="Val MAE", color="orange")
plt.title("Multi-Feature Model — MAE (Fixed)")
plt.xlabel("Epoch"); plt.ylabel("MAE")
plt.legend(); plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("data/multifeature_training_history.png")
plt.close()
print("Training chart saved")

# ── Predictions ──────────────────────────────────────
print("\nMaking predictions...")

mf_scaled = model.predict(X_test, verbose=0)

actual_usd = close_scaler.inverse_transform(
    y_test.reshape(-1, 1)).flatten()
mf_usd = close_scaler.inverse_transform(
    mf_scaled).flatten()

# Previous models
hybrid_scaled = np.load("data/hybrid_predictions_scaled.npy")
lstm_scaled   = np.load("data/predictions_scaled.npy")
hybrid_usd    = old_scaler.inverse_transform(hybrid_scaled).flatten()
lstm_usd      = old_scaler.inverse_transform(lstm_scaled).flatten()

# Align lengths
min_len    = min(len(actual_usd), len(hybrid_usd), len(lstm_usd))
actual_usd = actual_usd[:min_len]
mf_usd     = mf_usd[:min_len]
hybrid_usd = hybrid_usd[:min_len]
lstm_usd   = lstm_usd[:min_len]

print(f"\nPrediction range check:")
print(f"  Actual  : ${actual_usd.min():,.0f} → ${actual_usd.max():,.0f}")
print(f"  LSTM    : ${lstm_usd.min():,.0f} → ${lstm_usd.max():,.0f}")
print(f"  Hybrid  : ${hybrid_usd.min():,.0f} → ${hybrid_usd.max():,.0f}")
print(f"  Multi-F : ${mf_usd.min():,.0f} → ${mf_usd.max():,.0f}")

# ── Metrics ──────────────────────────────────────────
def get_metrics(actual, predicted):
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae  = mean_absolute_error(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    r2   = r2_score(actual, predicted)
    da   = np.mean(
               np.sign(np.diff(actual)) ==
               np.sign(np.diff(predicted))
           ) * 100
    return rmse, mae, mape, r2, da

m_lstm   = get_metrics(actual_usd, lstm_usd)
m_hybrid = get_metrics(actual_usd, hybrid_usd)
m_mf     = get_metrics(actual_usd, mf_usd)

# ── Comparison table ─────────────────────────────────
print("\n" + "="*72)
print("MODEL COMPARISON — ALL 3 MODELS")
print("="*72)
print(f"{'Metric':<18} {'Simple LSTM':>13} "
      f"{'Hybrid (1f)':>13} {'Multi-F (13f)':>13}  Best")
print("-"*72)

labels = [
    ("RMSE ($)",    True),
    ("MAE ($)",     True),
    ("MAPE (%)",    True),
    ("R²",          False),
    ("Dir.Acc (%)", False),
]
for i, (label, lower_better) in enumerate(labels):
    l, h, mf = m_lstm[i], m_hybrid[i], m_mf[i]
    scores   = [l, h, mf]
    names    = ["LSTM ✓", "Hybrid ✓", "Multi-F ✓"]
    best_idx = scores.index(min(scores) if lower_better else max(scores))
    print(f"{label:<18} {l:>13.2f} {h:>13.2f} "
          f"{mf:>13.2f}  {names[best_idx]}")

print("="*72)

rmse_vs_hybrid = ((m_hybrid[0] - m_mf[0]) / m_hybrid[0] * 100)
da_vs_hybrid   = m_mf[4] - m_hybrid[4]

print(f"\nVs Hybrid (1 feature):")
print(f"  RMSE      : {'improved' if rmse_vs_hybrid > 0 else 'worsened'} "
      f"{abs(rmse_vs_hybrid):.1f}%")
print(f"  Dir. Acc  : {'+' if da_vs_hybrid >= 0 else ''}{da_vs_hybrid:.1f}%")

# ── 3-Model chart ────────────────────────────────────
df = pd.read_csv("data/btc_data.csv", index_col=0, parse_dates=True)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

WINDOW_SIZE = 60
train_size  = int((len(df) - WINDOW_SIZE) * 0.80)
test_start  = WINDOW_SIZE + train_size
test_dates  = df.index[test_start : test_start + min_len]

plt.figure(figsize=(16, 8))
plt.plot(test_dates, actual_usd,
         color="blue", linewidth=2,
         label="Actual Price", zorder=4)
plt.plot(test_dates, lstm_usd,
         color="red", linewidth=1, linestyle="--", alpha=0.6,
         label=f"Simple LSTM       RMSE=${m_lstm[0]:,.0f}  DA={m_lstm[4]:.1f}%")
plt.plot(test_dates, hybrid_usd,
         color="orange", linewidth=1, linestyle="--", alpha=0.8,
         label=f"Hybrid LSTM-GRU   RMSE=${m_hybrid[0]:,.0f}  DA={m_hybrid[4]:.1f}%")
plt.plot(test_dates, mf_usd,
         color="green", linewidth=2, alpha=0.9,
         label=f"Multi-Feature(13) RMSE=${m_mf[0]:,.0f}  DA={m_mf[4]:.1f}%",
         zorder=3)

plt.title(
    "Bitcoin Price Prediction — 3 Model Comparison\n"
    "Simple LSTM  vs  Hybrid LSTM-GRU  vs  Multi-Feature (13f)",
    fontsize=13)
plt.xlabel("Date"); plt.ylabel("Price (USD)")
plt.legend(loc="upper left", fontsize=9)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("data/3model_comparison.png", dpi=150)
plt.close()
print("\nComparison chart saved: data/3model_comparison.png")

# ── Save ─────────────────────────────────────────────
np.save("data/multifeature_predictions_scaled.npy", mf_scaled)

metrics_dict = {
    "model"           : "Multi-Feature Hybrid LSTM-GRU (Fixed)",
    "features"        : 13,
    "rmse"            : round(m_mf[0], 2),
    "mae"             : round(m_mf[1], 2),
    "mape"            : round(m_mf[2], 2),
    "r2"              : round(m_mf[3], 4),
    "directional_acc" : round(m_mf[4], 2),
    "rmse_vs_hybrid"  : round(rmse_vs_hybrid, 2),
    "da_vs_hybrid"    : round(da_vs_hybrid, 2)
}
with open("data/multifeature_metrics.json", "w") as f:
    json.dump(metrics_dict, f, indent=2)
print("Metrics saved: data/multifeature_metrics.json")

print("\n--- Step 7 (Fixed) Complete ---")
print("Next: Step 8 — Forecast future prices + final dashboard!")