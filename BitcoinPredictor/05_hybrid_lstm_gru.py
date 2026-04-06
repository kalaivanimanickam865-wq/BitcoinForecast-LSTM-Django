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
from tensorflow.keras.layers import (
    LSTM, GRU, Dense, Dropout
)
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)
from tensorflow.keras.optimizers import Adam

# ── Load data ───────────────────────────────────────
print("Loading prepared data...")

X_train = np.load("data/X_train.npy")
y_train = np.load("data/y_train.npy")
X_test  = np.load("data/X_test.npy")
y_test  = np.load("data/y_test.npy")
scaler  = joblib.load("data/scaler.pkl")

print(f"X_train : {X_train.shape}")
print(f"X_test  : {X_test.shape}")

# ── Build Simpler Hybrid LSTM-GRU ───────────────────
# Key insight: simpler architecture works better
# with single feature input
print("\nBuilding Hybrid LSTM-GRU model...")

model = Sequential([

    # LSTM — captures long term trend
    LSTM(
        units=50,
        return_sequences=True,
        input_shape=(
            X_train.shape[1],
            X_train.shape[2]
        )
    ),
    Dropout(0.2),

    # GRU — captures short term movement
    # return_sequences=False — last layer before Dense
    GRU(
        units=50,
        return_sequences=False
    ),
    Dropout(0.2),

    # Output
    Dense(units=25, activation="relu"),
    Dense(units=1)
])

# Lower learning rate for more stable training
model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss="mean_squared_error",
    metrics=["mae"]
)

model.summary()
print(f"\nTotal parameters: {model.count_params():,}")

# ── Callbacks ───────────────────────────────────────
os.makedirs("data", exist_ok=True)

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        filepath="data/best_hybrid_model.keras",
        monitor="val_loss",
        save_best_only=True,
        verbose=0
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=7,
        min_lr=0.000001,
        verbose=1
    )
]

# ── Train ───────────────────────────────────────────
print("\nTraining Hybrid LSTM-GRU...")
print("Simpler architecture — better results!\n")

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=16,
    validation_split=0.1,
    callbacks=callbacks,
    verbose=1
)

print("\nTraining complete!")

# ── Training history chart ──────────────────────────
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["loss"],
         label="Train Loss", color="blue")
plt.plot(history.history["val_loss"],
         label="Val Loss", color="orange")
plt.title("Hybrid Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history["mae"],
         label="Train MAE", color="blue")
plt.plot(history.history["val_mae"],
         label="Val MAE", color="orange")
plt.title("Hybrid Model MAE")
plt.xlabel("Epoch")
plt.ylabel("MAE")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("data/hybrid_training_history.png")
plt.close()
print("Training chart saved")

# ── Predictions ─────────────────────────────────────
print("\nMaking predictions...")

hybrid_scaled = model.predict(X_test, verbose=0)

actual_usd = scaler.inverse_transform(
    y_test.reshape(-1, 1)
).flatten()

hybrid_usd = scaler.inverse_transform(
    hybrid_scaled
).flatten()

lstm_scaled = np.load("data/predictions_scaled.npy")
lstm_usd    = scaler.inverse_transform(
    lstm_scaled
).flatten()

# ── Metrics ─────────────────────────────────────────
def get_metrics(actual, predicted):
    rmse = np.sqrt(mean_squared_error(
        actual, predicted
    ))
    mae  = mean_absolute_error(actual, predicted)
    mape = np.mean(
               np.abs((actual - predicted) / actual)
           ) * 100
    r2   = r2_score(actual, predicted)
    da   = np.mean(
               np.sign(np.diff(actual)) ==
               np.sign(np.diff(predicted))
           ) * 100
    return rmse, mae, mape, r2, da

m_lstm   = get_metrics(actual_usd, lstm_usd)
m_hybrid = get_metrics(actual_usd, hybrid_usd)

# ── Comparison table ────────────────────────────────
print("\n" + "="*58)
print("MODEL COMPARISON")
print("="*58)
print(f"{'Metric':<20} {'Simple LSTM':>15} "
      f"{'Hybrid LSTM-GRU':>15}")
print("-"*58)

labels = [
    ("RMSE ($)",    True),
    ("MAE ($)",     True),
    ("MAPE (%)",    True),
    ("R²",          False),
    ("Dir.Acc (%)", False),
]
for i, (label, lower_better) in enumerate(labels):
    l = m_lstm[i]
    h = m_hybrid[i]
    if lower_better:
        better = "Hybrid ✓" if h < l else "LSTM ✓"
    else:
        better = "Hybrid ✓" if h > l else "LSTM ✓"
    print(f"{label:<20} {l:>15.2f} "
          f"{h:>15.2f}  {better}")

print("="*58)

improvement = ((m_lstm[0] - m_hybrid[0])
               / m_lstm[0] * 100)
if improvement > 0:
    print(f"\nRMSE improved by : {improvement:.1f}%")
else:
    print(f"\nRMSE difference  : {improvement:.1f}%")

# ── Comparison chart ────────────────────────────────
df = pd.read_csv(
    "data/btc_data.csv",
    index_col=0,
    parse_dates=True
)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

WINDOW_SIZE = 60
train_size  = int((len(df) - WINDOW_SIZE) * 0.80)
test_start  = WINDOW_SIZE + train_size
test_dates  = df.index[
    test_start : test_start + len(actual_usd)
]

plt.figure(figsize=(14, 7))

plt.plot(
    test_dates, actual_usd,
    color="blue", linewidth=2,
    label="Actual Price", zorder=3
)
plt.plot(
    test_dates, lstm_usd,
    color="red", linewidth=1,
    linestyle="--", alpha=0.7,
    label=f"Simple LSTM "
          f"RMSE=${m_lstm[0]:,.0f}"
)
plt.plot(
    test_dates, hybrid_usd,
    color="green", linewidth=1.5,
    linestyle="--", alpha=0.9,
    label=f"Hybrid LSTM-GRU "
          f"RMSE=${m_hybrid[0]:,.0f}",
    zorder=2
)

plt.title(
    "Bitcoin — Simple LSTM vs Hybrid LSTM-GRU"
)
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("data/lstm_vs_hybrid.png")
plt.close()
print("Comparison chart saved: data/lstm_vs_hybrid.png")

# ── Save ────────────────────────────────────────────
np.save("data/hybrid_predictions_scaled.npy",
        hybrid_scaled)

metrics_dict = {
    "model": "Hybrid LSTM-GRU",
    "rmse":  round(m_hybrid[0], 2),
    "mae":   round(m_hybrid[1], 2),
    "mape":  round(m_hybrid[2], 2),
    "r2":    round(m_hybrid[3], 4),
    "da":    round(m_hybrid[4], 2)
}
with open("data/hybrid_metrics.json", "w") as f:
    json.dump(metrics_dict, f, indent=2)
print("Metrics saved to data/hybrid_metrics.json")

print("\n--- Step 5 Complete ---")
print("Next: Step 6 — Add 13 features!")