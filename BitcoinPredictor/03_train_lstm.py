import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

# ── Load data ───────────────────────────────────────
print("Loading prepared data...")

X_train = np.load("data/X_train.npy")
y_train = np.load("data/y_train.npy")
X_test  = np.load("data/X_test.npy")
y_test  = np.load("data/y_test.npy")

print(f"X_train : {X_train.shape}")
print(f"X_test  : {X_test.shape}")

# ── Build LSTM model ────────────────────────────────
print("\nBuilding LSTM model...")

model = Sequential([

    LSTM(
        units=64,
        return_sequences=True,
        input_shape=(X_train.shape[1], X_train.shape[2])
    ),
    Dropout(0.2),

    LSTM(
        units=64,
        return_sequences=False
    ),
    Dropout(0.2),

    Dense(units=32, activation="relu"),
    Dense(units=1)
])

model.compile(
    optimizer="adam",
    loss="mean_squared_error",
    metrics=["mae"]
)

model.summary()

# ── Callbacks ───────────────────────────────────────
os.makedirs("data", exist_ok=True)

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        filepath="data/best_lstm_model.keras",
        monitor="val_loss",
        save_best_only=True,
        verbose=0
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=0.00001,
        verbose=1
    )
]

# ── Train ───────────────────────────────────────────
print("\nTraining started...")
print("Watch loss go down each epoch!\n")

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    callbacks=callbacks,
    verbose=1
)

print("\nTraining complete!")

# ── Plot training history ───────────────────────────
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["loss"],
         label="Train Loss", color="blue")
plt.plot(history.history["val_loss"],
         label="Val Loss", color="orange")
plt.title("LSTM Loss During Training")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history["mae"],
         label="Train MAE", color="blue")
plt.plot(history.history["val_mae"],
         label="Val MAE", color="orange")
plt.title("LSTM MAE During Training")
plt.xlabel("Epoch")
plt.ylabel("MAE")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("data/lstm_training_history.png")
plt.close()
print("Training chart saved to data/lstm_training_history.png")

# ── Evaluate ────────────────────────────────────────
print("\nEvaluating on test data...")
test_loss, test_mae = model.evaluate(
    X_test, y_test, verbose=0
)
print(f"Test Loss (MSE) : {test_loss:.6f}")
print(f"Test MAE        : {test_mae:.6f}")

# ── Predict ─────────────────────────────────────────
print("\nMaking predictions...")
predictions_scaled = model.predict(X_test, verbose=0)
np.save("data/predictions_scaled.npy", predictions_scaled)
print("Predictions saved to data/predictions_scaled.npy")

print("\n--- Step 3 Complete ---")
print("Model trained and saved!")
print("Next: Step 4 will show predictions in real USD")