import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

# ── Load everything ─────────────────────────────────
print("Loading data and predictions...")

y_test             = np.load("data/y_test.npy")
predictions_scaled = np.load("data/predictions_scaled.npy")
scaler             = joblib.load("data/scaler.pkl")

df = pd.read_csv(
    "data/btc_data.csv",
    index_col=0,
    parse_dates=True
)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
df.columns = [str(c).strip() for c in df.columns]

print(f"Test samples     : {len(y_test)}")

# ── Inverse transform to USD ────────────────────────
print("\nConverting to USD...")

actual_usd    = scaler.inverse_transform(
    y_test.reshape(-1, 1)
).flatten()

predicted_usd = scaler.inverse_transform(
    predictions_scaled
).flatten()

print(f"Actual range   : ${actual_usd.min():,.0f} "
      f"— ${actual_usd.max():,.0f}")
print(f"Predicted range: ${predicted_usd.min():,.0f} "
      f"— ${predicted_usd.max():,.0f}")

# ── Metrics in USD ───────────────────────────────────
print("\n--- Model Performance ---")

rmse = np.sqrt(mean_squared_error(
    actual_usd, predicted_usd
))
mae  = mean_absolute_error(
    actual_usd, predicted_usd
)
mape = np.mean(
    np.abs((actual_usd - predicted_usd) / actual_usd)
) * 100
r2   = r2_score(actual_usd, predicted_usd)

actual_dir    = np.sign(np.diff(actual_usd))
predicted_dir = np.sign(np.diff(predicted_usd))
da = np.mean(actual_dir == predicted_dir) * 100

print(f"RMSE  : ${rmse:,.2f}")
print(f"MAE   : ${mae:,.2f}")
print(f"MAPE  : {mape:.2f}%")
print(f"R²    : {r2:.4f}")
print(f"Dir.Accuracy : {da:.2f}%")

# ── Get test dates ───────────────────────────────────
WINDOW_SIZE = 60
total       = len(df) - WINDOW_SIZE
train_size  = int(total * 0.80)
test_start  = WINDOW_SIZE + train_size
test_dates  = df.index[
    test_start : test_start + len(actual_usd)
]

print(f"\nTest period: "
      f"{test_dates[0].date()} "
      f"to {test_dates[-1].date()}")

# ── Plot actual vs predicted ─────────────────────────
plt.figure(figsize=(14, 6))

plt.plot(
    test_dates, actual_usd,
    color="blue", linewidth=1.5,
    label="Actual Bitcoin Price"
)
plt.plot(
    test_dates, predicted_usd,
    color="orange", linewidth=1.5,
    linestyle="--",
    label="Predicted Bitcoin Price"
)

metrics_text = (
    f"RMSE : ${rmse:,.0f}\n"
    f"MAE  : ${mae:,.0f}\n"
    f"MAPE : {mape:.1f}%\n"
    f"R²   : {r2:.3f}\n"
    f"Dir. : {da:.1f}%"
)
plt.text(
    0.02, 0.97,
    metrics_text,
    transform=plt.gca().transAxes,
    fontsize=10,
    verticalalignment="top",
    bbox=dict(
        boxstyle="round",
        facecolor="white",
        alpha=0.8
    )
)

plt.title("Bitcoin — Actual vs Predicted (Simple LSTM)")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("data/actual_vs_predicted.png")
plt.close()
print("\nChart saved: data/actual_vs_predicted.png")

# ── Error analysis ───────────────────────────────────
errors = actual_usd - predicted_usd

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(
    test_dates, errors,
    color="red", linewidth=0.8, alpha=0.7
)
plt.axhline(y=0, color="black",
            linestyle="--", linewidth=1)
plt.fill_between(
    test_dates, errors, 0,
    where=(errors > 0),
    color="green", alpha=0.2,
    label="Underestimated"
)
plt.fill_between(
    test_dates, errors, 0,
    where=(errors < 0),
    color="red", alpha=0.2,
    label="Overestimated"
)
plt.title("Prediction Error Over Time")
plt.xlabel("Date")
plt.ylabel("Error (USD)")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.hist(
    errors, bins=40,
    color="purple", alpha=0.7,
    edgecolor="white"
)
plt.axvline(x=0, color="black",
            linestyle="--", linewidth=1)
plt.axvline(
    x=mae, color="orange",
    linestyle="--", linewidth=1.5,
    label=f"MAE: ${mae:,.0f}"
)
plt.axvline(
    x=-mae, color="orange",
    linestyle="--", linewidth=1.5
)
plt.title("Distribution of Errors")
plt.xlabel("Error (USD)")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("data/error_analysis.png")
plt.close()
print("Error chart saved: data/error_analysis.png")

# ── Best and worst predictions ───────────────────────
print("\n--- Best 3 Predictions ---")
abs_errors = np.abs(errors)
best_3     = np.argsort(abs_errors)[:3]
for i in best_3:
    print(f"  {test_dates[i].date()} | "
          f"Actual: ${actual_usd[i]:,.0f} | "
          f"Predicted: ${predicted_usd[i]:,.0f} | "
          f"Error: ${errors[i]:,.0f}")

print("\n--- Worst 3 Predictions ---")
worst_3 = np.argsort(abs_errors)[-3:]
for i in worst_3:
    print(f"  {test_dates[i].date()} | "
          f"Actual: ${actual_usd[i]:,.0f} | "
          f"Predicted: ${predicted_usd[i]:,.0f} | "
          f"Error: ${errors[i]:,.0f}")

# ── Save metrics for later comparison ───────────────
import json
metrics_dict = {
    "model":  "Simple LSTM",
    "rmse":   round(rmse, 2),
    "mae":    round(mae, 2),
    "mape":   round(mape, 2),
    "r2":     round(r2, 4),
    "da":     round(da, 2)
}
with open("data/lstm_metrics.json", "w") as f:
    json.dump(metrics_dict, f, indent=2)
print("\nMetrics saved to data/lstm_metrics.json")

print("\n--- Step 4 Complete ---")
print("These are your baseline scores.")
print("Step 5 Hybrid LSTM-GRU will beat these!")