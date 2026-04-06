import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import joblib
import os
from datetime import timedelta
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import load_model

print("=" * 60)
print("  STEP 8 (FIXED) — REALISTIC FORECAST + DASHBOARD")
print("=" * 60)

# ── Load everything ──────────────────────────────────
print("\nLoading model and data...")

model  = load_model("data/best_hybrid_model.keras")
scaler = joblib.load("data/scaler.pkl")

X_train = np.load("data/X_train.npy")
y_train = np.load("data/y_train.npy")
X_test  = np.load("data/X_test.npy")
y_test  = np.load("data/y_test.npy")

hybrid_scaled = np.load("data/hybrid_predictions_scaled.npy")
lstm_scaled   = np.load("data/predictions_scaled.npy")

df = pd.read_csv("data/btc_data.csv", index_col=0, parse_dates=True)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

print("Model and data loaded")

# ── Inverse transform ────────────────────────────────
actual_usd = scaler.inverse_transform(
    y_test.reshape(-1, 1)).flatten()
hybrid_usd = scaler.inverse_transform(hybrid_scaled).flatten()
lstm_usd   = scaler.inverse_transform(lstm_scaled).flatten()

min_len    = min(len(actual_usd), len(hybrid_usd), len(lstm_usd))
actual_usd = actual_usd[:min_len]
hybrid_usd = hybrid_usd[:min_len]
lstm_usd   = lstm_usd[:min_len]

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

# ── Test dates ───────────────────────────────────────
WINDOW_SIZE = 60
train_size  = int((len(df) - WINDOW_SIZE) * 0.80)
test_start  = WINDOW_SIZE + train_size
test_dates  = df.index[test_start : test_start + min_len]

# ════════════════════════════════════════════════════
# REALISTIC 30-DAY FORECAST
# Fix: blend model signal with recent momentum
# instead of pure autoregressive rollout
# ════════════════════════════════════════════════════
print("\nGenerating realistic 30-day forecast...")

close_col  = df[["Close"]]
scaled_all = scaler.transform(close_col.values)
current_price = df["Close"].iloc[-1]

# --- Raw model rollout ---
seed_seq = scaled_all[-WINDOW_SIZE:].reshape(1, WINDOW_SIZE, 1)
raw_scaled = []
seq = seed_seq.copy()
for _ in range(30):
    pred = model.predict(seq, verbose=0)[0, 0]
    raw_scaled.append(pred)
    seq = np.append(seq[:, 1:, :], [[[pred]]], axis=1)

raw_forecast = scaler.inverse_transform(
    np.array(raw_scaled).reshape(-1, 1)).flatten()

# --- Model implied daily changes ---
raw_changes = np.diff(raw_forecast, prepend=current_price) / \
              np.concatenate([[current_price], raw_forecast[:-1]])

# --- Recent 14-day momentum ---
recent      = df["Close"].iloc[-14:].values
ret         = np.diff(recent) / recent[:-1]
momentum    = float(np.mean(ret))
volatility  = float(np.std(ret))

print(f"  14-day momentum  : {momentum*100:+.3f}% / day")
print(f"  14-day volatility: {volatility*100:.3f}% / day")

# --- Blend: 40% model + 60% momentum (decaying) ---
forecast_prices = [current_price]
for i in range(30):
    decay   = np.exp(-i / 20)
    blended = (0.40 * raw_changes[i] +
               0.60 * momentum * decay)
    blended = np.clip(blended, -0.05, 0.05)   # cap at 5%/day
    forecast_prices.append(forecast_prices[-1] * (1 + blended))

forecast_usd = np.array(forecast_prices[1:])

# --- Confidence bands ---
widths     = np.array([volatility * np.sqrt(i + 1) * forecast_usd[i]
                       for i in range(30)])
upper_band = forecast_usd + widths
lower_band = np.maximum(forecast_usd - widths, 1000)

last_date    = df.index[-1]
future_dates = pd.date_range(
    start=last_date + timedelta(days=1),
    periods=30, freq="D"
)

change_pct = (forecast_usd[-1] - current_price) / current_price * 100

print(f"\n  Current price : ${current_price:,.0f}")
print(f"  D+7  forecast : ${forecast_usd[6]:,.0f}  "
      f"({(forecast_usd[6]-current_price)/current_price*100:+.1f}%)")
print(f"  D+14 forecast : ${forecast_usd[13]:,.0f}  "
      f"({(forecast_usd[13]-current_price)/current_price*100:+.1f}%)")
print(f"  D+30 forecast : ${forecast_usd[-1]:,.0f}  "
      f"({change_pct:+.1f}%)")
print(f"  Upper D+30    : ${upper_band[-1]:,.0f}")
print(f"  Lower D+30    : ${lower_band[-1]:,.0f}")

# ════════════════════════════════════════════════════
# DASHBOARD
# ════════════════════════════════════════════════════
print("\nBuilding dashboard...")

fig = plt.figure(figsize=(20, 24))
fig.patch.set_facecolor("#0d1117")

gs = gridspec.GridSpec(
    4, 2, figure=fig,
    hspace=0.45, wspace=0.3,
    top=0.93, bottom=0.04,
    left=0.07, right=0.97
)

DARK_BG  = "#161b22"
GRID_COL = "#30363d"
TEXT_COL = "#e6edf3"
BLUE     = "#58a6ff"
GREEN    = "#3fb950"
ORANGE   = "#f78166"
YELLOW   = "#e3b341"
PURPLE   = "#bc8cff"

def style_ax(ax, title):
    ax.set_facecolor(DARK_BG)
    ax.tick_params(colors=TEXT_COL, labelsize=9)
    ax.xaxis.label.set_color(TEXT_COL)
    ax.yaxis.label.set_color(TEXT_COL)
    ax.title.set_color(TEXT_COL)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COL)
    ax.grid(True, color=GRID_COL, alpha=0.5, linewidth=0.7)
    ax.set_title(title, fontsize=11, fontweight="bold",
                 color=TEXT_COL, pad=10)

fig.text(0.5, 0.965,
         "Bitcoin Price Prediction — Final Dashboard",
         ha="center", fontsize=20, fontweight="bold",
         color=TEXT_COL)
fig.text(0.5, 0.948,
         f"Hybrid LSTM-GRU  |  Data: 2018-2024  |  "
         f"Test: {test_dates[0].strftime('%b %Y')} - "
         f"{test_dates[-1].strftime('%b %Y')}",
         ha="center", fontsize=11, color="#8b949e")

# ── Chart 1: Full history ────────────────────────────
ax1 = fig.add_subplot(gs[0, :])
style_ax(ax1, "BTC Price — Full History & Model Predictions")

train_dates  = df.index[WINDOW_SIZE : WINDOW_SIZE + len(y_train)]
train_actual = scaler.inverse_transform(
    y_train.reshape(-1, 1)).flatten()

ax1.plot(train_dates, train_actual,
         color="#8b949e", linewidth=1, alpha=0.6,
         label="Historical (train)")
ax1.plot(test_dates, actual_usd,
         color=BLUE, linewidth=2, label="Actual (test)")
ax1.plot(test_dates, hybrid_usd,
         color=GREEN, linewidth=1.5, linestyle="--", alpha=0.9,
         label=f"Hybrid  RMSE=${m_hybrid[0]:,.0f}  R2={m_hybrid[3]:.3f}")
ax1.plot(test_dates, lstm_usd,
         color=ORANGE, linewidth=1, linestyle=":", alpha=0.7,
         label=f"Simple LSTM  RMSE=${m_lstm[0]:,.0f}")
ax1.set_ylabel("Price (USD)", color=TEXT_COL)
ax1.legend(loc="upper left", fontsize=9,
           facecolor=DARK_BG, labelcolor=TEXT_COL,
           edgecolor=GRID_COL)
ax1.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

# ── Chart 2: 30-day forecast ─────────────────────────
ax2 = fig.add_subplot(gs[1, :])
style_ax(ax2,
         "30-Day Future Forecast (Momentum-Anchored + Confidence Band)")

context_dates  = df.index[-90:]
context_prices = df["Close"].iloc[-90:].values

ax2.plot(context_dates, context_prices,
         color=BLUE, linewidth=2, label="Recent Actual")
ax2.plot(future_dates, forecast_usd,
         color=YELLOW, linewidth=2.5,
         label=f"Forecast  D+30=${forecast_usd[-1]:,.0f}  "
               f"({change_pct:+.1f}%)")
ax2.fill_between(future_dates, lower_band, upper_band,
                 color=YELLOW, alpha=0.15,
                 label="Confidence Band")
ax2.axvline(x=last_date, color=GRID_COL,
            linestyle="--", linewidth=1.2)

for day, label in [(6, "D+7"), (13, "D+14"), (29, "D+30")]:
    ax2.annotate(
        f"{label}\n${forecast_usd[day]:,.0f}",
        xy=(future_dates[day], forecast_usd[day]),
        xytext=(0, 18), textcoords="offset points",
        ha="center", fontsize=8, color=YELLOW,
        arrowprops=dict(arrowstyle="-", color=YELLOW, alpha=0.5)
    )

ax2.set_ylabel("Price (USD)", color=TEXT_COL)
ax2.legend(loc="upper left", fontsize=9,
           facecolor=DARK_BG, labelcolor=TEXT_COL,
           edgecolor=GRID_COL)
ax2.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

# ── Chart 3: Residuals ───────────────────────────────
ax3 = fig.add_subplot(gs[2, 0])
style_ax(ax3, "Prediction Error Over Time")

residuals = actual_usd - hybrid_usd
colors    = [GREEN if r >= 0 else ORANGE for r in residuals]
ax3.bar(test_dates, residuals, color=colors, alpha=0.7, width=1.5)
ax3.axhline(0, color=TEXT_COL, linewidth=0.8)
std_res = np.std(residuals)
ax3.axhline( std_res, color=YELLOW, linewidth=1,
             linestyle="--", alpha=0.7,
             label=f"+1 std=${std_res:,.0f}")
ax3.axhline(-std_res, color=YELLOW, linewidth=1,
             linestyle="--", alpha=0.7)
ax3.set_ylabel("Error (USD)", color=TEXT_COL)
ax3.legend(fontsize=8, facecolor=DARK_BG,
           labelcolor=TEXT_COL, edgecolor=GRID_COL)
ax3.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

# ── Chart 4: Scatter ─────────────────────────────────
ax4 = fig.add_subplot(gs[2, 1])
style_ax(ax4, "Actual vs Predicted")

ax4.scatter(actual_usd, hybrid_usd,
            color=PURPLE, alpha=0.4, s=8,
            label="Hybrid predictions")
mn = min(actual_usd.min(), hybrid_usd.min())
mx = max(actual_usd.max(), hybrid_usd.max())
ax4.plot([mn, mx], [mn, mx], color=TEXT_COL,
         linewidth=1, linestyle="--", alpha=0.6,
         label="Perfect prediction")
ax4.set_xlabel("Actual Price (USD)", color=TEXT_COL)
ax4.set_ylabel("Predicted Price (USD)", color=TEXT_COL)
ax4.legend(fontsize=8, facecolor=DARK_BG,
           labelcolor=TEXT_COL, edgecolor=GRID_COL)
ax4.xaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
ax4.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))

# ── Chart 5: Metrics bar ─────────────────────────────
ax5 = fig.add_subplot(gs[3, 0])
style_ax(ax5, "Model Metrics Comparison")

metrics_labels = ["RMSE ($k)", "MAE ($k)", "MAPE (%)", "Dir.Acc (%)"]
lstm_vals   = [m_lstm[0]/1000,   m_lstm[1]/1000,
               m_lstm[2],         m_lstm[4]]
hybrid_vals = [m_hybrid[0]/1000, m_hybrid[1]/1000,
               m_hybrid[2],       m_hybrid[4]]

x     = np.arange(len(metrics_labels))
width = 0.35
bars1 = ax5.bar(x - width/2, lstm_vals, width,
                label="Simple LSTM", color=ORANGE, alpha=0.8)
bars2 = ax5.bar(x + width/2, hybrid_vals, width,
                label="Hybrid LSTM-GRU", color=GREEN, alpha=0.8)
ax5.set_xticks(x)
ax5.set_xticklabels(metrics_labels, fontsize=9, color=TEXT_COL)
ax5.legend(fontsize=9, facecolor=DARK_BG,
           labelcolor=TEXT_COL, edgecolor=GRID_COL)
for bar in list(bars1) + list(bars2):
    ax5.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.02,
             f"{bar.get_height():.1f}",
             ha="center", va="bottom",
             fontsize=7, color=TEXT_COL)

# ── Panel 6: Summary card ────────────────────────────
ax6 = fig.add_subplot(gs[3, 1])
ax6.set_facecolor(DARK_BG)
for spine in ax6.spines.values():
    spine.set_edgecolor(GRID_COL)
ax6.set_xticks([]); ax6.set_yticks([])
ax6.set_title("Final Model Summary", fontsize=11,
              fontweight="bold", color=TEXT_COL, pad=10)

d7_pct  = (forecast_usd[6]  - current_price) / current_price * 100
d14_pct = (forecast_usd[13] - current_price) / current_price * 100

summary_lines = [
    ("Model",        "Hybrid LSTM-GRU"),
    ("Architecture", "LSTM(50)->GRU(50)->Dense"),
    ("Train Period", "2018 - 2022  (80%)"),
    ("Test Period",  "2023 - 2024  (20%)"),
    ("", ""),
    ("RMSE",         f"${m_hybrid[0]:,.0f}"),
    ("MAE",          f"${m_hybrid[1]:,.0f}"),
    ("MAPE",         f"{m_hybrid[2]:.2f}%"),
    ("R2",           f"{m_hybrid[3]:.4f}"),
    ("Dir. Acc",     f"{m_hybrid[4]:.1f}%"),
    ("", ""),
    ("Current",      f"${current_price:,.0f}"),
    ("D+7",          f"${forecast_usd[6]:,.0f}  ({d7_pct:+.1f}%)"),
    ("D+14",         f"${forecast_usd[13]:,.0f}  ({d14_pct:+.1f}%)"),
    ("D+30",         f"${forecast_usd[-1]:,.0f}  ({change_pct:+.1f}%)"),
]

y_pos = 0.97
for label, value in summary_lines:
    if label == "":
        y_pos -= 0.04
        continue
    val_color = TEXT_COL
    if label in ("D+7", "D+14", "D+30"):
        val_color = GREEN if "+" in value else ORANGE
    ax6.text(0.05, y_pos, label + ":",
             transform=ax6.transAxes,
             fontsize=9, color="#8b949e", va="top")
    ax6.text(0.45, y_pos, value,
             transform=ax6.transAxes,
             fontsize=9, color=val_color,
             fontweight="bold", va="top")
    y_pos -= 0.065

# ── Save ─────────────────────────────────────────────
os.makedirs("data", exist_ok=True)
plt.savefig("data/final_dashboard.png", dpi=150,
            facecolor=fig.get_facecolor())
plt.close()
print("Dashboard saved: data/final_dashboard.png")

# ── Forecast CSV ─────────────────────────────────────
forecast_df = pd.DataFrame({
    "Date"       : future_dates.strftime("%Y-%m-%d"),
    "Forecast"   : np.round(forecast_usd, 2),
    "Upper_Band" : np.round(upper_band, 2),
    "Lower_Band" : np.round(lower_band, 2),
    "Change_Pct" : np.round(
        (forecast_usd - current_price) / current_price * 100, 2)
})
forecast_df.to_csv("data/forecast_30day.csv", index=False)
print("Forecast saved: data/forecast_30day.csv")

# ── Print summary ────────────────────────────────────
print("\n" + "="*60)
print("  FINAL PROJECT SUMMARY")
print("="*60)
print(f"  Model         : Hybrid LSTM-GRU")
print(f"  RMSE          : ${m_hybrid[0]:,.0f}")
print(f"  MAE           : ${m_hybrid[1]:,.0f}")
print(f"  MAPE          : {m_hybrid[2]:.2f}%")
print(f"  R2            : {m_hybrid[3]:.4f}")
print(f"  Dir. Accuracy : {m_hybrid[4]:.1f}%")
print(f"  --")
print(f"  Current price : ${current_price:,.0f}")
print(f"  D+7  forecast : ${forecast_usd[6]:,.0f}  ({d7_pct:+.1f}%)")
print(f"  D+14 forecast : ${forecast_usd[13]:,.0f}  ({d14_pct:+.1f}%)")
print(f"  D+30 forecast : ${forecast_usd[-1]:,.0f}  ({change_pct:+.1f}%)")
print("="*60)
print("\nStep 8 Complete — Project Finished!")
print("\nOutputs saved:")
print("  data/final_dashboard.png")
print("  data/forecast_30day.csv")