import numpy as np
import pandas as pd
import joblib
import json
import os
import base64
import io
import requests
from datetime import timedelta, datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from django.shortcuts import render
from django.http import JsonResponse

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ── Paths ────────────────────────────────────────────
DATA_DIR   = r"C:\Projects\BitcoinPredictor\data"
MODEL_PATH = os.path.join(DATA_DIR, "best_hybrid_model.keras")

# ── Lazy-load model once ─────────────────────────────
_model  = None
_scaler = None

def get_model():
    global _model, _scaler
    if _model is None:
        from tensorflow.keras.models import load_model
        _model  = load_model(MODEL_PATH)
        _scaler = joblib.load(os.path.join(DATA_DIR, "scaler.pkl"))
    return _model, _scaler


# ════════════════════════════════════════════════════
# LIVE DATA FETCHERS
# ════════════════════════════════════════════════════

def get_live_btc_price():
    """Current BTC price + 24h change."""
    try:
        res  = requests.get(
            "https://api.coingecko.com/api/v3/simple/price",
            params={"ids":"bitcoin","vs_currencies":"usd",
                    "include_24hr_change":"true"},
            timeout=5
        )
        data = res.json()["bitcoin"]
        return {
            "price"     : data["usd"],
            "change_24h": round(data.get("usd_24h_change", 0), 2),
            "source"    : "CoinGecko (live)",
            "live"      : True
        }
    except Exception:
        try:
            df = load_btc_csv()
            return {
                "price"     : float(df["Close"].iloc[-1]),
                "change_24h": round(
                    (df["Close"].iloc[-1] - df["Close"].iloc[-2])
                    / df["Close"].iloc[-2] * 100, 2),
                "source"    : "CSV (offline)",
                "live"      : False
            }
        except Exception:
            return {"price":0,"change_24h":0,
                    "source":"unavailable","live":False}


def get_live_90day_prices():
    """
    Fetch last 90 days daily BTC prices from CoinGecko.
    Falls back to CSV last 90 days if API fails.
    Returns: (pd.Series with DatetimeIndex, is_live bool)
    """
    try:
        res  = requests.get(
            "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart",
            params={"vs_currency":"usd","days":"90","interval":"daily"},
            timeout=10
        )
        data   = res.json()
        prices = data["prices"]
        dates  = [datetime.fromtimestamp(p[0]/1000) for p in prices]
        vals   = [p[1] for p in prices]

        series = pd.Series(vals,
                           index=pd.DatetimeIndex(dates),
                           name="Close")
        series = series[~series.index.duplicated(keep="last")]
        series = series.sort_index()
        return series, True

    except Exception as e:
        print(f"[Fallback] CoinGecko failed ({e}) — using CSV")
        df = load_btc_csv()
        return df["Close"].tail(90), False


def load_btc_csv():
    df = pd.read_csv(
        os.path.join(DATA_DIR, "btc_data.csv"),
        index_col=0, parse_dates=True
    )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


# ════════════════════════════════════════════════════
# FORECAST — live 90-day seed + LSTM model
# ════════════════════════════════════════════════════

def generate_forecast(days=30):
    """
    1. Fetch last 90 days live from CoinGecko (fallback: CSV)
    2. Use last 60 days as model seed
    3. Blend model output with recent momentum
    4. Return forecast dict
    """
    model, scaler = get_model()

    # Step 1 — live history
    price_series, is_live = get_live_90day_prices()

    # Pad with CSV if not enough data
    if len(price_series) < 60:
        csv_df       = load_btc_csv()
        price_series = pd.concat([
            csv_df["Close"].tail(90 - len(price_series)),
            price_series
        ])

    current   = float(price_series.iloc[-1])
    last_date = price_series.index[-1]

    # Step 2 — scale + model rollout
    scaled_all = scaler.transform(
        price_series.values.reshape(-1, 1)
    )

    WINDOW = 60
    seq    = scaled_all[-WINDOW:].reshape(1, WINDOW, 1)
    raw_sc = []
    for _ in range(days):
        p = model.predict(seq, verbose=0)[0, 0]
        raw_sc.append(p)
        seq = np.append(seq[:, 1:, :], [[[p]]], axis=1)

    raw_usd = scaler.inverse_transform(
        np.array(raw_sc).reshape(-1, 1)).flatten()
    raw_chg = np.diff(raw_usd, prepend=current) / \
              np.concatenate([[current], raw_usd[:-1]])

    # Step 3 — momentum blend
    recent     = price_series.iloc[-14:].values
    ret        = np.diff(recent) / recent[:-1]
    momentum   = float(np.mean(ret))
    volatility = float(np.std(ret))

    prices_fc = [current]
    for i in range(days):
        decay   = np.exp(-i / 20)
        blended = 0.40 * raw_chg[i] + 0.60 * momentum * decay
        blended = np.clip(blended, -0.05, 0.05)
        prices_fc.append(prices_fc[-1] * (1 + blended))

    forecast = np.array(prices_fc[1:])
    widths   = np.array([volatility * np.sqrt(i+1) * forecast[i]
                         for i in range(days)])
    upper    = forecast + widths
    lower    = np.maximum(forecast - widths, 1000)

    future_dates = pd.date_range(
        start=last_date + timedelta(days=1),
        periods=days, freq="D"
    )

    return {
        "current"      : current,
        "forecast"     : forecast,
        "upper"        : upper,
        "lower"        : lower,
        "future_dates" : future_dates,
        "momentum"     : momentum,
        "volatility"   : volatility,
        "last_date"    : last_date,
        "price_series" : price_series,
        "is_live"      : is_live,
    }


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120,
                facecolor=fig.get_facecolor(),
                bbox_inches="tight")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img_b64


# ════════════════════════════════════════════════════
# VIEWS
# ════════════════════════════════════════════════════

def index(request):
    try:
        live         = get_live_btc_price()
        metrics_path = os.path.join(DATA_DIR, "hybrid_metrics.json")
        with open(metrics_path) as f:
            metrics = json.load(f)

        df             = load_btc_csv()
        fc_path        = os.path.join(DATA_DIR, "forecast_30day.csv")
        forecast_table = None
        if os.path.exists(fc_path):
            fc_df          = pd.read_csv(fc_path)
            forecast_table = fc_df.head(7).to_dict("records")

        context = {
            "metrics"        : metrics,
            "live_price"     : f"{live['price']:,.0f}",
            "live_change"    : live["change_24h"],
            "live_source"    : live["source"],
            "is_live"        : live["live"],
            "current_price"  : f"{float(df['Close'].iloc[-1]):,.0f}",
            "forecast_table" : forecast_table,
            "last_date"      : df.index[-1].strftime("%B %d, %Y"),
        }
    except Exception as e:
        context = {"error": str(e)}

    return render(request, "predictor/index.html", context)


def forecast_view(request):
    days = int(request.GET.get("days", 30))
    days = max(7, min(days, 90))

    try:
        result   = generate_forecast(days)
        live     = get_live_btc_price()

        fc       = result["forecast"]
        upper    = result["upper"]
        lower    = result["lower"]
        fdts     = result["future_dates"]
        curr     = result["current"]
        ps       = result["price_series"]
        is_live  = result["is_live"]

        change_pct = (fc[-1] - curr) / curr * 100

        # ── Chart — Past CSV + Live 90d + Forecast ────
        DARK_BG  = "#161b22"
        GRID_COL = "#30363d"
        TEXT_COL = "#e6edf3"
        GRAY     = "#8b949e"
        BLUE     = "#58a6ff"
        GREEN    = "#3fb950"
        ORANGE   = "#f78166"

        fig, ax = plt.subplots(figsize=(14, 5))
        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor(DARK_BG)
        ax.tick_params(colors=TEXT_COL, labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COL)
        ax.grid(True, color=GRID_COL, alpha=0.4)

        # ── 1. Past CSV data (2018 → 90 days ago) ────
        csv_df    = load_btc_csv()
        # Show last 365 days of CSV for context
        csv_tail  = csv_df["Close"].tail(365)
        # Only show CSV part that is BEFORE live data starts
        live_start = ps.index[0]
        csv_past   = csv_tail[csv_tail.index < live_start]
        if len(csv_past) > 0:
            ax.plot(csv_past.index, csv_past.values,
                    color=GRAY, linewidth=1.2, alpha=0.7,
                    label="Past Data (CSV 2018–2024)")

        # ── 2. Live 90-day data ───────────────────────
        live_label = "Live 90 Days (CoinGecko)" if is_live \
                     else "Recent CSV (offline)"
        ax.plot(ps.index, ps.values,
                color=BLUE, linewidth=2,
                label=live_label)

        # ── 3. Today marker ───────────────────────────
        ax.axvline(x=result["last_date"],
                   color=GRAY, linestyle="--",
                   linewidth=1, alpha=0.8)
        ax.text(result["last_date"],
                ps.values.min() * 0.995,
                "  Today", color=GRAY, fontsize=8)

        # ── 4. Forecast ───────────────────────────────
        fc_color = GREEN if change_pct >= 0 else ORANGE
        ax.plot(fdts, fc,
                color=fc_color, linewidth=2.5,
                label=f"{days}-Day Forecast  "
                      f"D+{days}=${fc[-1]:,.0f} "
                      f"({change_pct:+.1f}%)")
        ax.fill_between(fdts, lower, upper,
                        color=fc_color, alpha=0.12,
                        label="Confidence Band")

        # ── 5. Milestone annotations ──────────────────
        for d, lbl in [(6,"D+7"),(13,"D+14"),(days-1,f"D+{days}")]:
            if d < days:
                ax.annotate(
                    f"{lbl}\n${fc[d]:,.0f}",
                    xy=(fdts[d], fc[d]),
                    xytext=(0, 18), textcoords="offset points",
                    ha="center", fontsize=8, color=fc_color,
                    arrowprops=dict(arrowstyle="-",
                                   color=fc_color, alpha=0.5)
                )

        ax.set_title(
            f"BTC Price — Past History + Live Data + {days}-Day Forecast  "
            f"|  Current: ${curr:,.0f}  "
            f"|  {'LIVE' if is_live else 'OFFLINE'}",
            color=TEXT_COL, fontsize=11
        )
        ax.set_ylabel("Price (USD)", color=TEXT_COL)
        ax.legend(facecolor=DARK_BG, labelcolor=TEXT_COL,
                  edgecolor=GRID_COL, fontsize=9,
                  loc="upper left")
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        fig.tight_layout()
        chart_b64 = fig_to_base64(fig)

        # Forecast table
        table = [
            {
                "date"       : fdts[i].strftime("%Y-%m-%d"),
                "forecast"   : f"{fc[i]:,.0f}",
                "upper"      : f"{upper[i]:,.0f}",
                "lower"      : f"{lower[i]:,.0f}",
                "change_pct" : f"{(fc[i]-curr)/curr*100:+.1f}%",
            }
            for i in [0, 6, 13, 20, 27, days-1]
            if i < days
        ]

        context = {
            "days"        : days,
            "days_list"   : [7, 14, 30, 60, 90],
            "chart"       : chart_b64,
            "current"     : f"{curr:,.0f}",
            "live_price"  : f"{live['price']:,.0f}",
            "live_change" : live["change_24h"],
            "is_live"     : is_live,
            "data_source" : "CoinGecko Live" if is_live else "CSV Offline",
            "d30"         : f"{fc[min(29,days-1)]:,.0f}",
            "change_pct"  : f"{change_pct:+.1f}",
            "momentum"    : f"{result['momentum']*100:+.3f}",
            "volatility"  : f"{result['volatility']*100:.3f}",
            "table"       : table,
            "seed_days"   : len(ps),
        }

    except Exception as e:
        context = {"error": str(e)}

    return render(request, "predictor/forecast.html", context)


def dashboard_view(request):
    img_path = os.path.join(DATA_DIR, "final_dashboard.png")
    if os.path.exists(img_path):
        with open(img_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
        context = {"dashboard_img": img_b64}
    else:
        context = {"error": "Run 08_forecast_dashboard.py first."}
    return render(request, "predictor/dashboard.html", context)


def live_dashboard(request):
    return render(request, "predictor/live.html", {})


def api_forecast(request):
    days = int(request.GET.get("days", 30))
    days = max(7, min(days, 90))
    try:
        result = generate_forecast(days)
        live   = get_live_btc_price()
        fc     = result["forecast"]
        curr   = result["current"]
        fdts   = result["future_dates"]
        data   = {
            "live_price"   : live["price"],
            "live_change"  : live["change_24h"],
            "data_source"  : "CoinGecko Live" if result["is_live"]
                             else "CSV Offline",
            "current_price": round(curr, 2),
            "days"         : days,
            "momentum_pct" : round(result["momentum"]*100, 3),
            "seed_days"    : len(result["price_series"]),
            "forecast"     : [
                {
                    "date"      : fdts[i].strftime("%Y-%m-%d"),
                    "price"     : round(float(fc[i]), 2),
                    "upper"     : round(float(result["upper"][i]), 2),
                    "lower"     : round(float(result["lower"][i]), 2),
                    "change_pct": round((fc[i]-curr)/curr*100, 2),
                }
                for i in range(days)
            ]
        }
        return JsonResponse(data)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)