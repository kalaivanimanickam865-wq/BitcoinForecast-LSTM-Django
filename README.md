# BitcoinForecast-LSTM-Django

> Real-time Bitcoin price forecasting using Hybrid LSTM-GRU Deep Learning, deployed as a Django web application with live market data integration.

![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python)
![Django](https://img.shields.io/badge/Django-6.0-darkgreen?style=flat-square&logo=django)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-LSTM--GRU-red?style=flat-square)
![CoinGecko](https://img.shields.io/badge/API-CoinGecko-green?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

---

## What is this project?

**BitcoinForecast-LSTM-Django** is an end-to-end deep learning web application that:

- Trains a **Hybrid LSTM-GRU neural network** on 7 years of Bitcoin price data (2018–2025)
- Fetches **real-time market data** from CoinGecko API
- Generates **30-day price forecasts** with confidence bands
- Displays everything in a **live interactive Django dashboard**

---

## Live Demo Screenshots

### Home Page — Live BTC Price + Model Metrics
![Home](data/screenshots/home.png)

### Forecast Page — Past Data + Live Data + 30-Day Forecast
![Forecast](data/screenshots/forecast.png)

### Live Dashboard — Real-time Market Data
![Live](data/screenshots/live.png)

---

## Model Results

| Metric | Simple LSTM | Hybrid LSTM-GRU |
|--------|-------------|-----------------|
| RMSE | $6,016 | **$4,202** ✅ |
| MAE | $4,314 | **$3,154** ✅ |
| MAPE | 6.85% | **4.93%** ✅ |
| R² Score | 0.90 | **0.9535** ✅ |
| Directional Acc | 53.75% | 48.90% |

**Best Model: Hybrid LSTM-GRU** — 30% lower RMSE, R² of 0.9535

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Deep Learning | TensorFlow / Keras |
| Models | LSTM + GRU (Hybrid) |
| Web Framework | Django 6.0 |
| Live Data | CoinGecko API |
| Data Processing | Pandas, NumPy, Scikit-learn |
| Visualization | Matplotlib, Chart.js |
| Language | Python 3.12 |

---

## Project Structure

```
BitcoinForecast-LSTM-Django/
│
├── predictor/                      ← Django app
│   ├── views.py                    ← All views + forecast logic
│   ├── urls.py                     ← URL routing
│   └── templates/
│       └── predictor/
│           ├── base.html           ← Base template
│           ├── index.html          ← Home page
│           ├── forecast.html       ← Forecast page
│           ├── dashboard.html      ← Model dashboard
│           └── live.html           ← Live market data
│
├── bitcoinproject/                 ← Django project settings
│   ├── settings.py
│   └── urls.py
│
├── ml_models/                      ← Training scripts
│   ├── 01_explore_data.py
│   ├── 02_prepare_data.py
│   ├── 03_train_lstm.py
│   ├── 04_predict_and_plot.py
│   ├── 05_hybrid_lstm_gru.py
│   ├── 06_feature_engineering.py
│   ├── 07_train_multifeature.py
│   └── 08_forecast_dashboard.py
│
├── data/                           ← Model files (not pushed to git)
│   ├── best_hybrid_model.keras
│   ├── scaler.pkl
│   ├── btc_data.csv
│   └── forecast_30day.csv
│
├── requirements.txt
├── manage.py
└── README.md
```

---

## How the Forecast Works

```
Step 1 — Past Data (2018-2024)
    Historical CSV → Train LSTM-GRU model

Step 2 — Live Data (Today)
    CoinGecko API → Last 90 days real prices

Step 3 — Model Prediction
    Last 60 days (live) → LSTM-GRU → Next 30 days

Step 4 — Momentum Blend
    40% Model signal + 60% Recent market trend

Step 5 — Confidence Band
    ± volatility × √time (uncertainty grows over time)
```

---

## Model Architecture

```
Input: (60 days × 1 feature)
         │
    LSTM (50 units)     ← Long-term trend
         │
    Dropout (0.2)
         │
    GRU  (50 units)     ← Short-term momentum
         │
    Dropout (0.2)
         │
    Dense (25, ReLU)
         │
    Dense (1)           ← Predicted price
```

---

## Features Engineered (13 Total)

| # | Feature | Category |
|---|---------|----------|
| 1 | Close Price | Base |
| 2 | Volume | Base |
| 3 | RSI (14) | Momentum |
| 4 | MACD | Momentum |
| 5 | MACD Signal | Momentum |
| 6 | EMA 12 | Trend |
| 7 | EMA 26 | Trend |
| 8 | Bollinger Upper | Volatility |
| 9 | Bollinger Lower | Volatility |
| 10 | SMA 7 | Trend |
| 11 | SMA 30 | Trend |
| 12 | Price ROC (10d) | Momentum |
| 13 | OBV | Volume |

---

## Web App Pages

| URL | Page | Description |
|-----|------|-------------|
| `/` | Home | Live BTC price + model metrics + 7-day preview |
| `/forecast/` | Forecast | Full chart + 30-day prediction table |
| `/live/` | Live Dashboard | Real-time market data via CoinGecko |
| `/dashboard/` | ML Dashboard | Model evaluation charts |
| `/api/forecast/` | JSON API | Forecast data as JSON |

---

## Quickstart

### 1. Clone the repo
```bash
git clone https://github.com/kalaivanimanickam865-wq/BitcoinForecast-LSTM-Django.git
cd BitcoinForecast-LSTM-Django
```

### 2. Create virtual environment
```bash
python -m venv btcenv
btcenv\Scripts\activate        # Windows
# source btcenv/bin/activate   # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add model files
Download and place in `data/` folder:
- `best_hybrid_model.keras`
- `scaler.pkl`
- `btc_data.csv`

### 5. Run Django server
```bash
python manage.py runserver
```

### 6. Open browser
```
http://127.0.0.1:8000
```

---

## Requirements

```
tensorflow>=2.11
django>=6.0
scikit-learn>=1.3
numpy>=1.24
pandas>=2.0
matplotlib>=3.7
joblib>=1.3
yfinance>=0.2
requests>=2.28
```

Install all:
```bash
pip install -r requirements.txt
```

---

## API Usage

```bash
# 30-day forecast
GET /api/forecast/?days=30

# 7-day forecast
GET /api/forecast/?days=7
```

Response:
```json
{
  "live_price": 83420.50,
  "live_change": -1.23,
  "data_source": "CoinGecko Live",
  "current_price": 83420.50,
  "days": 30,
  "momentum_pct": -0.506,
  "forecast": [
    {
      "date": "2025-04-07",
      "price": 82800.00,
      "upper": 84200.00,
      "lower": 81400.00,
      "change_pct": -0.74
    }
  ]
}
```

---

## Disclaimer

> This project is for **educational purposes only**.
> Cryptocurrency markets are highly volatile and unpredictable.
> This is **NOT financial advice**.
> Do not make investment decisions based on model outputs.

---

## Author

**Kalaivani M**
- GitHub: [@kalaivanimanickam865-wq](https://github.com/kalaivanimanickam865-wq)

---

## License

MIT License — free to use, modify, and distribute.

---

*Built with TensorFlow, Keras, Django, Pandas, Matplotlib and CoinGecko API*
