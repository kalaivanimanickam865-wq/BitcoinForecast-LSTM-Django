cat > model_downloader.py << 'EOF'
import os
import requests

MODEL_FILES = {
    "best_hybrid_model.keras": "https://huggingface.co/datasets/Kalaivanimanickam865/bitcoinforecast_models/blob/main/best_hybrid_model.keras",
    "scaler.pkl"             : "https://huggingface.co/datasets/Kalaivanimanickam865/bitcoinforecast_models/blob/main/scaler.pkl",
    "hybrid_metrics.json"    : "https://huggingface.co/datasets/Kalaivanimanickam865/bitcoinforecast_models/blob/main/hybrid_metrics.json",
    "forecast_30day.csv"     : "https://huggingface.co/datasets/Kalaivanimanickam865/bitcoinforecast_models/blob/main/forecast_30day.csv",
}

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

def download_models():
    for filename, url in MODEL_FILES.items():
        filepath = os.path.join(DATA_DIR, filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            try:
                res = requests.get(url, stream=True, timeout=60)
                with open(filepath, "wb") as f:
                    for chunk in res.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"Done: {filename}")
            except Exception as e:
                print(f"Failed: {filename} - {e}")
        else:
            print(f"Already exists: {filename}")

if __name__ == "__main__":
    download_models()
EOF