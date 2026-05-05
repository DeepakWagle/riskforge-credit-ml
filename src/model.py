from pathlib import Path
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT.mkdir(parents=True, exist_ok=True)

def load_model(name):
    path = PROJECT_ROOT / 'models' / name
    return joblib.load(path)

def save_model(model, filename):
    path = PROJECT_ROOT / 'models' / filename
    joblib.dump(model, path)
    print(f"✅ Model saved at: {path}")