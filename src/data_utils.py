import pandas as pd
from pathlib import Path

DATA_DIR=Path(__file__).parent.parent/'data'

def load_raw(name):
    path=DATA_DIR/'raw'/name
    return pd.read_csv(path)

def load_interim(name):
    path=DATA_DIR/'interim'/name
    return pd.read_csv(name)

def save_interim(df, name):
    path=DATA_DIR/'interim'
    path.mkdir(parents=True, exist_ok=True)
    df.to_csv(path/name, index=False)

def save_processed(df, name):
    path=DATA_DIR/'processed'
    path.mkdir(parents=True, exist_ok=True)
    df.to_csv(path/name, index=False)