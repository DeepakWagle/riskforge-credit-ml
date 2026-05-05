import joblib
import pandas as pd

bundle = joblib.load("models/bundle.joblib")
pipe = bundle["model"]


cat_cols = pipe.named_steps["prep"].transformers_[0][2]
EXPECTED_FEATURES = set(pipe.feature_names_in_)

def risk_band(prob):
    if prob<0.2:
        return "Low Risk"
    elif prob<0.5:
        return "Medium Risk"
    else:
        return "High Risk"

def decision(prob):
    if prob<0.3:
        return "Approve"
    elif prob<0.6:
        return "Review"
    else:
        return "Reject"

def predict_customer(data: dict):
    input_keys = set(data.keys())

    extra = input_keys - EXPECTED_FEATURES
    if len(extra) > 0:
        raise ValueError(f"Unexpected features: {extra}")

    df = pd.DataFrame([data])
    df = df.reindex(columns=pipe.feature_names_in_, fill_value=0)

    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)

    prob = pipe.predict_proba(df)[:, 1][0]
    return prob