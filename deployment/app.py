
from fastapi import FastAPI, HTTPException
from src.inference import predict_customer, risk_band, decision

app=FastAPI()

@app.get("/")
def home():
    return {"message": "API is running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: dict):
    try:
        prob = predict_customer(data)
        return {
        "probability": float(prob),
        "risk_band": risk_band(prob),
        "decision": decision(prob)
    }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))