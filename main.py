from fastapi import FastAPI
from typing import Dict
from pathlib import Path
import joblib
import pandas as pd

app = FastAPI(title="Anomaly Detection API")

# Load model bundle at startup
bundle = joblib.load(Path("models/best_model.joblib"))
model = bundle["model"]
threshold = float(bundle["threshold"])
features = bundle["features"]

def to_frame(payload: Dict[str, float]) -> pd.DataFrame:
    """Turn JSON dict into a single-row DataFrame with the exact trained feature order.
    Missing features are filled with 0.0 so we can still score."""
    X = pd.DataFrame([payload])
    missing = [f for f in features if f not in X.columns]
    for f in missing:
        X[f] = 0.0
    # reorder to the training feature order
    X = X[features]
    return X, missing

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(payload: Dict[str, float]):
    X, missing = to_frame(payload)
    prob = float(model.predict_proba(X)[:, 1][0])
    pred = int(prob >= threshold)
    return {
        "prob": prob,
        "pred": pred,
        "threshold": threshold,
        "missing_features": missing,  # informative if caller didn’t supply everything
    }