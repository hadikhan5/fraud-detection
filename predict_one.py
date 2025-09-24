import joblib
import pandas as pd
from pathlib import Path

# Loads the trained model
bundle = joblib.load(Path("models/best_model.joblib"))
model = bundle["model"]
threshold = float(bundle["threshold"])
features = bundle["features"]

# Loads processed data
df = pd.read_csv(Path("data/processed/creditcard_processed.csv"))
X = df[features]
y = df["Class"].astype(int)

# Pick a test row: prefer a fraud example if available, else first row
fraud_idxs = df.index[df["Class"] == 1].tolist()
idx = fraud_idxs[0] if fraud_idxs else 0

x_one = X.loc[[idx]]
true_label = int(y.loc[idx])

# Prediction
prob = float(model.predict_proba(x_one)[:,1][0])
pred = int(prob >= threshold)

print({
  "index": int(idx),
  "prob": prob,
  "pred": pred,
  "true": true_label,
  "threshold": threshold
})