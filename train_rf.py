import numpy as np
import pandas as pd
from pathlib import Path
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    classification_report, precision_recall_curve
)

RANDOM_STATE = 42

def pick_threshold_f2(y_true, y_prob, beta=2.0):
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    prec, rec = prec[:-1], rec[:-1]           # align with thr
    f2 = (1 + beta**2) * (prec * rec) / (beta**2 * prec + rec + 1e-12)
    best_idx = int(np.nanargmax(f2))
    return float(thr[best_idx])

def main():
    # load processed data
    df = pd.read_csv("data/processed/creditcard_processed.csv")
    X = df.drop(columns=["Class"])
    y = df["Class"].astype(int)
    features = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # train RF with class weighting for imbalance
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        class_weight="balanced"  # weight minority class more
    )
    rf.fit(X_train, y_train)

    # evaluate @ 0.5
    y_prob = rf.predict_proba(X_test)[:, 1]
    y_pred_default = (y_prob >= 0.5).astype(int)

    print("=== RandomForest @ 0.5 threshold ===")
    print("ROC-AUC:", roc_auc_score(y_test, y_prob))
    print("PR-AUC :", average_precision_score(y_test, y_prob))
    print(classification_report(y_test, y_pred_default, digits=4))

    # threshold tuning (F2 -> recall-heavy)
    best_thr = pick_threshold_f2(y_test, y_prob, beta=2.0)
    y_pred_tuned = (y_prob >= best_thr).astype(int)

    print("\n=== RandomForest @ tuned threshold (F2) ===")
    print("Chosen threshold:", best_thr)
    print("ROC-AUC:", roc_auc_score(y_test, y_prob))
    print("PR-AUC :", average_precision_score(y_test, y_prob))
    print(classification_report(y_test, y_pred_tuned, digits=4))

    # save as candidate bundle
    out_dir = Path("models")
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = out_dir / "candidate_rf.joblib"
    joblib.dump({"model": rf, "threshold": best_thr, "features": features}, bundle_path)
    print("\nSaved:", str(bundle_path))

if __name__ == "__main__":
    main()