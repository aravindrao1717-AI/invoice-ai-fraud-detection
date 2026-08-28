from pathlib import Path
import json
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from src.fraud_engine import FraudEngine

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "synthetic_invoices.csv").sort_values("invoice_date").reset_index(drop=True)
train = df[df["synthetic_is_anomaly"] == False].copy()  # clean synthetic history only
engine = FraudEngine(train)
# Evaluate all injected anomaly scenarios plus a representative sample of normal records.
anomalies = df[df["synthetic_is_anomaly"] == True].copy()
normals = df[df["synthetic_is_anomaly"] == False].tail(60).copy()
# Re-submit normal behaviour as new invoices so duplicate detection does not trivially fire.
normals["invoice_number"] = [f"EVAL-NORMAL-{i:03d}" for i in range(len(normals))]
normals["invoice_date"] = "2026-08-25"
normals["due_date"] = "2026-09-24"
test = pd.concat([normals, anomalies], ignore_index=True)
y_true=[]; y_pred=[]
for _, row in test.iterrows():
    r = engine.analyze(row.to_dict())
    y_true.append(bool(row["synthetic_is_anomaly"]))
    y_pred.append(r.risk_level != "VERIFIED")
metrics = {
    "precision": float(precision_score(y_true, y_pred, zero_division=0)),
    "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    "accuracy": float(accuracy_score(y_true, y_pred)),
    "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[False, True]).tolist(),
    "evaluated_normal": int(len(normals)),
    "evaluated_anomalies": int(len(anomalies)),
    "note": "Synthetic scenario evaluation; not an estimate of production fraud performance."
}
(ROOT / "models" / "evaluation.json").write_text(json.dumps(metrics, indent=2))
print(json.dumps(metrics, indent=2))
