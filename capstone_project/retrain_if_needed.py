#!/usr/bin/env python
"""
Run nightly (cron or GitHub Action):
• Checks logs/classified_sample_log.csv
• Compares # new samples against models/current_model_metadata.json
• If threshold reached → trains next model version and promotes it
"""
import os, json, subprocess, joblib
import pandas as pd
from datetime import datetime

# ---------- paths ----------
BASE_DIR   = os.path.dirname(__file__)
LOG_PATH   = os.path.join(BASE_DIR, "logs",   "classified_sample_log.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
CUR_META   = os.path.join(MODEL_DIR, "current_model_metadata.json")
CUR_MODEL  = os.path.join(MODEL_DIR, "current_model.joblib")

THRESHOLD  = 50         # can be read from metadata if you prefer

# ---------- load state ----------
if not (os.path.exists(LOG_PATH) and os.path.exists(CUR_META)):
    print("❌ Missing log or current model metadata.")
    raise SystemExit(1)

log_df   = pd.read_csv(LOG_PATH, quotechar='"')
meta     = json.load(open(CUR_META))
trained_on_samples = meta.get("samples", 0)
new_samples       = len(log_df) - trained_on_samples

print(f"🗒️  New samples since last train: {new_samples} / {THRESHOLD}")

if new_samples < THRESHOLD:
    print("✅ Threshold not reached. No retrain.")
    raise SystemExit(0)

# ---------- determine next version ----------
existing = [
    f for f in os.listdir(MODEL_DIR)
    if f.startswith("model_v") and f.endswith(".joblib")
]
versions = sorted(
    int(f.split("_v")[1].split(".")[0]) for f in existing
)
next_v   = versions[-1] + 1 if versions else 1
print(f"🚀 Training model v{next_v}")

# ---------- call existing trainer ----------
data_path = os.path.join(BASE_DIR, "data", "raw", "microbial_counts.csv")
train_cmd = [
    "python", "train_model.py", data_path, f"v{next_v}"
]
subprocess.check_call(train_cmd)

# ---------- promote new model ----------
from src.pipeline import promote_model
promote_model(f"v{next_v}")

# ---------- log audit line ----------
with open(LOG_PATH, "a") as f:
    f.write(
        f'"{datetime.now():%Y-%m-%d %H:%M:%S}","AUTO","retrained_to_v{next_v}","system",""\n'
    )

print(f"🎉 Model v{next_v} trained and promoted.")
