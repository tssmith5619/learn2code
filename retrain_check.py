import sys
import os
import json
import pandas as pd
import subprocess
from ml_training.pipeline import check_if_retraining_needed

def main():
    if len(sys.argv) != 4:
        print("Usage: python retrain_check.py <model_version> <sample_threshold> <csv_path>")
        sys.exit(1)

    model_version = sys.argv[1]
    threshold = int(sys.argv[2])
    csv_path = sys.argv[3]

    if not os.path.exists(csv_path):
        print(f"❌ Data file not found: {csv_path}")
        sys.exit(1)

    if not check_if_retraining_needed(model_version, threshold):
        print("✅ No retraining needed.")
        return

    # Auto-trigger retraining
    new_version = f"v{int(model_version.strip('v')) + 1}"
    print(f"🔁 Retraining now as version {new_version}...")

    subprocess.run(["python", "train_model.py", csv_path, new_version])

    # Log retraining event
    log_path = "retraining_log.csv"
    event = pd.DataFrame([{
        "old_version": model_version,
        "new_version": new_version,
        "triggered_on": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "threshold": threshold
    }])
    if os.path.exists(log_path):
        event.to_csv(log_path, mode="a", header=False, index=False)
    else:
        event.to_csv(log_path, index=False)

    print(f"📋 Retraining event logged to {log_path}")

if __name__ == "__main__":
    main()

