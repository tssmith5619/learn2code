import sys
import pandas as pd
import joblib
from src.pipeline import classify_batch, log_prediction

def main():
    if len(sys.argv) != 2:
        print("Usage: python run_capstone.py <path/to/new_samples.csv>")
        sys.exit(1)

    csv_path = sys.argv[1]

    # Load new samples
    df = pd.read_csv(csv_path, index_col=0)

    # Load current production model
    model = joblib.load("models/current_model.joblib")

    # Predict
    results = classify_batch(df, model)

    # Log predictions
    for sample_id, row in results.iterrows():
        label = row["Cluster Label"]
        log_prediction(sample_id, label, "current")

    print(f"✅ Processed and logged predictions for {len(results)} samples.")

if __name__ == "__main__":
    main()
