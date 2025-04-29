import sys
import pandas as pd
import joblib
from src.pipeline import classify_batch, log_prediction


def main():
    if len(sys.argv) != 2:
        print("Usage: python run_capstone.py <path/to/new_samples.csv>")
        sys.exit(1)

    csv_path = sys.argv[1]

    # -------------------------------------------------
    # 1. load production model
    # -------------------------------------------------
    model = joblib.load("models/current_model.joblib")
    expected = list(model.feature_names_in_)          # ['Bif', 'Lactobacilli', …]

    # -------------------------------------------------
    # 2. load CSV  →  coerce to rows = microbes, columns = samples
    # -------------------------------------------------
    df_raw = pd.read_csv(csv_path, index_col=0)

    # Case 1 ─ microbes are already columns
    if set(model.feature_names_in_).issubset(df_raw.columns):
        df_ready = df_raw.loc[:, model.feature_names_in_]

    # Case 2 ─ microbes are rows  (need single transpose)
    elif set(model.feature_names_in_).issubset(df_raw.index):
        df_ready = df_raw.loc[model.feature_names_in_].T

    # Anything else → hard fail
    else:
        raise ValueError(
            "CSV does not contain expected microbe names "
            "either as columns or rows."
    )
    
    assert list(df_ready.columns) == list(model.feature_names_in_), "Column order mismatch"


    # -------------------------------------------------
    # 3. classify & log
    # -------------------------------------------------
    results = classify_batch(df_ready.T, model)

    for sample_id, row in results.iterrows():
        log_prediction(sample_id, row["Cluster Label"], "current")

    print(f"✅ Processed and logged predictions for {len(results)} samples.")


if __name__ == "__main__":
    main()
