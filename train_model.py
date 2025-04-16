import sys
import pandas as pd
from ml_training.pipeline import run_microbiome_pipeline

def main():
    if len(sys.argv) != 2:
        print("Usage: python train_model.py <model_version>")
        sys.exit(1)

    model_version = sys.argv[1]

    try:
        df = pd.read_csv("microbial_counts.csv", index_col=0)
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        sys.exit(1)

    try:
        run_microbiome_pipeline(df, model_version=model_version)
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
