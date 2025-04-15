import sys
import pandas as pd
import joblib
from ml_training.pipeline import classify_batch

def main():
    if len(sys.argv) < 2:
        print("Usage: python cli_classify.py path/to/your_data.csv")
        sys.exit(1)

    input_path = sys.argv[1]

    # Load new sample data
    try:
        df = pd.read_csv(input_path, index_col=0)
    except Exception as e:
        print(f"Failed to load input file: {e}")
        sys.exit(1)

    # Load trained model
    try:
        model = joblib.load("trained_microbiome_model.joblib")
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)

    # Run classification
    try:
        predictions = classify_batch(df, model)
        output_path = "predictions.csv"
        predictions.to_csv(output_path)
        print(f"\n✅ Classification complete. Results saved to {output_path}")
        print(predictions)
    except Exception as e:
        print(f"Classification failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
