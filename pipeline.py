import numpy as np
import pandas as pd

def classify_new_sample(new_sample_df, trained_model):
    """
    Clean and predict a new microbiome sample using a trained model.

    Parameters:
    - new_sample_df: pandas DataFrame with the same columns as the training data
    - trained_model: a pre-trained scikit-learn model

    Returns:
    - Predicted label/class (e.g., 'Dysbiotic', 'Healthy', etc.)
    """

    # Example cleaning logic: fill missing values with 0
    cleaned = new_sample_df.fillna(0)

    # Convert to NumPy array if needed
    X_new = cleaned.to_numpy()

    # Run prediction
    prediction = trained_model.predict(X_new)

    return prediction
