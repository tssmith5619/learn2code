import streamlit as st
import pandas as pd
import joblib
from ml_training.pipeline import classify_batch

st.title("Gut Microbiome Classifier")
st.write("Upload a CSV file containing new microbiome samples to classify them.")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# Only show this if a file is uploaded
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file, index_col=0)
        model = joblib.load("trained_microbiome_model.joblib")
        results = classify_batch(df, model)

        st.success("Classification complete!")
        st.dataframe(results)

        # Download link
        csv = results.to_csv().encode("utf-8")
        st.download_button(
            label="📥 Download predictions as CSV",
            data=csv,
            file_name="classified_predictions.csv",
            mime="text/csv",
        )
    except Exception as e:
        st.error(f"Something went wrong: {e}")
