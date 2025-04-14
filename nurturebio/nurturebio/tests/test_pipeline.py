import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import pandas as pd
from nurturebio.pipeline import classify_new_sample, run_microbiome_pipeline
df_count = pd.read_csv("microbial_counts.csv", index_col=0)
results = run_microbiome_pipeline(df_count)
model = results["model"]

# Test classify_new_sample function
def test_classify_new_sample_returns_valid_label():
    new_sample = pd.DataFrame({
        "Test Sample": [100, 70, 25, 10, 5, 5, 3]}, index=["Bif", "Lactobacilli", "Strep", "Staph","Colostridium", "Enterococcus", "Bacteroides"]) 
    label = classify_new_sample(new_sample, model)
    assert label in ["Balanced-like", "Dysbiotic-like"]

def test_classify_batch_length_matches_input():
    import numpy as np
    from nurturebio.pipeline import classify_batch, run_microbiome_pipeline

    df_count = pd.read_csv("microbial_counts.csv", index_col=0)
    results = run_microbiome_pipeline(df_count)

    np.random.seed(101)
    df_new_samples = pd.DataFrame({
        f"NewSample{i}": np.random.poisson(lam=[100, 70, 50, 25, 15, 10, 5])
        for i in range(1, 6)
    }, index=["Bif", "Lactobacilli", "Strep", "Staph", "Colostridium", "Enterococcus", "Bacteroides"])

    batch_results = classify_batch(df_new_samples, results["model"])

    assert len(batch_results) == df_new_samples.shape[1]
