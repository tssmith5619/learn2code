import pandas as pd
import numpy as np
from nurturebio.pipeline import (
    classify_new_sample,
    run_microbiome_pipeline,
    classify_batch
)

# Helper to set up and return trained model
def setup_pipeline():
    df_count = pd.read_csv("microbial_counts.csv", index_col=0)
    results = run_microbiome_pipeline(df_count)
    return results["model"]

# Test classify_new_sample returns valid label
def test_classify_new_sample_returns_valid_label():
    model = setup_pipeline()

    new_sample = pd.DataFrame({
        "Test Sample": [100, 70, 25, 10, 5, 5, 3]
    }, index=["Bif", "Lactobacilli", "Strep", "Staph", "Colostridium", "Enterococcus", "Bacteroides"])

    label = classify_new_sample(new_sample, model)
    assert label in ["Balanced-like", "Dysbiotic-like"]

# Test classify_batch returns a row for each input sample
def test_classify_batch_length_matches_input():
    model = setup_pipeline()

    df_new_samples = pd.DataFrame({
        f"NewSample{i}": np.random.poisson(lam=[100, 70, 50, 25, 15, 10, 5])
        for i in range(1, 6)
    }, index=["Bif", "Lactobacilli", "Strep", "Staph", "Colostridium", "Enterococcus", "Bacteroides"])

    batch_results = classify_batch(df_new_samples, model)
    assert len(batch_results) == df_new_samples.shape[1]

