import shap
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# Load trained model
model = joblib.load("models/model_v3.joblib")

# Load and preprocess data
df = pd.read_csv("microbial_counts.csv", index_col=0)
df = df.select_dtypes(include="number")
relative_abundance = df.div(df.sum(axis=0), axis=1) * 100
relative_abundance = relative_abundance.round(2)
X = relative_abundance.T.copy()

# Ensure features match what model was trained on
expected_features = model.feature_names_in_
X = X[expected_features]

# Set up SHAP for tree model
explainer = shap.TreeExplainer(model, model_output="probability")
shap_values = explainer.shap_values(X)

# Save global summary plot for class 1 (Balanced-like)
plt.figure()
shap.summary_plot(shap_values[1], X, show=False)
plt.savefig("shap_summary_plot_v3.png")
plt.close()

print("✅ SHAP summary plot saved as shap_summary_plot_v3.png")



