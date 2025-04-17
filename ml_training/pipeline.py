import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import os
import json
from datetime import date, datetime
import shap



def run_microbiome_pipeline(df_count, model_version="v1"):
    """
    Runs the full clustering and classification pipeline on a raw count matrix.

    Parameters:
        df_count (DataFrame): Raw counts of microbes (rows) x samples (columns)

    Returns:
        dict: {
            "clustered_data": DataFrame with cluster assignments,
            "model": trained RandomForestClassifier,
            "feature_importances": array of feature importances
        }
    """
    # Step 1: Preprocess
    df_count = df_count.select_dtypes(include="number")  # remove any stray non-numeric columns
    relative_abundance = df_count.div(df_count.sum(axis=0), axis=1) * 100
    relative_abundance = relative_abundance.round(2)
    X_features = relative_abundance.T

    # Step 2: Clustering
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(X_features)
    X_features["Cluster"] = kmeans.labels_
    X_features["Cluster Label"] = X_features["Cluster"].map({0: "Balanced-like", 1: "Dysbiotic-like"})

    # Step 3: Train classifier
    X_train, X_test, y_train, y_test = train_test_split(
        X_features.drop(columns=["Cluster", "Cluster Label"]),
        X_features["Cluster"],
        test_size=0.3,
        random_state=42,
        stratify=X_features["Cluster"]
    )
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    import os
    import joblib

    model_path = os.path.join("models", f"model_{model_version}.joblib")
    joblib.dump(clf, model_path)
    print(f"✅ Model saved as {model_path}")


    # Step 4: Report and feature importance
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    importances = clf.feature_importances_
    features = X_train.columns

    plt.figure(figsize=(8, 4))
    plt.barh(features, importances, color="skyblue")
    plt.xlabel("Feature Importance")
    plt.title("Microbial Features Driving Cluster Prediction")
    plt.tight_layout()
    plt.savefig(f"models/model_{model_version}_feature_importance.png")
    plt.close()


    # Step 5: Save metadata
    metadata = {
        "version": model_version,
        "trained_on": str(date.today()),
        "samples": df_count.shape[1],
        "features": df_count.shape[0],
        "accuracy": float((y_pred == y_test).mean()),
        "notes": "Initial production model"
    }

    meta_path = os.path.join("models", f"model_{model_version}_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Metadata saved as {meta_path}")

    return {
        "clustered_data": X_features,
        "model": clf,
        "feature_importances": importances
    }


def classify_new_sample(new_sample_df, trained_model):
    """
    Classifies a single new sample using a trained classifier.

    Parameters:
        new_sample_df (DataFrame): Raw counts (rows = microbes, 1 column = new sample)
        trained_model (sklearn model): A trained classifier (e.g., RandomForest)

    Returns:
        tuple: (cluster_number, human-readable label)
    """
    # Clean input: keep only numeric features
    new_sample_df = new_sample_df.select_dtypes(include="number")

    # Convert to relative abundance
    relative_abundance = new_sample_df.div(new_sample_df.sum(axis=0), axis=1) * 100
    relative_abundance = relative_abundance.round(2)
    X_features = relative_abundance.T

    # Predict
    predicted_cluster = trained_model.predict(X_features)[0]
    label_map = {0: "Balanced-like", 1: "Dysbiotic-like"}
    label = label_map.get(predicted_cluster, "Unknown")

    return label


def classify_batch(df_new_samples, trained_model):
    """
    Classifies a batch of new samples using a trained classifier.

    Parameters:
        df_new_samples (DataFrame): Raw count matrix (rows = microbes, columns = samples)
        trained_model (sklearn model): Trained classifier from run_microbiome_pipeline()

    Returns:
        DataFrame: Predictions including cluster number and label per sample
    """
    # Step 1: Clean input
    df_new_samples = df_new_samples.select_dtypes(include="number")

    # Step 2: Convert to relative abundance
    relative_abundance = df_new_samples.div(df_new_samples.sum(axis=0), axis=1) * 100
    relative_abundance = relative_abundance.round(2)
    X_features = relative_abundance.T

    # Step 3: Predict cluster for each sample
    cluster_preds = trained_model.predict(X_features)
    label_map = {0: "Balanced-like", 1: "Dysbiotic-like"}
    labels = [label_map.get(c, "Unknown") for c in cluster_preds]

    # Step 4: Return result table
    results_df = pd.DataFrame({
        "Cluster": cluster_preds,
        "Cluster Label": labels
    }, index=X_features.index)

    return results_df


def check_if_retraining_needed(current_version: str, threshold: int = 50):
    """
    Checks whether the number of new samples classified since the last model version
    exceeds a threshold. Returns True if retraining is recommended.
    """
    log_path = "classified_sample_log.csv"
    meta_path = os.path.join("models", f"model_{current_version}_metadata.json")

    if not os.path.exists(log_path) or not os.path.exists(meta_path):
        print("❌ Missing classification log or model metadata.")
        return False

    log_df = pd.read_csv(log_path)
    with open(meta_path, "r") as f:
        metadata = json.load(f)

    trained_on_samples = metadata.get("samples", 0)
    new_samples = len(log_df) - trained_on_samples

    print(f"📊 {new_samples} new samples since model version {current_version} was trained.")
    return new_samples >= threshold

def log_prediction(sample_id: str, prediction: str, model_version: str, shap_summary: str = None, log_path: str = "classified_sample_log.csv"):
    entry = pd.DataFrame([{
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sample_id": sample_id,
        "prediction": prediction,
        "model_version": model_version,
        "shap_summary": shap_summary or ""
    }])

    if os.path.exists(log_path):
        entry.to_csv(log_path, mode="a", header=False, index=False)
    else:
        entry.to_csv(log_path, index=False)

    print(f"📋 Logged prediction for {sample_id} with SHAP summary.")




def get_shap_summary_for_sample(sample_df: pd.DataFrame, trained_model) -> str:
    """
    Returns a summary of SHAP values for a single sample, formatted as 'Feature:+0.12, Feature:-0.03'
    """
    sample_df = sample_df.select_dtypes(include="number")
    relative_abundance = sample_df.div(sample_df.sum(axis=0), axis=1) * 100
    relative_abundance = relative_abundance.round(2)
    X_sample = relative_abundance.T

    # Ensure feature order matches model
    X_sample = X_sample[trained_model.feature_names_in_]

    explainer = shap.TreeExplainer(trained_model)
    shap_values = explainer.shap_values(X_sample)

    # Determine correct output shape
    if isinstance(shap_values, list) and len(shap_values) > 1:
        shap_vals = shap_values[1][0]  # class 1, sample 0
    else:
        shap_vals = shap_values[0]     # single-output classifier

    # Flatten the SHAP values for the single sample
    shap_vals_flat = shap_vals.flatten()

    # Create summary string
    summary = ", ".join([
        f"{feature}:{float(shap_vals_flat[i]):+.2f}" for i, feature in enumerate(X_sample.columns)
    ])


    return summary



