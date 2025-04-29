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
import csv
import joblib

def run_microbiome_pipeline(df_count, model_version="v1"):
    """
    Runs the full clustering and classification pipeline on a raw count matrix.
    """
    df_count = df_count.select_dtypes(include="number")
    relative_abundance = df_count.div(df_count.sum(axis=0), axis=1) * 100
    relative_abundance = relative_abundance.round(2)
    X_features = relative_abundance.T

    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(X_features)
    X_features["Cluster"] = kmeans.labels_
    X_features["Cluster Label"] = X_features["Cluster"].map({0: "Balanced-like", 1: "Dysbiotic-like"})

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

    import joblib
    model_path = os.path.join("models", f"model_{model_version}.joblib")
    joblib.dump(clf, model_path)
    print(f"✅ Model saved as {model_path}")

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
    new_sample_df = new_sample_df.select_dtypes(include="number")
    relative_abundance = new_sample_df.div(new_sample_df.sum(axis=0), axis=1) * 100
    relative_abundance = relative_abundance.round(2)
    X_features = relative_abundance.T

    predicted_cluster = trained_model.predict(X_features)[0]
    label_map = {0: "Balanced-like", 1: "Dysbiotic-like"}
    label = label_map.get(predicted_cluster, "Unknown")

    return label

def classify_batch(df_new_samples, trained_model):
    df_new_samples = df_new_samples.select_dtypes(include="number")
    relative_abundance = df_new_samples.div(df_new_samples.sum(axis=0), axis=1) * 100
    relative_abundance = relative_abundance.round(2)
    X_features = relative_abundance.T

    print("DEBUG—model expects:", trained_model.feature_names_in_)
    print("DEBUG—X_features columns:", list(X_features.columns[:10]))


    cluster_preds = trained_model.predict(X_features)
    label_map = {0: "Balanced-like", 1: "Dysbiotic-like"}
    labels = [label_map.get(c, "Unknown") for c in cluster_preds]

    results_df = pd.DataFrame({
        "Cluster": cluster_preds,
        "Cluster Label": labels
    }, index=X_features.index)

    return results_df

def check_if_retraining_needed(current_version: str, threshold: int = 50):
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

# ─── fixed log directory ─────────────────────────────────────────────────
LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)             # create logs/ if it doesn't exist
# ─────────────────────────────────────────────────────────────────────────

def log_prediction(
    sample_id: str,
    prediction: str,
    model_version: str,
    shap_summary: str | None = None,
    log_filename: str = "classified_sample_log.csv",
):
    """
    Append one prediction to logs/<log_filename> (CSV, quoted).
    """
    log_path = os.path.join(LOG_DIR, log_filename)

    entry = pd.DataFrame([{
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sample_id": sample_id,
        "prediction": prediction,
        "model_version": model_version,
        "shap_summary": shap_summary or ""
    }])

    entry.to_csv(
        log_path,
        mode="a" if os.path.exists(log_path) else "w",
        header=not os.path.exists(log_path),
        index=False,
        quoting=csv.QUOTE_ALL,
    )

    print(f"📋 Logged prediction for {sample_id} → {log_path}")
    
def get_shap_summary_for_sample(sample_df: pd.DataFrame, trained_model) -> str:
    sample_df = sample_df.select_dtypes(include="number")
    relative_abundance = sample_df.div(sample_df.sum(axis=0), axis=1) * 100
    relative_abundance = relative_abundance.round(2)
    X_sample = relative_abundance.T

    X_sample = X_sample[trained_model.feature_names_in_]

    explainer = shap.TreeExplainer(trained_model)
    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list) and len(shap_values) > 1:
        shap_vals = shap_values[1][0]
    else:
        shap_vals = shap_values[0]

    shap_vals_flat = shap_vals.flatten()

    summary = ", ".join([
        f"{feature}:{float(shap_vals_flat[i]):+.2f}" for i, feature in enumerate(X_sample.columns)
    ])

    return summary



def compare_models(model_path_a, model_path_b, data_path):
    """
    Compare two trained models on the same input data.

    Parameters:
        model_path_a (str): Path to first model (e.g., model_v2.joblib)
        model_path_b (str): Path to second model (e.g., model_v3.joblib)
        data_path (str): Path to CSV containing microbial counts

    Returns:
        str: model name of the better performing model
    """
    df = pd.read_csv(data_path, index_col=0)
    df = df.select_dtypes(include="number")

    relative_abundance = df.div(df.sum(axis=0), axis=1) * 100
    relative_abundance = relative_abundance.round(2)
    X = relative_abundance.T

    model_a = joblib.load(model_path_a)
    model_b = joblib.load(model_path_b)

    preds_a = model_a.predict(X)
    preds_b = model_b.predict(X)

    accuracy_a = (preds_a == preds_b).mean()
    accuracy_b = (preds_b == preds_a).mean()

    print(f"{model_path_a} vs {model_path_b}")
    print(f"✅ Agreement with each other: {accuracy_a:.2f}")

    return model_path_a if accuracy_a >= accuracy_b else model_path_b


def promote_model(version: str):
    """
    Promotes the specified model version to 'current_model.joblib' and 'current_model_metadata.json'.

    Parameters:
        version (str): The model version to promote (e.g., 'v3')
    """
    src_model = os.path.join("models", f"model_{version}.joblib")
    src_meta = os.path.join("models", f"model_{version}_metadata.json")

    dest_model = os.path.join("models", "current_model.joblib")
    dest_meta = os.path.join("models", "current_model_metadata.json")

    if not os.path.exists(src_model) or not os.path.exists(src_meta):
        print(f"❌ Model version {version} not found.")
        return

    # Copy both model and metadata
    import shutil
    shutil.copyfile(src_model, dest_model)
    shutil.copyfile(src_meta, dest_meta)

    print(f"🚀 Promoted model version {version} to current_model.joblib")
