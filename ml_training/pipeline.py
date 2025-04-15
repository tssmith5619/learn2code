import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

def run_microbiome_pipeline(df_count):
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
    plt.show()

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
