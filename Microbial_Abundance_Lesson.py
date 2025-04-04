import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

def run_microbiome_pipeline(df_count):
    # Step 1: Preprocess
    relative_abundance = df_count.div(df_count.sum(axis=0), axis=1) * 100
    relative_abundance = relative_abundance.round(2)
    X_features = relative_abundance.T

    # Step 2: Cluster
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(X_features)
    X_features["Cluster"] = kmeans.labels_
    X_features["Cluster Label"] = X_features["Cluster"].map({0: "Balanced-like", 1: "Dysbiotic-like"})

    # Step 3: Train Classifier
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

    # Step 4: Report and Plot
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

    # Step 5: Return key outputs
    return {
        "clustered_data": X_features,
        "model": clf,
        "feature_importances": importances
    }


def classify_new_sample(new_sample_df, trained_model):
    relative_abundance = new_sample_df.div(new_sample_df.sum(axis=0), axis=1) * 100
    relative_abundance = relative_abundance.round(2)
    X_features = relative_abundance.T

    predicted_cluster = trained_model.predict(X_features)[0]
    label_map = {0: "Balanced-like", 1: "Dysbiotic-like"}
    label = label_map.get(predicted_cluster, "Unknown")

    return predicted_cluster, label
