import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
# Load raw counts and compute relative abundance
df = pd.read_csv("microbial_counts.csv", index_col=0)
df = df.select_dtypes(include="number")
relative_abundance = df.div(df.sum(axis=0), axis=1) * 100
relative_abundance = relative_abundance.round(2)

# Transpose to make samples = rows, microbes = columns
X = relative_abundance.T

# Create target labels based on known logic (e.g. balanced if Bif > 100)
y = (X["Bif"] > 100).astype(int)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # binary classification
])

# Compile the model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Train the model
history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=8,
    validation_data=(X_test, y_test)
)
