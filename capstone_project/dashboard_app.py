import os, json, pandas as pd, streamlit as st
import altair as alt
from datetime import datetime

LOG_PATH   = os.path.join("logs", "classified_sample_log.csv")
META_PATH  = os.path.join("models", "current_model_metadata.json")

# ---------- data load ----------
if not os.path.exists(LOG_PATH):
    st.error("No log file found. Run the pipeline at least once.")
    st.stop()

log_df = pd.read_csv(LOG_PATH, quotechar='"')
log_df["timestamp"] = pd.to_datetime(log_df["timestamp"])

meta = json.load(open(META_PATH))
trained_on = meta.get("trained_on", "unknown")
initial_samples = meta.get("samples", 0)

# ---------- sidebar ----------
st.sidebar.header("Model info")
st.sidebar.write(f"**Version**: {meta['version']}")
st.sidebar.write(f"**Trained on**: {trained_on}")
st.sidebar.write(f"**Samples in training set**: {initial_samples}")

# ---------- recent predictions ----------
st.subheader("Most recent predictions")
st.dataframe(log_df.sort_values("timestamp", ascending=False).head(20))

# ---------- class balance ----------
st.subheader("Class balance")
class_chart = (
    alt.Chart(log_df)
    .mark_bar()
    .encode(
        x=alt.X("count()", title="Count"),
        y=alt.Y("prediction", sort="-x", title="Class"),
        color="prediction"
    )
)
st.altair_chart(class_chart, use_container_width=True)

# ---------- samples since last train ----------
since_train = len(log_df) - initial_samples
st.subheader("Samples since last training")
st.metric("New samples", since_train)

THRESHOLD = 50  # same as your retrain trigger
if since_train >= THRESHOLD:
    st.warning("🚨 Retrain threshold reached!")
else:
    st.info(f"{THRESHOLD - since_train} samples to go before retrain.")
