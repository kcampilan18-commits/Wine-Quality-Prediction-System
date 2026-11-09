"""Streamlit interface for the Wine Quality Prediction System."""

from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path("wine_model.pkl")
DATA_PATH = Path("wine_quality.csv")
METRICS_PATH = Path("evaluation_results.json")

st.set_page_config(
    page_title="Wine Quality Prediction System",
    page_icon="🍷",
    layout="wide",
)

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_dataset():
    if not DATA_PATH.exists():
        return None
    return pd.read_csv(DATA_PATH)

model = load_model()
data = load_dataset()

st.title("🍷 Wine Quality Prediction System")
st.caption(
    "Section B classroom prototype | Preliminary red-wine quality classification"
)

if model is None or data is None:
    st.error(
        "The model or dataset is missing. First run `python train_model.py` "
        "in this project folder, then refresh this page."
    )
    st.stop()

if METRICS_PATH.exists():
    metrics = pd.read_json(METRICS_PATH, typ="series")
    c1, c2, c3 = st.columns(3)
    c1.metric("Dataset records", f"{len(data):,}")
    c2.metric("Average original quality", f"{data['quality'].mean():.2f}")
    c3.metric(
        "Good-quality records",
        f"{int((data['quality'] >= 6).sum()):,}",
    )
else:
    st.info("Evaluation file not found. Run train_model.py to generate it.")

st.subheader("Dataset preview")
st.dataframe(data.head(10), use_container_width=True)

required_columns = list(model.feature_names_in_)

st.subheader("Predict uploaded wine samples")
st.write(
    "Upload a CSV containing the 11 physicochemical predictor columns. "
    "Do not include the original `quality` column."
)

sample_template = data[required_columns].head(5)
st.download_button(
    "Download sample input CSV",
    data=sample_template.to_csv(index=False).encode("utf-8"),
    file_name="sample_wine_input.csv",
    mime="text/csv",
)

uploaded_file = st.file_uploader("Choose a wine sample CSV", type=["csv"])

if uploaded_file is not None:
    try:
        new_data = pd.read_csv(uploaded_file)
        missing = [column for column in required_columns if column not in new_data.columns]

        if missing:
            st.error("Missing required columns: " + ", ".join(missing))
        elif new_data.empty:
            st.warning("The uploaded CSV contains no data rows.")
        else:
            # Select only the required predictors in the training order.
            prediction_input = new_data[required_columns].copy()
            predictions = model.predict(prediction_input)

            results = new_data.copy()
            results["Prediction"] = [
                "GOOD QUALITY" if value == 1 else "NEEDS IMPROVEMENT"
                for value in predictions
            ]

            st.subheader("Prediction results")
            st.dataframe(results, use_container_width=True)

            good_count = int((predictions == 1).sum())
            needs_count = int((predictions == 0).sum())
            col1, col2 = st.columns(2)
            col1.metric("Predicted GOOD QUALITY", good_count)
            col2.metric("Predicted NEEDS IMPROVEMENT", needs_count)

            st.download_button(
                "Download prediction results",
                data=results.to_csv(index=False).encode("utf-8"),
                file_name="wine_quality_predictions.csv",
                mime="text/csv",
            )
    except Exception as exc:
        st.error(f"Unable to process this CSV file: {exc}")

st.divider()
st.caption(
    "Responsible use: Predictions are preliminary screening results only. "
    "They do not replace laboratory testing, sensory evaluation, food-safety "
    "procedures, or professional quality-control decisions."
)
