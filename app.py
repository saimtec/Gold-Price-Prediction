import streamlit as st
import json
from pathlib import Path

from gold_model import (
    load_and_prepare_data,
    load_model,
    predict_price,
    get_feature_ranges,
    MODEL_PATH,
)


st.set_page_config(page_title="Gold Price Prediction (GLD)", layout="wide")

st.title("Gold Price Prediction App (GLD)")
st.write("Predict daily GLD price from a few market indicators.")

# --- Load data and model artifact ---

@st.cache_data
def _load_data():
    gold, X, y, _, _, _, _ = load_and_prepare_data()
    return gold, X, y


@st.cache_resource
def _load_model():
    return load_model(MODEL_PATH)


gold, X, y = _load_data()
mins, maxs = get_feature_ranges(X)
metrics_path = Path("artifacts") / "metrics.json"
metrics_dict = None

if metrics_path.exists():
    metrics_dict = json.loads(metrics_path.read_text(encoding="utf-8"))

# --- Sidebar: concise project info and full metrics ---

st.sidebar.title("Gold Price Project")
st.sidebar.write("Random Forest regression on historical GLD data.")

st.sidebar.subheader("Model")
st.sidebar.write("RandomForestRegressor (100 trees, random_state=3)")

st.sidebar.subheader("Test Metrics (saved model)")
if metrics_dict:
    st.sidebar.write(f"R2: {metrics_dict['r2']:.4f}")
    st.sidebar.write(f"MAE: {metrics_dict['mae']:.4f}")
    st.sidebar.write(f"RMSE: {metrics_dict['rmse']:.4f}")
else:
    st.sidebar.info("Run `python train_model.py` to generate metrics.")

st.sidebar.markdown("---")
st.sidebar.write(f"Rows: {len(gold)} | Features: {X.shape[1]}")

if not MODEL_PATH.exists():
    st.error(
        "No trained model found. Run `python train_model.py` first to create artifacts/gold_price_model.joblib."
    )
    st.stop()

model = _load_model()

# --- Main layout: tabs for EDA and Prediction ---

tab_eda, tab_predict = st.tabs(["EDA", "Make a Prediction"])

with tab_eda:
    st.subheader("Dataset Preview")
    st.write(gold.head())

    st.subheader("Summary Statistics")
    st.write(gold.describe())

with tab_predict:
    st.subheader("Enter Features and Predict")

    with st.form("prediction_form"):
        user_values = {}

        for feature in X.columns:
            f_min = float(mins[feature])
            f_max = float(maxs[feature])
            f_mean = float(X[feature].mean())

            user_values[feature] = st.number_input(
                f"{feature}",
                min_value=f_min,
                max_value=f_max,
                value=f_mean,
            )

        submit_prediction = st.form_submit_button("Predict GLD Price")

    if submit_prediction:
        prediction = predict_price(model, user_values)

        st.success(f"Predicted GLD Price: {prediction:.2f}")

        st.caption(
            "Model trained on historical data with a random train/test split. "
            "These predictions are for learning and experimentation, not trading."
        )
