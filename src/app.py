import streamlit as st
import os
import requests

try:
    from .gold_model import (
        load_and_prepare_data,
        get_feature_ranges,
    )
except ImportError:
    from gold_model import (
        load_and_prepare_data,
        get_feature_ranges,
    )


@st.cache_data
def _load_data():
    gold, X, y, _, _, _, _ = load_and_prepare_data()
    return gold, X, y


def _backend_url() -> str:
    env_url = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
    try:
        return st.secrets.get("BACKEND_URL", env_url)
    except Exception:
        return env_url


def _predict_from_api(user_values: dict) -> float:
    payload = {
        "spx": float(user_values["SPX"]),
        "uso": float(user_values["USO"]),
        "slv": float(user_values["SLV"]),
        "eurUsd": float(user_values["EUR/USD"]),
    }

    response = requests.post(
        f"{_backend_url().rstrip('/')}/predict",
        json=payload,
        timeout=20,
    )
    response.raise_for_status()
    data = response.json()
    return float(data["predicted_gld_price"])


def main():
    st.set_page_config(page_title="Gold Price Prediction (GLD)", layout="wide")

    st.title("Gold Price Prediction App (GLD)")
    st.write("Predict daily GLD price from a few market indicators.")

    gold, X, y = _load_data()
    mins, maxs = get_feature_ranges(X)

    st.sidebar.title("Gold Price Project")
    st.sidebar.write("Frontend app that sends prediction requests to FastAPI backend.")
    st.sidebar.subheader("Backend API")
    st.sidebar.write(_backend_url())
    st.sidebar.caption("Set BACKEND_URL in Streamlit Cloud secrets to your Render URL.")

    st.sidebar.markdown("---")
    st.sidebar.write(f"Rows: {len(gold)} | Features: {X.shape[1]}")

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
            try:
                prediction = _predict_from_api(user_values)
                st.success(f"Predicted GLD Price: {prediction:.2f}")
                st.caption(
                    "Prediction is served by the backend API."
                )
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")


if __name__ == "__main__":
    main()
