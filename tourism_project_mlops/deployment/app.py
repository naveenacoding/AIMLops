from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model.joblib"
SCHEMA_PATH = ROOT / "feature_schema.json"

st.set_page_config(
    page_title="Tourism Prediction",
    page_icon="✈️",
)

st.title("Tourism Package Purchase Prediction")
st.write(
    "Enter the customer details below to predict whether the customer "
    "is likely to purchase the tourism package."
)

if not MODEL_PATH.exists() or not SCHEMA_PATH.exists():
    st.error(
        "The trained model has not been generated yet. "
        "Run the GitHub Actions pipeline first."
    )
    st.stop()

model = joblib.load(MODEL_PATH)
schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

values = {}

with st.form("prediction_form"):
    for column in schema["columns"]:
        default = schema["defaults"][column]

        if column in schema["numeric"]:
            values[column] = st.number_input(
                column,
                value=float(default),
            )
        else:
            options = schema["categories"].get(column, [])
            if not options:
                options = [str(default)]

            default_text = str(default)
            index = options.index(default_text) if default_text in options else 0

            values[column] = st.selectbox(
                column,
                options,
                index=index,
            )

    submitted = st.form_submit_button("Predict Purchase")

if submitted:
    input_df = pd.DataFrame([values], columns=schema["columns"])

    prediction = int(model.predict(input_df)[0])
    probability = float(model.predict_proba(input_df)[0][1])

    if prediction == 1:
        st.success("Customer is likely to purchase the package.")
    else:
        st.warning("Customer is unlikely to purchase the package.")

    st.metric("Purchase Probability", f"{probability:.2%}")
