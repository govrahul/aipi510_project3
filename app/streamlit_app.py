"""Streamlit front end for the water potability model.

Run from the repository root:
    streamlit run app/streamlit_app.py
"""
import pandas as pd
import streamlit as st

import predict as wm

st.set_page_config(page_title="Water Potability Predictor", page_icon="💧", layout="wide")


@st.cache_resource(show_spinner="Loading model...")
def get_pipeline():
    return wm.load_pipeline()


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("💧 About")
    st.write(
        "This app uses the project's **random forest** model to estimate whether a "
        "water sample is potable from five measurements you can test at home."
    )
    st.markdown(
        "- Test accuracy ≈ **0.65**, F1 ≈ **0.52**\n"
        "- Trained on a **synthetic** Kaggle dataset\n"
        "- Missing measurements are filled with the training-set median"
    )
    st.warning(
        "For learning purposes only. Do not use this to decide whether water is safe "
        "to drink; use a certified lab test.",
        icon="⚠️",
    )

st.title("💧 Water Potability Predictor")
st.caption("AIPI 510 Project 3 · Rahul Govil and Daniel Yaari")

try:
    pipeline = get_pipeline()
except FileNotFoundError as missing:
    st.error(f"Model file not found: `{missing}`")
    st.markdown(
        "Create it first:\n\n"
        "- **Docker:** run the `app` step; it builds the model automatically.\n"
        "- **Locally:** run `python models/export_pipeline_model.py` from the repo root."
    )
    st.stop()


def show_result(prediction: int, probability: float) -> None:
    if prediction == 1:
        st.success(f"### ✅ Likely potable\nModel confidence it's potable: **{probability:.0%}**")
    else:
        st.error(f"### ❌ Likely not potable\nModel confidence it's potable: **{probability:.0%}**")
    st.progress(probability, text=f"P(potable) = {probability:.2f}")


single_tab, batch_tab, about_tab = st.tabs(["🔬 Single sample", "📄 Batch (CSV)", "ℹ️ How it works"])

# ---------------------------------------------------------------- single sample
with single_tab:
    st.subheader("Enter your measurements")
    st.write("Tick **Not measured** for anything you don't have; the model will fill it in.")

    values: dict[str, float | None] = {}
    columns = st.columns(len(wm.FEATURES))
    for column, feature in zip(columns, wm.FEATURES):
        with column:
            unit = f" ({feature.unit})" if feature.unit else ""
            not_measured = st.checkbox("Not measured", key=f"skip_{feature.name}")
            value = st.number_input(
                f"{feature.label}{unit}",
                min_value=feature.min_value,
                max_value=feature.max_value,
                value=feature.default,
                step=feature.step,
                disabled=not_measured,
                help=feature.guideline,
                key=f"value_{feature.name}",
            )
            values[feature.name] = None if not_measured else float(value)
            if wm.outside_guideline(feature, values[feature.name]):
                st.caption(f"⚠️ Outside guideline. {feature.guideline}")
            else:
                st.caption(feature.guideline)

    if all(v is None for v in values.values()):
        st.info("Enter at least one measurement to get a prediction.")
    elif st.button("Predict", type="primary"):
        prediction, probability = wm.predict_one(pipeline, values)
        show_result(prediction, probability)
        skipped = [f.label for f in wm.FEATURES if values[f.name] is None]
        if skipped:
            st.caption(f"Filled with training medians: {', '.join(skipped)}")

# ---------------------------------------------------------------- batch
with batch_tab:
    st.subheader("Predict many samples at once")
    st.write(
        "Upload a CSV with the columns `ph`, `Hardness`, `Chloramines`, `Sulfate`, "
        "`Conductivity` (any order, case doesn't matter). Blank cells are filled with "
        "the training median; other columns are kept as they are."
    )
    st.download_button(
        "Download a template CSV",
        data=wm.template_csv(),
        file_name="water_samples_template.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Upload CSV", type="csv")
    if uploaded is not None:
        try:
            samples = pd.read_csv(uploaded)
        except Exception as error:  # bad file
            st.error(f"Couldn't read that file: {error}")
            st.stop()

        samples, missing_columns = wm.normalize_columns(samples)
        if samples.empty:
            st.warning("The file has no rows.")
        elif len(missing_columns) == len(wm.FEATURE_NAMES):
            st.error(f"None of the expected columns were found: {', '.join(wm.FEATURE_NAMES)}")
        else:
            if missing_columns:
                st.warning(
                    f"Missing columns {', '.join(missing_columns)}; "
                    "they'll be filled with training medians."
                )
            results = wm.predict(pipeline, samples)
            output = samples.copy()
            output["prediction"] = results["prediction"].map({1: "potable", 0: "not potable"})
            output["probability_potable"] = results["probability_potable"].round(3)

            potable = int((results["prediction"] == 1).sum())
            c1, c2, c3 = st.columns(3)
            c1.metric("Samples", len(output))
            c2.metric("Likely potable", potable)
            c3.metric("Likely not potable", len(output) - potable)

            st.dataframe(output, hide_index=True)
            st.download_button(
                "Download predictions",
                data=output.to_csv(index=False).encode(),
                file_name="water_potability_predictions.csv",
                mime="text/csv",
                type="primary",
            )

# ---------------------------------------------------------------- about
with about_tab:
    st.subheader("How the prediction works")
    st.markdown(
        """
1. **Your measurements** for pH, hardness, chloramines, sulfate and conductivity go into
   the same preprocessing used in training: missing values are filled with the training
   median, then everything is standardized.
2. **A random forest** (200 trees, max depth 10, balanced class weights) predicts whether
   the sample is potable and how confident it is.
3. The model only uses features you could test at home; lab-only measurements such as
   organic carbon and trihalomethanes were left out.

**Limitations.** The training data is synthetically generated, and the model is right
about two times in three on held-out data. Treat the output as a demonstration of the
pipeline, not as a water-safety assessment.
"""
    )
    st.markdown(
        "Guidelines shown next to each input are commonly cited reference values for "
        "context only; the model does not use them."
    )
