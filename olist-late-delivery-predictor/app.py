import json
import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="Olist Late Delivery Predictor",
    page_icon="📦",
    layout="wide"
)


BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

@st.cache_resource
def load_artifacts():
    model = joblib.load(ARTIFACTS_DIR / "late_delivery_model.joblib")

    with open(ARTIFACTS_DIR / "model_metadata.json", "r") as f:
        metadata = json.load(f)

    with open(ARTIFACTS_DIR / "feature_schema.json", "r") as f:
        schema = json.load(f)

    return model, metadata, schema


model, metadata, schema = load_artifacts()

st.title("Olist Late Delivery Predictor")
st.caption("Machine Learning Based Delivery Risk Assessment")
st.info(
    "Risk scores are model estimates, not calibrated delivery probabilities. "
    "Use them to prioritize review rather than as guarantees."
)

st.divider()

st.subheader("Model Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Model", "Random Forest")
col2.metric("Features", len(schema["features"]))
col3.metric("ROC-AUC", "0.7625")

st.divider()

st.subheader("Order Information")

with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    with col1:
        item_count = st.number_input(
            "Number of Items",
            min_value=1,
            max_value=100,
            value=1
        )

        total_price = st.number_input(
            "Total Price",
            min_value=0.0,
            value=137.04
        )

        total_freight = st.number_input(
            "Total Freight",
            min_value=0.0,
            value=22.78
        )

        mean_item_price = st.number_input(
            "Mean Item Price",
            min_value=0.0,
            value=120.0
        )

        mean_item_freight = st.number_input(
            "Mean Item Freight",
            min_value=0.0,
            value=19.0
        )

        customer_state = st.selectbox(
            "Customer State",
            schema["categorical_options"]["customer_state"]
        )

        main_category = st.selectbox(
            "Product Category",
            schema["categorical_options"]["main_category"]
        )

    with col2:
        mean_distance_km = st.number_input(
            "Mean Distance (KM)",
            min_value=0.0,
            value=600.0
        )

        max_distance_km = st.number_input(
            "Maximum Distance (KM)",
            min_value=0.0,
            value=600.0
        )

        distance_coverage = st.number_input(
            "Distance Coverage",
            min_value=0.0,
            max_value=1.0,
            value=1.0
        )

        all_distances_missing = st.selectbox(
            "All Distances Missing",
            [0, 1]
        )

        seller_count_items = st.number_input(
            "Seller Count per Items",
            min_value=1,
            value=1
        )

        seller_city_count = st.number_input(
            "Seller City Count",
            min_value=1,
            value=1
        )

        seller_state_count = st.number_input(
            "Seller State Count",
            min_value=1,
            value=1
        )

        category_count = st.number_input(
            "Category Count",
            min_value=1,
            value=1
        )

    col3, col4 = st.columns(2)

    with col3:
        purchase_year = st.number_input(
            "Purchase Year",
            min_value=2016,
            max_value=2018,
            value=2018
        )

        purchase_month = st.number_input(
            "Purchase Month",
            min_value=1,
            max_value=12,
            value=6
        )

        purchase_dayofweek = st.number_input(
            "Purchase Day of Week",
            min_value=0,
            max_value=6,
            value=2
        )

    with col4:
        purchase_hour = st.number_input(
            "Purchase Hour",
            min_value=0,
            max_value=23,
            value=14
        )

        promised_delivery_days = st.number_input(
            "Promised Delivery Days",
            min_value=0.1,
            value=23.7
        )

    threshold_mode = st.radio(
        "Prediction Strategy",
        ["Balanced", "High Recall"],
        horizontal=True
    )

    submitted = st.form_submit_button(
        "Predict Delivery Risk",
        type="primary",
        use_container_width=True
    )


if submitted:

    input_data = {
        "item_count": item_count,
        "total_price": total_price,
        "total_freight": total_freight,
        "mean_item_price": mean_item_price,
        "mean_item_freight": mean_item_freight,
        "mean_distance_km": mean_distance_km,
        "max_distance_km": max_distance_km,
        "distance_coverage": distance_coverage,
        "all_distances_missing": all_distances_missing,
        "seller_count_items": seller_count_items,
        "seller_city_count": seller_city_count,
        "seller_state_count": seller_state_count,
        "category_count": category_count,
        "purchase_year": purchase_year,
        "purchase_month": purchase_month,
        "purchase_dayofweek": purchase_dayofweek,
        "purchase_hour": purchase_hour,
        "promised_delivery_days": promised_delivery_days,
        "customer_state": customer_state,
        "main_category": main_category
    }

    input_df = pd.DataFrame(
        [input_data],
        columns=schema["features"]
    )

    threshold = metadata["thresholds"][
        "balanced" if threshold_mode == "Balanced"
        else "high_recall"
    ]

    probability = model.predict_proba(input_df)[0, 1]

    prediction = int(probability >= threshold)

    st.divider()
    st.subheader("Prediction Results")

    col1, col2 = st.columns(2)

    col1.metric(
        "Predicted Late Risk Score",
        f"{probability:.2%}"
    )

    col2.metric(
        "Selected Threshold",
        f"{threshold:.4f}"
    )

    if prediction == 1:
        st.error("The order is flagged as potentially late.")
    else:
        st.success("The order is not flagged as potentially late.")

    st.caption(
        "This is a model-based risk estimate, not a guarantee "
        "of actual delivery performance."
    )

st.divider()

st.header("Batch Prediction")
st.write("Upload a CSV file containing engineered order features.")

uploaded_file = st.file_uploader(
    "Upload Orders CSV",
    type=["csv"]
)
template_data = {
    "item_count": [1],
    "total_price": [137.04],
    "total_freight": [22.78],
    "mean_item_price": [137.04],
    "mean_item_freight": [22.78],
    "mean_distance_km": [600.0],
    "max_distance_km": [600.0],
    "distance_coverage": [1.0],
    "all_distances_missing": [0],
    "seller_count_items": [1],
    "seller_city_count": [1],
    "seller_state_count": [1],
    "category_count": [1],
    "purchase_year": [2018],
    "purchase_month": [6],
    "purchase_dayofweek": [2],
    "purchase_hour": [14],
    "promised_delivery_days": [23.7],
    "customer_state": ["SP"],
    "main_category": ["beleza_saude"]
}

template_df = pd.DataFrame(
    template_data,
    columns=schema["features"]
)

st.download_button(
    "Download CSV Template",
    data=template_df.to_csv(index=False).encode("utf-8-sig"),
    file_name="olist_prediction_template.csv",
    mime="text/csv"
)

if uploaded_file is not None:

    try:
        batch_df = pd.read_csv(uploaded_file)

        required_features = schema["features"]

        missing_columns = [
            col for col in required_features
            if col not in batch_df.columns
        ]

        if missing_columns:
            st.error(
                f"Missing columns: {missing_columns}"
            )

        elif batch_df.empty:
            st.warning("The uploaded file is empty.")

        else:
            batch_features = batch_df[required_features].copy()

            numeric_features = schema["numerical_features"]

            for col in numeric_features:
                batch_features[col] = pd.to_numeric(
                    batch_features[col],
                    errors="coerce"
                )

            if batch_features[numeric_features].isna().any().any():
                st.error(
                    "Some numerical values are invalid or missing."
                )

            else:
                batch_threshold = st.selectbox(
                    "Batch Prediction Strategy",
                    ["Balanced", "High Recall"],
                    key="batch_strategy"
                )

                threshold = metadata["thresholds"][
                    "balanced"
                    if batch_threshold == "Balanced"
                    else "high_recall"
                ]

                if st.button("Run Batch Prediction"):

                    probabilities = model.predict_proba(
                        batch_features
                    )[:, 1]

                    predictions = (
                        probabilities >= threshold
                    ).astype(int)

                    results = batch_df.copy()

                    results["risk_score"] = probabilities
                    results["prediction"] = predictions

                    results["risk_status"] = [
                        "Flagged as Potentially Late"
                        if p == 1 else "Not Flagged"
                        for p in predictions
                    ]

                    col1, col2, col3 = st.columns(3)

                    col1.metric(
                        "Total Orders",
                        len(results)
                    )

                    col2.metric(
                        "Flagged Orders",
                        int(predictions.sum())
                    )

                    col3.metric(
                        "Flagged Rate",
                        f"{predictions.mean():.2%}"
                    )

                    st.subheader("Prediction Results")

                    st.dataframe(
                        results,
                        use_container_width=True
                    )

                    st.subheader("Batch Analytics")

                    analytics_col1, analytics_col2 = st.columns(2)

                    with analytics_col1:
                        st.write("Orders by Prediction Status")
                        status_counts = (
                            results["risk_status"]
                            .value_counts()
                            .rename_axis("Prediction Status")
                            .to_frame("Orders")
                        )
                        st.bar_chart(status_counts)

                    with analytics_col2:
                        st.write("Risk Score Distribution")
                        risk_bins = pd.cut(
                            results["risk_score"],
                            bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                            include_lowest=True
                        )
                        risk_distribution = (
                            risk_bins.value_counts(sort=False)
                            .rename_axis("Risk Score Range")
                            .to_frame("Orders")
                        )
                        st.bar_chart(risk_distribution)

                    if "customer_state" in results.columns:
                        st.write("Flagged Rate by Customer State")
                        state_summary = (
                            results.groupby("customer_state", dropna=False)
                            .agg(
                                total_orders=("prediction", "size"),
                                flagged_orders=("prediction", "sum")
                            )
                        )
                        state_summary["flagged_rate"] = (
                            state_summary["flagged_orders"]
                            / state_summary["total_orders"]
                        )
                        st.bar_chart(
                            state_summary[["flagged_rate"]].sort_values(
                                "flagged_rate", ascending=False
                            )
                        )
                        st.dataframe(
                            state_summary.sort_values(
                                "flagged_rate", ascending=False
                            ).style.format({"flagged_rate": "{:.2%}"}),
                            use_container_width=True
                        )

                    csv_data = results.to_csv(
                        index=False
                    ).encode("utf-8-sig")

                    st.download_button(
                        "Download Predictions CSV",
                        data=csv_data,
                        file_name="late_delivery_predictions.csv",
                        mime="text/csv"
                    )

    except Exception as e:
        st.error(f"Error processing file: {e}")
