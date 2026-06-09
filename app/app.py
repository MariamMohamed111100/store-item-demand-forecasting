import streamlit as st
import joblib
import pandas as pd
import os
import plotly.express as px
import numpy as np
from huggingface_hub import hf_hub_download

# ==========================================
# PATHS
# ==========================================

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

ARTIFACTS_DIR = os.path.join(
    BASE_DIR,
    "artifacts"
)

os.makedirs(
    ARTIFACTS_DIR,
    exist_ok=True
)

OUTPUT_DIR = os.path.join(BASE_DIR, "output")

importance_df = pd.read_csv(
    os.path.join(
        OUTPUT_DIR,
        "feature_importance.csv"
    )
)

train_df = pd.read_csv(
    os.path.join(BASE_DIR, "Data", "train.csv"),
    parse_dates=["date"]
)

submission_df = pd.read_csv(
    os.path.join(BASE_DIR, "output", "submission.csv")
)

# ==========================================
# DOWNLOAD MODELS FROM HUGGING FACE
# ==========================================

REPO_ID = "Mariam1095/store-item-demand-models"

MODEL_FILES = [
    "ensemble_config.pkl",
    "features.pkl",
    "project_metadata.pkl",
    "log_models.pkl",
    "poisson_models.pkl",
    "tweedie_models.pkl"
]

for file_name in MODEL_FILES:

    local_file = os.path.join(
        ARTIFACTS_DIR,
        file_name
    )

    if not os.path.exists(local_file):

        with st.spinner(
            f"Downloading {file_name}..."
        ):

            hf_hub_download(
                repo_id=REPO_ID,
                filename=file_name,
                local_dir=ARTIFACTS_DIR,
                local_dir_use_symlinks=False
            )

# ==========================================
# LOAD ARTIFACTS
# ==========================================

metadata = joblib.load(
    os.path.join(
        ARTIFACTS_DIR,
        "project_metadata.pkl"
    )
)

ensemble_config = joblib.load(
    os.path.join(
        ARTIFACTS_DIR,
        "ensemble_config.pkl"
    )
)

log_models = joblib.load(
    os.path.join(
        ARTIFACTS_DIR,
        "log_models.pkl"
    )
)

poisson_models = joblib.load(
    os.path.join(
        ARTIFACTS_DIR,
        "poisson_models.pkl"
    )
)

tweedie_models = joblib.load(
    os.path.join(
        ARTIFACTS_DIR,
        "tweedie_models.pkl"
    )
)

features = joblib.load(
    os.path.join(
        ARTIFACTS_DIR,
        "features.pkl"
    )
)

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Store Item Demand Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>

    .main {
        padding-top: 1rem;
    }

    div[data-testid="metric-container"] {
        border-radius: 12px;
        padding: 15px;
        border: 1px solid rgba(200,200,200,0.2);
    }

    .stAlert {
        border-radius: 12px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("📈 Navigation")

page = st.sidebar.radio(
    "Go To",
    [
        "🏠 Overview",
        "📊 Forecast Explorer",
        "📈 Feature Importance",
        "⚙️ Model Information",
        "ℹ️ About"
    ]
)
st.sidebar.divider()

st.sidebar.markdown("### 📊 Project Metrics")

st.sidebar.metric(
    "CV SMAPE",
    f"{metadata['cv_smape']:.2f}%"
)

st.sidebar.metric(
    "Models",
    len(log_models)
    +
    len(poisson_models)
    +
    len(tweedie_models)
)

# ==========================================
# OVERVIEW PAGE
# ==========================================

if page == "🏠 Overview":

    st.title("📈 Store Item Demand Forecasting")

    st.success(
    """
    🏆 Advanced Retail Demand Forecasting System

    Forecasting daily product demand across 10 stores and 50 products
    using an ensemble of 48 LightGBM models and 59 engineered features.
    """
    )

    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "CV SMAPE",
            f"{metadata['cv_smape']:.2f}%"
        )

    with col2:
        st.metric(
            "Features",
            metadata["n_features"]
        )

    with col3:
        st.metric(
            "Validation Folds",
            metadata["n_folds"]
        )

    with col4:
        st.metric(
            "Random Seeds",
            metadata["n_seeds"]
        )
    st.divider()

    st.subheader("📦 Dataset Overview")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Train Records",
            f"{len(train_df):,}"
        )

    with c2:
        st.metric(
            "Stores",
            train_df["store"].nunique()
        )

    with c3:
        st.metric(
            "Items",
            train_df["item"].nunique()
        )

    st.divider()

    st.subheader("⚙️ Ensemble Configuration")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.info(
            f"Regression\n\n{ensemble_config['log_weight']:.2f}"
        )

    with c2:
        st.success(
            f"Poisson\n\n{ensemble_config['poisson_weight']:.2f}"
        )

    with c3:
        st.warning(
            f"Tweedie\n\n{ensemble_config['tweedie_weight']:.2f}"
        )

    st.divider()

    st.subheader("📋 Model Summary")

    st.markdown("""
    ### Models Used

    - LightGBM Regression
    - LightGBM Poisson
    - LightGBM Tweedie

    ### Validation Strategy

    - Time-Based Validation
    - 4 Validation Folds
    - 4 Random Seeds

    ### Feature Engineering

    - Calendar Features
    - Cyclic Features
    - Lag Features
    - Rolling Statistics
    - EWMA Features
    - Trend Features
    - Aggregation Features
    """) 
    st.divider()



    st.subheader("📈 Forecast Statistics")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Mean Forecast",
            f"{submission_df['sales'].mean():.2f}"
        )

    with c2:
        st.metric(
            "Std Forecast",
            f"{submission_df['sales'].std():.2f}"
        )

    with c3:
        st.metric(
            "Min Forecast",
            f"{submission_df['sales'].min():.2f}"
        )

    with c4:
        st.metric(
            "Max Forecast",
            f"{submission_df['sales'].max():.2f}"
        )

    st.divider()

    st.subheader("⬇️ Download Forecasts")

    csv = submission_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="Download Submission File",
        data=csv,
        file_name="submission.csv",
        mime="text/csv"
    )

# ==========================================
# FORECAST PAGE
# ==========================================

elif page == "📊 Forecast Explorer":

    st.title("📊 Forecast Explorer")

    st.markdown("""
    Explore historical demand patterns and generate ensemble forecasts
    for any Store-Item combination.
    """)

    # ==========================================
    # STORE / ITEM SELECTORS
    # ==========================================

    col1, col2 = st.columns(2)

    with col1:
        selected_store = st.selectbox(
            "🏪 Select Store",
            sorted(train_df["store"].unique())
        )

    with col2:
        selected_item = st.selectbox(
            "📦 Select Item",
            sorted(train_df["item"].unique())
        )

    filtered = train_df[
        (train_df["store"] == selected_store)
        &
        (train_df["item"] == selected_item)
    ].copy()

    filtered = filtered.sort_values("date")

    st.divider()

    # ==========================================
    # KPI CARDS
    # ==========================================

    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.metric(
            "Average Sales",
            f"{filtered['sales'].mean():.2f}"
        )

    with k2:
        st.metric(
            "Max Sales",
            int(filtered["sales"].max())
        )

    with k3:
        st.metric(
            "Min Sales",
            int(filtered["sales"].min())
        )

    with k4:
        st.metric(
            "Records",
            len(filtered)
        )

    st.divider()

    # ==========================================
    # HISTORICAL SALES
    # ==========================================

    st.subheader("📈 Historical Demand")

    fig_hist = px.line(
        filtered,
        x="date",
        y="sales",
        title=f"Store {selected_store} - Item {selected_item}"
    )

    fig_hist.update_layout(
        xaxis_title="Date",
        yaxis_title="Sales"
    )

    st.plotly_chart(
        fig_hist,
        width='stretch'
    )

    # ==========================================
    # MONTHLY PATTERN
    # ==========================================

    st.subheader("📅 Monthly Seasonality")

    monthly = (
        filtered
        .groupby(
            filtered["date"].dt.month
        )["sales"]
        .mean()
        .reset_index()
    )

    monthly.columns = [
        "Month",
        "Average Sales"
    ]

    fig_month = px.bar(
        monthly,
        x="Month",
        y="Average Sales",
        title="Average Monthly Demand"
    )

    st.plotly_chart(
        fig_month,
        width='stretch'
    )

    # ==========================================
    # WEEKDAY PATTERN
    # ==========================================

    st.subheader("🗓 Weekly Pattern")

    weekday = (
        filtered
        .groupby(
            filtered["date"].dt.day_name()
        )["sales"]
        .mean()
        .reset_index()
    )

    weekday_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ]

    weekday["day_name"] = pd.Categorical(
        weekday["date"],
        categories=weekday_order,
        ordered=True
    )

    weekday = weekday.sort_values(
        "day_name"
    )

    fig_week = px.bar(
        weekday,
        x="date",
        y="sales",
        title="Average Sales by Day"
    )

    st.plotly_chart(
        fig_week,
        width='stretch'
    )

    st.divider()

    # ==========================================
    # FORECAST SIMULATOR
    # ==========================================

    st.subheader("🚀 Ensemble Prediction Demo")

    latest_row = (
        filtered
        .sort_values("date")
        .tail(1)
    )

    st.write(
        "Latest Available Observation"
    )

    st.dataframe(
        latest_row[
            ["date","store","item","sales"]
        ],
        width='stretch'
    )

    if st.button(
        "Generate Forecast"
    ):

        X = latest_row.copy()

        X = X.reindex(
            columns=features,
            fill_value=0
        )

        # ==========================
        # LOG MODELS
        # ==========================

        log_preds = []

        for model in log_models:

            pred = np.expm1(
                model.predict(X)[0]
            )

            log_preds.append(pred)

        # ==========================
        # POISSON MODELS
        # ==========================

        poi_preds = []

        for model in poisson_models:

            pred = model.predict(X)[0]

            poi_preds.append(pred)

        # ==========================
        # TWEEDIE MODELS
        # ==========================

        twe_preds = []

        for model in tweedie_models:

            pred = model.predict(X)[0]

            twe_preds.append(pred)

        log_pred = np.mean(log_preds)

        poi_pred = np.mean(poi_preds)

        twe_pred = np.mean(twe_preds)

        final_forecast = (

            ensemble_config["log_weight"]
            * log_pred

            +

            ensemble_config["poisson_weight"]
            * poi_pred

            +

            ensemble_config["tweedie_weight"]
            * twe_pred

        )

        st.success(
            f"""
            Forecasted Demand

            {final_forecast:.2f} Units
            """
        )

        model_preds = (
            log_preds
            +
            poi_preds
            +
            twe_preds
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Forecast",
                f"{final_forecast:.2f}"
            )

        with c2:
            st.metric(
                "Min Prediction",
                f"{np.min(model_preds):.2f}"
            )

        with c3:
            st.metric(
                "Max Prediction",
                f"{np.max(model_preds):.2f}"
            )

        st.info(
            f"""
            Ensemble Forecast Generated Using

            • {len(log_models)} Log Models

            • {len(poisson_models)} Poisson Models

            • {len(tweedie_models)} Tweedie Models
            """
        )


# ==========================================
# FEATURE IMPORTANCE
# ==========================================

elif page == "📈 Feature Importance":

    st.title("📈 Feature Importance")

    st.markdown("""
    Feature importance extracted from the trained LightGBM model.
    """)

    top_n = st.slider(
        "Select Number of Features",
        min_value=10,
        max_value=50,
        value=20
    )

    top_features = (
        importance_df
        .sort_values(
            "Importance",
            ascending=False
        )
        .head(top_n)
    )

    fig = px.bar(
        top_features.sort_values("Importance"),
        x="Importance",
        y="Feature",
        orientation="h",
        title=f"Top {top_n} Features"
    )

    st.plotly_chart(
        fig,
        width='stretch'
    )

    st.dataframe(
        top_features,
        width='stretch'
    )

# ==========================================
# MODEL INFO
# ==========================================

elif page == "⚙️ Model Information":

    st.title("⚙️ Model Information")

    st.markdown("""
    Technical details about the forecasting pipeline.
    """)

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("📊 Model Metrics")

        st.metric(
            "CV SMAPE",
            f"{metadata['cv_smape']:.2f}%"
        )

        st.metric(
            "Number of Features",
            metadata["n_features"]
        )

        st.metric(
            "Validation Folds",
            metadata["n_folds"]
        )

        st.metric(
            "Random Seeds",
            metadata["n_seeds"]
        )

    with col2:

        st.subheader("⚖️ Ensemble Weights")

        st.progress(
            float(
                ensemble_config["log_weight"]
            )
        )

        st.write(
            f"Log Regression: {ensemble_config['log_weight']:.2f}"
        )

        st.progress(
            float(
                ensemble_config["poisson_weight"]
            )
        )

        st.write(
            f"Poisson: {ensemble_config['poisson_weight']:.2f}"
        )

        st.progress(
            float(
                ensemble_config["tweedie_weight"]
            )
        )

        st.write(
            f"Tweedie: {ensemble_config['tweedie_weight']:.2f}"
        )

    st.divider()

    st.subheader("🧠 Training Strategy")

    st.markdown("""
    ### Forecasting Models

    - LightGBM Regression
    - LightGBM Poisson
    - LightGBM Tweedie

    ### Validation

    - Rolling Time-Series Validation
    - Leakage-Free Splits
    - Multi-Fold Evaluation

    ### Robustness

    - Multi-Seed Training
    - Ensemble Averaging
    - Weight Optimization

    ### Feature Engineering

    - Calendar Features
    - Cyclic Features
    - Lag Features
    - Rolling Statistics
    - EWMA Features
    - Trend Features
    - Aggregation Features
    """)

# ==========================================
# ABOUT
# ==========================================

elif page == "ℹ️ About":

    st.title("ℹ️ About This Project")

    st.markdown("""
    ## 📦 Store Item Demand Forecasting

    This project focuses on forecasting future retail sales
    using advanced machine learning and time-series techniques.

    The goal is to help retailers:

    - Reduce stock shortages
    - Minimize overstock costs
    - Improve inventory planning
    - Improve supply chain efficiency
    """)

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("📊 Dataset")

        st.markdown("""
        - 913,000 Training Records
        - 45,000 Test Records
        - 10 Stores
        - 50 Products
        - Daily Sales Data
        - 5 Years of Historical Data
        """)

    with col2:

        st.subheader("🧠 Forecasting Approach")

        st.markdown("""
        - Advanced Feature Engineering
        - Time-Based Validation
        - Multi-Fold Training
        - Multi-Seed Training
        - Ensemble Learning
        - SHAP Explainability
        """)

    st.divider()

st.divider()

st.caption(
    """
    Store Item Demand Forecasting System

    Built with:
    Streamlit • LightGBM • SHAP • Plotly
    """
)