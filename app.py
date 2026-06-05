# -*- coding: utf-8 -*-
"""
Streamlit Web App for XGBoost Model Development

Main functions:
1. Download Excel data template
2. Upload completed Excel file
3. Select train/test ratio
4. Train or rerun XGBoost model
5. Show only R2 plot and R2 values
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import time

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


# ============================================================
# Streamlit page setting
# ============================================================
st.set_page_config(
    page_title="XGBoost Web-Based Model",
    layout="wide"
)

st.title("Web-Based XGBoost Model Development App")

st.write(
    "Download the Excel template, fill your experimental data, upload the completed file, "
    "select the train/test ratio, and run or rerun the XGBoost model."
)


# ============================================================
# Excel template
# ============================================================
template_columns = [
    "MW (g mol-1)",
    "TPSA",
    "HBDC",
    "HBAC",
    "Monoisotopic Mass",
    "Formal Charge",
    "Bandgap Energy (eV)",
    "Specific Surface Area (m2/g)",
    "Photocatalyst dosage (g/L)",
    "Pollutant dosage (mg/L)",
    "pH",
    "Light Wavelength",
    "t (min)",
    "Degradation (%)"
]


def create_excel_template():
    template_df = pd.DataFrame(columns=template_columns)

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        template_df.to_excel(writer, index=False, sheet_name="Data Template")

    output.seek(0)
    return output


st.subheader("Step 1: Download Excel Template")

template_file = create_excel_template()

st.download_button(
    label="Download Data Template",
    data=template_file,
    file_name="Data_Template.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

st.info(
    "Fill the downloaded Excel file. The last column, 'Degradation (%)', must be the model output/target."
)


# ============================================================
# Upload completed Excel file
# ============================================================
st.subheader("Step 2: Upload Completed Excel File")

uploaded_file = st.file_uploader(
    "Upload the completed Excel file",
    type=["xlsx", "xls"]
)

if uploaded_file is not None:

    try:
        df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error reading Excel file: {e}")
        st.stop()

    st.success("Excel file uploaded successfully.")

    st.write("Preview of uploaded data:")
    st.dataframe(df.head())

    # --------------------------------------------------------
    # Check columns
    # --------------------------------------------------------
    missing_columns = [col for col in template_columns if col not in df.columns]

    if len(missing_columns) > 0:
        st.error("Your uploaded Excel file does not match the template.")
        st.write("Missing columns:")
        st.write(missing_columns)
        st.stop()

    # --------------------------------------------------------
    # Remove empty rows and convert to numeric
    # --------------------------------------------------------
    df = df[template_columns].copy()
    df = df.dropna()

    for col in template_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna()

    if df.shape[0] < 10:
        st.error("The uploaded file has too few valid rows. Please add more data.")
        st.stop()

    st.write(f"Valid data rows: **{df.shape[0]}**")

    # ========================================================
    # Model settings
    # ========================================================
    st.subheader("Step 3: Select Model Settings")

    train_percent = st.slider(
        "Training data percentage (%)",
        min_value=50,
        max_value=90,
        value=80,
        step=5
    )

    test_percent = 100 - train_percent

    st.write(f"Train/Test split: **{train_percent}/{test_percent}**")

    n_estimators = st.sidebar.slider(
        "n_estimators",
        min_value=50,
        max_value=1000,
        value=300,
        step=50
    )

    max_depth = st.sidebar.slider(
        "max_depth",
        min_value=1,
        max_value=20,
        value=5,
        step=1
    )

    learning_rate = st.sidebar.slider(
        "learning_rate",
        min_value=0.001,
        max_value=0.300,
        value=0.050,
        step=0.001,
        format="%.3f"
    )

    subsample = st.sidebar.slider(
        "subsample",
        min_value=0.50,
        max_value=1.00,
        value=0.80,
        step=0.05
    )

    colsample_bytree = st.sidebar.slider(
        "colsample_bytree",
        min_value=0.50,
        max_value=1.00,
        value=0.80,
        step=0.05
    )

    # ========================================================
    # Session state for rerun attempt
    # ========================================================
    if "attempt" not in st.session_state:
        st.session_state.attempt = 0

    if st.button("Run / Rerun XGBoost Model"):
        st.session_state.attempt += 1

        # Different random seed every click
        random_seed = int(time.time()) + st.session_state.attempt

        # ----------------------------------------------------
        # Split X and Y
        # ----------------------------------------------------
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            train_size=train_percent / 100,
            random_state=random_seed,
            shuffle=True
        )

        # ----------------------------------------------------
        # Train XGBoost model
        # ----------------------------------------------------
        model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            objective="reg:squarederror",
            random_state=random_seed,
            n_jobs=-1,
            verbosity=0
        )

        model.fit(X_train, y_train)

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        y_all = np.concatenate([y_train, y_test])
        y_all_pred = np.concatenate([y_train_pred, y_test_pred])

        # ----------------------------------------------------
        # R2 values
        # ----------------------------------------------------
        r2_train = r2_score(y_train, y_train_pred)
        r2_test = r2_score(y_test, y_test_pred)
        r2_all = r2_score(y_all, y_all_pred)

        st.subheader("Model Results")

        col1, col2, col3 = st.columns(3)

        col1.metric("Train R²", f"{r2_train:.4f}")
        col2.metric("Test R²", f"{r2_test:.4f}")
        col3.metric("All Data R²", f"{r2_all:.4f}")

        st.write(f"Model attempt number: **{st.session_state.attempt}**")

        # ====================================================
        # R2 plot only
        # ====================================================
        st.subheader("R² Plot")

        fig, ax = plt.subplots(figsize=(7, 7))

        ax.scatter(
            y_train,
            y_train_pred,
            label="Train data",
            s=35,
            marker="o"
        )

        ax.scatter(
            y_test,
            y_test_pred,
            label="Test data",
            s=35,
            marker="s"
        )

        min_val = min(y_all.min(), y_all_pred.min())
        max_val = max(y_all.max(), y_all_pred.max())

        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            "--",
            label="45° line"
        )

        # Best-fit line
        p = np.polyfit(y_all, y_all_pred, 1)
        fit_line = np.polyval(p, y_all)

        sorted_index = np.argsort(y_all)
        ax.plot(
            y_all[sorted_index],
            fit_line[sorted_index],
            "-",
            label="Best-fit line"
        )

        ax.set_xlabel("Experimental Degradation (%)", fontsize=12)
        ax.set_ylabel("Predicted Degradation (%)", fontsize=12)
        ax.set_title("XGBoost Model R² Plot", fontsize=14)

        ax.text(
            0.05,
            0.95,
            f"Train R² = {r2_train:.4f}\nTest R² = {r2_test:.4f}\nAll R² = {r2_all:.4f}",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
        )

        ax.legend()
        ax.grid(False)

        st.pyplot(fig)

        # ====================================================
        # Model parameters
        # ====================================================
        st.subheader("Model Parameters")

        params_df = pd.DataFrame([{
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "random_seed": random_seed,
            "train_percent": train_percent,
            "test_percent": test_percent
        }])

        st.dataframe(params_df)

        # ====================================================
        # Message for rerun
        # ====================================================
        if r2_test < 0.80:
            st.warning(
                "The test R² is relatively low. You can click 'Run / Rerun XGBoost Model' again "
                "to try another random train/test split."
            )
        else:
            st.success("The model result is acceptable based on the current train/test split.")

else:
    st.warning("Please upload the completed Excel file after filling the template.")