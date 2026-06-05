# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 17:01:11 2026

@author: 24550372
"""

# -*- coding: utf-8 -*-
"""
Streamlit Web App for GBM + Optional PSO Model Development (Developed by Amir Dashti, amirdashti13681990@gmail.com)

Functions:
1. Download Excel data template
2. Upload completed Excel file
3. Select train/test ratio
4. Train or rerun Gradient Boosting model
5. Optional PSO hyperparameter optimization
6. Results 
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import time

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score
from pyswarms.single.global_best import GlobalBestPSO


# ============================================================
# Streamlit page setting
# ============================================================
st.set_page_config(
    page_title="GBM Web-Based Model",
    layout="wide"
)

st.title("Web-Based Gradient Boosting Model Development App")

st.write(
    "Download the Excel template, fill your experimental data, upload the completed file, "
    "select the train/test ratio, and run or rerun the Gradient Boosting model."
)


# ============================================================
# Metric functions
# ============================================================
def compute_rmse(y_true, y_pred):
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def compute_mape(y_true, y_pred):
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    return np.mean(
        np.divide(
            np.abs(y_true - y_pred),
            np.abs(y_true),
            out=np.zeros_like(y_true, dtype=np.float64),
            where=y_true != 0
        )
    ) * 100


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


# ============================================================
# Step 1: Download template
# ============================================================
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
# Step 2: Upload completed Excel file
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
        st.error("Your uploaded Excel file does not match the required template.")
        st.write("Missing columns:")
        st.write(missing_columns)
        st.stop()

    # --------------------------------------------------------
    # Clean data
    # --------------------------------------------------------
    df = df[template_columns].copy()

    for col in template_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna()

    if df.shape[0] < 10:
        st.error("The uploaded file has too few valid rows. Please add more data.")
        st.stop()

    st.write(f"Valid data rows: **{df.shape[0]}**")

    # ========================================================
    # Step 3: Model settings
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

    # ========================================================
    # Sidebar settings
    # ========================================================
    st.sidebar.header("Model Settings")

    use_pso = st.sidebar.checkbox("Use PSO optimization", value=True)

    if use_pso:
        st.sidebar.subheader("PSO Settings")

        n_particles = st.sidebar.slider(
            "Number of particles",
            min_value=5,
            max_value=50,
            value=5,
            step=5
        )

        pso_iterations = st.sidebar.slider(
            "PSO iterations",
            min_value=5,
            max_value=200,
            value=100,
            step=5
        )

        c1 = st.sidebar.number_input(
            "PSO cognitive coefficient c1",
            min_value=0.1,
            max_value=5.0,
            value=1.5,
            step=0.1
        )

        c2 = st.sidebar.number_input(
            "PSO social coefficient c2",
            min_value=0.1,
            max_value=5.0,
            value=1.5,
            step=0.1
        )

        w = st.sidebar.number_input(
            "PSO inertia weight w",
            min_value=0.1,
            max_value=2.0,
            value=0.7,
            step=0.1
        )

        cv_folds = st.sidebar.slider(
            "Cross-validation folds",
            min_value=2,
            max_value=10,
            value=5,
            step=1
        )

    else:
        st.sidebar.subheader("Manual GBM Parameters")

        n_estimators = st.sidebar.slider(
            "n_estimators",
            min_value=50,
            max_value=1000,
            value=300,
            step=50
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

        min_samples_split = st.sidebar.slider(
            "min_samples_split",
            min_value=2,
            max_value=100,
            value=2,
            step=1
        )

        min_samples_leaf = st.sidebar.slider(
            "min_samples_leaf",
            min_value=1,
            max_value=50,
            value=1,
            step=1
        )

        max_depth = st.sidebar.slider(
            "max_depth",
            min_value=1,
            max_value=30,
            value=5,
            step=1
        )

    # ========================================================
    # Session state for rerun attempt
    # ========================================================
    if "attempt" not in st.session_state:
        st.session_state.attempt = 0

    # ========================================================
    # Step 4: Run model
    # ========================================================
    st.subheader("Step 4: Run or Rerun Model")

    if st.button("Run / Rerun GBM Model"):

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
        # PSO optimization
        # ----------------------------------------------------
        if use_pso:

            st.write("Running PSO optimization...")

            progress_placeholder = st.empty()

            def objective_function(params):
                n_particles_local = params.shape[0]
                scores = []

                for i in range(n_particles_local):

                    n_estimators_i = int(np.round(params[i, 0]))
                    learning_rate_i = float(params[i, 1])
                    subsample_i = float(params[i, 2])
                    min_samples_split_i = int(np.round(params[i, 3]))
                    min_samples_leaf_i = int(np.round(params[i, 4]))
                    max_depth_i = int(np.round(params[i, 5]))

                    # Safety correction
                    min_samples_split_i = max(min_samples_split_i, 2)
                    min_samples_leaf_i = max(min_samples_leaf_i, 1)
                    max_depth_i = max(max_depth_i, 1)

                    temp_model = GradientBoostingRegressor(
                        n_estimators=n_estimators_i,
                        learning_rate=learning_rate_i,
                        subsample=subsample_i,
                        min_samples_split=min_samples_split_i,
                        min_samples_leaf=min_samples_leaf_i,
                        max_depth=max_depth_i,
                        random_state=random_seed
                    )

                    score = cross_val_score(
                        temp_model,
                        X_train,
                        y_train,
                        cv=cv_folds,
                        scoring="r2",
                        n_jobs=-1
                    ).mean()

                    scores.append(-score)

                return np.array(scores)

            # These are the 6 optimized GBM hyperparameters:
            # 1. n_estimators
            # 2. learning_rate
            # 3. subsample
            # 4. min_samples_split
            # 5. min_samples_leaf
            # 6. max_depth
            bounds = (
                [50, 0.01, 0.50, 2, 1, 1],
                [300, 0.30, 1.00, 100, 50, 30]
            )

            optimizer = GlobalBestPSO(
                n_particles=n_particles,
                dimensions=6,
                options={
                    "c1": c1,
                    "c2": c2,
                    "w": w
                },
                bounds=bounds
            )

            best_cost, best_pos = optimizer.optimize(
                objective_function,
                iters=pso_iterations,
                verbose=False
            )

            best_params = {
                "n_estimators": int(np.round(best_pos[0])),
                "learning_rate": float(best_pos[1]),
                "subsample": float(best_pos[2]),
                "min_samples_split": int(np.round(best_pos[3])),
                "min_samples_leaf": int(np.round(best_pos[4])),
                "max_depth": int(np.round(best_pos[5]))
            }

            # Safety correction
            best_params["min_samples_split"] = max(best_params["min_samples_split"], 2)
            best_params["min_samples_leaf"] = max(best_params["min_samples_leaf"], 1)
            best_params["max_depth"] = max(best_params["max_depth"], 1)

            progress_placeholder.success("PSO optimization completed.")

        else:

            best_params = {
                "n_estimators": int(n_estimators),
                "learning_rate": float(learning_rate),
                "subsample": float(subsample),
                "min_samples_split": int(min_samples_split),
                "min_samples_leaf": int(min_samples_leaf),
                "max_depth": int(max_depth)
            }

        # ----------------------------------------------------
        # Final GBM model
        # ----------------------------------------------------
        model = GradientBoostingRegressor(
            **best_params,
            random_state=random_seed
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
        # R2
        # ----------------------------------------------------
        r2_train = r2_score(y_train, y_train_pred)
        r2_test = r2_score(y_test, y_test_pred)
        r2_all = r2_score(y_all, y_all_pred)

        # ----------------------------------------------------
        # RMSE
        # ----------------------------------------------------
        rmse_train = compute_rmse(y_train, y_train_pred)
        rmse_test = compute_rmse(y_test, y_test_pred)
        rmse_all = compute_rmse(y_all, y_all_pred)

        # ----------------------------------------------------
        # MAPE
        # ----------------------------------------------------
        mape_train = compute_mape(y_train, y_train_pred)
        mape_test = compute_mape(y_test, y_test_pred)
        mape_all = compute_mape(y_all, y_all_pred)

        # ====================================================
        # Results
        # ====================================================
        st.subheader("Model Results")

        st.write(f"Model attempt number: **{st.session_state.attempt}**")

        col1, col2, col3 = st.columns(3)

        col1.metric("Train R²", f"{r2_train:.4f}")
        col2.metric("Test R²", f"{r2_test:.4f}")
        col3.metric("All Data R²", f"{r2_all:.4f}")

        col4, col5, col6 = st.columns(3)

        col4.metric("Train RMSE", f"{rmse_train:.4f}")
        col5.metric("Test RMSE", f"{rmse_test:.4f}")
        col6.metric("All Data RMSE", f"{rmse_all:.4f}")

        col7, col8, col9 = st.columns(3)

        col7.metric("Train MAPE (%)", f"{mape_train:.2f}")
        col8.metric("Test MAPE (%)", f"{mape_test:.2f}")
        col9.metric("All Data MAPE (%)", f"{mape_all:.2f}")

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
        ax.set_title("GBM-PSO R² Plot", fontsize=14)

        ax.text(
            0.05,
            0.95,
            f"Train R² = {r2_train:.4f}\n"
            f"Test R² = {r2_test:.4f}\n"
            f"All R² = {r2_all:.4f}\n"
            f"Test RMSE = {rmse_test:.4f}\n"
            f"Test MAPE = {mape_test:.2f}%",
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(
                boxstyle="round",
                facecolor="white",
                alpha=0.8
            )
        )

        ax.legend()
        ax.grid(False)

        st.pyplot(fig)

        # ====================================================
        # Model parameters
        # ====================================================
        st.subheader("Model Parameters")

        if use_pso:
            params_df = pd.DataFrame([{
                **best_params,
                "use_pso": use_pso,
                "n_particles": n_particles,
                "dimensions": 6,
                "pso_iterations": pso_iterations,
                "c1": c1,
                "c2": c2,
                "w": w,
                "cv_folds": cv_folds,
                "random_seed": random_seed,
                "train_percent": train_percent,
                "test_percent": test_percent
            }])
        else:
            params_df = pd.DataFrame([{
                **best_params,
                "use_pso": use_pso,
                "random_seed": random_seed,
                "train_percent": train_percent,
                "test_percent": test_percent
            }])

        st.dataframe(params_df)

        # ====================================================
        # Rerun message
        # ====================================================
        if r2_test < 0.80:
            st.warning(
                "The test R² is relatively low. Click 'Run / Rerun GBM Model' again "
                "to try another random train/test split."
            )
        else:
            st.success(
                "The model result is acceptable based on the current train/test split."
            )

else:
    st.warning("Please upload the completed Excel file after filling the template.")