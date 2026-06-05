# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 16:10:11 2026

@author: 24550372
"""

# -*- coding: utf-8 -*-
"""
Streamlit Web App for XGBoost Modeling
- Upload Excel/CSV data
- Select input columns, target column, optional reference column
- Select train/test ratio
- Train XGBoost model
- Optional PSO hyperparameter optimization
- Show metrics, model parameters, plots
- Download results as Excel and trained model as PKL
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import io
import tempfile

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from pyswarms.single.global_best import GlobalBestPSO
from scipy.stats import norm


# ============================================================
# Page setting
# ============================================================
st.set_page_config(
    page_title="XGBoost Modeling App",
    layout="wide"
)

st.title("XGBoost Modeling Web App")
st.write(
    "Upload your Excel/CSV file, select input and output variables, choose train/test ratio, "
    "train an XGBoost model, and download predictions, metrics, and model parameters."
)


# ============================================================
# Helper functions
# ============================================================
def compute_aard(y_true, y_pred):
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    return np.mean(
        np.divide(
            np.abs(y_pred - y_true),
            np.abs(y_true),
            out=np.zeros_like(y_true, dtype=np.float64),
            where=y_true != 0
        )
    ) * 100


def compute_std(y_true, y_pred):
    return np.std(np.asarray(y_pred).flatten() - np.asarray(y_true).flatten())


def calculate_metrics(y_train, y_train_pred, y_test, y_test_pred):
    y_all = np.concatenate([y_train, y_test])
    y_all_pred = np.concatenate([y_train_pred, y_test_pred])

    metrics = {
        "R2_train": r2_score(y_train, y_train_pred),
        "R2_test": r2_score(y_test, y_test_pred),
        "R2_all": r2_score(y_all, y_all_pred),

        "MSE_train": mean_squared_error(y_train, y_train_pred),
        "MSE_test": mean_squared_error(y_test, y_test_pred),
        "MSE_all": mean_squared_error(y_all, y_all_pred),

        "RMSE_train": np.sqrt(mean_squared_error(y_train, y_train_pred)),
        "RMSE_test": np.sqrt(mean_squared_error(y_test, y_test_pred)),
        "RMSE_all": np.sqrt(mean_squared_error(y_all, y_all_pred)),

        "MAE_train": mean_absolute_error(y_train, y_train_pred),
        "MAE_test": mean_absolute_error(y_test, y_test_pred),
        "MAE_all": mean_absolute_error(y_all, y_all_pred),

        "AARD_train_%": compute_aard(y_train, y_train_pred),
        "AARD_test_%": compute_aard(y_test, y_test_pred),
        "AARD_all_%": compute_aard(y_all, y_all_pred),

        "STD_train": compute_std(y_train, y_train_pred),
        "STD_test": compute_std(y_test, y_test_pred),
        "STD_all": compute_std(y_all, y_all_pred),
    }

    return metrics


def make_actual_vs_predicted_plot(y_train, y_train_pred, y_test, y_test_pred):
    fig, ax = plt.subplots(figsize=(7, 6))

    y_all = np.concatenate([y_train, y_test])
    y_all_pred = np.concatenate([y_train_pred, y_test_pred])

    ax.scatter(y_train, y_train_pred, label="Train", s=20)
    ax.scatter(y_test, y_test_pred, label="Test", s=20)

    min_val = min(y_all.min(), y_all_pred.min())
    max_val = max(y_all.max(), y_all_pred.max())

    ax.plot([min_val, max_val], [min_val, max_val], "--", label="45° line")

    ax.set_xlabel("Experimental")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted")
    ax.legend()
    ax.grid(False)

    return fig


def make_index_plot(y_train, y_train_pred, y_test, y_test_pred):
    fig, ax = plt.subplots(figsize=(11, 5))

    train_index = np.arange(1, len(y_train) + 1)
    test_index = np.arange(len(y_train) + 1, len(y_train) + len(y_test) + 1)

    ax.plot(train_index, y_train, "o", label="Train Experimental", markersize=4)
    ax.plot(train_index, y_train_pred, "-o", label="Train Predicted", markersize=3)

    ax.plot(test_index, y_test, "s", label="Test Experimental", markersize=4)
    ax.plot(test_index, y_test_pred, "-s", label="Test Predicted", markersize=3)

    ax.set_xlabel("Data Index")
    ax.set_ylabel("Target Value")
    ax.set_title("Experimental and Predicted Values")
    ax.legend()
    ax.grid(False)

    return fig


def make_error_histogram(y_train, y_train_pred, y_test, y_test_pred):
    fig, ax = plt.subplots(figsize=(8, 5))

    y_all = np.concatenate([y_train, y_test])
    y_all_pred = np.concatenate([y_train_pred, y_test_pred])
    errors = y_all_pred - y_all

    counts, bins = np.histogram(errors, bins=40)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    if counts.sum() > 0:
        normalized_counts = 100 * counts / counts.sum()
    else:
        normalized_counts = counts

    ax.bar(bin_centers, normalized_counts, width=bins[1] - bins[0], edgecolor="black", alpha=0.7)

    mu = np.mean(errors)
    sigma = np.std(errors)

    if sigma > 0:
        ax.plot(
            bin_centers,
            norm.pdf(bin_centers, mu, sigma) * (bins[1] - bins[0]) * 100,
            linewidth=2,
            label="Normal fit"
        )

    ax.set_xlabel("Prediction Error")
    ax.set_ylabel("Frequency (%)")
    ax.set_title("Error Distribution")
    ax.legend()
    ax.grid(False)

    return fig


def make_relative_error_plot(y_train, y_train_pred, y_test, y_test_pred):
    fig, ax = plt.subplots(figsize=(8, 5))

    rel_train = np.divide(
        y_train_pred - y_train,
        y_train,
        out=np.zeros_like(y_train, dtype=np.float64),
        where=y_train != 0
    )

    rel_test = np.divide(
        y_test_pred - y_test,
        y_test,
        out=np.zeros_like(y_test, dtype=np.float64),
        where=y_test != 0
    )

    ax.scatter(y_train, rel_train, label="Train Relative Error", s=20)
    ax.scatter(y_test, rel_test, label="Test Relative Error", s=20)
    ax.axhline(0, linestyle="--")

    ax.set_xlabel("Experimental")
    ax.set_ylabel("Relative Error")
    ax.set_title("Relative Error Plot")
    ax.legend()
    ax.grid(False)

    return fig


def create_excel_download(train_df, test_df, metrics, best_params):
    output = io.BytesIO()

    all_results = pd.concat([train_df, test_df], ignore_index=True)

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        all_results.to_excel(writer, sheet_name="Predictions", index=False)
        pd.DataFrame([metrics]).to_excel(writer, sheet_name="Metrics", index=False)
        pd.DataFrame([best_params]).to_excel(writer, sheet_name="Model_Parameters", index=False)

    output.seek(0)
    return output


# ============================================================
# Upload data
# ============================================================
uploaded_file = st.file_uploader(
    "Upload Excel or CSV file",
    type=["xlsx", "xls", "csv"]
)

if uploaded_file is not None:

    # ------------------------------
    # Read file
    # ------------------------------
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            header_option = st.checkbox("My file has column headers", value=True)

            if header_option:
                df = pd.read_excel(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file, header=None)

        st.success("File uploaded successfully.")

    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    st.subheader("Data Preview")
    st.dataframe(df.head())

    st.write(f"Data shape: **{df.shape[0]} rows × {df.shape[1]} columns**")

    # ------------------------------
    # Column selection
    # ------------------------------
    st.subheader("Select Input and Output Columns")

    all_columns = list(df.columns)

    target_column = st.selectbox(
        "Select target/output column",
        all_columns,
        index=len(all_columns) - 1
    )

    use_reference = st.checkbox("My file has a reference column", value=False)

    ref_column = None
    if use_reference:
        ref_column = st.selectbox(
            "Select reference column",
            all_columns,
            index=len(all_columns) - 1
        )

    default_features = [
        col for col in all_columns
        if col != target_column and col != ref_column
    ]

    feature_columns = st.multiselect(
        "Select input feature columns",
        all_columns,
        default=default_features
    )

    if len(feature_columns) == 0:
        st.warning("Please select at least one input feature.")
        st.stop()

    # ------------------------------
    # Remove missing values
    # ------------------------------
    selected_columns = feature_columns + [target_column]
    if ref_column is not None:
        selected_columns.append(ref_column)

    data = df[selected_columns].copy()
    data = data.dropna()

    X = data[feature_columns].to_numpy(dtype=float)
    y = data[target_column].to_numpy(dtype=float).flatten()

    if ref_column is not None:
        ref = data[ref_column].to_numpy()
    else:
        ref = None

    # ========================================================
    # Sidebar settings
    # ========================================================
    st.sidebar.header("Model Settings")

    train_percent = st.sidebar.slider(
        "Training percentage (%)",
        min_value=50,
        max_value=95,
        value=80,
        step=5
    )

    test_size = 1 - train_percent / 100

    random_seed = st.sidebar.number_input(
        "Random seed",
        min_value=0,
        max_value=999999,
        value=42,
        step=1
    )

    use_pso = st.sidebar.checkbox("Use PSO for XGBoost tuning", value=True)

    if use_pso:
        st.sidebar.subheader("PSO Settings")

        n_particles = st.sidebar.slider(
            "Number of particles",
            min_value=5,
            max_value=50,
            value=10,
            step=5
        )

        pso_iterations = st.sidebar.slider(
            "PSO iterations",
            min_value=5,
            max_value=200,
            value=30,
            step=5
        )

        cv_folds = st.sidebar.slider(
            "Cross-validation folds",
            min_value=2,
            max_value=10,
            value=5,
            step=1
        )

    else:
        st.sidebar.subheader("Manual XGBoost Parameters")

        manual_n_estimators = st.sidebar.slider("n_estimators", 50, 1000, 300, 50)
        manual_learning_rate = st.sidebar.slider("learning_rate", 0.001, 0.5, 0.05, 0.001)
        manual_max_depth = st.sidebar.slider("max_depth", 1, 20, 5, 1)
        manual_subsample = st.sidebar.slider("subsample", 0.3, 1.0, 0.8, 0.05)
        manual_colsample_bytree = st.sidebar.slider("colsample_bytree", 0.3, 1.0, 0.8, 0.05)
        manual_gamma = st.sidebar.slider("gamma", 0.0, 20.0, 0.0, 0.1)
        manual_min_child_weight = st.sidebar.slider("min_child_weight", 1.0, 30.0, 1.0, 0.5)

    # ========================================================
    # Train model
    # ========================================================
    if st.button("Train XGBoost Model"):

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=int(random_seed),
            shuffle=True
        )

        st.subheader("Train/Test Split")
        st.write(f"Training samples: **{X_train.shape[0]}**")
        st.write(f"Testing samples: **{X_test.shape[0]}**")

        # ----------------------------------------------------
        # PSO optimization
        # ----------------------------------------------------
        if use_pso:

            st.subheader("PSO Hyperparameter Optimization")
            st.write("Running PSO optimization...")

            progress_bar = st.progress(0)

            def objective_function(params):
                n_particles_local = params.shape[0]
                scores = []

                for i in range(n_particles_local):
                    n_estimators = int(params[i, 0])
                    learning_rate = params[i, 1]
                    subsample = params[i, 2]
                    max_depth = int(params[i, 3])
                    min_child_weight = params[i, 4]
                    gamma = params[i, 5]
                    colsample_bytree = params[i, 6]

                    model = XGBRegressor(
                        n_estimators=n_estimators,
                        learning_rate=learning_rate,
                        subsample=subsample,
                        max_depth=max_depth,
                        min_child_weight=min_child_weight,
                        gamma=gamma,
                        colsample_bytree=colsample_bytree,
                        objective="reg:squarederror",
                        n_jobs=-1,
                        random_state=int(random_seed),
                        verbosity=0
                    )

                    score = cross_val_score(
                        model,
                        X_train,
                        y_train,
                        cv=cv_folds,
                        scoring="r2",
                        n_jobs=-1
                    ).mean()

                    scores.append(-score)

                return np.array(scores)

            bounds = (
                [50, 0.01, 0.5, 2, 1, 0, 0.3],
                [500, 0.3, 1.0, 15, 20, 10, 1.0]
            )

            optimizer = GlobalBestPSO(
                n_particles=n_particles,
                dimensions=7,
                options={"c1": 1.5, "c2": 1.5, "w": 0.7},
                bounds=bounds
            )

            best_cost, best_pos = optimizer.optimize(
                objective_function,
                iters=pso_iterations,
                verbose=False
            )

            progress_bar.progress(100)

            best_params = {
                "n_estimators": int(best_pos[0]),
                "learning_rate": float(best_pos[1]),
                "subsample": float(best_pos[2]),
                "max_depth": int(best_pos[3]),
                "min_child_weight": float(best_pos[4]),
                "gamma": float(best_pos[5]),
                "colsample_bytree": float(best_pos[6]),
            }

            st.success("PSO optimization completed.")

            # PSO convergence curve
            st.subheader("PSO Convergence Curve")

            fig_conv, ax_conv = plt.subplots(figsize=(8, 4))
            ax_conv.plot(optimizer.cost_history, marker="o")
            ax_conv.set_xlabel("Iteration")
            ax_conv.set_ylabel("Best Cost = -R²")
            ax_conv.set_title("PSO Convergence Curve")
            ax_conv.grid(False)
            st.pyplot(fig_conv)

        else:
            best_params = {
                "n_estimators": manual_n_estimators,
                "learning_rate": manual_learning_rate,
                "max_depth": manual_max_depth,
                "subsample": manual_subsample,
                "colsample_bytree": manual_colsample_bytree,
                "gamma": manual_gamma,
                "min_child_weight": manual_min_child_weight,
            }

        # ----------------------------------------------------
        # Final model training
        # ----------------------------------------------------
        model = XGBRegressor(
            **best_params,
            objective="reg:squarederror",
            n_jobs=-1,
            random_state=int(random_seed),
            verbosity=0
        )

        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        metrics = calculate_metrics(
            y_train,
            y_train_pred,
            y_test,
            y_test_pred
        )

        # ====================================================
        # Results
        # ====================================================
        st.subheader("Model Parameters")
        st.dataframe(pd.DataFrame([best_params]))

        st.subheader("Model Performance Metrics")
        metrics_df = pd.DataFrame([metrics]).T
        metrics_df.columns = ["Value"]
        st.dataframe(metrics_df)

        # ====================================================
        # Prediction tables
        # ====================================================
        train_df = pd.DataFrame(X_train, columns=feature_columns)
        train_df[target_column] = y_train
        train_df["Prediction"] = y_train_pred
        train_df["Set"] = "Train"

        test_df = pd.DataFrame(X_test, columns=feature_columns)
        test_df[target_column] = y_test
        test_df["Prediction"] = y_test_pred
        test_df["Set"] = "Test"

        st.subheader("Prediction Results")
        st.dataframe(pd.concat([train_df, test_df], ignore_index=True))

        # ====================================================
        # Plots
        # ====================================================
        st.subheader("Plots")

        col1, col2 = st.columns(2)

        with col1:
            st.pyplot(make_actual_vs_predicted_plot(
                y_train,
                y_train_pred,
                y_test,
                y_test_pred
            ))

        with col2:
            st.pyplot(make_relative_error_plot(
                y_train,
                y_train_pred,
                y_test,
                y_test_pred
            ))

        col3, col4 = st.columns(2)

        with col3:
            st.pyplot(make_index_plot(
                y_train,
                y_train_pred,
                y_test,
                y_test_pred
            ))

        with col4:
            st.pyplot(make_error_histogram(
                y_train,
                y_train_pred,
                y_test,
                y_test_pred
            ))

        # ====================================================
        # Downloads
        # ====================================================
        st.subheader("Download Results")

        excel_file = create_excel_download(
            train_df,
            test_df,
            metrics,
            best_params
        )

        st.download_button(
            label="Download Excel Results",
            data=excel_file,
            file_name="XGBoost_Modeling_Results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        model_buffer = io.BytesIO()
        joblib.dump(model, model_buffer)
        model_buffer.seek(0)

        st.download_button(
            label="Download Trained XGBoost Model (.pkl)",
            data=model_buffer,
            file_name="XGBoost_Model.pkl",
            mime="application/octet-stream"
        )

else:
    st.info("Please upload an Excel or CSV file to start.")