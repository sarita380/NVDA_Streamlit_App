# app.py
# NVDA Stock Price Prediction – Streamlit Dashboard (just in case Python 3.9 compatible)

import io
import os
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import streamlit as st
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# Optional tree models
HAS_XGB = True
HAS_LGBM = True
try:
    from xgboost import XGBRegressor
except Exception:
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor
except Exception:
    HAS_LGBM = False

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
st.set_page_config(
    page_title="NVDA Stock Prediction Dashboard",
    layout="wide"
)

# Base directory = folder containing this app.py
BASE_DIR = os.path.dirname(__file__)
# Safe path to default CSV
DEFAULT_CSV_PATH = os.path.join(
    BASE_DIR, "Data", "nvda-daily-stock-prices-augmented.csv"
)

FEATURE_COLS: List[str] = ["Open", "High", "Low", "Volume", "Sentiment"]
TARGET_COL = "Close"

# -------------------------------------------------------------------
# CACHING DECORATOR (works on old/new Streamlit)
# -------------------------------------------------------------------
if hasattr(st, "cache_data"):
    cache_decorator = st.cache_data       # Streamlit ≥ 1.18
else:
    cache_decorator = st.cache            # Older Streamlit versions


# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------
def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    return rmse, mae, mape


@cache_decorator
def load_dataset(from_uploaded_bytes: Optional[bytes], default_path: str) -> pd.DataFrame:
    """
    Load NVDA dataset either from uploaded file or from default CSV path.
    - from_uploaded_bytes: contents of an uploaded CSV file, or None
    - default_path: path to fallback CSV on disk
    """
    if from_uploaded_bytes is not None:
        buffer = io.BytesIO(from_uploaded_bytes)
        raw_df = pd.read_csv(buffer)
    else:
        raw_df = pd.read_csv(default_path)

    df = raw_df.copy()

    # Find a date-like column
    date_col = None
    for col in df.columns:
        try:
            pd.to_datetime(df[col].head(3))
            date_col = col
            break
        except Exception:
            continue

    if date_col is None:
        raise ValueError("No date-like column found in the CSV.")

    # Normalize Date column to YYYY-MM-DD string
    df["Date"] = pd.to_datetime(df[date_col])
    df = df.sort_values("Date").reset_index(drop=True)
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

    # Ensure required price columns exist
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in CSV.")

    # Sentiment: if not present, set to 0
    sentiment_col = None
    for col in df.columns:
        if "sent" in col.lower():
            sentiment_col = col
            break

    if sentiment_col is None:
        df["Sentiment"] = 0.0
    else:
        df["Sentiment"] = df[sentiment_col].astype(float)

    # Keep only the relevant columns
    keep_cols = ["Date", "Open", "High", "Low", "Close", "Volume", "Sentiment"]
    df = df[keep_cols]

    return df


@cache_decorator
def train_all_models(dataframe: pd.DataFrame, test_ratio: float = 0.2):
    """
    Time-based split, train multiple models, return predictions & metrics.
    No Streamlit UI calls here (pure compute) to keep caching clean.
    """
    df_sorted = dataframe.sort_values("Date").reset_index(drop=True)

    # Handle missing values
    # Drop rows where target is NaN
    df_sorted = df_sorted.dropna(subset=[TARGET_COL])

    # For feature columns: forward-fill, back-fill, then fill remaining with 0
    df_sorted[FEATURE_COLS] = (
        df_sorted[FEATURE_COLS]
        .ffill()
        .bfill()
        .fillna(0.0)
    )

    X = df_sorted[FEATURE_COLS].values
    y = df_sorted[TARGET_COL].values

    split_idx = int(len(df_sorted) * (1 - test_ratio))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    test_dates = df_sorted["Date"].iloc[split_idx:].reset_index(drop=True)

    predictions: Dict[str, np.ndarray] = {}
    metrics_rows: List[Dict[str, Any]] = []

    # ---------------- Linear Regression ----------------
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    yhat_lr = lr.predict(X_test)
    predictions["Linear Regression"] = yhat_lr
    rmse, mae, mape = eval_metrics(y_test, yhat_lr)
    metrics_rows.append(
        {"Model": "Linear Regression", "RMSE": rmse, "MAE": mae, "MAPE_%": mape}
    )

    # ---------------- Random Forest ----------------
    rf = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    yhat_rf = rf.predict(X_test)
    predictions["Random Forest"] = yhat_rf
    rmse, mae, mape = eval_metrics(y_test, yhat_rf)
    metrics_rows.append(
        {"Model": "Random Forest", "RMSE": rmse, "MAE": mae, "MAPE_%": mape}
    )

    # ---------------- XGBoost ----------------
    if HAS_XGB:
        try:
            xgb = XGBRegressor(
                n_estimators=300,
                learning_rate=0.07,
                max_depth=5,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                objective="reg:squarederror",
                n_jobs=-1,
            )
            xgb.fit(X_train, y_train)
            yhat_xgb = xgb.predict(X_test)
            predictions["XGBoost"] = yhat_xgb
            rmse, mae, mape = eval_metrics(y_test, yhat_xgb)
            metrics_rows.append(
                {"Model": "XGBoost", "RMSE": rmse, "MAE": mae, "MAPE_%": mape}
            )
        except Exception:
            # Skip XGBoost if it fails
            pass

    # ---------------- LightGBM ----------------
    if HAS_LGBM:
        try:
            lgbm = LGBMRegressor(
                n_estimators=300,
                learning_rate=0.07,
                random_state=42,
            )
            lgbm.fit(X_train, y_train)
            yhat_lgb = lgbm.predict(X_test)
            predictions["LightGBM"] = yhat_lgb
            rmse, mae, mape = eval_metrics(y_test, yhat_lgb)
            metrics_rows.append(
                {"Model": "LightGBM", "RMSE": rmse, "MAE": mae, "MAPE_%": mape}
            )
        except Exception:
            # Skip LightGBM if it fails
            pass

    metrics_df = pd.DataFrame(metrics_rows).sort_values("RMSE").reset_index(drop=True)

    return predictions, metrics_df, test_dates, y_test


# -------------------------------------------------------------------
# UI LAYOUT
# -------------------------------------------------------------------
st.title(" NVDA Stock Price Prediction Dashboard")
st.write(
    "Interactive dashboard to explore multiple ML models predicting "
    "**NVDA** closing prices using price features and sentiment."
)

# Sidebar: data + settings
st.sidebar.header("Data & Settings")

uploaded_file = st.sidebar.file_uploader(
    "Upload custom CSV (optional)",
    type=["csv"],
    help="If empty, the default NVDA augmented CSV is used."
)

test_ratio_value = st.sidebar.slider(
    "Test set size (time-based split)",
    min_value=0.1,
    max_value=0.4,
    step=0.05,
    value=0.2,
)

last_n_days = st.sidebar.slider(
    "Days to show in charts (test set)",
    min_value=20,
    max_value=200,
    step=10,
    value=60,
)

# -------------------------------------------------------------------
# Load data
# -------------------------------------------------------------------
try:
    uploaded_bytes: Optional[bytes] = uploaded_file.read() if uploaded_file is not None else None
    df_loaded = load_dataset(
        from_uploaded_bytes=uploaded_bytes,
        default_path=DEFAULT_CSV_PATH,
    )
    st.success("Dataset loaded successfully.")
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

st.subheader(" Dataset Preview")
st.dataframe(df_loaded.head())

with st.expander("Descriptive Statistics"):
    # Drop Date for numeric stats
    numeric_df = df_loaded.drop(columns=["Date"], errors="ignore")
    st.write(numeric_df.describe())

# -------------------------------------------------------------------
# Train models
# -------------------------------------------------------------------
st.subheader(" Training Models")
with st.spinner("Training models on NVDA data..."):
    predictions_dict, metrics_df, test_dates_series, y_test_array = train_all_models(
        df_loaded, test_ratio_value
    )

st.success("Training complete.")


#here
# -------------------------------------------------------------------
# Metrics Table + Accuracy Chart
# -------------------------------------------------------------------
st.subheader(" Model Performance (Test Set)")

# 1) Add Accuracy_% column (100 - MAPE)
metrics_df["Accuracy_%"] = 100.0 - metrics_df["MAPE_%"]

# 2) Show table (include Accuracy_% in formatting)
st.dataframe(
    metrics_df.style.format(
        {
            "RMSE": "{:.3f}",
            "MAE": "{:.3f}",
            "MAPE_%": "{:.2f}",
            "Accuracy_%": "{:.2f}",
        }
    )
)

# 3) Accuracy bar chart
st.subheader(" Accuracy by Model (100% - MAPE)")

fig_acc, ax_acc = plt.subplots(figsize=(8, 4))

models = metrics_df["Model"].tolist()
accuracies = metrics_df["Accuracy_%"].tolist()

ax_acc.bar(models, accuracies)
ax_acc.set_ylim(0, 100)
ax_acc.set_ylabel("Accuracy (%)")
ax_acc.set_title("Model Accuracy (higher is better)")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()

st.pyplot(fig_acc)

#to here

# -------------------------------------------------------------------
# Prediction Explorer
# -------------------------------------------------------------------
st.subheader(" Prediction Explorer")

selected_model = st.selectbox("Select model to inspect:", list(predictions_dict.keys()))
y_hat_selected = predictions_dict[selected_model]

# ensure last_n_days doesn't exceed test length
last_n = min(last_n_days, len(test_dates_series))
plot_dates = test_dates_series.iloc[-last_n:]
plot_actual = y_test_array[-last_n:]
plot_pred = y_hat_selected[-last_n:]

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(plot_dates, plot_actual, label="Actual", linewidth=2)
ax.plot(plot_dates, plot_pred, "--", label=f"{selected_model} Predicted", alpha=0.9)
ax.set_title(f"{selected_model} — Actual vs Predicted (last {last_n} test days)")
ax.set_xlabel("Date")
ax.set_ylabel("Close Price")
ax.legend()
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig)

# -------------------------------------------------------------------
# Single-day inspection (last test date)
# -------------------------------------------------------------------
st.subheader(" Last Test Day – Model Comparison")

last_date = test_dates_series.iloc[-1]  # this is already a string like '2020-01-15'
last_actual_value = float(y_test_array[-1])
rows: List[Dict[str, Any]] = []

for model_name_key, preds in predictions_dict.items():
    last_pred_value = float(preds[-1])
    abs_err_value = last_pred_value - last_actual_value
    pct_err_value = abs(abs_err_value) / last_actual_value * 100
    rows.append(
        {
            "Model": model_name_key,
            "Date": last_date,  # <-- FIX: use string directly, no .date()
            "Actual Close": round(last_actual_value, 2),
            "Predicted Close": round(last_pred_value, 2),
            "Abs Error": round(abs_err_value, 2),
            "Abs % Error": round(pct_err_value, 2),
        }
    )


comparison_df = pd.DataFrame(rows).sort_values("Abs % Error").reset_index(drop=True)
st.dataframe(comparison_df)
