"""
Asset Forecasting Dashboard

This page allows the user to:
- Select an asset to forecast
- Select optional exogenous variables (other columns)
- Choose the historical window and forecast horizon
- Train a lightweight ML model (Ridge Regression with autoregressive lags)
- Plot actual vs predicted + forecast

The model is intentionally simple, fast and robust for daily use.
"""

# ============================================================
# Imports & Config
# ============================================================
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.data_pipeline import df
from src.utils import apply_theme, date_range_picker, section, asset_picker_dropdown

# Theme
apply_theme()

# ============================================================
# Base data
# ============================================================
BASE = df.copy()
BASE["date"] = pd.to_datetime(BASE["date"], errors="coerce")
BASE = BASE.sort_values("date")

# ============================================================
# Asset selection
# ============================================================
section("Selecione o ativo para previsão", "O modelo irá aprender automaticamente seus padrões", "🔮")

ASSETS_MAP = {col: col for col in BASE.columns if col not in ["date"]}

target_col, _ = asset_picker_dropdown(
    BASE,
    ASSETS_MAP,
    state_key="forecast_target",
)

st.divider()

# ============================================================
# Exogenous feature selection
# ============================================================
section("Variáveis explicativas (opcionais)", "Selecione colunas adicionais para ajudar na previsão", "🧩")

exog_cols = st.multiselect(
    "Variáveis adicionais",
    options=[c for c in BASE.columns if c not in ["date", target_col]],
    default=[],
)

st.divider()

# ============================================================
# Parameters
# ============================================================
section("Parâmetros do modelo", "Janelas e horizonte de previsão", "⚙️")

lags = st.slider("Número de lags (autoregressivos)", min_value=3, max_value=60, value=20)
horizon = st.slider("Horizonte de previsão (dias)", min_value=1, max_value=30, value=7)

start_date, end_date = date_range_picker(
    BASE["date"], state_key="range_forecast", default_days=800
)

st.divider()

# ============================================================
# Prepare dataset
# ============================================================
df_view = BASE[(BASE["date"] >= start_date) & (BASE["date"] <= end_date)].copy()
df_view = df_view.dropna(subset=[target_col]).reset_index(drop=True)

if df_view.empty:
    st.error("Sem dados suficientes para o período selecionado.")
    st.stop()

# Build supervised learning table
def build_lagged_matrix(df, target, exog, lags):
    X, y = [], []
    for i in range(lags, len(df)):
        row = []

        # target lags
        for k in range(1, lags + 1):
            row.append(df[target].iloc[i - k])

        # exogenous variables
        for col in exog:
            row.append(df[col].iloc[i])

        X.append(row)
        y.append(df[target].iloc[i])

    return np.array(X), np.array(y)

X, y = build_lagged_matrix(df_view, target_col, exog_cols, lags)

# Train/Test split (last "horizon" points reserved)
X_train, X_test = X[:-horizon], X[-horizon:]
y_train, y_test = y[:-horizon], y[-horizon:]

# ============================================================
# Train model
# ============================================================
model = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", Ridge(alpha=1.0)),
])

model.fit(X_train, y_train)

# ============================================================
# Forecast last horizon
# ============================================================
y_pred = model.predict(X_test)

# Future forecast (recursive)
future_preds = []
last_window = list(y[-lags:])  # last lags of the target

for step in range(horizon):
    features = last_window[-lags:]

    # add exogenous future values? (Unavailable → using last known)
    for col in exog_cols:
        features.append(df_view[col].iloc[-1])

    features = np.array(features).reshape(1, -1)
    next_pred = model.predict(features)[0]

    future_preds.append(next_pred)
    last_window.append(next_pred)

# ============================================================
# Plot
# ============================================================
fig = go.Figure()

# Historical
fig.add_trace(go.Scatter(
    x=df_view["date"].iloc[-(horizon + 100):],
    y=df_view[target_col].iloc[-(horizon + 100):],
    mode="lines",
    name="Histórico",
    line=dict(color="#1f77b4"),
))

# In-sample last predictions
fig.add_trace(go.Scatter(
    x=df_view["date"].iloc[-horizon:],
    y=y_pred,
    mode="lines",
    name="Previsão (in-sample)",
    line=dict(color="#ff7f0e", dash="dot"),
))

# Future forecast
future_dates = pd.date_range(df_view["date"].iloc[-1], periods=horizon + 1, freq="D")[1:]

fig.add_trace(go.Scatter(
    x=future_dates,
    y=future_preds,
    mode="lines+markers",
    name="Forecast futuro",
    line=dict(color="#2ca02c", width=3),
))

fig.update_layout(
    title=f"Previsão para {target_col.upper()} (horizon = {horizon} dias)",
    xaxis_title="Data",
    yaxis_title="Preço",
    margin=dict(t=80),
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
)

st.plotly_chart(fig, use_container_width=True)

# ============================================================
# Metrics
# ============================================================
from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

st.subheader("📌 Métricas do modelo")
st.write(f"**MAE:** {mae:,.4f}")
st.write(f"**RMSE:** {rmse:,.4f}")
