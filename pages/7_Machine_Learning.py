"""Machine Learning Forecasting

This page lets the user:
- select an asset (oil, meal, palm, FX, crypto, etc.),
- choose a historical window,
- configure the forecast horizon (up to 45 business days),
- choose an ML model (Ridge, Random Forest, XGBoost if available),
- and forecast the selected asset using lag features.

The page shows:
- how the model performs on recent history (backtest: real vs predicted),
- and the multi-step forecast into the future.
"""

# ============================================================
# Imports & Config
# ============================================================
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# XGBoost is optional: only enabled if installed
try:
    from xgboost import XGBRegressor  # type: ignore
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from src.data_pipeline import df
from src.utils import (
    apply_theme,
    asset_picker_dropdown,
    date_range_picker,
    section,
)

# --- Theme
apply_theme()

# ============================================================
# Helpers
# ============================================================
def make_supervised(
    series: pd.Series,
    n_lags: int,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Convert a time series into supervised data using lags.

    For each time t, the features are:
        [y_{t-1}, y_{t-2}, ..., y_{t-n_lags}]
    and the target is y_t.
    """
    df_sup = pd.DataFrame({"target": series.astype(float)})
    for lag in range(1, n_lags + 1):
        df_sup[f"lag_{lag}"] = df_sup["target"].shift(lag)

    df_sup = df_sup.dropna().copy()
    X = df_sup[[f"lag_{lag}" for lag in range(1, n_lags + 1)]].values
    y = df_sup["target"].values
    idx = df_sup.index  # dates aligned with y

    return X, y, idx


def train_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> object:
    """Instantiate and fit the chosen model."""
    if model_name == "Ridge Regression":
        model = Ridge(alpha=1.0, random_state=42)
    elif model_name == "Random Forest":
        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
        )
    elif model_name == "XGBoost (if available)" and HAS_XGB:
        model = XGBRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )
    else:
        # Fallback to Ridge if something odd happens
        model = Ridge(alpha=1.0, random_state=42)

    model.fit(X_train, y_train)
    return model


def multi_step_forecast(
    model: object,
    history: np.ndarray,
    n_lags: int,
    horizon: int,
) -> np.ndarray:
    """Recursive multi-step forecast.

    Uses the last n_lags values of the series (including predicted ones)
    to forecast the next value, and repeats this up to `horizon` steps.
    """
    history = list(history.astype(float))
    preds = []

    for _ in range(horizon):
        # last n_lags values, most recent first
        if len(history) < n_lags:
            break

        last_vals = history[-n_lags:]
        # features are [y_{t-1}, y_{t-2}, ..., y_{t-n_lags}]
        x_input = np.array(last_vals[::-1]).reshape(1, -1)
        y_hat = model.predict(x_input)[0]

        preds.append(y_hat)
        history.append(y_hat)

    return np.array(preds)


# ============================================================
# Base data
# ============================================================
BASE = df.copy()
BASE["date"] = pd.to_datetime(BASE["date"], errors="coerce")
BASE = BASE.sort_values("date").reset_index(drop=True)

# ============================================================
# Asset selection
# ============================================================
section(
    "Previsão com Machine Learning",
    "Selecione um ativo, um período de histórico e um modelo para gerar previsões.",
    "🤖",
)

ASSETS_MAP = {
    "Flat do óleo de soja (BRL - C1)": "oleo_flat_brl",
    "Flat do óleo de soja (USD - C1)": "oleo_flat_usd",
    "Flat do farelo de soja (BRL - C1)": "farelo_flat_brl",
    "Flat do farelo de soja (USD - C1)": "farelo_flat_usd",
    "Óleo de soja (BOC1)": "boc1",
    "Farelo de soja (SMC1)": "smc1",
    "Óleo - Prêmio C1": "so-premp-c1",
    "Farelo - Prêmio C1": "sm-premp-c1",
    "Soja (SC1)": "sc1",
    "Milho (CC1)": "cc1",
    "RIN D4": "rin-d4-us",
    "Óleo de palma (FCPOC1)": "fcpoc1",
    "Brent (LCOC1)": "lcoc1",
    "Heating Oil (HOC1)": "hoc1",
    "Dólar": "brl=",
    "Bitcoin": "btc=",
}

target_col, _ = asset_picker_dropdown(
    BASE,
    ASSETS_MAP,
    state_key="ml_target_col",
)
target_label = next(
    (label for label, col in ASSETS_MAP.items() if col == target_col),
    target_col,
)

st.divider()

# ============================================================
# Date range
# ============================================================
section("Período de histórico", "Defina o histórico usado para treinar o modelo.", "📅")
start_date, end_date = date_range_picker(
    BASE["date"],
    state_key="ml_range",
    default_days=365 * 2,
)

date_series = BASE["date"].dt.date
mask = date_series.between(start_date, end_date)
df_view = BASE.loc[mask, ["date", target_col]].dropna().copy()

if df_view.empty:
    st.warning("Sem dados no período selecionado para este ativo.")
    st.stop()

df_view = df_view.sort_values("date").reset_index(drop=True)
df_view.set_index("date", inplace=True)

st.divider()

# ============================================================
# Model & Forecast settings
# ============================================================
section("Configuração do modelo", "Escolha o modelo e os parâmetros de previsão.", "⚙️")

model_options = ["Ridge Regression", "Random Forest"]
if HAS_XGB:
    model_options.append("XGBoost (if available)")

c1, c2, c3 = st.columns([1, 1, 1])

with c1:
    model_name = st.selectbox("Modelo", options=model_options)

with c2:
    n_lags = st.slider(
        "Número de lags (dias) usados como features",
        min_value=5,
        max_value=30,
        value=10,
        help="Cada lag representa o valor do ativo em um dia anterior. "
             "Por exemplo, lag 1 = ontem, lag 2 = anteontem, etc.",
    )

with c3:
    forecast_horizon = st.slider(
        "Horizonte de previsão (dias úteis)",
        min_value=5,
        max_value=45,
        value=22,
        help="Quantidade de dias úteis à frente para prever (máx. 45).",
    )

test_size_fraction = st.slider(
    "Proporção da amostra reservada para teste (backtest)",
    min_value=0.1,
    max_value=0.4,
    value=0.2,
    step=0.05,
    help="Parte final do histórico usada para avaliar o modelo (previsão vs real).",
)

st.divider()

# ============================================================
# Build supervised dataset
# ============================================================
y_series = df_view[target_col].copy()
X, y, idx = make_supervised(y_series, n_lags=n_lags)

if len(y) < 50:
    st.warning("Poucos dados após a criação dos lags. Tente ampliar o período de histórico.")
    st.stop()

# Split train/test by time (no shuffling)
n_samples = len(y)
test_size = max(1, int(n_samples * test_size_fraction))
train_size = n_samples - test_size

X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
idx_train, idx_test = idx[:train_size], idx[train_size:]

# Train model
model = train_model(model_name, X_train, y_train)

# In-sample backtest: predictions on test set
y_pred_test = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred_test)
rmse = mean_squared_error(y_test, y_pred_test, squared=False)
r2 = r2_score(y_test, y_pred_test)

# Build full series for plotting backtest
y_full = y.copy()
y_pred_full = np.full_like(y_full, fill_value=np.nan, dtype=float)
y_pred_full[train_size:] = y_pred_test  # only test window has predictions

dates_full = idx

# ============================================================
# Multi-step forecast into the future
# ============================================================
history_vals = y_series.values  # full history (after filtering by date)
forecast_vals = multi_step_forecast(
    model=model,
    history=history_vals,
    n_lags=n_lags,
    horizon=forecast_horizon,
)

if len(forecast_vals) > 0:
    last_date = df_view.index.max()
    forecast_index = pd.bdate_range(
        last_date, periods=forecast_horizon + 1, freq="B"
    )[1:]  # skip last_date itself
else:
    forecast_index = pd.DatetimeIndex([])

# ============================================================
# Plot: real vs backtest vs forecast
# ============================================================
section("Resultado da previsão", None, "📈")

fig = go.Figure()

# Real prices (full supervised window)
fig.add_trace(
    go.Scatter(
        x=dates_full,
        y=y_full,
        mode="lines",
        name=f"{target_label} - Real",
        line=dict(width=2),
    )
)

# Backtest predictions (only on test window)
fig.add_trace(
    go.Scatter(
        x=dates_full,
        y=y_pred_full,
        mode="lines",
        name=f"{target_label} - Previsão (backtest)",
        line=dict(width=2, dash="dash"),
    )
)

# Future forecast (multi-step)
if len(forecast_vals) > 0:
    fig.add_trace(
        go.Scatter(
            x=forecast_index,
            y=forecast_vals,
            mode="lines+markers",
            name=f"{target_label} - Forecast futuro",
            line=dict(width=2, dash="dot"),
        )
    )

fig.update_layout(
    title=dict(
        text=f"Previsão para {target_label}",
        x=0.0,
        xanchor="left",
        y=0.98,
        yanchor="top",
        pad=dict(b=12),
    ),
    xaxis_title="Data",
    yaxis_title=target_label,
    margin=dict(t=80),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0.0,
    ),
)

st.plotly_chart(fig, use_container_width=True)

# ============================================================
# Metrics & explanations
# ============================================================
st.markdown("### Métricas do modelo (backtest na parte final da amostra)")

c_mae, c_rmse, c_r2 = st.columns(3)
with c_mae:
    st.metric("MAE", f"{mae:,.2f}")
with c_rmse:
    st.metric("RMSE", f"{rmse:,.2f}")
with c_r2:
    st.metric("R²", f"{r2:,.3f}")

st.caption(
    "- **MAE (Mean Absolute Error)**: erro médio absoluto entre previsão e valor real. "
    "Quanto menor, melhor.\n"
    "- **RMSE (Root Mean Squared Error)**: penaliza mais erros grandes. Também quanto menor, melhor.\n"
    "- **R² (Coeficiente de determinação)**: mede quanto da variação do alvo é explicada pelo modelo. "
    "Valores próximos de 1 são bons; **R² negativo** significa que o modelo foi pior do que uma "
    "previsão ingênua usando a média histórica."
)

st.markdown("---")

with st.expander("Entenda o que o modelo está fazendo 🧠", expanded=False):
    st.markdown(
        """
        **1. Lags (defasagens)**  
        - Um *lag* é simplesmente o valor passado da série.  
        - `lag 1` = valor de ontem, `lag 2` = anteontem, e assim por diante.  
        - O modelo aprende padrões do tipo:  
          > "quando o preço dos últimos 10 dias estava assim, o próximo dia tende a ser assado".

        **2. Como o forecasting é feito (multi-step)**  
        - Primeiro, treinamos o modelo para prever **apenas o próximo dia** usando os lags.  
        - Depois, para prever vários dias à frente (*multi-step*), fazemos de forma recursiva:
            1. Usamos o histórico real para prever o dia `t+1`.  
            2. Em seguida, usamos essa previsão como parte dos lags para prever `t+2`.  
            3. Repetimos até o horizonte escolhido (por exemplo, 22 ou 45 dias úteis).

        - Isso é chamado de **multi-step recursive forecast**, e é um padrão comum em séries temporais.

        **3. Backtest vs Forecast futuro**  
        - A parte final do histórico é separada como **janela de teste**.  
        - Nela, comparamos previsão vs valor real (linha tracejada no gráfico).  
        - Depois do último ponto real, mostramos o **forecast futuro**, onde só temos a previsão do modelo.
        """
    )
