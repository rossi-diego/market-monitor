"""Machine Learning – Price Forecast

This page allows the user to:
- select a target asset (price column),
- choose a date range,
- choose explanatory variables (features),
and then fit a simple Ridge Regression model to forecast the asset.

The model:
- uses only past data within the selected date range,
- performs a time-based train/test split (80% train, 20% test),
- evaluates the model on the test set (MAE, RMSE, R²),
- generates a one-step-ahead forecast using the last available row of features.
"""

# ============================================================
# Imports & Config
# ============================================================
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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
# Base data
# ============================================================
BASE = df.copy()
BASE["date"] = pd.to_datetime(BASE["date"], errors="coerce")
BASE = BASE.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

# ============================================================
# Asset map (target options)
# ============================================================
ASSETS_MAP = {
    "Flat do óleo de soja (BRL - C1)": "oleo_flat_brl",
    "Flat do óleo de soja (USD - C1)": "oleo_flat_usd",
    "Flat do farelo de soja (BRL - C1)": "farelo_flat_brl",
    "Flat do farelo de soja (USD - C1)": "farelo_flat_usd",
    "Óleo de soja (BOC1)": "boc1",
    "Farelo de soja (SMC1)": "smc1",
    "Soja (SC1)": "sc1",
    "Milho (CC1)": "cc1",
    "Óleo de palma (FCPOC1)": "fcpoc1",
    "Brent (LCOC1)": "lcoc1",
    "Heating Oil (HOC1)": "hoc1",
    "Dólar": "brl=",
    "Bitcoin": "btc=",
}

# ============================================================
# 1) Target selection
# ============================================================
section(
    "Selecione o ativo alvo (target)",
    "Esse será o preço que o modelo tentará prever.",
    "🎯",
)

target_col, _ = asset_picker_dropdown(
    BASE,
    ASSETS_MAP,
    state_key="ml_target_col",
)
target_label = next(
    (label for label, col in ASSETS_MAP.items() if col == target_col),
    target_col,
)

if target_col not in BASE.columns:
    st.error(f"A coluna de target '{target_col}' não existe em BASE.")
    st.stop()

st.divider()

# ============================================================
# 2) Date range
# ============================================================
section(
    "Período de treino/validação",
    "O modelo será treinado somente com os dados dentro desse intervalo.",
    "🗓️",
)
start_date, end_date = date_range_picker(
    BASE["date"], state_key="ml_range", default_days=365 * 3
)

date_series = BASE["date"].dt.date
mask = date_series.between(start_date, end_date)
df_view = BASE.loc[mask].copy()

if df_view.empty:
    st.warning("Sem dados no período selecionado.")
    st.stop()

st.write(
    f"Período selecionado: **{start_date}** até **{end_date}** "
    f"({len(df_view)} observações)."
)
st.divider()

# ============================================================
# 3) Feature selection
# ============================================================
section(
    "Selecione as variáveis explicativas (features)",
    "Escolha quais colunas o modelo pode usar como informação para prever o ativo.",
    "🧠",
)

# Candidate features = numeric columns, excluding date and the target
numeric_cols = df_view.select_dtypes(include=[np.number]).columns.tolist()
feature_candidates = [c for c in numeric_cols if c != target_col]

if not feature_candidates:
    st.error("Não há colunas numéricas disponíveis para usar como features.")
    st.stop()

default_features = [
    c for c in feature_candidates
    if c in {"boc1", "smc1", "brl=", "oleo_flat_usd", "farelo_flat_usd"}
]

features_selected = st.multiselect(
    "Features para o modelo",
    options=feature_candidates,
    default=default_features or feature_candidates[:3],
    help="Selecione uma ou mais variáveis que o modelo poderá usar como input.",
)

if not features_selected:
    st.warning("Selecione pelo menos uma feature.")
    st.stop()

st.caption(
    "Dica: incluir variáveis relacionadas (ex.: câmbio, outros contratos, flats) "
    "geralmente ajuda o modelo a capturar melhor a dinâmica de preço."
)
st.divider()

# ============================================================
# 4) Treino do modelo
# ============================================================
section(
    "Treino do modelo de Regressão (Ridge)",
    "O modelo é treinado usando uma divisão temporal: 80% treino / 20% teste.",
    "⚙️",
)

# Monta X e y
df_model = df_view.dropna(subset=[target_col] + features_selected).copy()
if df_model.empty:
    st.warning(
        "Após remover valores ausentes, não sobraram linhas suficientes para o treino."
    )
    st.stop()

X = df_model[features_selected].values
y = df_model[target_col].values

n_samples = len(df_model)
if n_samples < 30:
    st.warning(
        f"Apenas {n_samples} observações após o filtro. "
        "Isso pode não ser suficiente para um modelo robusto."
    )

# Time-based split: 80% train, 20% test
split_idx = int(n_samples * 0.8)
if split_idx == 0 or split_idx >= n_samples:
    st.error("Não foi possível criar uma divisão treino/teste adequada.")
    st.stop()

X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]
dates_train = df_model["date"].iloc[:split_idx]
dates_test = df_model["date"].iloc[split_idx:]

alpha = st.slider(
    "Parâmetro de regularização (alpha)",
    min_value=0.0,
    max_value=10.0,
    value=1.0,
    step=0.1,
)

model = Ridge(alpha=alpha)
model.fit(X_train, y_train)

# Previsão no conjunto de teste
y_pred = model.predict(X_test)

# Métricas
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

st.subheader("Desempenho no conjunto de teste")
st.write(f"- MAE:  **{mae:,.4f}**")
st.write(f"- RMSE: **{rmse:,.4f}**")
st.write(f"- R²:   **{r2:,.4f}**")

st.caption(
    "Treino/teste são separados no tempo (sem embaralhar), "
    "para refletir melhor o uso real do modelo em dados futuros."
)

st.divider()

# ============================================================
# 5) Gráfico – valor real vs previsto no teste
# ============================================================
section(
    "Gráfico – valor real vs previsto (conjunto de teste)",
    None,
    "📈",
)

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=dates_test,
        y=y_test,
        mode="lines",
        name="Real",
    )
)

fig.add_trace(
    go.Scatter(
        x=dates_test,
        y=y_pred,
        mode="lines",
        name="Previsto (modelo)",
    )
)

fig.update_layout(
    title=dict(
        text=f"{target_label}: valores reais vs previstos (teste)",
        x=0.0,
        xanchor="left",
        y=0.98,
        yanchor="top",
        pad=dict(b=12),
    ),
    xaxis_title="Data",
    yaxis_title=target_label,
    margin=dict(t=80),
)

st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 6) Previsão one-step-ahead
# ============================================================
section(
    "Previsão para o próximo ponto (one-step-ahead)",
    "Usa a última linha de features disponível no período selecionado.",
    "🔮",
)

last_row = df_model.iloc[[-1]]  # mantém DataFrame
X_last = last_row[features_selected].values
next_pred = model.predict(X_last)[0]

st.write(
    f"Última data disponível no período: **{last_row['date'].iloc[0].date()}**"
)
st.write(
    f"Último valor observado de **{target_label}**: "
    f"**{last_row[target_col].iloc[0]:,.4f}**"
)
st.write(
    f"Previsão do modelo para o próximo ponto (baseada nos mesmos features): "
    f"**{next_pred:,.4f}**"
)

st.caption(
    "Essa previsão é ilustrativa: ela assume que as features usadas permanecem "
    "compatíveis com o último ponto observado. Para previsões de múltiplos passos "
    "à frente, seria necessário um modelo de séries temporais mais estruturado "
    "(ex.: ARIMA/SARIMAX, modelos com lags, etc.)."
)
