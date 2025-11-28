"""Machine Learning – Forecast de Ativos

Nesta página, você pode:
- escolher um ativo como alvo (target),
- selecionar o período de treino,
- escolher variáveis explicativas (features),
- escolher o modelo de Machine Learning,
- treinar o modelo e gerar previsão para frente (multi-step).

O modelo usa lags do próprio ativo (autoregressivo) e as features selecionadas,
faz um split temporal em treino/teste, calcula métricas (MAE, RMSE, R²) e projeta
o preço para alguns dias úteis à frente.
"""

# ============================================================
# Imports & Config
# ============================================================
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

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
BASE = BASE.sort_values("date").reset_index(drop=True)

# ============================================================
# Helpers
# ============================================================
def build_lag_features(
    data: pd.DataFrame,
    target_col: str,
    extra_features: list[str] | None = None,
    n_lags: int = 5,
) -> pd.DataFrame:
    """
    Cria lags do ativo alvo e adiciona features extras.

    target_col: coluna a ser prevista.
    extra_features: outras colunas numéricas a usar como features (nível atual).
    n_lags: número de lags do target a incluir (t-1, t-2, ..., t-n_lags).
    """
    df_model = data[["date", target_col]].copy()

    # Cria lags do próprio ativo (autoregressivo)
    for lag in range(1, n_lags + 1):
        df_model[f"{target_col}_lag{lag}"] = df_model[target_col].shift(lag)

    # Adiciona features extras (nível atual)
    if extra_features:
        for col in extra_features:
            if col in data.columns:
                df_model[col] = data[col].values

    # Remove linhas com NaN (principalmente por causa dos lags iniciais)
    df_model = df_model.dropna().reset_index(drop=True)

    return df_model


def train_test_split_time(
    df_model: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
):
    """Split temporal simples: últimas `test_size` fração vão para teste."""
    n = len(df_model)
    if n < 20:
        raise ValueError("Poucos dados para treinar/testar (menos de 20 linhas).")

    split_idx = int(n * (1 - test_size))
    if split_idx <= 0 or split_idx >= n:
        raise ValueError("Split temporal inválido. Ajuste o tamanho de teste.")

    feature_cols = [c for c in df_model.columns if c not in ("date", target_col)]
    X = df_model[feature_cols].values
    y = df_model[target_col].values

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    dates_train = df_model["date"].iloc[:split_idx]
    dates_test = df_model["date"].iloc[split_idx:]

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        dates_train.reset_index(drop=True),
        dates_test.reset_index(drop=True),
        feature_cols,
    )


def build_model(model_name: str) -> Pipeline:
    """Retorna um Pipeline sklearn de acordo com o modelo escolhido."""
    if model_name == "Linear Regression":
        model = LinearRegression()
        scaler = StandardScaler()
        return Pipeline([("scaler", scaler), ("model", model)])

    if model_name == "Ridge Regression":
        model = Ridge(alpha=1.0, random_state=42)
        scaler = StandardScaler()
        return Pipeline([("scaler", scaler), ("model", model)])

    if model_name == "Random Forest":
        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
        )
        # Random Forest não precisa de scaler, mas não atrapalha
        return Pipeline([("scaler", StandardScaler()), ("model", model)])

    raise ValueError(f"Modelo não suportado: {model_name}")


def recursive_forecast(
    last_row: pd.Series,
    pipeline: Pipeline,
    feature_cols: list[str],
    target_col: str,
    horizon_days: int,
) -> pd.DataFrame:
    """
    Faz previsão multi-step recursiva para frente (horizon_days dias úteis).

    Usa os lags do target; a cada passo, o valor previsto entra como novo lag1,
    e os lags anteriores são "empurrados" (lag1 → lag2, etc.).
    Features exógenas são mantidas constantes com o último valor conhecido.
    """
    # Construir calendário de dias úteis à frente
    last_date = last_row["date"]
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=horizon_days)

    # Vamos trabalhar com um dict mutável das features
    current_features = last_row[feature_cols].copy()

    forecasts = []
    for future_date in future_dates:
        # Previsão para o próximo passo
        X_curr = current_features.values.reshape(1, -1)
        y_pred = pipeline.predict(X_curr)[0]

        forecasts.append({"date": future_date, "forecast": y_pred})

        # Atualiza lags do target dentro de current_features
        # Ex.: target_lag1, target_lag2, ..., target_lagN
        lag_names = [c for c in feature_cols if c.startswith(f"{target_col}_lag")]
        # ordena lag1, lag2, ...
        lag_names_sorted = sorted(
            lag_names,
            key=lambda x: int(x.split("lag")[-1])
        )

        # shift: lag_{k} <- lag_{k-1}, e lag1 recebe o novo y_pred
        # começamos de trás (maior lag)
        for i in range(len(lag_names_sorted) - 1, 0, -1):
            prev_lag = lag_names_sorted[i - 1]
            curr_lag = lag_names_sorted[i]
            current_features[curr_lag] = current_features[prev_lag]

        if lag_names_sorted:
            current_features[lag_names_sorted[0]] = y_pred
        # features exógenas ficam constantes

    return pd.DataFrame(forecasts)


# ============================================================
# UI – Seção: Seleção do ativo e período
# ============================================================
section(
    "Previsão com Machine Learning",
    "Escolha o ativo, período, features e modelo para gerar previsões.",
    "🤖",
)

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

# Período
section(
    "Período de treino",
    "Selecione o intervalo de datas usado para treinar e avaliar o modelo.",
    "📅",
)
start_date, end_date = date_range_picker(
    BASE["date"],
    state_key="ml_range",
    default_days=365 * 3,
)

date_series = BASE["date"].dt.date
mask_period = date_series.between(start_date, end_date)
DATA_PERIOD = BASE.loc[mask_period].copy()

if DATA_PERIOD[target_col].dropna().empty:
    st.warning("Sem dados do ativo selecionado no período escolhido.")
    st.stop()

st.divider()

# ============================================================
# UI – Seção: Features, Modelo e Horizonte
# ============================================================
section(
    "Configuração do modelo",
    "Escolha as variáveis explicativas, o tipo de modelo e o horizonte de previsão.",
    "⚙️",
)

# Features disponíveis: todas numéricas, exceto 'date' e target
numeric_cols = [
    c for c in DATA_PERIOD.columns
    if c not in ("date", target_col) and np.issubdtype(DATA_PERIOD[c].dtype, np.number)
]

default_feats = [col for col in numeric_cols if col in ("brl=", "boc1", "smc1")]
extra_features = st.multiselect(
    "Features adicionais (opcionais)",
    options=numeric_cols,
    default=default_feats,
    help="Features usadas como variáveis explicativas (nível atual). "
         "Além disso, o modelo sempre usa lags do próprio ativo.",
)

c1, c2, c3 = st.columns(3)
with c1:
    model_name = st.selectbox(
        "Modelo de Machine Learning",
        options=["Linear Regression", "Ridge Regression", "Random Forest"],
        index=1,
    )
with c2:
    n_lags = st.slider(
        "Número de lags do ativo",
        min_value=1,
        max_value=10,
        value=5,
        step=1,
        help="Número de dias defasados do próprio ativo usados como features.",
    )
with c3:
    forecast_days = st.slider(
        "Horizonte de previsão (dias úteis)",
        min_value=5,
        max_value=60,
        value=22,
        step=1,
        help="Quantidade de dias úteis para projetar à frente (≈ 22 dias úteis ~ 1 mês).",
    )

st.markdown("---")

# ============================================================
# Build dataset & train model
# ============================================================
try:
    df_model = build_lag_features(
        DATA_PERIOD,
        target_col=target_col,
        extra_features=extra_features,
        n_lags=n_lags,
    )
except Exception as e:
    st.error(f"Erro ao construir o dataset do modelo: {e}")
    st.stop()

if df_model.empty:
    st.warning("Dataset do modelo ficou vazio após a criação de lags/features.")
    st.stop()

try:
    (
        X_train,
        X_test,
        y_train,
        y_test,
        dates_train,
        dates_test,
        feature_cols,
    ) = train_test_split_time(df_model, target_col=target_col, test_size=0.2)
except Exception as e:
    st.error(f"Erro ao fazer o split temporal de treino/teste: {e}")
    st.stop()

pipeline = build_model(model_name)

with st.spinner("Treinando o modelo..."):
    pipeline.fit(X_train, y_train)
    y_pred_test = pipeline.predict(X_test)

# ============================================================
# Métricas
# ============================================================
mae = mean_absolute_error(y_test, y_pred_test)
rmse = mean_squared_error(y_test, y_pred_test, squared=False)
r2 = r2_score(y_test, y_pred_test)

section(
    "Desempenho do modelo (dados de teste)",
    "As métricas abaixo são calculadas na parte final do período (hold-out temporal).",
    "📈",
)

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("MAE", f"{mae:,.2f}")
with c2:
    st.metric("RMSE", f"{rmse:,.2f}")
with c3:
    st.metric("R²", f"{r2:,.3f}")

st.caption(
    "- **MAE (Mean Absolute Error)**: erro médio absoluto, na mesma unidade do ativo. "
    "Quanto menor, melhor.\n"
    "- **RMSE (Root Mean Squared Error)**: similar ao MAE, mas penaliza mais erros grandes, "
    "por elevar ao quadrado antes da média.\n"
    "- **R² (coeficiente de determinação)**: mede a fração da variância explicada pelo modelo. "
    "Valores próximos de 1 indicam bom ajuste; valores próximos de 0 indicam desempenho "
    "similar a prever a média; valores **negativos** indicam que o modelo está pior que "
    "um modelo ingênuo que sempre prevê a média histórica do período de teste."
)

if r2 < 0:
    st.warning(
        "O R² ficou negativo, indicando que o modelo está pior do que prever a média "
        "no conjunto de teste. Considere revisar o período, as features ou o tipo de modelo."
    )

st.markdown("---")

# ============================================================
# Forecast multi-step
# ============================================================
section(
    "Previsão para frente",
    "Projeção de preços usando o modelo treinado (multi-step recursiva).",
    "🔮",
)

# Última linha disponível no df_model, usada como base para previsão
last_row = df_model.iloc[-1].copy()

try:
    df_forecast = recursive_forecast(
        last_row=last_row,
        pipeline=pipeline,
        feature_cols=feature_cols,
        target_col=target_col,
        horizon_days=forecast_days,
    )
except Exception as e:
    st.error(f"Erro ao gerar a previsão para frente: {e}")
    st.stop()

# ============================================================
# Plot – histórico + previsão
# ============================================================
# Histórico no período selecionado (usando a série original)
hist_view = DATA_PERIOD[["date", target_col]].dropna().copy()

# Para visualizar melhor, pode ser interessante focar nos últimos N dias históricos
N_HISTORY_DAYS = 250
if len(hist_view) > N_HISTORY_DAYS:
    hist_view = hist_view.iloc[-N_HISTORY_DAYS:].copy()

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=hist_view["date"],
        y=hist_view[target_col],
        mode="lines",
        name=f"Histórico – {target_label}",
    )
)

fig.add_trace(
    go.Scatter(
        x=df_forecast["date"],
        y=df_forecast["forecast"],
        mode="lines+markers",
        name=f"Previsão ({model_name})",
    )
)

fig.update_layout(
    title=dict(
        text=f"Previsão de {target_label} ({model_name})",
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

st.caption(
    f"A linha azul mostra o histórico recente de **{target_label}**; "
    "a linha de previsão estende a série para frente com base no modelo treinado, "
    f"usando lags do ativo e as features selecionadas. Horizonte: {forecast_days} dias úteis."
)
