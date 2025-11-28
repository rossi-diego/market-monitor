"""Forecast dashboard for commodities, FX and crypto.

This page lets the user:
- select an asset from the data pipeline,
- choose a historical lookback window (training sample),
- define a forecast horizon (in business days),
- select a confidence level,

and then builds a simple geometric-random-walk forecast based on
historical log-returns (drift + volatility), with an expected path
and a confidence band for the final price.
"""

# ============================================================
# Imports & Config
# ============================================================
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

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

# ============================================================
# Asset map (same style as other pages)
# ============================================================
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

# ============================================================
# 1) Asset selection
# ============================================================
section(
    "Previsão de ativos",
    "Selecione o ativo, a janela histórica e o horizonte de previsão.",
    "🔮",
)

close_col, _assets = asset_picker_dropdown(
    BASE,
    ASSETS_MAP,
    state_key="forecast_close_col",
)
st.divider()

asset_label = next(
    (label for label, col in ASSETS_MAP.items() if col == close_col),
    close_col,
)

if close_col not in BASE.columns:
    st.warning(f"A coluna selecionada ('{close_col}') não está disponível nos dados.")
    st.stop()

# Filtra série do ativo
series_df = (
    BASE[["date", close_col]]
    .dropna()
    .sort_values("date")
    .reset_index(drop=True)
)

if series_df.empty:
    st.warning("Sem dados disponíveis para o ativo selecionado.")
    st.stop()

# ============================================================
# 2) Parameters – lookback & horizon
# ============================================================
section("Configurações de previsão", None, "⚙️")

c1, c2, c3 = st.columns([1, 1, 1])

with c1:
    lookback_days = st.slider(
        "Janela histórica (dias úteis)",
        min_value=60,
        max_value=750,
        value=252,
        step=10,
        help="Número de dias úteis usados para estimar drift e volatilidade (log-retornos).",
    )

with c2:
    horizon_days = st.slider(
        "Horizonte de previsão (dias úteis)",
        min_value=5,
        max_value=180,
        value=60,
        step=5,
        help="Número de dias úteis à frente para projetar o preço.",
    )

with c3:
    conf_label = st.selectbox(
        "Nível de confiança",
        options=["68%", "95%"],
        index=1,
        help="Usado para o intervalo de confiança do preço projetado.",
    )

conf_map = {"68%": 1.0, "95%": 1.96}
z = conf_map[conf_label]

st.caption(
    "O modelo usa log-retornos históricos para estimar drift (média) e volatilidade da série. "
    "A projeção assume um passeio aleatório geométrico (retornos i.i.d.), gerando um "
    "preço esperado e um intervalo de confiança aproximado para o final do horizonte."
)

st.divider()

# ============================================================
# 3) Prepare training sample
# ============================================================
# Consider only the last `lookback_days` business days (or all, if shorter)
series_df = series_df.set_index("date").asfreq("B")  # força calendário de dias úteis
series_df[close_col] = series_df[close_col].ffill()
series_df = series_df.dropna(subset=[close_col])

if len(series_df) < lookback_days:
    train = series_df.copy()
else:
    train = series_df.iloc[-lookback_days:].copy()

train = train.reset_index().rename(columns={"index": "date"})

# Compute log-returns
train["log_ret"] = np.log(train[close_col] / train[close_col].shift(1))
train = train.dropna(subset=["log_ret"])

if train.empty:
    st.warning("Não foi possível calcular log-retornos na janela selecionada.")
    st.stop()

mu = train["log_ret"].mean()
sigma = train["log_ret"].std(ddof=1)

last_date = train["date"].iloc[-1]
last_price = train[close_col].iloc[-1]

# ============================================================
# 4) Build forecast path & confidence band
# ============================================================
# Future business-day calendar
future_dates = pd.bdate_range(
    last_date + pd.offsets.BDay(1),
    periods=horizon_days,
    name="date",
)

steps = np.arange(1, horizon_days + 1)

# Expected log-return over t steps
exp_log_returns = mu * steps
exp_path = last_price * np.exp(exp_log_returns)

# Confidence band for FINAL price (horizon)
# Var(sum log-ret) = sigma^2 * T
total_mu = mu * horizon_days
total_sigma = sigma * np.sqrt(horizon_days)

final_expected = last_price * np.exp(total_mu)
final_lower = last_price * np.exp(total_mu - z * total_sigma)
final_upper = last_price * np.exp(total_mu + z * total_sigma)

# ============================================================
# 5) Plot – history + expected path + band at horizon
# ============================================================
section("Resultado da previsão", None, "📈")

hist_df = series_df.reset_index().rename(columns={"index": "date"})
hist_df = hist_df[hist_df["date"] <= last_date]

fig = go.Figure()

# Historical prices
fig.add_trace(
    go.Scatter(
        x=hist_df["date"],
        y=hist_df[close_col],
        mode="lines",
        name=f"{asset_label} (histórico)",
        line=dict(width=2),
        hovertemplate="%{x|%Y-%m-%d}<br>Preço: %{y:.2f}<extra></extra>",
    )
)

# Forecast expected path
fig.add_trace(
    go.Scatter(
        x=future_dates,
        y=exp_path,
        mode="lines",
        name="Preço esperado",
        line=dict(width=2, dash="dash"),
        hovertemplate="%{x|%Y-%m-%d}<br>Preço esperado: %{y:.2f}<extra></extra>",
    )
)

# Confidence band at horizon (plot as horizontal band from last_date to final_date)
final_date = future_dates[-1]

fig.add_trace(
    go.Scatter(
        x=[final_date, final_date],
        y=[final_lower, final_upper],
        mode="lines",
        name=f"Intervalo {conf_label}",
        line=dict(width=8, dash="solid"),
        opacity=0.25,
        hovertemplate=(
            f"{conf_label} IC<br>Inferior: %{y[0]:.2f}<br>Superior: %{y[1]:.2f}<extra></extra>"
        ),
        showlegend=True,
    )
)

fig.update_layout(
    title=dict(
        text=f"Previsão de {asset_label}",
        x=0.0,
        xanchor="left",
        y=0.98,
        yanchor="top",
        pad=dict(b=12),
    ),
    xaxis=dict(title="Data"),
    yaxis=dict(title=asset_label),
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
# 6) Summary stats
# ============================================================
st.markdown("### Resumo estatístico")

col_a, col_b, col_c, col_d = st.columns(4)

ret_annualized = mu * 252  # log-retorno anualizado
vol_annualized = sigma * np.sqrt(252)

with col_a:
    st.metric("Último preço", f"{last_price:,.2f}")

with col_b:
    st.metric("Preço esperado (horizonte)", f"{final_expected:,.2f}")

with col_c:
    st.metric(
        f"Intervalo {conf_label} (mín)",
        f"{final_lower:,.2f}",
    )

with col_d:
    st.metric(
        f"Intervalo {conf_label} (máx)",
        f"{final_upper:,.2f}",
    )

st.caption(
    f"Drift anualizado (log): {ret_annualized:.2%} | "
    f"Volatilidade anualizada: {vol_annualized:.2%} "
    f"(baseado em {len(train)} observações de log-retorno)."
)

with st.expander("Detalhes do modelo e hipóteses"):
    st.markdown(
        """
- Os log-retornos diários são calculados como \\( r_t = \\ln(P_t / P_{t-1}) \\).
- A média \\(\\mu\\) e o desvio-padrão \\(\\sigma\\) desses log-retornos são estimados na janela histórica selecionada.
- A projeção assume retornos i.i.d. (passeio aleatório geométrico), de forma que:
  - Drift acumulado em \\(T\\) dias: \\( \\mu_T = \\mu \\cdot T \\)
  - Volatilidade acumulada: \\( \\sigma_T = \\sigma \\sqrt{T} \\)
- O intervalo de confiança usa a aproximação normal sobre a soma dos log-retornos:
  \\[
  \\ln(P_T / P_0) \\sim \\mathcal{N}(\\mu_T, \\sigma_T^2)
  \\]
- Isso gera um preço esperado e um intervalo de confiança para o preço ao final do horizonte.
"""
    )
