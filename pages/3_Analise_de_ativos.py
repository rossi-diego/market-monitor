# ============================================================
# Imports & Config
# ============================================================
import pandas as pd
import streamlit as st
import datetime as dt

from src.data_pipeline import df
from src.visualization import plot_price_rsi_plotly
from src.utils import apply_theme, asset_picker, asset_picker_dropdown, date_range_picker, ma_picker, rsi, section

# --- Theme
apply_theme()

# ============================================================
# Base
# ============================================================
BASE = df.copy()
# (se o data_pipeline já garante datetime, a linha abaixo é opcional)
BASE["date"] = pd.to_datetime(BASE["date"], errors="coerce")

# ============================================================
# Seleção do ativo
# ============================================================
section("Selecione o ativo", "Favoritos + busca", "🧭")

ASSETS_MAP = {
    "Flat do óleo de soja (BRL - C1)": "oleo_flat_brl",
    "Flat do óleo de soja (USD - C1)": "oleo_flat_usd",
    "Óleo de soja (BOC1)": "boc1",
    "Flat do farelo de soja (BRL - C1)": "farelo_flat_brl",
    "Flat do farelo de soja (USD - C1)": "farelo_flat_usd",
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
}

CLOSE, _assets = asset_picker_dropdown(
    BASE,
    ASSETS_MAP,
    state_key="close_col",
    # favorites opcional; se quiser customizar a ordem:
    # favorites=["Óleo de soja (BOC1)", "Flat do óleo de soja (BRL - C1)", "Farelo de soja (SMC1)"]
)
st.divider()

# ============================================================
# Período
# ============================================================
section("Selecione o período do gráfico", "Use presets ou ajuste no slider", "🗓️")
start_date, end_date = date_range_picker(BASE["date"], state_key="range", default_days=365)

# ============================================================
# Parâmetros do gráfico
# ============================================================
section("Parâmetros", None, "⚙️")
ma_window = ma_picker(options=(20, 50, 200), default=90, state_key="ma_window")
st.caption(f"Média móvel selecionada: **{ma_window}** períodos")
st.divider()

# ============================================================
# Filtra e plota
# ============================================================
mask = (BASE["date"].dt.date >= start_date) & (BASE["date"].dt.date <= end_date)
df_view = BASE.loc[mask, ["date", CLOSE]].dropna().copy()

if df_view.empty:
    st.info("Sem dados no período selecionado.")
else:
    # RSI é calculado dentro da função (mais clean)
    fig = plot_price_rsi_plotly(
        df_view,
        title=CLOSE.upper(),
        date_col="date",
        close_col=CLOSE,
        rsi_col=None,
        rsi_fn=rsi,
        rsi_len=14,
        ma_window=ma_window,
        show_bollinger=False,
        bands_window=20,
        bands_sigma=2.0,
    )
    # gap extra no título (opcional)
    fig.update_layout(
        title=dict(text=CLOSE.upper(), x=0.0, xanchor="left", y=0.98, yanchor="top", pad=dict(b=12)),
        margin=dict(t=80)
    )
    st.plotly_chart(fig, use_container_width=True)
