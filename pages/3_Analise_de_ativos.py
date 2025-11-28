"""Price + RSI dashboard for commodity and FX assets.

This page lets the user:
- select an asset (oil, meal, palm, FX, crypto, etc.),
- choose a date range,
- configure a moving average window,
and then plots Price + RSI using Plotly.
"""

# ============================================================
# Imports & Config
# ============================================================
import pandas as pd
import streamlit as st

from src.data_pipeline import df
from src.visualization import plot_price_rsi_plotly
from src.utils import (
    apply_theme,
    asset_picker,
    asset_picker_dropdown,
    date_range_picker,
    ma_picker,
    rsi,
    section,
)

# --- Theme
apply_theme()

# ============================================================
# Base data
# ============================================================
BASE = df.copy()
# If `date` is already datetime in the pipeline, this is just a safeguard.
BASE["date"] = pd.to_datetime(BASE["date"], errors="coerce")

# ============================================================
# Asset selection
# ============================================================
section(
    "Selecione o ativo",
    "Favoritos abaixo, caso queira outro ativo, selecione no dropdown",
    "🧭",
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

close_col, _assets = asset_picker_dropdown(
    BASE,
    ASSETS_MAP,
    state_key="close_col",
    # Optionally: favorites=["Óleo de soja (BOC1)", ...]
)
st.divider()

# Derive a nice label for the selected asset (for the chart title)
asset_label = next(
    (label for label, col in ASSETS_MAP.items() if col == close_col),
    close_col,
)

# ============================================================
# Date range
# ============================================================
section("Selecione o período do gráfico", "Use presets ou ajuste no slider", "🗓️")
start_date, end_date = date_range_picker(
    BASE["date"],
    state_key="range",
    default_days=365,
)

# ============================================================
# Chart parameters
# ============================================================
section("Parâmetros", None, "⚙️")
ma_window = ma_picker(
    options=(20, 50, 200),
    default=90,
    state_key="ma_window",
)
st.caption(f"Média móvel selecionada: **{ma_window}** períodos")
st.divider()

# ============================================================
# Filter & plot
# ============================================================
if close_col not in BASE.columns:
    st.warning(f"A coluna selecionada ('{close_col}') não está disponível nos dados.")
else:
    # Filter by date range
    date_series = BASE["date"].dt.date
    mask = date_series.between(start_date, end_date)

    df_view = BASE.loc[mask, ["date", close_col]].dropna().copy()

    if df_view.empty:
        st.info("Sem dados no período selecionado.")
    else:
        # RSI is computed inside the plotting function (clean API)
        fig = plot_price_rsi_plotly(
            df_view,
            title=asset_label,
            date_col="date",
            close_col=close_col,
            rsi_col=None,
            rsi_fn=rsi,
            rsi_len=14,
            ma_window=ma_window,
            show_bollinger=False,
            bands_window=20,
            bands_sigma=2.0,
        )

        # Small tweak on title position and margin
        fig.update_layout(
            title=dict(
                text=asset_label,
                x=0.0,
                xanchor="left",
                y=0.98,
                yanchor="top",
                pad=dict(b=12),
            ),
            margin=dict(t=80),
        )

        st.plotly_chart(fig, use_container_width=True)
