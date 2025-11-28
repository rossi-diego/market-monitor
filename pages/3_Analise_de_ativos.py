"""Price + RSI dashboard for commodity and FX assets.

This page lets the user:
- select an asset (oil, meal, palm, FX, crypto, etc.),
- choose a date range,
- configure a moving average window,
and then plots Price + RSI using Plotly.

If a second asset is selected for comparison, the chart switches to a
price-only view (no moving average and no RSI), with two y-axes.
"""

# ============================================================
# Imports & Config
# ============================================================
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

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

# First asset
close_col, _assets = asset_picker_dropdown(
    BASE,
    ASSETS_MAP,
    state_key="close_col",
)
st.divider()

# Nice label for first asset
asset_label = next(
    (label for label, col in ASSETS_MAP.items() if col == close_col),
    close_col,
)

# ============================================================
# Optional: second asset for comparison
# ============================================================
section(
    "Comparação",
    "Opcional: selecione um segundo ativo para comparar no mesmo gráfico.",
    "📊",
)
compare_two = st.checkbox(
    "Comparar com segundo ativo",
    value=False,
    key="compare_two_assets",
)

second_col = None
second_label = None

if compare_two:
    # Build list of labels, excluding the first asset (optional)
    asset_labels = list(ASSETS_MAP.keys())
    asset_labels_no_first = [
        lbl for lbl in asset_labels if ASSETS_MAP[lbl] != close_col
    ]

    second_label = st.selectbox(
        "Segundo ativo",
        options=asset_labels_no_first,
        key="second_asset_select",
    )

    second_col = ASSETS_MAP[second_label]

st.divider()

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

if not compare_two:
    # Only show MA / RSI parameters in single-asset mode
    ma_window = ma_picker(
        options=(20, 50, 200),
        default=90,
        state_key="ma_window",
    )
    st.caption(f"Média móvel selecionada: **{ma_window}** períodos")
else:
    st.caption(
        "Média móvel e RSI desativados no modo de comparação entre dois ativos."
    )

st.divider()

# ============================================================
# Filter & plot
# ============================================================
if close_col not in BASE.columns:
    st.warning(f"A coluna selecionada ('{close_col}') não está disponível nos dados.")
else:
    date_series = BASE["date"].dt.date
    mask = date_series.between(start_date, end_date)

    # -------------------------
    # Single-asset mode (Price + RSI)
    # -------------------------
    if not compare_two:
        df_view = BASE.loc[mask, ["date", close_col]].dropna().copy()

        if df_view.empty:
            st.info("Sem dados no período selecionado.")
        else:
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

    # -------------------------
    # Two-asset mode (price-only comparison)
    # -------------------------
    else:
        if not second_col:
            st.info("Selecione um segundo ativo para comparação.")
        elif second_col not in BASE.columns:
            st.warning(
                f"A coluna do segundo ativo selecionado ('{second_col}') não está disponível nos dados."
            )
        else:
            cols = ["date", close_col, second_col]

            # Como tratar datas faltantes entre os dois ativos
            merge_mode = st.radio(
                "Tratamento de datas faltantes",
                options=(
                    "Datas em comum (sem preenchimento)",
                    "Preencher pequenos gaps com último valor (ffill)",
                ),
                key="merge_mode",
            )

            if merge_mode.startswith("Datas em comum"):
                # Usa apenas dias em que os DOIS ativos têm preço
                tmp = (
                    BASE.loc[mask, cols]
                    .dropna(subset=cols)  # exige date, close_col e second_col
                    .copy()
                )
            else:
                # Reindexa em calendário de dias úteis e faz ffill limitado
                GAP_LIMIT = 3  # máx. dias consecutivos de preenchimento

                tmp = (
                    BASE.loc[mask, cols]
                    .dropna(subset=["date"])
                    .copy()
                    .sort_values("date")
                )

                if tmp.empty:
                    tmp = pd.DataFrame(columns=cols)
                else:
                    idx = pd.bdate_range(
                        tmp["date"].min(), tmp["date"].max(), name="date"
                    )

                    tmp = (
                        tmp.set_index("date")
                        .reindex(idx)
                        .ffill(limit=GAP_LIMIT)
                        .dropna(how="all")  # remove linhas totalmente vazias
                        .reset_index()
                        .rename(columns={"index": "date"})
                    )

                    # Garante que ambos tenham valor depois do preenchimento
                    tmp = tmp.dropna(subset=[close_col, second_col])

            df_view = tmp

            if df_view.empty:
                st.info(
                    "Sem dados no período selecionado (após tratamento de datas)."
                )
            else:
                # --- Normalization option (index = 100 at start) ---
                normalize = st.checkbox(
                    "Normalizar ambos os ativos (início do período = 100)",
                    value=False,
                    key="normalize_compare",
                )

                yaxis_title_left = asset_label
                yaxis_title_right = second_label

                if normalize:
                    for col in [close_col, second_col]:
                        series = df_view[col].dropna()
                        if not series.empty:
                            base = series.iloc[0]
                            if base != 0:
                                df_view[col] = df_view[col] / base * 100

                    yaxis_title_left = (
                        f"{asset_label} (índice, início = 100)"
                    )
                    yaxis_title_right = (
                        f"{second_label} (índice, início = 100)"
                    )

                # --- Plot two-asset comparison (price-only) ---
                fig = go.Figure()

                # First asset (left y-axis)
                fig.add_trace(
                    go.Scatter(
                        x=df_view["date"],
                        y=df_view[close_col],
                        mode="lines",
                        name=asset_label,
                        yaxis="y1",
                    )
                )

                # Second asset (right y-axis)
                fig.add_trace(
                    go.Scatter(
                        x=df_view["date"],
                        y=df_view[second_col],
                        mode="lines",
                        name=second_label,
                        yaxis="y2",
                    )
                )

                fig.update_layout(
                    title=dict(
                        text=f"{asset_label} vs {second_label}",
                        x=0.0,
                        xanchor="left",
                        y=0.98,
                        yanchor="top",
                        pad=dict(b=12),
                    ),
                    xaxis=dict(title="Data"),
                    yaxis=dict(
                        title=yaxis_title_left,
                        side="left",
                        showgrid=True,
                    ),
                    yaxis2=dict(
                        title=yaxis_title_right,
                        side="right",
                        overlaying="y",
                        showgrid=False,
                    ),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="left",
                        x=0.0,
                    ),
                    margin=dict(t=80),
                )

                st.plotly_chart(fig, use_container_width=True)
