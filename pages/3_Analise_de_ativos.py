"""
Asset Analysis Page

Features:
- Select one asset to analyze (Price + RSI chart)
- OR compare two assets side-by-side
- Configure date ranges, moving averages, and normalization options
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
    asset_picker_dropdown,
    date_range_picker,
    ma_picker,
    rsi,
    section,
)

# Apply theme
apply_theme()

# ============================================================
# Configuration
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
    "Gold": "gcc1",
    "Silver": "sagc1",
}

# ============================================================
# Helper Functions
# ============================================================
def prepare_base_data(dataframe):
    """Prepare base dataframe with datetime conversion."""
    base = dataframe.copy()
    base["date"] = pd.to_datetime(base["date"], errors="coerce")
    return base


def get_asset_label(column_name, assets_map):
    """Get friendly label for asset column name."""
    return next(
        (label for label, col in assets_map.items() if col == column_name),
        column_name,
    )


def plot_single_asset(data, asset_col, asset_label, ma_window):
    """Create price + RSI chart for single asset."""
    if data.empty:
        st.info("Sem dados no período selecionado.")
        return

    fig = plot_price_rsi_plotly(
        data,
        title=asset_label,
        date_col="date",
        close_col=asset_col,
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


def prepare_comparison_data(base, cols, mask, merge_mode):
    """Prepare data for two-asset comparison with gap handling."""
    if merge_mode == "Datas em comum (sem preenchimento)":
        # Only use dates where both assets have prices
        return base.loc[mask, cols].dropna(subset=cols).copy()

    # Fill small gaps with forward fill (limited)
    GAP_LIMIT = 3

    tmp = base.loc[mask, cols].dropna(subset=["date"]).copy().sort_values("date")

    if tmp.empty:
        return pd.DataFrame(columns=cols)

    # Create business day index and forward fill gaps
    idx = pd.bdate_range(tmp["date"].min(), tmp["date"].max(), name="date")

    tmp = (
        tmp.set_index("date")
        .reindex(idx)
        .ffill(limit=GAP_LIMIT)
        .dropna(how="all")
        .reset_index()
        .rename(columns={"index": "date"})
    )

    # Ensure both columns have values after filling
    return tmp.dropna(subset=[c for c in cols if c != "date"])


def normalize_comparison_data(data, col1, col2):
    """Normalize both columns to start at 100."""
    for col in [col1, col2]:
        series = data[col].dropna()
        if not series.empty:
            base_value = series.iloc[0]
            if base_value != 0:
                data[col] = data[col] / base_value * 100
    return data


def plot_comparison_chart(data, col1, col2, label1, label2, normalized):
    """Create comparison chart for two assets."""
    fig = go.Figure()

    # Add traces
    fig.add_trace(
        go.Scatter(
            x=data["date"],
            y=data[col1],
            mode="lines",
            name=label1,
            yaxis="y1",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=data["date"],
            y=data[col2],
            mode="lines",
            name=label2,
            yaxis="y1" if normalized else "y2",
        )
    )

    # Layout configuration
    y1_title = "Índice (início = 100)" if normalized else label1
    y2_title = label2

    layout = dict(
        title=dict(
            text=f"{label1} vs {label2}",
            x=0.0,
            xanchor="left",
            y=0.98,
            yanchor="top",
            pad=dict(b=12),
        ),
        xaxis=dict(title="Data"),
        yaxis=dict(
            title=y1_title,
            side="left",
            showgrid=True,
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

    # Add second y-axis if not normalized
    if not normalized:
        layout["yaxis2"] = dict(
            title=y2_title,
            side="right",
            overlaying="y",
            showgrid=False,
        )

    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# Main Page Logic
# ============================================================
def main():
    """Main page logic."""
    # Prepare data
    BASE = prepare_base_data(df)

    # ============================================================
    # Asset Selection
    # ============================================================
    section(
        "Selecione o ativo",
        "Favoritos abaixo, caso queira outro ativo, selecione no dropdown",
        "🧭",
    )

    close_col, _assets = asset_picker_dropdown(
        BASE,
        ASSETS_MAP,
        state_key="close_col",
    )

    asset_label = get_asset_label(close_col, ASSETS_MAP)
    st.divider()

    # ============================================================
    # Optional: Two-Asset Comparison
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
        # Build list excluding first asset
        asset_labels = list(ASSETS_MAP.keys())
        available_for_comparison = [
            lbl for lbl in asset_labels if ASSETS_MAP[lbl] != close_col
        ]

        second_label = st.selectbox(
            "Segundo ativo",
            options=available_for_comparison,
            key="second_asset_select",
        )
        second_col = ASSETS_MAP[second_label]

    st.divider()

    # ============================================================
    # Date Range Selection
    # ============================================================
    section("Selecione o período do gráfico", "Use presets ou ajuste no slider", "🗓️")
    start_date, end_date = date_range_picker(
        BASE["date"],
        state_key="range",
        default_days=365,
    )

    # ============================================================
    # Chart Parameters
    # ============================================================
    section("Parâmetros", None, "⚙️")

    if not compare_two:
        ma_window = ma_picker(
            options=(20, 50, 200),
            default=90,
            state_key="ma_window",
        )
        st.caption(f"Média móvel selecionada: **{ma_window}** períodos")
    else:
        st.caption("Média móvel e RSI desativados no modo de comparação.")

    st.divider()

    # ============================================================
    # Filter Data and Plot
    # ============================================================
    if close_col not in BASE.columns:
        st.warning(f"A coluna selecionada ('{close_col}') não está disponível.")
        return

    # Filter by date range
    mask = BASE["date"].dt.date.between(start_date, end_date)

    # -------------------------
    # Single-Asset Mode
    # -------------------------
    if not compare_two:
        df_view = BASE.loc[mask, ["date", close_col]].dropna().copy()
        plot_single_asset(df_view, close_col, asset_label, ma_window)

    # -------------------------
    # Two-Asset Comparison Mode
    # -------------------------
    else:
        if not second_col:
            st.info("Selecione um segundo ativo para comparação.")
            return

        if second_col not in BASE.columns:
            st.warning(f"A coluna '{second_col}' não está disponível.")
            return

        cols = ["date", close_col, second_col]

        # Gap handling options
        st.markdown("**Tratamento de dados faltantes**")
        merge_mode = st.radio(
            "",
            options=(
                "Datas em comum (sem preenchimento)",
                "Preencher pequenos gaps com último valor (ffill)",
            ),
            key="merge_mode",
        )
        st.caption(
            "• *Datas em comum*: usa apenas dias em que os dois ativos têm preço.\n"
            "• *ffill*: preenche gaps de até 3 dias com o último valor."
        )

        # Prepare data
        df_view = prepare_comparison_data(BASE, cols, mask, merge_mode)

        if df_view.empty:
            st.info("Sem dados no período selecionado.")
            return

        st.markdown("---")

        # Normalization option
        st.markdown("**Normalização (opcional)**")
        normalize = st.checkbox(
            "Normalizar ambos os ativos (início do período = 100)",
            value=False,
            key="normalize_compare",
        )
        st.caption(
            "A normalização reescala cada série para começar em **100**, "
            "facilitando a comparação de variação percentual."
        )
        st.markdown("<br>", unsafe_allow_html=True)

        # Apply normalization if requested
        if normalize:
            df_view = normalize_comparison_data(df_view, close_col, second_col)

        # Plot comparison
        plot_comparison_chart(
            df_view,
            close_col,
            second_col,
            asset_label,
            second_label,
            normalize
        )


# Run main function
main()
