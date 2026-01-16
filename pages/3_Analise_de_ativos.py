"""
Asset Analysis Page - Professional Edition

Features:
- Select one asset to analyze (Price + RSI chart)
- OR compare two assets side-by-side
- Summary statistics and key metrics
- Professional visual layout with containers
- Export functionality (CSV and PNG)
- Data quality indicators
"""

# ============================================================
# Imports & Config
# ============================================================
import pandas as pd
import numpy as np
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


def calculate_statistics(data, column):
    """Calculate key statistics for the data."""
    if data.empty or column not in data.columns:
        return None

    series = data[column].dropna()
    if series.empty:
        return None

    current_price = series.iloc[-1]
    first_price = series.iloc[0]

    # Calculate returns
    returns = series.pct_change().dropna()

    stats = {
        "current": current_price,
        "period_change": ((current_price - first_price) / first_price * 100),
        "min": series.min(),
        "max": series.max(),
        "mean": series.mean(),
        "volatility": returns.std() * np.sqrt(252) * 100,  # Annualized
        "last_update": data["date"].max(),
        "data_points": len(series),
    }

    return stats


def display_metric_card(label, value, delta=None, help_text=None):
    """Display a professional metric card."""
    col = st.container()
    with col:
        if delta is not None:
            st.metric(
                label=label,
                value=value,
                delta=delta,
                help=help_text
            )
        else:
            st.metric(
                label=label,
                value=value,
                help=help_text
            )


def display_statistics_panel(data, column, label):
    """Display statistics panel with key metrics."""
    stats = calculate_statistics(data, column)

    if not stats:
        st.warning("Dados insuficientes para calcular estatísticas.")
        return

    st.markdown(f"### 📊 Estatísticas - {label}")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        display_metric_card(
            "Preço Atual",
            f"{stats['current']:.2f}",
            f"{stats['period_change']:+.2f}%",
            "Preço mais recente no período"
        )

    with col2:
        display_metric_card(
            "Volatilidade (anual)",
            f"{stats['volatility']:.1f}%",
            help_text="Volatilidade anualizada dos retornos"
        )

    with col3:
        display_metric_card(
            "Mínimo",
            f"{stats['min']:.2f}",
            help_text="Menor preço no período"
        )

    with col4:
        display_metric_card(
            "Máximo",
            f"{stats['max']:.2f}",
            help_text="Maior preço no período"
        )

    # Data quality indicator
    st.caption(f"📅 Última atualização: {stats['last_update'].strftime('%d/%m/%Y')} | "
               f"📈 {stats['data_points']} pontos de dados")


def calculate_correlation(data, col1, col2):
    """Calculate correlation between two series."""
    if data.empty or col1 not in data.columns or col2 not in data.columns:
        return None

    return data[[col1, col2]].corr().iloc[0, 1]


def display_comparison_stats(data, col1, col2, label1, label2):
    """Display comparison statistics between two assets."""
    stats1 = calculate_statistics(data, col1)
    stats2 = calculate_statistics(data, col2)
    corr = calculate_correlation(data, col1, col2)

    if not stats1 or not stats2:
        return

    st.markdown("### 📊 Estatísticas Comparativas")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown(f"**{label1}**")
        st.metric("Variação no período", f"{stats1['period_change']:+.2f}%")
        st.metric("Volatilidade", f"{stats1['volatility']:.1f}%")

    with col_b:
        st.markdown(f"**{label2}**")
        st.metric("Variação no período", f"{stats2['period_change']:+.2f}%")
        st.metric("Volatilidade", f"{stats2['volatility']:.1f}%")

    with col_c:
        st.markdown("**Correlação**")
        if corr is not None:
            corr_pct = corr * 100
            st.metric(
                "Coeficiente",
                f"{corr:.3f}",
                f"{corr_pct:+.1f}%",
                help="Correlação de Pearson entre os ativos"
            )

            # Interpretation
            if abs(corr) > 0.7:
                st.success("Correlação forte")
            elif abs(corr) > 0.3:
                st.info("Correlação moderada")
            else:
                st.warning("Correlação fraca")


def export_data_to_csv(data, filename="data.csv"):
    """Create CSV download button."""
    csv = data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Baixar dados (CSV)",
        data=csv,
        file_name=filename,
        mime="text/csv",
    )


def plot_single_asset(data, asset_col, asset_label, ma_window):
    """Create price + RSI chart for single asset."""
    if data.empty:
        st.info("Sem dados no período selecionado.")
        return None

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
        height=600,
    )

    st.plotly_chart(fig, use_container_width=True)
    return fig


def prepare_comparison_data(base, cols, mask, merge_mode):
    """Prepare data for two-asset comparison with gap handling."""
    if merge_mode == "Datas em comum (sem preenchimento)":
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

    return tmp.dropna(subset=[c for c in cols if c != "date"])


def normalize_comparison_data(data, col1, col2):
    """Normalize both columns to start at 100."""
    data = data.copy()
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
            line=dict(width=2),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=data["date"],
            y=data[col2],
            mode="lines",
            name=label2,
            yaxis="y1" if normalized else "y2",
            line=dict(width=2),
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
        height=600,
        hovermode='x unified',
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
    return fig


# ============================================================
# Main Page Logic
# ============================================================
def main():
    """Main page logic."""
    # Page header
    st.markdown("# 📈 Análise de Ativos")
    st.markdown("Análise técnica e comparativa de commodities e ativos financeiros")
    st.divider()

    # Prepare data
    BASE = prepare_base_data(df)

    # ============================================================
    # Asset Selection (in container)
    # ============================================================
    with st.container(border=True):
        section(
            "Selecione o ativo principal",
            "Escolha o ativo que deseja analisar",
            "🎯",
        )

        close_col, _assets = asset_picker_dropdown(
            BASE,
            ASSETS_MAP,
            state_key="close_col",
        )

        asset_label = get_asset_label(close_col, ASSETS_MAP)

    st.divider()

    # ============================================================
    # Comparison Mode Toggle
    # ============================================================
    with st.container(border=True):
        section(
            "Modo de comparação",
            "Compare dois ativos no mesmo gráfico",
            "📊",
        )

        compare_two = st.checkbox(
            "Ativar comparação entre dois ativos",
            value=False,
            key="compare_two_assets",
        )

        second_col = None
        second_label = None

        if compare_two:
            asset_labels = list(ASSETS_MAP.keys())
            available_for_comparison = [
                lbl for lbl in asset_labels if ASSETS_MAP[lbl] != close_col
            ]

            second_label = st.selectbox(
                "Segundo ativo para comparação",
                options=available_for_comparison,
                key="second_asset_select",
            )
            second_col = ASSETS_MAP[second_label]

    st.divider()

    # ============================================================
    # Date Range and Parameters
    # ============================================================
    with st.container(border=True):
        section("Configurações do gráfico", "Ajuste período e parâmetros", "⚙️")

        col1, col2 = st.columns([2, 1])

        with col1:
            start_date, end_date = date_range_picker(
                BASE["date"],
                state_key="range",
                default_days=365,
            )

        with col2:
            if not compare_two:
                ma_window = ma_picker(
                    options=(20, 50, 90, 200),
                    default=90,
                    state_key="ma_window",
                )
                st.caption(f"Média móvel: **{ma_window}** períodos")
            else:
                st.info("MA e RSI desativados em modo de comparação")

    st.divider()

    # ============================================================
    # Validate and Filter Data
    # ============================================================
    if close_col not in BASE.columns:
        st.error(f"❌ A coluna '{close_col}' não está disponível nos dados.")
        return

    # Filter by date range
    mask = BASE["date"].dt.date.between(start_date, end_date)

    # ============================================================
    # Single-Asset Mode
    # ============================================================
    if not compare_two:
        df_view = BASE.loc[mask, ["date", close_col]].dropna().copy()

        if df_view.empty:
            st.warning("⚠️ Sem dados no período selecionado.")
            return

        # Display statistics
        display_statistics_panel(df_view, close_col, asset_label)
        st.divider()

        # Plot chart
        fig = plot_single_asset(df_view, close_col, asset_label, ma_window)

        # Export options
        st.markdown("### 📥 Exportar dados")
        col_exp1, col_exp2 = st.columns(2)

        with col_exp1:
            export_data_to_csv(
                df_view,
                f"{asset_label.replace('/', '_')}_{start_date}_{end_date}.csv"
            )

        with col_exp2:
            if fig:
                # Create a buffer for the image
                img_bytes = fig.to_image(format="png", width=1200, height=600)
                st.download_button(
                    label="📥 Baixar gráfico (PNG)",
                    data=img_bytes,
                    file_name=f"{asset_label.replace('/', '_')}_chart.png",
                    mime="image/png",
                )

    # ============================================================
    # Two-Asset Comparison Mode
    # ============================================================
    else:
        if not second_col or second_col not in BASE.columns:
            st.error("❌ Selecione um segundo ativo válido para comparação.")
            return

        cols = ["date", close_col, second_col]

        # Gap handling options (in expander)
        with st.expander("⚙️ Opções avançadas de tratamento de dados"):
            merge_mode = st.radio(
                "Tratamento de dados faltantes",
                options=(
                    "Datas em comum (sem preenchimento)",
                    "Preencher pequenos gaps com último valor (ffill)",
                ),
                key="merge_mode",
            )
            st.caption(
                "• *Datas em comum*: usa apenas dias com dados em ambos.\n"
                "• *ffill*: preenche gaps de até 3 dias úteis."
            )

        # Prepare data
        df_view = prepare_comparison_data(BASE, cols, mask, merge_mode)

        if df_view.empty:
            st.warning("⚠️ Sem dados no período selecionado.")
            return

        # Normalization option
        with st.container(border=True):
            normalize = st.checkbox(
                "🔄 Normalizar ambos os ativos (base 100)",
                value=False,
                key="normalize_compare",
                help="Reescala ambas as séries para começarem em 100"
            )

        # Apply normalization if requested
        if normalize:
            df_view = normalize_comparison_data(df_view, close_col, second_col)

        # Display comparison statistics
        display_comparison_stats(df_view, close_col, second_col, asset_label, second_label)
        st.divider()

        # Plot comparison
        fig = plot_comparison_chart(
            df_view,
            close_col,
            second_col,
            asset_label,
            second_label,
            normalize
        )

        # Export options
        st.markdown("### 📥 Exportar dados")
        col_exp1, col_exp2 = st.columns(2)

        with col_exp1:
            export_data_to_csv(
                df_view,
                f"comparison_{asset_label}_{second_label}_{start_date}_{end_date}.csv".replace("/", "_")
            )

        with col_exp2:
            if fig:
                img_bytes = fig.to_image(format="png", width=1200, height=600)
                st.download_button(
                    label="📥 Baixar gráfico (PNG)",
                    data=img_bytes,
                    file_name=f"comparison_{asset_label}_{second_label}.png".replace("/", "_"),
                    mime="image/png",
                )


# Run main function
main()
