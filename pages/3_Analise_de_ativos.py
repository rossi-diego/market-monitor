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

# Asset categories for better UX
ASSET_CATEGORIES = {
    "🌾 Complexo Soja": [
        ("Flat do óleo de soja (BRL - C1)", "oleo_flat_brl"),
        ("Flat do óleo de soja (USD - C1)", "oleo_flat_usd"),
        ("Flat do farelo de soja (BRL - C1)", "farelo_flat_brl"),
        ("Flat do farelo de soja (USD - C1)", "farelo_flat_usd"),
        ("Óleo de soja (BOC1)", "boc1"),
        ("Farelo de soja (SMC1)", "smc1"),
        ("Soja (SC1)", "sc1"),
        ("Óleo - Prêmio C1", "so-premp-c1"),
        ("Farelo - Prêmio C1", "sm-premp-c1"),
        ("Soja - Prêmio C1", "sb-premp-c1"),
    ],
    "🌍 Soja Regional": [
        ("Soybean Oil Brazil Paranagua A1", "bo-brzpar-a1"),
        ("Soybean Oil Brazil Paranagua B1", "bo-brzpar-b1"),
        ("Soybean Brazil Paranagua A1", "s-brzpar-a1"),
        ("Soybean Brazil Paranagua B1", "s-brzpar-b1"),
        ("Soybean Meal Brazil Paranagua A1", "sm-brzpar-a1"),
        ("Soybean Meal Brazil Paranagua B1", "sm-brzpar-b1"),
        ("Soybean Oil Lib C1", "sb-oilib-c1"),
        ("Soybean Argentina Ref", "soy-arg-ref"),
    ],
    "🌾 Grãos": [
        ("Milho (CC1)", "cc1"),
        ("Wheat (WC1)", "wc1"),
        ("Kansas Wheat (KWC1)", "kwc1"),
        ("Rice (RBC1)", "rbc1"),
    ],
    "🌴 Óleos & Gorduras": [
        ("Óleo de palma - Ringgit (FCPOC1)", "fcpoc1"),
        ("Óleo de palma - USD (FUPOC1)", "fupoc1"),
        ("Palm Oil Malaysia Crude (P1)", "palm-mycrd-p1"),
        ("Palm Oil Malaysia RBD (P1)", "palm-myrbd-p1"),
        ("Palm Oil SEA (OCRDM)", "palm-sea-ocrdm"),
        ("CPO Indonesia (M1)", "cpo-id-m1"),
        ("CPO Malaysia South (M1)", "cpo-mysth-m1"),
        ("Soybean Oil SEA (DGCF)", "soil-sea-dgcf"),
        ("Sunflower Oil SEA (CRM)", "sunf-sea-crm"),
        ("Canola (RSC1)", "rsc1"),
    ],
    "⚡ Energia": [
        ("Brent (LCOC1)", "lcoc1"),
        ("WTI Crude Oil (CLC1)", "clc1"),
        ("Heating Oil (HOC1)", "hoc1"),
        ("Natural Gas (NGC1)", "ngc1"),
    ],
    "☕ Softs": [
        ("Coffee (KCC1)", "kcc1"),
        ("Cotton (CTC1)", "ctc1"),
        ("Sugar (SBC1)", "sbc1"),
        ("Orange Juice (OJC1)", "ojc1"),
        ("Cocoa (CCC1)", "ccc1"),
    ],
    "🐄 Pecuária": [
        ("Cattle (LCC1)", "lcc1"),
        ("Lean Hogs (LHC1)", "lhc1"),
        ("Feeder Cattle (FCC1)", "fcc1"),
    ],
    "🥇 Metais": [
        ("Gold (GCC1)", "gcc1"),
        ("Silver (SIC1)", "sic1"),
        ("Copper (HGC1)", "hgc1"),
        ("Platinum (PLC1)", "plc1"),
        ("Palladium (PAC1)", "pac1"),
    ],
    "💱 Moedas": [
        ("Dólar (BRL=)", "brl="),
        ("Malaysian Ringgit (MYR=)", "myr="),
    ],
    "₿ Cripto": [
        ("Bitcoin (BTC=)", "btc="),
    ],
}

# Flatten to old format for compatibility
ASSETS_MAP = {
    # Flats (calculated)
    "Flat do óleo de soja (BRL - C1)": "oleo_flat_brl",
    "Flat do óleo de soja (USD - C1)": "oleo_flat_usd",
    "Flat do farelo de soja (BRL - C1)": "farelo_flat_brl",
    "Flat do farelo de soja (USD - C1)": "farelo_flat_usd",

    # Soy Complex
    "Óleo de soja (BOC1)": "boc1",
    "Farelo de soja (SMC1)": "smc1",
    "Soja (SC1)": "sc1",

    # Soy Premiums C1
    "Óleo - Prêmio C1": "so-premp-c1",
    "Farelo - Prêmio C1": "sm-premp-c1",
    "Soja - Prêmio C1": "sb-premp-c1",

    # Soy Regional Prices
    "Soybean Oil Brazil Paranagua A1": "bo-brzpar-a1",
    "Soybean Oil Brazil Paranagua B1": "bo-brzpar-b1",
    "Soybean Brazil Paranagua A1": "s-brzpar-a1",
    "Soybean Brazil Paranagua B1": "s-brzpar-b1",
    "Soybean Meal Brazil Paranagua A1": "sm-brzpar-a1",
    "Soybean Meal Brazil Paranagua B1": "sm-brzpar-b1",
    "Soybean Oil Lib C1": "sb-oilib-c1",
    "Soybean Argentina Ref": "soy-arg-ref",

    # Grains
    "Milho (CC1)": "cc1",
    "Wheat (WC1)": "wc1",
    "Kansas Wheat (KWC1)": "kwc1",
    "Rice (RBC1)": "rbc1",

    # Oils & Fats
    "Óleo de palma - Ringgit (FCPOC1)": "fcpoc1",
    "Óleo de palma - USD (FUPOC1)": "fupoc1",
    "Palm Oil Malaysia Crude (P1)": "palm-mycrd-p1",
    "Palm Oil Malaysia RBD (P1)": "palm-myrbd-p1",
    "Palm Oil SEA (OCRDM)": "palm-sea-ocrdm",
    "CPO Indonesia (M1)": "cpo-id-m1",
    "CPO Malaysia South (M1)": "cpo-mysth-m1",
    "Soybean Oil SEA (DGCF)": "soil-sea-dgcf",
    "Sunflower Oil SEA (CRM)": "sunf-sea-crm",
    "Canola (RSC1)": "rsc1",

    # Energy
    "Brent (LCOC1)": "lcoc1",
    "WTI Crude Oil (CLC1)": "clc1",
    "Heating Oil (HOC1)": "hoc1",
    "Natural Gas (NGC1)": "ngc1",

    # Softs
    "Coffee (KCC1)": "kcc1",
    "Cotton (CTC1)": "ctc1",
    "Sugar (SBC1)": "sbc1",
    "Orange Juice (OJC1)": "ojc1",
    "Cocoa (CCC1)": "ccc1",

    # Livestock
    "Cattle (LCC1)": "lcc1",
    "Lean Hogs (LHC1)": "lhc1",
    "Feeder Cattle (FCC1)": "fcc1",

    # Metals
    "Gold (GCC1)": "gcc1",
    "Silver (SIC1)": "sic1",
    "Copper (HGC1)": "hgc1",
    "Platinum (PLC1)": "plc1",
    "Palladium (PAC1)": "pac1",

    # Currencies
    "Dólar (BRL=)": "brl=",
    "Malaysian Ringgit (MYR=)": "myr=",

    # Crypto
    "Bitcoin (BTC=)": "btc=",
}

# Favorites for quick access
FAVORITE_ASSETS = [
    "Flat do óleo de soja (BRL - C1)",
    "Óleo de soja (BOC1)",
    "Farelo de soja (SMC1)",
    "Soja (SC1)",
    "Óleo de palma - Ringgit (FCPOC1)",
    "Dólar (BRL=)",
]


# ============================================================
# Custom Asset Picker with Categories
# ============================================================
def categorized_asset_picker(df_data, state_key="asset_col", show_favorites=True):
    """
    Asset picker with category filter for better UX.
    Returns: (selected_column, label)
    """
    # Filter available assets
    available_map = {label: col for label, col in ASSETS_MAP.items() if col in df_data.columns}

    if not available_map:
        st.error("Nenhum ativo disponível")
        st.stop()

    # Initialize state
    if state_key not in st.session_state:
        # Default to first favorite that exists
        default = next((ASSETS_MAP[f] for f in FAVORITE_ASSETS if f in available_map), next(iter(available_map.values())))
        st.session_state[state_key] = default

    # Category selection
    col_cat, col_search = st.columns([1, 2])

    with col_cat:
        # Build category filter options
        category_options = ["📂 Todas as categorias"]
        for cat_name in ASSET_CATEGORIES.keys():
            # Check if category has any available assets
            cat_assets = [col for label, col in ASSET_CATEGORIES[cat_name] if col in df_data.columns]
            if cat_assets:
                count = len(cat_assets)
                category_options.append(f"{cat_name} ({count})")

        selected_category = st.selectbox(
            "Categoria",
            options=category_options,
            key=f"{state_key}_category",
            help="Filtre por categoria para encontrar ativos mais facilmente"
        )

    # Filter assets by category
    if selected_category == "📂 Todas as categorias":
        filtered_map = available_map
    else:
        # Extract category name (remove count)
        cat_name = selected_category.rsplit(" (", 1)[0]
        filtered_map = {}
        for label, col in ASSET_CATEGORIES.get(cat_name, []):
            if col in df_data.columns:
                filtered_map[label] = col

    # Favorites row (only if show_favorites and not filtered by category)
    if show_favorites and selected_category == "📂 Todas as categorias":
        st.markdown("**⭐ Favoritos**")
        fav_cols = st.columns(len(FAVORITE_ASSETS))
        for i, fav_label in enumerate(FAVORITE_ASSETS):
            if fav_label in available_map:
                fav_col_name = available_map[fav_label]
                with fav_cols[i]:
                    is_selected = (st.session_state[state_key] == fav_col_name)
                    if st.button(
                        fav_label.split("(")[0].strip(),  # Short name
                        key=f"{state_key}_fav_{i}",
                        type="primary" if is_selected else "secondary",
                        use_container_width=True,
                    ):
                        st.session_state[state_key] = fav_col_name
                        st.rerun()

    # Searchable dropdown
    with col_search:
        labels = list(filtered_map.keys())
        current_col = st.session_state[state_key]

        # Find current label
        current_label = next((lbl for lbl, col in filtered_map.items() if col == current_col), labels[0] if labels else None)

        if not labels:
            st.warning("Nenhum ativo disponível nesta categoria")
            return st.session_state[state_key], current_label

        selected_label = st.selectbox(
            "Buscar ativo",
            options=labels,
            index=labels.index(current_label) if current_label in labels else 0,
            key=f"{state_key}_select",
            help="Digite para buscar ou role para navegar"
        )

        # Update state if changed
        new_col = filtered_map[selected_label]
        if new_col != st.session_state[state_key]:
            st.session_state[state_key] = new_col

    return st.session_state[state_key], selected_label


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

    # Calculate number of trading days in period
    num_days = len(series)

    # Volatility calculations (period and annualized)
    vol_period = returns.std() * 100  # Volatility in the selected period
    vol_annual = returns.std() * np.sqrt(252) * 100  # Annualized volatility

    # Sharpe-like metric (assuming risk-free rate = 0 for simplicity)
    mean_return = returns.mean()
    sharpe = (mean_return / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0

    # Max drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min() * 100

    # Distance from mean (Z-score)
    mean_price = series.mean()
    std_price = series.std()
    z_score = (current_price - mean_price) / std_price if std_price > 0 else 0

    stats = {
        "current": current_price,
        "period_change": ((current_price - first_price) / first_price * 100),
        "min": series.min(),
        "max": series.max(),
        "mean": mean_price,
        "vol_period": vol_period,
        "vol_annual": vol_annual,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "z_score": z_score,
        "last_update": data["date"].max(),
        "data_points": num_days,
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

    # Main metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        display_metric_card(
            "Preço Atual",
            f"{stats['current']:.2f}",
            f"{stats['period_change']:+.2f}%",
            f"Preço mais recente e variação no período selecionado ({stats['data_points']} dias)"
        )

    with col2:
        display_metric_card(
            "Volatilidade Diária",
            f"{stats['vol_period']:.2f}%",
            help_text=f"Volatilidade diária no período: {stats['vol_period']:.2f}% | Anualizada (252 dias): {stats['vol_annual']:.1f}%"
        )

    with col3:
        display_metric_card(
            "Min / Max (período)",
            f"{stats['min']:.2f}",
            f"Max: {stats['max']:.2f}",
            "Range de preços no período selecionado"
        )

    with col4:
        display_metric_card(
            "Média (período)",
            f"{stats['mean']:.2f}",
            help_text="Preço médio no período selecionado"
        )

    # Advanced metrics row
    st.markdown("#### 🎯 Métricas para Trading")
    col5, col6, col7, col8 = st.columns(4)

    with col5:
        # Z-Score interpretation
        z_interp = "Sobrecomprado" if stats['z_score'] > 1.5 else "Sobrevendido" if stats['z_score'] < -1.5 else "Neutro"
        z_color = "🔴" if stats['z_score'] > 1.5 else "🟢" if stats['z_score'] < -1.5 else "🟡"
        display_metric_card(
            "Z-Score (período)",
            f"{stats['z_score']:.2f}",
            f"{z_color} {z_interp}",
            "Distância do preço atual da média DO PERÍODO (em desvios padrão). >1.5: caro, <-1.5: barato"
        )

    with col6:
        sharpe_color = "🟢" if stats['sharpe'] > 1 else "🟡" if stats['sharpe'] > 0 else "🔴"
        display_metric_card(
            "Sharpe (anualizado)",
            f"{stats['sharpe']:.2f}",
            f"{sharpe_color}",
            "Retorno ajustado ao risco ANUALIZADO (252 dias). >1: bom, >2: muito bom"
        )

    with col7:
        dd_color = "🟢" if stats['max_drawdown'] > -10 else "🟡" if stats['max_drawdown'] > -20 else "🔴"
        display_metric_card(
            "Max DD (período)",
            f"{stats['max_drawdown']:.1f}%",
            f"{dd_color}",
            "Maior queda do pico ao vale NO PERÍODO selecionado"
        )

    with col8:
        # Distance from mean in percentage
        dist_from_mean = ((stats['current'] - stats['mean']) / stats['mean'] * 100)
        dist_color = "↗️" if dist_from_mean > 0 else "↘️"
        display_metric_card(
            "vs Média (período)",
            f"{dist_from_mean:+.1f}%",
            f"{dist_color}",
            "Distância % do preço atual da média DO PERÍODO selecionado"
        )

    # Data quality indicator
    st.caption(f"📅 Última atualização: {stats['last_update'].strftime('%d/%m/%Y')} | "
               f"📈 {stats['data_points']} pontos de dados no período")


def calculate_correlation(data, col1, col2):
    """Calculate Pearson and Spearman correlations between two series."""
    if data.empty or col1 not in data.columns or col2 not in data.columns:
        return None, None

    # Pearson correlation (linear relationship)
    pearson = data[[col1, col2]].corr(method='pearson').iloc[0, 1]

    # Spearman correlation (monotonic relationship)
    spearman = data[[col1, col2]].corr(method='spearman').iloc[0, 1]

    return pearson, spearman


def calculate_beta(data, col1, col2):
    """Calculate beta of col1 relative to col2 (col2 is the market/benchmark)."""
    if data.empty or col1 not in data.columns or col2 not in data.columns:
        return None

    # Calculate returns
    returns1 = data[col1].pct_change().dropna()
    returns2 = data[col2].pct_change().dropna()

    # Align the series
    aligned = pd.DataFrame({'asset': returns1, 'benchmark': returns2}).dropna()

    if aligned.empty or len(aligned) < 2:
        return None

    # Beta = Cov(asset, benchmark) / Var(benchmark)
    covariance = aligned['asset'].cov(aligned['benchmark'])
    variance = aligned['benchmark'].var()

    if variance == 0:
        return None

    beta = covariance / variance
    return beta


def display_comparison_stats(data, col1, col2, label1, label2):
    """Display comparison statistics between two assets."""
    stats1 = calculate_statistics(data, col1)
    stats2 = calculate_statistics(data, col2)
    pearson, spearman = calculate_correlation(data, col1, col2)
    beta = calculate_beta(data, col1, col2)

    if not stats1 or not stats2:
        return

    st.markdown("### 📊 Estatísticas Comparativas")

    # Correlation and Beta section
    with st.container(border=True):
        st.markdown("#### 🔗 Relação entre os ativos")

        col_corr1, col_corr2, col_beta = st.columns(3)

        with col_corr1:
            st.markdown("**Correlação de Pearson**")
            if pearson is not None:
                corr_color = "🟢" if abs(pearson) > 0.7 else "🟡" if abs(pearson) > 0.3 else "🔴"
                st.metric(
                    "Linear",
                    f"{pearson:.3f}",
                    f"{corr_color}",
                    help="Mede relação LINEAR entre os ativos. 1 = movem juntos, -1 = movem opostos, 0 = sem relação"
                )
                if abs(pearson) > 0.7:
                    st.caption("✅ Forte relação linear")
                elif abs(pearson) > 0.3:
                    st.caption("⚠️ Relação linear moderada")
                else:
                    st.caption("❌ Relação linear fraca")

        with col_corr2:
            st.markdown("**Correlação de Spearman**")
            if spearman is not None:
                spear_color = "🟢" if abs(spearman) > 0.7 else "🟡" if abs(spearman) > 0.3 else "🔴"
                st.metric(
                    "Monotônica",
                    f"{spearman:.3f}",
                    f"{spear_color}",
                    help="Mede relação MONOTÔNICA (mesma direção, mas não necessariamente linear). Mais robusta a outliers"
                )
                if abs(spearman) > 0.7:
                    st.caption("✅ Forte relação monotônica")
                elif abs(spearman) > 0.3:
                    st.caption("⚠️ Relação monotônica moderada")
                else:
                    st.caption("❌ Relação monotônica fraca")

        with col_beta:
            st.markdown(f"**Beta ({label1} vs {label2})**")
            if beta is not None:
                beta_interp = "Alta sensibilidade" if abs(beta) > 1.5 else "Moderada" if abs(beta) > 0.5 else "Baixa"
                beta_color = "🔴" if abs(beta) > 1.5 else "🟡" if abs(beta) > 0.5 else "🟢"
                st.metric(
                    "Sensibilidade",
                    f"{beta:.2f}",
                    f"{beta_color} {beta_interp}",
                    help=f"Quando {label2} varia 1%, {label1} tende a variar {beta:.2f}%. Beta>1: mais volátil, Beta<1: menos volátil"
                )
                st.caption(f"Se {label2} sobe 1%, {label1} {'sobe' if beta > 0 else 'desce'} ~{abs(beta):.1f}%")
            else:
                st.metric("Sensibilidade", "N/A", help="Dados insuficientes para calcular Beta")

    # Side-by-side metrics comparison
    st.markdown("#### 📈 Métricas de Trading")

    # Basic metrics
    with st.container(border=True):
        st.markdown("**Preço e Variação**")
        col_a1, col_a2, col_a3, col_a4 = st.columns(4)

        with col_a1:
            st.metric(f"{label1} - Atual", f"{stats1['current']:.2f}")
        with col_a2:
            change_color1 = "🟢" if stats1['period_change'] > 0 else "🔴"
            st.metric(f"{label1} - Variação", f"{stats1['period_change']:+.2f}%", f"{change_color1}")
        with col_a3:
            st.metric(f"{label2} - Atual", f"{stats2['current']:.2f}")
        with col_a4:
            change_color2 = "🟢" if stats2['period_change'] > 0 else "🔴"
            st.metric(f"{label2} - Variação", f"{stats2['period_change']:+.2f}%", f"{change_color2}")

    # Volatility and risk metrics
    with st.container(border=True):
        st.markdown("**Volatilidade e Risco**")
        col_b1, col_b2, col_b3, col_b4 = st.columns(4)

        with col_b1:
            st.metric(f"{label1} - Vol. Período", f"{stats1['vol_period']:.1f}%",
                     help="Volatilidade do período selecionado")
        with col_b2:
            st.metric(f"{label1} - Vol. Anual", f"{stats1['vol_annual']:.1f}%",
                     help="Volatilidade anualizada (252 dias)")
        with col_b3:
            st.metric(f"{label2} - Vol. Período", f"{stats2['vol_period']:.1f}%",
                     help="Volatilidade do período selecionado")
        with col_b4:
            st.metric(f"{label2} - Vol. Anual", f"{stats2['vol_annual']:.1f}%",
                     help="Volatilidade anualizada (252 dias)")

    # Trading signals
    with st.container(border=True):
        st.markdown("**Sinais de Trading**")
        col_c1, col_c2, col_c3, col_c4 = st.columns(4)

        with col_c1:
            z1_interp = "Caro" if stats1['z_score'] > 1.5 else "Barato" if stats1['z_score'] < -1.5 else "Justo"
            z1_color = "🔴" if stats1['z_score'] > 1.5 else "🟢" if stats1['z_score'] < -1.5 else "🟡"
            st.metric(f"{label1} - Z-Score", f"{stats1['z_score']:.2f}", f"{z1_color} {z1_interp}",
                     help="Distância da média em desvios padrão")

        with col_c2:
            sharpe1_color = "🟢" if stats1['sharpe'] > 1 else "🟡" if stats1['sharpe'] > 0 else "🔴"
            st.metric(f"{label1} - Sharpe", f"{stats1['sharpe']:.2f}", f"{sharpe1_color}",
                     help="Retorno ajustado ao risco")

        with col_c3:
            z2_interp = "Caro" if stats2['z_score'] > 1.5 else "Barato" if stats2['z_score'] < -1.5 else "Justo"
            z2_color = "🔴" if stats2['z_score'] > 1.5 else "🟢" if stats2['z_score'] < -1.5 else "🟡"
            st.metric(f"{label2} - Z-Score", f"{stats2['z_score']:.2f}", f"{z2_color} {z2_interp}",
                     help="Distância da média em desvios padrão")

        with col_c4:
            sharpe2_color = "🟢" if stats2['sharpe'] > 1 else "🟡" if stats2['sharpe'] > 0 else "🔴"
            st.metric(f"{label2} - Sharpe", f"{stats2['sharpe']:.2f}", f"{sharpe2_color}",
                     help="Retorno ajustado ao risco")

    # Max drawdown
    with st.container(border=True):
        st.markdown("**Drawdown e Posição**")
        col_d1, col_d2, col_d3, col_d4 = st.columns(4)

        with col_d1:
            dd1_color = "🟢" if stats1['max_drawdown'] > -10 else "🟡" if stats1['max_drawdown'] > -20 else "🔴"
            st.metric(f"{label1} - Max DD", f"{stats1['max_drawdown']:.1f}%", f"{dd1_color}",
                     help="Maior queda do pico ao vale")

        with col_d2:
            dist1 = ((stats1['current'] - stats1['mean']) / stats1['mean'] * 100)
            dist1_color = "↗️" if dist1 > 0 else "↘️"
            st.metric(f"{label1} - vs Média", f"{dist1:+.1f}%", f"{dist1_color}",
                     help="Distância percentual da média")

        with col_d3:
            dd2_color = "🟢" if stats2['max_drawdown'] > -10 else "🟡" if stats2['max_drawdown'] > -20 else "🔴"
            st.metric(f"{label2} - Max DD", f"{stats2['max_drawdown']:.1f}%", f"{dd2_color}",
                     help="Maior queda do pico ao vale")

        with col_d4:
            dist2 = ((stats2['current'] - stats2['mean']) / stats2['mean'] * 100)
            dist2_color = "↗️" if dist2 > 0 else "↘️"
            st.metric(f"{label2} - vs Média", f"{dist2:+.1f}%", f"{dist2_color}",
                     help="Distância percentual da média")


def export_data_to_csv(data, filename="data.csv", key_suffix=""):
    """Create CSV download button."""
    if data.empty:
        st.warning("Sem dados para exportar")
        return

    try:
        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Baixar dados (CSV)",
            data=csv,
            file_name=filename,
            mime="text/csv",
            key=f"download_csv_{key_suffix}",
        )
    except Exception as e:
        st.error(f"Erro ao exportar CSV: {str(e)}")


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

        close_col, asset_label = categorized_asset_picker(
            BASE,
            state_key="close_col",
            show_favorites=True,
        )

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
            st.markdown("#### Segundo Ativo")
            second_col, second_label = categorized_asset_picker(
                BASE,
                state_key="second_col",
                show_favorites=False,  # Hide favorites for second asset to save space
            )

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

        # Explanatory notes
        with st.expander("ℹ️ Como interpretar o gráfico", expanded=False):
            st.markdown("""
            ### 📊 Componentes do Gráfico

            **Painel Superior - Preço:**
            - **Linha Azul**: Preço de fechamento do ativo ao longo do tempo
            - **Linha Laranja**: Média móvel (MA) - suaviza flutuações e mostra a tendência
              - Preço acima da MA = tendência de alta
              - Preço abaixo da MA = tendência de baixa
              - Cruzamentos podem indicar mudança de tendência

            **Painel Inferior - RSI (Relative Strength Index):**
            - **Escala**: 0 a 100
            - **Interpretação**:
              - RSI > 70: Ativo pode estar **sobrecomprado** (caro) - possível correção
              - RSI < 30: Ativo pode estar **sobrevendido** (barato) - possível recuperação
              - RSI entre 30-70: Zona neutra
            - **Linha Roxa**: RSI atual
            - **Linhas Pontilhadas**: Zonas de 30 (suporte) e 70 (resistência)

            ### 🎯 Dicas de Trading

            - **Sinal de Compra**: RSI < 30 + preço tocando ou abaixo da MA
            - **Sinal de Venda**: RSI > 70 + preço muito acima da MA
            - **Divergências**: RSI sobe enquanto preço cai (ou vice-versa) = possível reversão
            - Combine com as métricas acima (Z-Score, Sharpe, etc.) para decisões mais informadas
            """)

        # Export options
        st.markdown("### 📥 Exportar dados")
        col_exp1, col_exp2 = st.columns(2)

        with col_exp1:
            export_data_to_csv(
                df_view,
                f"{asset_label.replace('/', '_')}_{start_date}_{end_date}.csv",
                key_suffix="single_asset"
            )

        with col_exp2:
            if fig:
                try:
                    # Create a buffer for the image
                    img_bytes = fig.to_image(format="png", width=1200, height=600)
                    st.download_button(
                        label="📥 Baixar gráfico (PNG)",
                        data=img_bytes,
                        file_name=f"{asset_label.replace('/', '_')}_chart.png",
                        mime="image/png",
                        key="download_png_single",
                    )
                except Exception:
                    st.warning("⚠️ Export de PNG não disponível neste ambiente. Use screenshot do navegador.")

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

        # Explanatory notes for comparison
        with st.expander("ℹ️ Como interpretar a comparação", expanded=False):
            st.markdown(f"""
            ### 📊 Gráfico de Comparação

            **Modo de Visualização:**
            - {'**Normalizado (Base 100)**: Ambos os ativos começam em 100, facilitando comparação de performance relativa' if normalize else '**Escalas Separadas**: Cada ativo usa sua própria escala (eixo Y esquerdo e direito)'}
            - Compare a movimentação e tendências dos ativos ao longo do tempo

            ### 📈 Métricas Apresentadas

            **Correlações:**
            - **Pearson**: Mede relação LINEAR entre os ativos
              - +1: Movem-se perfeitamente juntos
              - -1: Movem-se perfeitamente opostos
              - 0: Sem relação linear
            - **Spearman**: Mede relação MONOTÔNICA (mesma direção, mas não necessariamente proporcional)
              - Mais robusta a outliers que Pearson
              - Use quando a relação não é perfeitamente linear

            **Beta ({asset_label} vs {second_label}):**
            - Mede quanto {asset_label} tende a variar quando {second_label} varia
            - Beta = 1.0: Movem-se na mesma proporção
            - Beta > 1.0: {asset_label} é mais volátil que {second_label}
            - Beta < 1.0: {asset_label} é menos volátil que {second_label}
            - Beta negativo: Movem-se em direções opostas

            ### 🎯 Estratégias com Comparação

            **Arbitragem:**
            - Se correlação é alta mas os ativos divergem temporariamente, pode haver oportunidade
            - Z-Score alto em um e baixo no outro = possível convergência futura

            **Hedge:**
            - Beta negativo indica potencial de hedge (proteção)
            - Exemplo: se beta = -0.8, quando um cai 10%, o outro tende a subir 8%

            **Pair Trading:**
            - Procure ativos com correlação forte (>0.7)
            - Quando um está "Caro" (Z > 1.5) e outro "Barato" (Z < -1.5)
            - Venda o caro, compre o barato, esperando convergência
            """)

        # Export options
        st.markdown("### 📥 Exportar dados")
        col_exp1, col_exp2 = st.columns(2)

        with col_exp1:
            export_data_to_csv(
                df_view,
                f"comparison_{asset_label}_{second_label}_{start_date}_{end_date}.csv".replace("/", "_"),
                key_suffix="comparison"
            )

        with col_exp2:
            if fig:
                try:
                    img_bytes = fig.to_image(format="png", width=1200, height=600)
                    st.download_button(
                        label="📥 Baixar gráfico (PNG)",
                        data=img_bytes,
                        file_name=f"comparison_{asset_label}_{second_label}.png".replace("/", "_"),
                        mime="image/png",
                        key="download_png_comparison",
                    )
                except Exception:
                    st.warning("⚠️ Export de PNG não disponível neste ambiente. Use screenshot do navegador.")


# Run main function
main()
