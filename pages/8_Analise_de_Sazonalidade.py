# ============================================================
# Seasonality Analysis - Commodity Futures
# ============================================================
"""
Professional seasonality analysis for commodity futures contracts.
Generates actionable insights based on historical patterns.
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional
from datetime import datetime

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from src.utils import apply_theme
from src.data_pipeline import df as BASE_DF

# ============================================================
# Configuration
# ============================================================
CONFIG = {
    # Data source configuration
    "data_source": "pipeline",  # "pipeline" or "csv"
    "csv_path": "data/futures_data.csv",  # If using CSV

    # Column names mapping (WIDE format)
    "date_col": "date",

    # Month code to number mapping (CME standard)
    "month_code_map": {
        'f': 1, 'g': 2, 'h': 3, 'j': 4, 'k': 5, 'm': 6,
        'n': 7, 'q': 8, 'u': 9, 'v': 10, 'x': 11, 'z': 12
    },

    # Asset prefix mapping (order matters - check 'sm' before 's')
    "asset_prefixes": [
        ("sm", "Farelo de Soja"),
        ("bo", "Óleo de Soja"),
        ("s", "Soja"),
        ("c", "Milho"),
        ("w", "Trigo"),
    ],

    # Analysis parameters
    "min_samples_per_bucket": 20,  # Minimum observations required per bucket
    "rolling_window": 5,  # Rolling average window for smoothing
    "hit_rate_threshold": 0.55,  # Threshold for "consistent" periods
    "std_percentile_low": 40,  # Low volatility threshold (percentile)
    "std_percentile_high": 75,  # High volatility threshold (percentile)
    "top_n_periods": 5,  # Number of top/bottom periods to highlight
}

# Apply theme
apply_theme()

# ============================================================
# Data Parsing Functions
# ============================================================
def parse_year(year_str: str) -> Optional[int]:
    """
    Parse year from contract string.

    Rules:
    - Single digit 0-3: 2020-2023
    - Single digit 4-9: 2014-2019
    - Two+ digits: 2000 + value (e.g., 24 -> 2024)

    Examples:
        '6' -> 2016
        '0' -> 2020
        '24' -> 2024
        '26' -> 2026
    """
    if not year_str:
        return None

    try:
        val = int(year_str)
        if len(year_str) == 1:
            # Single digit
            if 0 <= val <= 3:
                return 2020 + val
            elif 4 <= val <= 9:
                return 2010 + val
            else:
                return None
        else:
            # Two or more digits
            return 2000 + val
    except:
        return None


def parse_contract_column(colname: str) -> Optional[Dict]:
    """
    Parse contract column name to extract asset, month_code, year.

    Format: [prefix][month_letter][year_digits]
    Examples:
        'bok26' -> {asset: 'Óleo de Soja', prefix: 'bo', month_code: 'k', month_num: 5, year: 2026}
        'smh25' -> {asset: 'Farelo de Soja', prefix: 'sm', month_code: 'h', month_num: 3, year: 2025}
        'sh27' -> {asset: 'Soja', prefix: 's', month_code: 'h', month_num: 3, year: 2027}

    Args:
        colname: Column name to parse

    Returns:
        Dictionary with parsed information or None if invalid
    """
    colname_lower = colname.lower().strip()

    # Try to match each asset prefix (order matters - check 'sm' before 's')
    for prefix, asset_name in CONFIG["asset_prefixes"]:
        if colname_lower.startswith(prefix):
            # Extract month code and year after prefix
            remainder = colname_lower[len(prefix):]

            if len(remainder) < 2:
                return None

            month_code = remainder[0]
            year_str = remainder[1:]

            # Validate month code
            if month_code not in CONFIG["month_code_map"]:
                return None

            # Parse year
            year = parse_year(year_str)
            if year is None:
                return None

            return {
                'asset': asset_name,
                'prefix': prefix,
                'month_code': month_code.upper(),
                'month_num': CONFIG["month_code_map"][month_code],
                'year': year,
                'contract_id': colname
            }

    return None


# ============================================================
# Data Loading Functions
# ============================================================
@st.cache_data(ttl=3600)
def load_data_from_pipeline() -> pd.DataFrame:
    """Load data from the existing data pipeline (WIDE format)."""
    return BASE_DF.copy()


@st.cache_data(ttl=3600)
def load_data_from_csv(path: str) -> pd.DataFrame:
    """Load data from CSV file (WIDE format)."""
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        st.error(f"Error loading CSV: {str(e)}")
        return pd.DataFrame()


@st.cache_data
def transform_wide_to_long(df_wide: pd.DataFrame) -> pd.DataFrame:
    """
    Transform WIDE format data to LONG format for seasonality analysis.

    WIDE format: date | bok26 | smh25 | sh27 | ...
    LONG format: date | contract_id | price | asset | month_code | year | ...

    Args:
        df_wide: Dataframe in wide format

    Returns:
        Dataframe in long format with contract metadata
    """
    date_col = CONFIG["date_col"]

    if date_col not in df_wide.columns:
        st.error(f"Date column '{date_col}' not found in data")
        return pd.DataFrame()

    # Identify contract columns
    contract_cols = []
    contract_metadata = {}

    for col in df_wide.columns:
        if col == date_col:
            continue

        parsed = parse_contract_column(col)
        if parsed:
            contract_cols.append(col)
            contract_metadata[col] = parsed

    if not contract_cols:
        st.warning("No valid contract columns found in data")
        return pd.DataFrame()

    # Melt to long format
    df_long = pd.melt(
        df_wide,
        id_vars=[date_col],
        value_vars=contract_cols,
        var_name='contract_id',
        value_name='price'
    )

    # Drop NaN prices
    df_long = df_long.dropna(subset=['price'])

    # Add metadata columns
    df_long['asset'] = df_long['contract_id'].map(lambda x: contract_metadata[x]['asset'])
    df_long['prefix'] = df_long['contract_id'].map(lambda x: contract_metadata[x]['prefix'])
    df_long['month_code'] = df_long['contract_id'].map(lambda x: contract_metadata[x]['month_code'])
    df_long['month_num'] = df_long['contract_id'].map(lambda x: contract_metadata[x]['month_num'])
    df_long['contract_year'] = df_long['contract_id'].map(lambda x: contract_metadata[x]['year'])

    return df_long


@st.cache_data
def preprocess_data(
    df_long: pd.DataFrame,
    asset_filter: Optional[str] = None,
    month_codes: Optional[List[str]] = None,
    min_year: Optional[int] = None,
    max_year: Optional[int] = None
) -> pd.DataFrame:
    """
    Preprocess and filter data for seasonality analysis.

    Args:
        df_long: Dataframe in long format
        asset_filter: Asset to filter (e.g., 'Óleo de Soja')
        month_codes: List of month codes to include (e.g., ['F', 'H', 'K'])
        min_year: Minimum year
        max_year: Maximum year

    Returns:
        Preprocessed dataframe with additional columns
    """
    df = df_long.copy()

    # Ensure date is datetime
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])

    # Filter by asset
    if asset_filter:
        df = df[df['asset'] == asset_filter].copy()

    # Filter by month codes
    if month_codes and 'All' not in month_codes:
        # Convert to uppercase for comparison
        month_codes_upper = [m.upper() for m in month_codes]
        df = df[df['month_code'].isin(month_codes_upper)].copy()

    # Add time-based columns from date
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['weekofyear'] = df['date'].dt.isocalendar().week
    df['dayofyear'] = df['date'].dt.dayofyear

    # Filter by year range (of the date, not contract year)
    if min_year:
        df = df[df['year'] >= min_year].copy()
    if max_year:
        df = df[df['year'] <= max_year].copy()

    # Calculate returns (grouped by contract_id)
    df = df.sort_values(['contract_id', 'date'])
    df['return'] = df.groupby('contract_id')['price'].pct_change() * 100

    return df.dropna(subset=['return'])


# ============================================================
# Seasonality Computation
# ============================================================
@st.cache_data
def compute_seasonality(
    df: pd.DataFrame,
    granularity: str = 'dayofyear',
    apply_smoothing: bool = False,
    rolling_window: int = 5
) -> pd.DataFrame:
    """
    Compute seasonality metrics aggregated by time bucket.

    Args:
        df: Preprocessed dataframe with returns
        granularity: 'dayofyear', 'weekofyear', or 'month'
        apply_smoothing: Whether to apply rolling average smoothing
        rolling_window: Window size for smoothing

    Returns:
        Aggregated seasonality metrics by bucket
    """
    bucket_col = granularity

    # Group by bucket
    grouped = df.groupby(bucket_col)['return']

    # Compute aggregations
    agg_dict = {
        'mean_return': grouped.mean(),
        'median_return': grouped.median(),
        'std_return': grouped.std(),
        'count': grouped.count(),
        'min_return': grouped.min(),
        'max_return': grouped.max(),
        'p05': grouped.quantile(0.05),
        'p25': grouped.quantile(0.25),
        'p75': grouped.quantile(0.75),
        'p95': grouped.quantile(0.95),
    }

    result = pd.DataFrame(agg_dict).reset_index()

    # Hit rate (% positive returns)
    hit_rates = df.groupby(bucket_col)['return'].apply(lambda x: (x > 0).sum() / len(x) if len(x) > 0 else 0)
    result['hit_rate'] = hit_rates.values

    # Sharpe-like ratio
    result['sharpe_like'] = np.where(
        result['std_return'] > 0,
        result['mean_return'] / result['std_return'],
        0
    )

    # Statistical significance (if scipy available)
    if HAS_SCIPY:
        p_values = []
        for bucket_val in result[bucket_col]:
            bucket_returns = df[df[bucket_col] == bucket_val]['return'].values
            if len(bucket_returns) > 2:
                _, p_val = stats.ttest_1samp(bucket_returns, 0)
                p_values.append(p_val)
            else:
                p_values.append(1.0)
        result['p_value'] = p_values
        result['is_significant'] = result['p_value'] < 0.05

    # Skewness and kurtosis
    if HAS_SCIPY:
        skew_vals = df.groupby(bucket_col)['return'].apply(lambda x: stats.skew(x) if len(x) > 2 else 0)
        kurt_vals = df.groupby(bucket_col)['return'].apply(lambda x: stats.kurtosis(x) if len(x) > 2 else 0)
        result['skewness'] = skew_vals.values
        result['kurtosis'] = kurt_vals.values

    # Filter by minimum samples
    result = result[result['count'] >= CONFIG['min_samples_per_bucket']].copy()

    # Apply smoothing if requested
    if apply_smoothing and len(result) > rolling_window:
        result['mean_return_smoothed'] = result['mean_return'].rolling(window=rolling_window, center=True).mean()
        result['std_return_smoothed'] = result['std_return'].rolling(window=rolling_window, center=True).mean()

    return result


# ============================================================
# Insight Generation
# ============================================================
def generate_actionable_insights(
    seasonality_df: pd.DataFrame,
    granularity: str
) -> List[str]:
    """
    Generate actionable insights from seasonality analysis.

    Args:
        seasonality_df: Seasonality metrics dataframe
        granularity: Time granularity used

    Returns:
        List of insight strings
    """
    insights = []

    if seasonality_df.empty:
        return ["⚠️ Insufficient data to generate insights."]

    bucket_col = granularity
    bucket_label = {
        'dayofyear': 'Day',
        'weekofyear': 'Week',
        'month': 'Month'
    }.get(granularity, granularity)

    # Best periods by Sharpe-like
    top_sharpe = seasonality_df.nlargest(CONFIG['top_n_periods'], 'sharpe_like')
    if not top_sharpe.empty:
        best_period = top_sharpe.iloc[0]
        insights.append(
            f"🟢 **Best Period**: {bucket_label} {int(best_period[bucket_col])} shows the strongest risk-adjusted return "
            f"(Sharpe-like: {best_period['sharpe_like']:.2f}, Mean Return: {best_period['mean_return']:.2f}%, "
            f"Hit Rate: {best_period['hit_rate']*100:.1f}%)"
        )

    # Worst periods by Sharpe-like
    bottom_sharpe = seasonality_df.nsmallest(CONFIG['top_n_periods'], 'sharpe_like')
    if not bottom_sharpe.empty:
        worst_period = bottom_sharpe.iloc[0]
        insights.append(
            f"🔴 **Worst Period**: {bucket_label} {int(worst_period[bucket_col])} shows the weakest risk-adjusted return "
            f"(Sharpe-like: {worst_period['sharpe_like']:.2f}, Mean Return: {worst_period['mean_return']:.2f}%, "
            f"Hit Rate: {worst_period['hit_rate']*100:.1f}%)"
        )

    # Consistent periods (high hit rate, low std)
    std_threshold = seasonality_df['std_return'].quantile(CONFIG['std_percentile_low'] / 100)
    consistent = seasonality_df[
        (seasonality_df['hit_rate'] >= CONFIG['hit_rate_threshold']) &
        (seasonality_df['std_return'] <= std_threshold)
    ]

    if not consistent.empty:
        best_consistent = consistent.nlargest(1, 'mean_return').iloc[0]
        insights.append(
            f"✅ **Consistent Period**: {bucket_label} {int(best_consistent[bucket_col])} is historically consistent "
            f"with {best_consistent['hit_rate']*100:.1f}% hit rate and below-median volatility "
            f"({best_consistent['std_return']:.2f}%)"
        )

    # High-risk periods (high std, negative skew)
    std_high_threshold = seasonality_df['std_return'].quantile(CONFIG['std_percentile_high'] / 100)
    risky = seasonality_df[seasonality_df['std_return'] >= std_high_threshold]

    if not risky.empty and HAS_SCIPY:
        risky_with_skew = risky[risky.get('skewness', 0) < -0.5]
        if not risky_with_skew.empty:
            most_risky = risky_with_skew.iloc[0]
            insights.append(
                f"⚠️ **High-Risk Period**: {bucket_label} {int(most_risky[bucket_col])} shows elevated volatility "
                f"({most_risky['std_return']:.2f}%) with negative skewness ({most_risky.get('skewness', 0):.2f}). "
                f"5th percentile return: {most_risky['p05']:.2f}%"
            )

    # Statistical significance
    if HAS_SCIPY and 'is_significant' in seasonality_df.columns:
        significant_positive = seasonality_df[
            (seasonality_df['is_significant']) &
            (seasonality_df['mean_return'] > 0)
        ]

        if not significant_positive.empty:
            top_sig = significant_positive.nlargest(1, 'mean_return').iloc[0]
            insights.append(
                f"📊 **Statistically Significant**: {bucket_label} {int(top_sig[bucket_col])} has a mean return of "
                f"{top_sig['mean_return']:.2f}% that is statistically significant (p-value: {top_sig['p_value']:.3f})"
            )

    # Overall hit rate summary
    overall_hit_rate = seasonality_df['hit_rate'].mean()
    above_avg = seasonality_df[seasonality_df['hit_rate'] > overall_hit_rate]
    insights.append(
        f"📈 **Overall Pattern**: {len(above_avg)} out of {len(seasonality_df)} periods show above-average hit rates "
        f"(avg: {overall_hit_rate*100:.1f}%)"
    )

    # Volatility regime
    high_vol_periods = len(seasonality_df[seasonality_df['std_return'] >= std_high_threshold])
    low_vol_periods = len(seasonality_df[seasonality_df['std_return'] <= std_threshold])
    insights.append(
        f"📉 **Volatility Distribution**: {high_vol_periods} high-volatility periods vs {low_vol_periods} low-volatility periods. "
        f"Median volatility: {seasonality_df['std_return'].median():.2f}%"
    )

    # Range of returns
    best_return = seasonality_df['mean_return'].max()
    worst_return = seasonality_df['mean_return'].min()
    insights.append(
        f"🎯 **Return Range**: Mean returns range from {worst_return:.2f}% to {best_return:.2f}% across all periods, "
        f"spanning {abs(best_return - worst_return):.2f}% points"
    )

    return insights


# ============================================================
# Visualization Functions
# ============================================================
def plot_seasonality_main(
    seasonality_df: pd.DataFrame,
    granularity: str,
    metric: str = 'mean_return',
    show_bands: bool = True
) -> go.Figure:
    """
    Create main seasonality line chart with confidence bands.

    Args:
        seasonality_df: Seasonality metrics dataframe
        granularity: Time granularity
        metric: Metric to plot
        show_bands: Whether to show ±1 std bands

    Returns:
        Plotly figure
    """
    bucket_col = granularity

    fig = go.Figure()

    # Use smoothed if available, otherwise raw
    y_col = f"{metric}_smoothed" if f"{metric}_smoothed" in seasonality_df.columns else metric

    # Main line
    fig.add_trace(
        go.Scatter(
            x=seasonality_df[bucket_col],
            y=seasonality_df[y_col],
            mode='lines+markers',
            name=metric.replace('_', ' ').title(),
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=6)
        )
    )

    # Confidence bands (±1 std)
    if show_bands and 'std_return' in seasonality_df.columns:
        upper = seasonality_df[y_col] + seasonality_df['std_return']
        lower = seasonality_df[y_col] - seasonality_df['std_return']

        fig.add_trace(
            go.Scatter(
                x=seasonality_df[bucket_col],
                y=upper,
                mode='lines',
                name='+1 Std',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            )
        )

        fig.add_trace(
            go.Scatter(
                x=seasonality_df[bucket_col],
                y=lower,
                mode='lines',
                name='-1 Std',
                line=dict(width=0),
                fillcolor='rgba(31, 119, 180, 0.2)',
                fill='tonexty',
                showlegend=True,
                hoverinfo='skip'
            )
        )

    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

    # Highlight significant periods
    if HAS_SCIPY and 'is_significant' in seasonality_df.columns:
        sig_periods = seasonality_df[seasonality_df['is_significant']]
        if not sig_periods.empty:
            fig.add_trace(
                go.Scatter(
                    x=sig_periods[bucket_col],
                    y=sig_periods[y_col],
                    mode='markers',
                    name='Statistically Significant (p<0.05)',
                    marker=dict(
                        size=12,
                        color='red',
                        symbol='star',
                        line=dict(color='darkred', width=1)
                    )
                )
            )

    bucket_label = {
        'dayofyear': 'Day of Year',
        'weekofyear': 'Week of Year',
        'month': 'Month'
    }.get(granularity, granularity)

    fig.update_layout(
        title=f"Seasonality Pattern - {metric.replace('_', ' ').title()}",
        xaxis_title=bucket_label,
        yaxis_title=metric.replace('_', ' ').title() + ' (%)',
        hovermode='x unified',
        height=500,
        template='plotly_white'
    )

    return fig


def plot_hit_rate_bars(seasonality_df: pd.DataFrame, granularity: str) -> go.Figure:
    """
    Create bar chart of hit rates by period.

    Args:
        seasonality_df: Seasonality metrics dataframe
        granularity: Time granularity

    Returns:
        Plotly figure
    """
    bucket_col = granularity

    # Color bars by hit rate level
    colors = ['green' if x >= 0.6 else 'orange' if x >= 0.5 else 'red'
              for x in seasonality_df['hit_rate']]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=seasonality_df[bucket_col],
            y=seasonality_df['hit_rate'] * 100,
            marker_color=colors,
            name='Hit Rate'
        )
    )

    # Reference line at 50%
    fig.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.5)

    bucket_label = {
        'dayofyear': 'Day of Year',
        'weekofyear': 'Week of Year',
        'month': 'Month'
    }.get(granularity, granularity)

    fig.update_layout(
        title="Hit Rate by Period",
        xaxis_title=bucket_label,
        yaxis_title="Hit Rate (%)",
        height=400,
        template='plotly_white',
        showlegend=False
    )

    return fig


def plot_heatmap_year_month(df: pd.DataFrame) -> go.Figure:
    """
    Create heatmap of returns by Year x Month.

    Args:
        df: Preprocessed dataframe with returns

    Returns:
        Plotly figure
    """
    # Pivot table: rows = year, cols = month
    pivot = df.pivot_table(
        values='return',
        index='year',
        columns='month',
        aggfunc='mean'
    )

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale='RdYlGn',
            zmid=0,
            text=np.round(pivot.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Mean Return (%)")
        )
    )

    fig.update_layout(
        title="Return Heatmap: Year vs Month",
        xaxis_title="Month",
        yaxis_title="Year",
        height=400,
        template='plotly_white'
    )

    return fig


# ============================================================
# Streamlit Page
# ============================================================
def main():
    st.markdown("# 📅 Análise de Sazonalidade")
    st.markdown("Identifique padrões sazonais em contratos futuros de commodities e gere insights acionáveis baseados em estatísticas históricas")
    st.divider()

    # Load data (WIDE format)
    with st.spinner("Carregando dados..."):
        if CONFIG["data_source"] == "pipeline":
            raw_df_wide = load_data_from_pipeline()
        else:
            raw_df_wide = load_data_from_csv(CONFIG["csv_path"])

        if raw_df_wide.empty:
            st.error("❌ No data available. Check your data source configuration.")
            st.stop()

        # Transform WIDE to LONG format
        df_long = transform_wide_to_long(raw_df_wide)

        if df_long.empty:
            st.error("❌ No valid contract columns found in data.")
            st.stop()

    # Get available assets and metadata
    available_assets = sorted(df_long['asset'].unique())
    available_month_codes = sorted(df_long['month_code'].unique())

    # Sidebar configuration
    with st.sidebar:
        st.markdown("## ⚙️ Configuração da Análise")

        # Asset selection
        selected_asset = st.selectbox(
            "Ativo",
            options=available_assets,
            help="Selecione o ativo para análise de sazonalidade"
        )

        # Month code selection
        selected_months = st.multiselect(
            "Contratos (Month Code)",
            options=['All'] + available_month_codes,
            default=['All'],
            help="Selecione os meses de vencimento dos contratos"
        )

        # Year range
        if 'date' in df_long.columns:
            min_year_data = int(df_long['date'].dt.year.min())
            max_year_data = int(df_long['date'].dt.year.max())

            year_range = st.slider(
                "Período (anos)",
                min_value=min_year_data,
                max_value=max_year_data,
                value=(min_year_data, max_year_data)
            )
            min_year, max_year = year_range
        else:
            min_year, max_year = None, None

        # Granularity
        granularity = st.selectbox(
            "Granularidade",
            options=['dayofyear', 'weekofyear', 'month'],
            format_func=lambda x: {
                'dayofyear': 'Dia do Ano (DoY)',
                'weekofyear': 'Semana do Ano (WoY)',
                'month': 'Mês'
            }[x]
        )

        # Metric
        metric_options = {
            'mean_return': 'Retorno Médio',
            'median_return': 'Retorno Mediano',
            'std_return': 'Volatilidade (Std)',
            'hit_rate': 'Taxa de Acerto (%)',
            'sharpe_like': 'Sharpe-like Ratio'
        }
        selected_metric = st.selectbox(
            "Métrica",
            options=list(metric_options.keys()),
            format_func=lambda x: metric_options[x]
        )

        # Smoothing
        apply_smoothing = st.checkbox(
            "Aplicar suavização (rolling 5d)",
            value=False,
            help="Suaviza a série com média móvel de 5 períodos"
        )

        st.divider()

        generate_report = st.button("🔍 Gerar Análise", type="primary", use_container_width=True)

    # Main content
    if not generate_report:
        st.info("👈 Configure os parâmetros na barra lateral e clique em **Gerar Análise**")

        # Show data preview
        with st.expander("📊 Preview dos Dados (LONG format)"):
            st.dataframe(df_long.head(100), use_container_width=True)

        st.stop()

    # Process data
    with st.spinner("Processando dados..."):
        processed_df = preprocess_data(
            df_long,
            asset_filter=selected_asset,
            month_codes=selected_months,
            min_year=min_year,
            max_year=max_year
        )

    if processed_df.empty:
        st.error("❌ Nenhum dado disponível após aplicar os filtros. Tente ajustar os parâmetros.")
        st.stop()

    # Compute seasonality
    with st.spinner("Calculando métricas de sazonalidade..."):
        seasonality_df = compute_seasonality(
            processed_df,
            granularity=granularity,
            apply_smoothing=apply_smoothing,
            rolling_window=CONFIG["rolling_window"]
        )

    if seasonality_df.empty:
        st.warning("⚠️ Dados insuficientes para análise de sazonalidade. Reduza o filtro de samples mínimos ou amplie o período.")
        st.stop()

    # KPIs
    st.markdown("## 📊 KPIs - Visão Geral")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        best_period = seasonality_df.nlargest(1, 'sharpe_like').iloc[0]
        st.metric(
            "Melhor Período (Sharpe)",
            f"{granularity.upper()}: {int(best_period[granularity])}",
            f"{best_period['mean_return']:.2f}%"
        )

    with col2:
        worst_period = seasonality_df.nsmallest(1, 'sharpe_like').iloc[0]
        st.metric(
            "Pior Período (Sharpe)",
            f"{granularity.upper()}: {int(worst_period[granularity])}",
            f"{worst_period['mean_return']:.2f}%"
        )

    with col3:
        avg_hit_rate = seasonality_df['hit_rate'].mean()
        st.metric(
            "Taxa de Acerto Média",
            f"{avg_hit_rate*100:.1f}%"
        )

    with col4:
        avg_sharpe = seasonality_df['sharpe_like'].mean()
        st.metric(
            "Sharpe-like Médio",
            f"{avg_sharpe:.2f}"
        )

    st.divider()

    # Main chart
    st.markdown(f"## 📈 Padrão de Sazonalidade - {metric_options[selected_metric]}")

    fig_main = plot_seasonality_main(
        seasonality_df,
        granularity,
        metric=selected_metric,
        show_bands=(selected_metric in ['mean_return', 'median_return'])
    )
    st.plotly_chart(fig_main, use_container_width=True)

    # Hit rate chart
    st.markdown("## 🎯 Taxa de Acerto por Período")
    fig_hit_rate = plot_hit_rate_bars(seasonality_df, granularity)
    st.plotly_chart(fig_hit_rate, use_container_width=True)

    # Heatmap (if granularity is month)
    if granularity == 'month' and len(processed_df) > 100:
        st.markdown("## 🔥 Heatmap de Retornos (Ano x Mês)")
        fig_heatmap = plot_heatmap_year_month(processed_df)
        st.plotly_chart(fig_heatmap, use_container_width=True)

    st.divider()

    # Rankings
    st.markdown("## 🏆 Rankings de Períodos")

    col_rank1, col_rank2 = st.columns(2)

    with col_rank1:
        st.markdown("### 🟢 Top 5 - Melhores Períodos (Sharpe)")
        top_periods = seasonality_df.nlargest(5, 'sharpe_like')[[
            granularity, 'mean_return', 'hit_rate', 'std_return', 'sharpe_like', 'count'
        ]].copy()
        top_periods['hit_rate'] = (top_periods['hit_rate'] * 100).round(1)
        st.dataframe(top_periods, use_container_width=True, hide_index=True)

    with col_rank2:
        st.markdown("### 🔴 Bottom 5 - Piores Períodos (Sharpe)")
        bottom_periods = seasonality_df.nsmallest(5, 'sharpe_like')[[
            granularity, 'mean_return', 'hit_rate', 'std_return', 'sharpe_like', 'count'
        ]].copy()
        bottom_periods['hit_rate'] = (bottom_periods['hit_rate'] * 100).round(1)
        st.dataframe(bottom_periods, use_container_width=True, hide_index=True)

    st.divider()

    # Actionable Insights
    st.markdown("## 💡 Insights Acionáveis")

    with st.spinner("Gerando insights..."):
        insights = generate_actionable_insights(seasonality_df, granularity)

    for insight in insights:
        st.markdown(insight)

    st.divider()

    # Export
    st.markdown("## 📥 Exportar Dados")

    col_exp1, col_exp2 = st.columns(2)

    with col_exp1:
        # Export seasonality metrics
        export_df = seasonality_df.copy()
        export_df['hit_rate'] = (export_df['hit_rate'] * 100).round(2)

        csv_seasonality = export_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Baixar Métricas de Sazonalidade (CSV)",
            data=csv_seasonality,
            file_name=f"seasonality_{selected_asset or 'all'}_{granularity}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

    with col_exp2:
        # Export insights as text
        insights_text = "\n".join(insights)
        st.download_button(
            "📥 Baixar Insights (TXT)",
            data=insights_text.encode('utf-8'),
            file_name=f"insights_{selected_asset or 'all'}_{granularity}_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain"
        )

    # Footer info
    st.divider()
    st.caption(f"""
    **Configuração da Análise:**
    - Ativo: {selected_asset or 'Todos'}
    - Contratos: {', '.join(selected_months)}
    - Período: {min_year} - {max_year}
    - Granularidade: {metric_options.get(granularity, granularity)}
    - Total de observações: {len(processed_df):,}
    - Períodos analisados: {len(seasonality_df)}
    - Mínimo de amostras por bucket: {CONFIG['min_samples_per_bucket']}
    """)


# ============================================================
# Run
# ============================================================
if __name__ == "__main__":
    main()
