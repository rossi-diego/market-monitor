# ============================================================
# Análise de Sazonalidade - Contratos Futuros
# ============================================================
"""
Análise profissional de sazonalidade em contratos futuros de commodities.
Analisa padrões históricos por mês de vencimento específico (ex: todos os Janeiro - F).
"""

# ============================================================
# Imports & Config
# ============================================================
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from src.data_pipeline import df as BASE_DF
from src.utils import apply_theme, section

# Apply theme
apply_theme()

# Page header
st.markdown("# 📅 Análise de Sazonalidade - Contratos Futuros")
st.markdown("Identifique padrões sazonais em contratos futuros específicos e descubra os melhores períodos para negociar")
st.divider()

# ============================================================
# Configuration
# ============================================================
# Asset prefix mapping (order matters - check 'sm' before 's')
ASSET_PREFIXES = [
    ("sm", "Farelo de Soja"),
    ("bo", "Óleo de Soja"),
    ("s", "Soja"),
    ("c", "Milho"),
    ("w", "Trigo"),
]

# Month code mapping
MONTH_CODES = {
    'f': ('Janeiro', 1),
    'g': ('Fevereiro', 2),
    'h': ('Março', 3),
    'j': ('Abril', 4),
    'k': ('Maio', 5),
    'm': ('Junho', 6),
    'n': ('Julho', 7),
    'q': ('Agosto', 8),
    'u': ('Setembro', 9),
    'v': ('Outubro', 10),
    'x': ('Novembro', 11),
    'z': ('Dezembro', 12),
}

# ============================================================
# Helper Functions
# ============================================================
def parse_year(year_str: str) -> int:
    """
    Parse year from contract string.

    Rules:
    - 0-3 (single digit): 2020-2023
    - 4-9 (single digit): 2014-2019
    - 24, 25, 26... (2+ digits): 2024, 2025, 2026...
    """
    if not year_str:
        return None

    try:
        val = int(year_str)
        if len(year_str) == 1:
            if 0 <= val <= 3:
                return 2020 + val
            elif 4 <= val <= 9:
                return 2010 + val
        else:
            return 2000 + val
    except:
        return None


def parse_contract_column(colname: str) -> dict:
    """
    Parse contract column name to extract metadata.

    Format: [prefix][month_letter][year]
    Examples:
        'bok26' -> {asset: 'Óleo de Soja', month_code: 'K', month_name: 'Maio', year: 2026}
        'smf25' -> {asset: 'Farelo de Soja', month_code: 'F', month_name: 'Janeiro', year: 2025}
    """
    colname_lower = colname.lower().strip()

    for prefix, asset_name in ASSET_PREFIXES:
        if colname_lower.startswith(prefix):
            remainder = colname_lower[len(prefix):]

            if len(remainder) < 2:
                return None

            month_code = remainder[0]
            year_str = remainder[1:]

            if month_code not in MONTH_CODES:
                return None

            year = parse_year(year_str)
            if year is None:
                return None

            month_name, month_num = MONTH_CODES[month_code]

            return {
                'contract_id': colname,
                'asset': asset_name,
                'month_code': month_code.upper(),
                'month_name': month_name,
                'month_num': month_num,
                'year': year
            }

    return None


@st.cache_data
def extract_futures_contracts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract and parse all futures contracts from the dataframe.

    Returns:
        DataFrame with columns: date, contract_id, price, asset, month_code, month_name, year
    """
    date_col = 'date'

    # Identify contract columns
    contract_data = []

    for col in df.columns:
        if col == date_col:
            continue

        parsed = parse_contract_column(col)
        if parsed:
            contract_data.append(parsed)

    if not contract_data:
        return pd.DataFrame()

    # Melt to long format
    contract_cols = [c['contract_id'] for c in contract_data]
    df_long = pd.melt(
        df,
        id_vars=[date_col],
        value_vars=contract_cols,
        var_name='contract_id',
        value_name='price'
    )

    # Drop NaN prices
    df_long = df_long.dropna(subset=['price'])

    # Add metadata
    metadata_map = {c['contract_id']: c for c in contract_data}
    df_long['asset'] = df_long['contract_id'].map(lambda x: metadata_map[x]['asset'])
    df_long['month_code'] = df_long['contract_id'].map(lambda x: metadata_map[x]['month_code'])
    df_long['month_name'] = df_long['contract_id'].map(lambda x: metadata_map[x]['month_name'])
    df_long['month_num'] = df_long['contract_id'].map(lambda x: metadata_map[x]['month_num'])
    df_long['contract_year'] = df_long['contract_id'].map(lambda x: metadata_map[x]['year'])

    # Convert date
    df_long['date'] = pd.to_datetime(df_long['date'])

    return df_long


@st.cache_data
def calculate_seasonality_by_contract_month(
    df_contracts: pd.DataFrame,
    asset: str,
    month_code: str,
    years: list
) -> pd.DataFrame:
    """
    Calculate seasonality statistics for a specific contract month across multiple years.

    Args:
        df_contracts: Long format dataframe with all contracts
        asset: Asset name (e.g., 'Óleo de Soja')
        month_code: Month code (e.g., 'F' for January)
        years: List of years to analyze

    Returns:
        DataFrame with seasonality statistics by day of year
    """
    # Filter by asset, month, and years
    df_filtered = df_contracts[
        (df_contracts['asset'] == asset) &
        (df_contracts['month_code'] == month_code) &
        (df_contracts['contract_year'].isin(years))
    ].copy()

    if df_filtered.empty:
        return pd.DataFrame()

    # Calculate returns by contract
    df_filtered = df_filtered.sort_values(['contract_id', 'date'])
    df_filtered['return'] = df_filtered.groupby('contract_id')['price'].pct_change() * 100
    df_filtered = df_filtered.dropna(subset=['return'])

    # Add calendar features
    df_filtered['dayofyear'] = df_filtered['date'].dt.dayofyear
    df_filtered['month'] = df_filtered['date'].dt.month

    # Group by day of year
    grouped = df_filtered.groupby('dayofyear')['return']

    # Calculate statistics
    stats_df = pd.DataFrame({
        'dia_do_ano': grouped.apply(lambda x: x.name).index,
        'retorno_medio': grouped.mean(),
        'retorno_mediano': grouped.median(),
        'desvio_padrao': grouped.std(),
        'minimo': grouped.min(),
        'maximo': grouped.max(),
        'observacoes': grouped.count(),
    }).reset_index(drop=True)

    # Hit rate (% positive returns)
    stats_df['taxa_acerto'] = df_filtered.groupby('dayofyear')['return'].apply(
        lambda x: (x > 0).sum() / len(x) * 100 if len(x) > 0 else 0
    ).values

    # Sharpe-like ratio
    stats_df['sharpe_like'] = np.where(
        stats_df['desvio_padrao'] > 0,
        stats_df['retorno_medio'] / stats_df['desvio_padrao'],
        0
    )

    # Statistical significance
    if HAS_SCIPY:
        p_values = []
        for doy in stats_df['dia_do_ano']:
            doy_returns = df_filtered[df_filtered['dayofyear'] == doy]['return'].values
            if len(doy_returns) > 2:
                _, p_val = stats.ttest_1samp(doy_returns, 0)
                p_values.append(p_val)
            else:
                p_values.append(1.0)
        stats_df['p_valor'] = p_values
        stats_df['significante'] = stats_df['p_valor'] < 0.05

    # Filter by minimum observations
    stats_df = stats_df[stats_df['observacoes'] >= 3].copy()

    # Add cumulative return
    stats_df['retorno_acumulado'] = stats_df['retorno_medio'].cumsum()

    return stats_df


def get_date_label(day_of_year: int) -> str:
    """Convert day of year to readable date format (DD/MM)."""
    try:
        date = pd.Timestamp(f'2024-01-01') + pd.Timedelta(days=int(day_of_year) - 1)
        return date.strftime('%d/%m')
    except:
        return str(int(day_of_year))


# ============================================================
# Load and parse contracts
# ============================================================
with st.spinner("Carregando e processando contratos futuros..."):
    df_contracts = extract_futures_contracts(BASE_DF)

if df_contracts.empty:
    st.error("❌ Nenhum contrato futuro encontrado no dataset. Verifique o formato dos dados.")
    st.stop()

# Get available options
available_assets = sorted(df_contracts['asset'].unique())
available_months = sorted(
    df_contracts[['month_code', 'month_name', 'month_num']].drop_duplicates().values.tolist(),
    key=lambda x: x[2]
)
available_years = sorted(df_contracts['contract_year'].unique())

st.success(f"✅ {len(df_contracts):,} observações de contratos futuros carregadas com sucesso!")

# ============================================================
# Sidebar - Configuration
# ============================================================
with st.sidebar:
    section("Configuração", "Parâmetros da análise", "⚙️")

    # Asset selection
    st.markdown("### 📦 Ativo")
    selected_asset = st.selectbox(
        "Escolha o ativo",
        options=available_assets,
        help="Commodity para análise de sazonalidade"
    )

    # Month selection
    st.markdown("### 📅 Mês do Contrato Futuro")
    month_options = {f"{code} - {name}": code for code, name, _ in available_months}
    selected_month_display = st.selectbox(
        "Escolha o mês de vencimento",
        options=list(month_options.keys()),
        help="Mês de vencimento do contrato futuro (ex: F = Janeiro)"
    )
    selected_month_code = month_options[selected_month_display]
    selected_month_name = selected_month_display.split(' - ')[1]

    # Year selection
    st.markdown("### 📆 Anos para Análise")

    # Get available years for this asset/month combination
    available_years_filtered = sorted(
        df_contracts[
            (df_contracts['asset'] == selected_asset) &
            (df_contracts['month_code'] == selected_month_code)
        ]['contract_year'].unique()
    )

    if len(available_years_filtered) < 2:
        st.error("❌ Menos de 2 anos disponíveis para esta combinação. Selecione outro ativo/mês.")
        st.stop()

    selected_years = st.multiselect(
        "Selecione os anos",
        options=available_years_filtered,
        default=available_years_filtered,
        help="Anos dos contratos a serem incluídos na análise"
    )

    if len(selected_years) < 2:
        st.warning("⚠️ Selecione pelo menos 2 anos para análise estatística robusta.")
        st.stop()

    st.markdown("---")
    st.caption(f"""
    **Resumo da Seleção:**
    - Ativo: {selected_asset}
    - Contrato: {selected_month_display}
    - Anos: {len(selected_years)} contratos
    """)

# ============================================================
# Calculate Seasonality
# ============================================================
with st.spinner("Calculando padrões de sazonalidade..."):
    seasonality_stats = calculate_seasonality_by_contract_month(
        df_contracts,
        selected_asset,
        selected_month_code,
        selected_years
    )

if seasonality_stats.empty:
    st.error("❌ Dados insuficientes para análise. Tente selecionar mais anos ou outro contrato.")
    st.stop()

# ============================================================
# KPIs Summary
# ============================================================
st.markdown("## 📊 Resumo Executivo")

with st.container(border=True):
    st.markdown(f"### Análise do Contrato **{selected_asset} - {selected_month_name} ({selected_month_code})**")
    st.caption(f"Baseado em {len(selected_years)} contratos: {', '.join(map(str, selected_years))}")

    col1, col2, col3, col4, col5 = st.columns(5)

    # Best trading period
    best_idx = seasonality_stats['sharpe_like'].idxmax()
    best_period = seasonality_stats.loc[best_idx]

    with col1:
        st.metric(
            "Melhor Período",
            get_date_label(best_period['dia_do_ano']),
            f"{best_period['retorno_medio']:.2f}%",
            help=f"Dia do ano com melhor retorno ajustado por risco (Sharpe: {best_period['sharpe_like']:.2f})"
        )

    # Worst trading period
    worst_idx = seasonality_stats['sharpe_like'].idxmin()
    worst_period = seasonality_stats.loc[worst_idx]

    with col2:
        st.metric(
            "Pior Período",
            get_date_label(worst_period['dia_do_ano']),
            f"{worst_period['retorno_medio']:.2f}%",
            help=f"Dia do ano com pior retorno ajustado por risco (Sharpe: {worst_period['sharpe_like']:.2f})"
        )

    # Average hit rate
    avg_hit_rate = seasonality_stats['taxa_acerto'].mean()
    hit_color = "🟢" if avg_hit_rate > 55 else "🟡" if avg_hit_rate > 50 else "🔴"

    with col3:
        st.metric(
            "Taxa de Acerto Média",
            f"{avg_hit_rate:.1f}%",
            f"{hit_color}",
            help="% médio de dias com retorno positivo"
        )

    # Cumulative return
    total_cumulative = seasonality_stats['retorno_acumulado'].iloc[-1]
    cum_color = "🟢" if total_cumulative > 0 else "🔴"

    with col4:
        st.metric(
            "Retorno Acumulado",
            f"{total_cumulative:.2f}%",
            f"{cum_color}",
            help="Soma dos retornos médios ao longo do ano"
        )

    # Total observations
    total_obs = seasonality_stats['observacoes'].sum()

    with col5:
        st.metric(
            "Total de Observações",
            f"{total_obs:,}",
            help=f"{len(seasonality_stats)} dias analisados"
        )

st.divider()

# ============================================================
# Main Chart - Seasonality Pattern
# ============================================================
st.markdown("## 📈 Padrão de Sazonalidade Anual")

with st.container(border=True):
    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.45, 0.30, 0.25],
        subplot_titles=(
            "Retorno Médio Diário (%)",
            "Retorno Acumulado (%)",
            "Taxa de Acerto (%)"
        ),
        vertical_spacing=0.08
    )

    # Panel 1: Mean return with confidence bands
    fig.add_trace(
        go.Scatter(
            x=seasonality_stats['dia_do_ano'],
            y=seasonality_stats['retorno_medio'],
            mode='lines',
            name='Retorno Médio',
            line=dict(color='#1f77b4', width=2),
            hovertemplate='Dia %{x}: %{y:.2f}%<extra></extra>'
        ),
        row=1, col=1
    )

    # Confidence bands
    upper = seasonality_stats['retorno_medio'] + seasonality_stats['desvio_padrao']
    lower = seasonality_stats['retorno_medio'] - seasonality_stats['desvio_padrao']

    fig.add_trace(
        go.Scatter(
            x=seasonality_stats['dia_do_ano'],
            y=upper,
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=seasonality_stats['dia_do_ano'],
            y=lower,
            mode='lines',
            line=dict(width=0),
            fillcolor='rgba(31, 119, 180, 0.15)',
            fill='tonexty',
            name='±1 Desvio Padrão',
            hoverinfo='skip'
        ),
        row=1, col=1
    )

    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.3, row=1, col=1)

    # Highlight significant periods
    if HAS_SCIPY and 'significante' in seasonality_stats.columns:
        sig_periods = seasonality_stats[seasonality_stats['significante']]
        if not sig_periods.empty:
            fig.add_trace(
                go.Scatter(
                    x=sig_periods['dia_do_ano'],
                    y=sig_periods['retorno_medio'],
                    mode='markers',
                    name='Significante (p<0.05)',
                    marker=dict(size=8, color='red', symbol='star'),
                    hovertemplate='Dia %{x}: %{y:.2f}% (p<0.05)<extra></extra>'
                ),
                row=1, col=1
            )

    # Panel 2: Cumulative return
    fig.add_trace(
        go.Scatter(
            x=seasonality_stats['dia_do_ano'],
            y=seasonality_stats['retorno_acumulado'],
            mode='lines',
            name='Retorno Acumulado',
            line=dict(color='green', width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 128, 0, 0.1)',
            hovertemplate='Dia %{x}: %{y:.2f}%<extra></extra>'
        ),
        row=2, col=1
    )

    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.3, row=2, col=1)

    # Panel 3: Hit rate
    colors = ['green' if x >= 60 else 'orange' if x >= 50 else 'red'
              for x in seasonality_stats['taxa_acerto']]

    fig.add_trace(
        go.Bar(
            x=seasonality_stats['dia_do_ano'],
            y=seasonality_stats['taxa_acerto'],
            marker_color=colors,
            name='Taxa de Acerto',
            hovertemplate='Dia %{x}: %{y:.1f}%<extra></extra>',
            showlegend=False
        ),
        row=3, col=1
    )

    fig.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.5, row=3, col=1)

    # Update layout
    fig.update_xaxes(title_text="Dia do Ano", row=3, col=1)
    fig.update_yaxes(title_text="%", row=1, col=1)
    fig.update_yaxes(title_text="%", row=2, col=1)
    fig.update_yaxes(title_text="%", row=3, col=1)

    fig.update_layout(
        height=900,
        hovermode='x unified',
        template='plotly_white',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("ℹ️ Como interpretar os gráficos"):
        st.markdown(f"""
        ### 📊 Interpretação dos Painéis

        **Painel 1 - Retorno Médio Diário:**
        - Mostra o retorno médio esperado para cada dia do ano
        - **Banda cinza**: Volatilidade (±1 desvio padrão) - quanto maior, mais incerto
        - **Estrelas vermelhas**: Dias com retorno estatisticamente significante
        - **Acima de zero**: Dias historicamente positivos para compra
        - **Abaixo de zero**: Dias historicamente negativos

        **Painel 2 - Retorno Acumulado:**
        - Soma cumulativa dos retornos médios ao longo do ano
        - **Tendência de alta**: Períodos favoráveis para posições compradas
        - **Tendência de baixa**: Períodos desfavoráveis
        - **Picos e vales**: Indicam melhores momentos para entrar/sair

        **Painel 3 - Taxa de Acerto:**
        - **Verde (≥60%)**: Alta probabilidade de retorno positivo
        - **Laranja (50-60%)**: Probabilidade neutra
        - **Vermelho (<50%)**: Alta probabilidade de retorno negativo

        ### 🎯 Como Usar para Trading

        1. **Identificar janelas de alta probabilidade**: Dias com retorno médio positivo + taxa de acerto >60%
        2. **Avaliar risco**: Bandas largas = maior incerteza, aguardar confirmação
        3. **Planejar entradas**: Entrar antes dos picos de retorno acumulado
        4. **Planejar saídas**: Sair próximo aos topos ou antes de quedas históricas
        5. **Validar com significância**: Estrelas vermelhas indicam padrões estatisticamente robustos
        """)

st.divider()

# ============================================================
# Price Bands Analysis
# ============================================================
st.markdown("## 📊 Bandas de Preço Esperadas")

with st.container(border=True):
    # Calculate percentiles
    percentiles = [10, 25, 50, 75, 90]

    st.markdown(f"""
    ### Faixas de Retorno Diário - {selected_asset} {selected_month_name}

    Com base nos dados históricos, aqui estão as faixas de retorno esperadas para cada dia:
    """)

    col_p1, col_p2, col_p3 = st.columns(3)

    with col_p1:
        p10 = seasonality_stats['retorno_medio'].quantile(0.10)
        p25 = seasonality_stats['retorno_medio'].quantile(0.25)
        st.metric("Retorno P10-P25", f"{p10:.2f}% a {p25:.2f}%", help="10% dos dias têm retorno nesta faixa ou abaixo")

    with col_p2:
        p25 = seasonality_stats['retorno_medio'].quantile(0.25)
        p75 = seasonality_stats['retorno_medio'].quantile(0.75)
        st.metric("Retorno P25-P75 (IQR)", f"{p25:.2f}% a {p75:.2f}%", help="50% dos dias têm retorno nesta faixa (zona normal)")

    with col_p3:
        p75 = seasonality_stats['retorno_medio'].quantile(0.75)
        p90 = seasonality_stats['retorno_medio'].quantile(0.90)
        st.metric("Retorno P75-P90", f"{p75:.2f}% a {p90:.2f}%", help="10% dos dias têm retorno nesta faixa ou acima")

    st.info("""
    💡 **Como usar essas bandas:**
    - **P25-P75 (IQR)**: Zona de retorno "normal" - 50% dos dias estão aqui
    - **Abaixo de P10**: Dias com retorno excepcionalmente baixo (oportunidades de compra?)
    - **Acima de P90**: Dias com retorno excepcionalmente alto (oportunidades de realização?)
    """)

st.divider()

# ============================================================
# Top/Bottom Periods
# ============================================================
st.markdown("## 🏆 Melhores e Piores Períodos para Negociar")

with st.container(border=True):
    col_rank1, col_rank2 = st.columns(2)

    with col_rank1:
        st.markdown("### 🟢 Top 10 - Melhores Dias")
        st.caption("Ordenado por Sharpe-like (retorno ajustado por risco)")

        top10 = seasonality_stats.nlargest(10, 'sharpe_like').copy()
        top10['data'] = top10['dia_do_ano'].apply(get_date_label)

        display_top = top10[['data', 'retorno_medio', 'taxa_acerto', 'desvio_padrao', 'sharpe_like', 'observacoes']].copy()
        display_top.columns = ['Data (DD/MM)', 'Ret. Médio (%)', 'Tx. Acerto (%)', 'Volatilidade (%)', 'Sharpe-like', 'Obs.']

        st.dataframe(
            display_top.style.format({
                'Ret. Médio (%)': '{:.2f}',
                'Tx. Acerto (%)': '{:.1f}',
                'Volatilidade (%)': '{:.2f}',
                'Sharpe-like': '{:.2f}',
                'Obs.': '{:.0f}'
            }).background_gradient(subset=['Sharpe-like'], cmap='RdYlGn', vmin=-1, vmax=2),
            use_container_width=True,
            hide_index=True
        )

        st.caption("✅ Dias com Sharpe-like > 1.0 são excelentes oportunidades")

    with col_rank2:
        st.markdown("### 🔴 Bottom 10 - Piores Dias")
        st.caption("Ordenado por Sharpe-like (retorno ajustado por risco)")

        bottom10 = seasonality_stats.nsmallest(10, 'sharpe_like').copy()
        bottom10['data'] = bottom10['dia_do_ano'].apply(get_date_label)

        display_bottom = bottom10[['data', 'retorno_medio', 'taxa_acerto', 'desvio_padrao', 'sharpe_like', 'observacoes']].copy()
        display_bottom.columns = ['Data (DD/MM)', 'Ret. Médio (%)', 'Tx. Acerto (%)', 'Volatilidade (%)', 'Sharpe-like', 'Obs.']

        st.dataframe(
            display_bottom.style.format({
                'Ret. Médio (%)': '{:.2f}',
                'Tx. Acerto (%)': '{:.1f}',
                'Volatilidade (%)': '{:.2f}',
                'Sharpe-like': '{:.2f}',
                'Obs.': '{:.0f}'
            }).background_gradient(subset=['Sharpe-like'], cmap='RdYlGn_r', vmin=-2, vmax=1),
            use_container_width=True,
            hide_index=True
        )

        st.caption("⚠️ Evite ou use hedge nestes períodos historicamente fracos")

st.divider()

# ============================================================
# Actionable Insights
# ============================================================
st.markdown("## 💡 Insights Profissionais")

with st.container(border=True):
    insights = []

    # Best trading window
    best_windows = seasonality_stats[
        (seasonality_stats['sharpe_like'] > 0.5) &
        (seasonality_stats['taxa_acerto'] > 55)
    ]

    if not best_windows.empty:
        best_start = best_windows.iloc[0]['dia_do_ano']
        best_end = best_windows.iloc[-1]['dia_do_ano']
        avg_return = best_windows['retorno_medio'].mean()
        avg_hit = best_windows['taxa_acerto'].mean()

        insights.append(f"""
        **🎯 Janela de Oportunidade Identificada:**

        Período entre **{get_date_label(best_start)}** e **{get_date_label(best_end)}** ({len(best_windows)} dias) apresenta:
        - Retorno médio diário: **{avg_return:.2f}%**
        - Taxa de acerto: **{avg_hit:.1f}%**
        - Sharpe-like médio: **{best_windows['sharpe_like'].mean():.2f}**

        💰 **Estratégia sugerida**: Considere posições compradas neste período, com stop loss baseado no desvio padrão.
        """)

    # High volatility periods
    high_vol = seasonality_stats.nlargest(5, 'desvio_padrao')
    if not high_vol.empty:
        vol_period = high_vol.iloc[0]
        insights.append(f"""
        **⚠️ Período de Alta Volatilidade:**

        O dia **{get_date_label(vol_period['dia_do_ano'])}** historicamente apresenta a maior volatilidade:
        - Desvio padrão: **{vol_period['desvio_padrao']:.2f}%**
        - Range de retornos: **{vol_period['minimo']:.2f}%** a **{vol_period['maximo']:.2f}%**

        ⚡ **Atenção**: Período de alta incerteza - ajuste o tamanho da posição ou aguarde confirmação.
        """)

    # Consistent patterns
    consistent = seasonality_stats[
        (seasonality_stats['taxa_acerto'] >= 60) &
        (seasonality_stats['desvio_padrao'] < seasonality_stats['desvio_padrao'].median())
    ]

    if not consistent.empty:
        best_consistent = consistent.nlargest(1, 'retorno_medio').iloc[0]
        insights.append(f"""
        **✅ Padrão Mais Consistente:**

        O dia **{get_date_label(best_consistent['dia_do_ano'])}** combina:
        - Retorno médio: **{best_consistent['retorno_medio']:.2f}%**
        - Taxa de acerto: **{best_consistent['taxa_acerto']:.1f}%**
        - Baixa volatilidade: **{best_consistent['desvio_padrao']:.2f}%**

        🛡️ **Recomendação**: Ideal para estratégias de menor risco ou posições maiores.
        """)

    # Negative zones
    negative_zone = seasonality_stats[seasonality_stats['retorno_medio'] < -0.1]
    if not negative_zone.empty:
        worst_run_start = negative_zone.iloc[0]['dia_do_ano']
        worst_run_end = negative_zone.iloc[-1]['dia_do_ano']
        avg_neg = negative_zone['retorno_medio'].mean()

        insights.append(f"""
        **🔴 Zona de Risco Identificada:**

        Período entre **{get_date_label(worst_run_start)}** e **{get_date_label(worst_run_end)}** ({len(negative_zone)} dias):
        - Retorno médio: **{avg_neg:.2f}%** (negativo)
        - Taxa de acerto: **{negative_zone['taxa_acerto'].mean():.1f}%**

        🛑 **Ação recomendada**: Evite posições compradas ou considere proteção com hedge neste período.
        """)

    # Statistical robustness
    if HAS_SCIPY and 'significante' in seasonality_stats.columns:
        sig_count = seasonality_stats['significante'].sum()
        sig_pct = sig_count / len(seasonality_stats) * 100

        if sig_pct >= 20:
            insights.append(f"""
            **📈 Robustez Estatística:**

            **{sig_count}** de **{len(seasonality_stats)}** dias ({sig_pct:.0f}%) apresentam padrões estatisticamente significantes (p < 0.05).

            ✅ **Confiabilidade**: Os padrões observados têm boa robustez estatística e não são fruto apenas do acaso.
            """)

    # Display insights
    for insight in insights:
        st.info(insight)

st.divider()

# ============================================================
# Export
# ============================================================
st.markdown("## 📥 Exportar Análise")

col_exp1, col_exp2 = st.columns(2)

with col_exp1:
    # Export statistics
    export_stats = seasonality_stats.copy()
    export_stats['data'] = export_stats['dia_do_ano'].apply(get_date_label)

    csv_data = export_stats.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Baixar Estatísticas (CSV)",
        data=csv_data,
        file_name=f"sazonalidade_{selected_asset}_{selected_month_code}_{min(selected_years)}_{max(selected_years)}.csv",
        mime="text/csv",
        key="download_stats"
    )

with col_exp2:
    # Export insights
    insights_text = f"""ANÁLISE DE SAZONALIDADE - {selected_asset} {selected_month_name} ({selected_month_code})
{'='*80}

Anos Analisados: {', '.join(map(str, selected_years))}
Total de Observações: {total_obs:,}
Dias Analisados: {len(seasonality_stats)}

{'='*80}
RESUMO EXECUTIVO
{'='*80}

Melhor Período: {get_date_label(best_period['dia_do_ano'])} (Sharpe: {best_period['sharpe_like']:.2f})
Pior Período: {get_date_label(worst_period['dia_do_ano'])} (Sharpe: {worst_period['sharpe_like']:.2f})
Taxa de Acerto Média: {avg_hit_rate:.1f}%
Retorno Acumulado: {total_cumulative:.2f}%

{'='*80}
INSIGHTS ACIONÁVEIS
{'='*80}

""" + "\n\n".join(insights)

    st.download_button(
        "📥 Baixar Insights (TXT)",
        data=insights_text.encode('utf-8'),
        file_name=f"insights_{selected_asset}_{selected_month_code}_{min(selected_years)}_{max(selected_years)}.txt",
        mime="text/plain",
        key="download_insights"
    )

# Footer
st.divider()
st.caption(f"""
📊 **Análise de Sazonalidade:** {selected_asset} - {selected_month_name} ({selected_month_code}) |
Contratos: {', '.join(map(str, selected_years))} |
Observações: {total_obs:,} | Dias: {len(seasonality_stats)}
""")
