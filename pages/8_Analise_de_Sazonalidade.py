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
st.markdown("# 📅 Análise de Sazonalidade - Decisão de Fixação de Contratos")
st.markdown("Identifique os melhores períodos para fixar contratos frame com base em padrões históricos")
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
    - Format: DIGIT(S) + OPTIONAL(^1 or ^2)
    - Single digit (0-9): 2020-2029 if has ^2, else 2010-2019 if has ^1
    - Two+ digits: 2000 + value (ex: '24' = 2024, '25' = 2025)

    Examples:
        '6^1' -> 2016 (decade marker ^1 = 2010s)
        '0^2' -> 2020 (decade marker ^2 = 2020s)
        '24' or '24^2' -> 2024
        '26' -> 2026
    """
    if not year_str:
        return None

    try:
        # Check for decade marker
        has_marker_1 = '^1' in year_str
        has_marker_2 = '^2' in year_str

        # Extract numeric part only
        clean_year = year_str.split('^')[0]
        val = int(clean_year)

        if len(clean_year) == 1:
            # Single digit: determine decade by marker
            if has_marker_1:
                return 2010 + val  # ^1 = 2010s (2016, 2017, etc.)
            elif has_marker_2:
                return 2020 + val  # ^2 = 2020s (2020, 2021, 2022, 2023)
            else:
                # No marker: assume 2020s for 0-3, 2010s for 4-9 (legacy logic)
                return 2020 + val if val <= 3 else 2010 + val
        else:
            # Two+ digits: add 2000
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


@st.cache_data
def calculate_fixation_metrics(
    df_contracts: pd.DataFrame,
    asset: str,
    month_code: str,
    years: list
) -> pd.DataFrame:
    """
    Calculate fixation metrics: probability of success by calendar month.

    For each calendar month, calculate:
    - Avg return from fixation to contract maturity
    - Win rate (% of times price increased)
    - Expected drawdown
    - Best/worst case scenarios

    Returns:
        DataFrame with fixation metrics by calendar month
    """
    # Filter contracts
    df_filtered = df_contracts[
        (df_contracts['asset'] == asset) &
        (df_contracts['month_code'] == month_code) &
        (df_contracts['contract_year'].isin(years))
    ].copy()

    if df_filtered.empty:
        return pd.DataFrame()

    # Add calendar month
    df_filtered['calendar_month'] = df_filtered['date'].dt.month
    df_filtered['calendar_month_name'] = df_filtered['date'].dt.strftime('%B')

    # For each contract, calculate return from each calendar month to maturity
    results = []

    for contract_id in df_filtered['contract_id'].unique():
        df_contract = df_filtered[df_filtered['contract_id'] == contract_id].sort_values('date')

        # Get maturity price (last available price)
        maturity_price = df_contract['price'].iloc[-1]

        # For each calendar month in the contract's life
        for cal_month in range(1, 13):
            df_month = df_contract[df_contract['calendar_month'] == cal_month]

            if df_month.empty:
                continue

            # Get first price in this calendar month
            fixation_price = df_month['price'].iloc[0]

            # Calculate return to maturity
            ret_to_maturity = (maturity_price - fixation_price) / fixation_price * 100

            # Calculate drawdown during the period
            future_prices = df_contract[df_contract['date'] >= df_month['date'].iloc[0]]['price']
            if len(future_prices) > 1:
                drawdown = ((future_prices.min() - fixation_price) / fixation_price * 100)
            else:
                drawdown = 0

            results.append({
                'contract_id': contract_id,
                'calendar_month': cal_month,
                'fixation_price': fixation_price,
                'maturity_price': maturity_price,
                'return_to_maturity': ret_to_maturity,
                'drawdown': drawdown,
                'win': ret_to_maturity > 0
            })

    if not results:
        return pd.DataFrame()

    df_results = pd.DataFrame(results)

    # Aggregate by calendar month
    monthly_stats = df_results.groupby('calendar_month').agg({
        'return_to_maturity': ['mean', 'median', 'std', 'min', 'max'],
        'drawdown': ['mean', 'min'],
        'win': ['sum', 'count']
    }).reset_index()

    # Flatten columns
    monthly_stats.columns = [
        'mes_calendario',
        'retorno_medio',
        'retorno_mediano',
        'volatilidade',
        'pior_caso',
        'melhor_caso',
        'drawdown_medio',
        'drawdown_maximo',
        'vitorias',
        'total_obs'
    ]

    # Calculate win rate
    monthly_stats['taxa_sucesso'] = (monthly_stats['vitorias'] / monthly_stats['total_obs'] * 100)

    # Add month names
    month_names = {
        1: 'Janeiro', 2: 'Fevereiro', 3: 'Março', 4: 'Abril',
        5: 'Maio', 6: 'Junho', 7: 'Julho', 8: 'Agosto',
        9: 'Setembro', 10: 'Outubro', 11: 'Novembro', 12: 'Dezembro'
    }
    monthly_stats['mes_nome'] = monthly_stats['mes_calendario'].map(month_names)

    # Calculate risk-adjusted return
    monthly_stats['sharpe_like'] = np.where(
        monthly_stats['volatilidade'] > 0,
        monthly_stats['retorno_medio'] / monthly_stats['volatilidade'],
        0
    )

    return monthly_stats


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
# Calculate Seasonality & Fixation Metrics
# ============================================================
with st.spinner("Calculando padrões de sazonalidade e métricas de fixação..."):
    seasonality_stats = calculate_seasonality_by_contract_month(
        df_contracts,
        selected_asset,
        selected_month_code,
        selected_years
    )

    fixation_metrics = calculate_fixation_metrics(
        df_contracts,
        selected_asset,
        selected_month_code,
        selected_years
    )

if seasonality_stats.empty:
    st.error("❌ Dados insuficientes para análise. Tente selecionar mais anos ou outro contrato.")
    st.stop()

if fixation_metrics.empty:
    st.warning("⚠️ Não foi possível calcular métricas de fixação. Continuando com análise de sazonalidade básica.")
    has_fixation_data = False
else:
    has_fixation_data = True

# ============================================================
# KPIs Summary - DECISÃO DE FIXAÇÃO
# ============================================================
st.markdown("## 📊 Resumo Executivo - Decisão de Fixação")

with st.container(border=True):
    st.markdown(f"### Análise do Contrato **{selected_asset} - {selected_month_name} ({selected_month_code})**")
    st.caption(f"Baseado em {len(selected_years)} contratos: {', '.join(map(str, selected_years))}")

    if has_fixation_data:
        # Fixation-focused KPIs
        col1, col2, col3, col4, col5 = st.columns(5)

        # Best month to fix
        best_month_idx = fixation_metrics['sharpe_like'].idxmax()
        best_month = fixation_metrics.loc[best_month_idx]

        with col1:
            st.metric(
                "Melhor Mês para Fixar",
                best_month['mes_nome'],
                f"+{best_month['retorno_medio']:.2f}%",
                help=f"Mês com melhor retorno médio até vencimento (Taxa de sucesso: {best_month['taxa_sucesso']:.0f}%)"
            )

        # Highest win rate month
        highest_wr_idx = fixation_metrics['taxa_sucesso'].idxmax()
        highest_wr_month = fixation_metrics.loc[highest_wr_idx]

        with col2:
            wr_color = "🟢" if highest_wr_month['taxa_sucesso'] >= 70 else "🟡" if highest_wr_month['taxa_sucesso'] >= 60 else "🔴"
            st.metric(
                "Maior Probabilidade",
                highest_wr_month['mes_nome'],
                f"{highest_wr_month['taxa_sucesso']:.0f}% {wr_color}",
                help=f"Mês com maior taxa de sucesso (retorno médio: {highest_wr_month['retorno_medio']:.2f}%)"
            )

        # Worst month (highest risk)
        worst_month_idx = fixation_metrics['drawdown_maximo'].idxmin()
        worst_month = fixation_metrics.loc[worst_month_idx]

        with col3:
            st.metric(
                "Maior Risco (Drawdown)",
                worst_month['mes_nome'],
                f"{worst_month['drawdown_maximo']:.2f}%",
                delta_color="inverse",
                help=f"Mês com maior drawdown máximo observado historicamente"
            )

        # Average return to maturity
        avg_return_all = fixation_metrics['retorno_medio'].mean()
        ret_color = "🟢" if avg_return_all > 0 else "🔴"

        with col4:
            st.metric(
                "Retorno Médio (Fixação→Vcto)",
                f"{avg_return_all:.2f}% {ret_color}",
                help="Retorno médio de todas as fixações até o vencimento"
            )

        # Total observations
        total_fixations = fixation_metrics['total_obs'].sum()

        with col5:
            st.metric(
                "Total de Fixações",
                f"{total_fixations:,}",
                help=f"Total de observações de fixação analisadas"
            )

    else:
        # Fallback to basic seasonality KPIs if no fixation data
        col1, col2, col3, col4, col5 = st.columns(5)

        best_idx = seasonality_stats['sharpe_like'].idxmax()
        best_period = seasonality_stats.loc[best_idx]

        with col1:
            st.metric(
                "Melhor Dia (Sharpe)",
                get_date_label(best_period['dia_do_ano']),
                f"{best_period['retorno_medio']:.2f}%",
                help=f"Dia com melhor retorno ajustado por risco"
            )

        avg_hit_rate = seasonality_stats['taxa_acerto'].mean()
        hit_color = "🟢" if avg_hit_rate > 55 else "🟡" if avg_hit_rate > 50 else "🔴"

        with col2:
            st.metric(
                "Taxa de Acerto Média",
                f"{avg_hit_rate:.1f}% {hit_color}",
                help="% médio de dias com retorno positivo"
            )

        total_obs = seasonality_stats['observacoes'].sum()

        with col3:
            st.metric(
                "Observações",
                f"{total_obs:,}",
                help=f"{len(seasonality_stats)} dias analisados"
            )

st.divider()

# ============================================================
# Fixation Analysis by Calendar Month
# ============================================================
if has_fixation_data:
    st.markdown("## 🎯 Análise de Fixação por Mês Calendário")
    st.markdown("**Pergunta-chave:** *Em qual mês devo fixar este contrato para maximizar retorno até o vencimento?*")

    with st.container(border=True):
        # Create fixation heatmap table
        st.markdown("### 📊 Matriz de Decisão de Fixação")

        # Prepare display dataframe
        display_fix = fixation_metrics[['mes_nome', 'retorno_medio', 'taxa_sucesso', 'drawdown_maximo', 'volatilidade', 'total_obs']].copy()
        display_fix.columns = ['Mês de Fixação', 'Retorno Médio (%)', 'Taxa de Sucesso (%)', 'Drawdown Máx (%)', 'Volatilidade (%)', 'Observações']

        # Style the dataframe
        def color_returns(val):
            if val > 2:
                return 'background-color: #00cc66; color: white'
            elif val > 0:
                return 'background-color: #90ee90'
            elif val > -2:
                return 'background-color: #ffcc99'
            else:
                return 'background-color: #ff6666; color: white'

        def color_success_rate(val):
            if val >= 70:
                return 'background-color: #00cc66; color: white'
            elif val >= 60:
                return 'background-color: #90ee90'
            elif val >= 50:
                return 'background-color: #ffcc99'
            else:
                return 'background-color: #ff6666; color: white'

        styled_df = display_fix.style.format({
            'Retorno Médio (%)': '{:.2f}',
            'Taxa de Sucesso (%)': '{:.0f}',
            'Drawdown Máx (%)': '{:.2f}',
            'Volatilidade (%)': '{:.2f}',
            'Observações': '{:.0f}'
        }).applymap(color_returns, subset=['Retorno Médio (%)']) \
          .applymap(color_success_rate, subset=['Taxa de Sucesso (%)'])

        st.dataframe(styled_df, use_container_width=True, hide_index=True)

        col_info1, col_info2 = st.columns(2)

        with col_info1:
            st.info("""
            💡 **Como usar esta tabela:**
            - **Verde escuro**: Excelente período para fixação (>70% sucesso ou >2% retorno)
            - **Verde claro**: Bom período (60-70% sucesso ou 0-2% retorno)
            - **Laranja**: Neutro/arriscado (50-60% sucesso ou -2 a 0% retorno)
            - **Vermelho**: Evitar fixação (<50% sucesso ou <-2% retorno)
            """)

        with col_info2:
            best_fix_month = fixation_metrics.loc[fixation_metrics['sharpe_like'].idxmax()]
            worst_fix_month = fixation_metrics.loc[fixation_metrics['sharpe_like'].idxmin()]

            st.success(f"""
            ✅ **Recomendação:**
            - **Melhor mês**: {best_fix_month['mes_nome']} (retorno: {best_fix_month['retorno_medio']:.2f}%, sucesso: {best_fix_month['taxa_sucesso']:.0f}%)
            - **Evitar**: {worst_fix_month['mes_nome']} (retorno: {worst_fix_month['retorno_medio']:.2f}%, sucesso: {worst_fix_month['taxa_sucesso']:.0f}%)
            """)

        # Visualization: Bar chart of returns by month
        st.markdown("### 📊 Retorno Esperado por Mês de Fixação")

        fig_fix = go.Figure()

        # Add bars with color coding
        colors_bars = ['green' if x > 0 else 'red' for x in fixation_metrics['retorno_medio']]

        fig_fix.add_trace(go.Bar(
            x=fixation_metrics['mes_nome'],
            y=fixation_metrics['retorno_medio'],
            marker_color=colors_bars,
            name='Retorno Médio',
            text=fixation_metrics['retorno_medio'].round(2),
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Retorno: %{y:.2f}%<br>Taxa Sucesso: %{customdata:.0f}%<extra></extra>',
            customdata=fixation_metrics['taxa_sucesso']
        ))

        # Add error bars (volatility)
        fig_fix.add_trace(go.Scatter(
            x=fixation_metrics['mes_nome'],
            y=fixation_metrics['retorno_medio'],
            error_y=dict(
                type='data',
                array=fixation_metrics['volatilidade'],
                visible=True,
                color='gray'
            ),
            mode='markers',
            marker=dict(size=0),
            showlegend=False,
            hoverinfo='skip'
        ))

        fig_fix.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

        fig_fix.update_layout(
            xaxis_title="Mês de Fixação",
            yaxis_title="Retorno Médio até Vencimento (%)",
            height=400,
            template='plotly_white',
            hovermode='x unified'
        )

        st.plotly_chart(fig_fix, use_container_width=True)

        st.caption("📌 As barras de erro representam a volatilidade (±1 desvio padrão). Maior barra = maior incerteza.")

    st.divider()

# ============================================================
# Main Chart - Seasonality Pattern
# ============================================================
st.markdown("## 📈 Padrão de Sazonalidade Diário (Detalhado)")

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
# Price Bands Analysis - IMPROVED
# ============================================================
st.markdown("## 📊 Distribuição de Retornos - Análise de Risco")

with st.container(border=True):
    st.markdown(f"""
    ### Faixas de Retorno Esperadas ao Longo do Ano
    **Aplicação:** Estabelecer limites de stop-loss e take-profit baseados em histórico
    """)

    # Calculate percentile bands by day of year
    fig_bands = go.Figure()

    # Add P90 (upper bound)
    fig_bands.add_trace(go.Scatter(
        x=seasonality_stats['dia_do_ano'],
        y=seasonality_stats['maximo'],
        mode='lines',
        name='Máximo Histórico',
        line=dict(color='rgba(0, 200, 0, 0.3)', width=1, dash='dot'),
        hovertemplate='Dia %{x}: %{y:.2f}%<extra></extra>'
    ))

    # Add mean + 1 std (P75 approx)
    upper_band = seasonality_stats['retorno_medio'] + seasonality_stats['desvio_padrao']
    fig_bands.add_trace(go.Scatter(
        x=seasonality_stats['dia_do_ano'],
        y=upper_band,
        mode='lines',
        name='Média + 1σ (≈P75)',
        line=dict(color='green', width=2),
        hovertemplate='Dia %{x}: %{y:.2f}%<extra></extra>'
    ))

    # Add mean
    fig_bands.add_trace(go.Scatter(
        x=seasonality_stats['dia_do_ano'],
        y=seasonality_stats['retorno_medio'],
        mode='lines',
        name='Retorno Médio (P50)',
        line=dict(color='blue', width=3),
        fill='tonexty',
        fillcolor='rgba(0, 200, 0, 0.1)',
        hovertemplate='Dia %{x}: %{y:.2f}%<extra></extra>'
    ))

    # Add mean - 1 std (P25 approx)
    lower_band = seasonality_stats['retorno_medio'] - seasonality_stats['desvio_padrao']
    fig_bands.add_trace(go.Scatter(
        x=seasonality_stats['dia_do_ano'],
        y=lower_band,
        mode='lines',
        name='Média - 1σ (≈P25)',
        line=dict(color='orange', width=2),
        fill='tonexty',
        fillcolor='rgba(255, 165, 0, 0.1)',
        hovertemplate='Dia %{x}: %{y:.2f}%<extra></extra>'
    ))

    # Add P10 (lower bound)
    fig_bands.add_trace(go.Scatter(
        x=seasonality_stats['dia_do_ano'],
        y=seasonality_stats['minimo'],
        mode='lines',
        name='Mínimo Histórico',
        line=dict(color='rgba(200, 0, 0, 0.3)', width=1, dash='dot'),
        fill='tonexty',
        fillcolor='rgba(200, 0, 0, 0.1)',
        hovertemplate='Dia %{x}: %{y:.2f}%<extra></extra>'
    ))

    fig_bands.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

    fig_bands.update_layout(
        xaxis_title="Dia do Ano",
        yaxis_title="Retorno Esperado (%)",
        height=500,
        template='plotly_white',
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig_bands, use_container_width=True)

    # Summary statistics
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)

    with col_stat1:
        worst_day_return = seasonality_stats['minimo'].min()
        st.metric(
            "Pior Dia Histórico",
            f"{worst_day_return:.2f}%",
            delta_color="inverse",
            help="Maior perda diária observada em toda a série"
        )

    with col_stat2:
        best_day_return = seasonality_stats['maximo'].max()
        st.metric(
            "Melhor Dia Histórico",
            f"{best_day_return:.2f}%",
            help="Maior ganho diário observado em toda a série"
        )

    with col_stat3:
        avg_volatility = seasonality_stats['desvio_padrao'].mean()
        st.metric(
            "Volatilidade Média",
            f"{avg_volatility:.2f}%",
            help="Desvio padrão médio dos retornos diários"
        )

    with col_stat4:
        percentile_range = seasonality_stats['retorno_medio'].quantile(0.75) - seasonality_stats['retorno_medio'].quantile(0.25)
        st.metric(
            "Range Interquartil (IQR)",
            f"{percentile_range:.2f}%",
            help="Diferença entre P75 e P25 - mede dispersão dos retornos"
        )

    st.info("""
    💡 **Aplicação para Gestão de Risco:**
    - **Stop-Loss**: Posicione próximo à banda Média - 1σ (linha laranja) ou Mínimo Histórico
    - **Take-Profit**: Considere realizar ganhos próximo à banda Média + 1σ (linha verde)
    - **Sizing**: Reduza tamanho de posição em períodos com alta volatilidade (bandas largas)
    - **Outliers**: Movimentos além do mínimo/máximo histórico podem sinalizar reversão ou nova tendência
    """)

st.divider()

# ============================================================
# Trading Windows - Opportunity Identification
# ============================================================
st.markdown("## 🎯 Janelas de Oportunidade e Zonas de Risco")

with st.container(border=True):
    st.markdown("### Períodos Estratégicos para Gestão de Risco")

    # Identify opportunity windows (high win rate + positive return)
    opportunity_windows = seasonality_stats[
        (seasonality_stats['taxa_acerto'] >= 65) &
        (seasonality_stats['retorno_medio'] > 0)
    ].copy()

    # Identify risk zones (low win rate or negative return)
    risk_zones = seasonality_stats[
        (seasonality_stats['taxa_acerto'] < 45) |
        (seasonality_stats['retorno_medio'] < -0.2)
    ].copy()

    col_opp, col_risk = st.columns(2)

    with col_opp:
        st.markdown("#### 🟢 Janelas de Oportunidade")
        st.caption(f"Períodos com taxa de acerto ≥65% e retorno positivo")

        if not opportunity_windows.empty:
            # Group consecutive days
            opp_groups = []
            current_group = []

            for idx, row in opportunity_windows.iterrows():
                if not current_group:
                    current_group = [row]
                elif row['dia_do_ano'] - current_group[-1]['dia_do_ano'] <= 5:  # Within 5 days
                    current_group.append(row)
                else:
                    opp_groups.append(current_group)
                    current_group = [row]

            if current_group:
                opp_groups.append(current_group)

            # Display top 5 windows
            for i, group in enumerate(opp_groups[:5]):
                start_day = group[0]['dia_do_ano']
                end_day = group[-1]['dia_do_ano']
                avg_return = np.mean([r['retorno_medio'] for r in group])
                avg_hit = np.mean([r['taxa_acerto'] for r in group])

                st.success(f"""
                **Janela #{i+1}:** {get_date_label(start_day)} - {get_date_label(end_day)}
                - Retorno médio: **{avg_return:.2f}%**
                - Taxa de acerto: **{avg_hit:.0f}%**
                - Duração: {len(group)} dias
                """)
        else:
            st.info("Nenhuma janela de alta probabilidade identificada com os critérios atuais.")

    with col_risk:
        st.markdown("#### 🔴 Zonas de Risco")
        st.caption(f"Períodos com taxa de acerto <45% ou retorno negativo")

        if not risk_zones.empty:
            # Group consecutive days
            risk_groups = []
            current_group = []

            for idx, row in risk_zones.iterrows():
                if not current_group:
                    current_group = [row]
                elif row['dia_do_ano'] - current_group[-1]['dia_do_ano'] <= 5:
                    current_group.append(row)
                else:
                    risk_groups.append(current_group)
                    current_group = [row]

            if current_group:
                risk_groups.append(current_group)

            # Display top 5 risk zones
            for i, group in enumerate(risk_groups[:5]):
                start_day = group[0]['dia_do_ano']
                end_day = group[-1]['dia_do_ano']
                avg_return = np.mean([r['retorno_medio'] for r in group])
                avg_hit = np.mean([r['taxa_acerto'] for r in group])

                st.error(f"""
                **Zona #{i+1}:** {get_date_label(start_day)} - {get_date_label(end_day)}
                - Retorno médio: **{avg_return:.2f}%**
                - Taxa de acerto: **{avg_hit:.0f}%**
                - Duração: {len(group)} dias
                """)
        else:
            st.info("Nenhuma zona de alto risco identificada com os critérios atuais.")

    st.markdown("---")
    st.info("""
    💡 **Aplicação para Risk Management:**
    - **Long Book**: Priorize fixações nas janelas de oportunidade (verde)
    - **Short Book**: Considere proteção/hedge nas zonas de risco (vermelho)
    - **Neutral**: Aguarde confirmação de mercado em períodos sem padrão claro
    """)

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
    # Recalculate summary stats for export
    total_obs_export = seasonality_stats['observacoes'].sum()
    avg_hit_rate_export = seasonality_stats['taxa_acerto'].mean()
    best_period_export = seasonality_stats.loc[seasonality_stats['sharpe_like'].idxmax()]
    worst_period_export = seasonality_stats.loc[seasonality_stats['sharpe_like'].idxmin()]

    insights_text = f"""ANÁLISE DE SAZONALIDADE - {selected_asset} {selected_month_name} ({selected_month_code})
{'='*80}

Anos Analisados: {', '.join(map(str, selected_years))}
Total de Observações: {total_obs_export:,}
Dias Analisados: {len(seasonality_stats)}

{'='*80}
RESUMO EXECUTIVO
{'='*80}

Melhor Dia (Sharpe): {get_date_label(best_period_export['dia_do_ano'])} (Sharpe: {best_period_export['sharpe_like']:.2f}, Retorno: {best_period_export['retorno_medio']:.2f}%)
Pior Dia (Sharpe): {get_date_label(worst_period_export['dia_do_ano'])} (Sharpe: {worst_period_export['sharpe_like']:.2f}, Retorno: {worst_period_export['retorno_medio']:.2f}%)
Taxa de Acerto Média: {avg_hit_rate_export:.1f}%
"""

    # Add fixation metrics if available
    if has_fixation_data:
        best_fix_month_export = fixation_metrics.loc[fixation_metrics['sharpe_like'].idxmax()]
        insights_text += f"""
{'='*80}
MÉTRICAS DE FIXAÇÃO
{'='*80}

Melhor Mês para Fixar: {best_fix_month_export['mes_nome']} (Retorno: {best_fix_month_export['retorno_medio']:.2f}%, Sucesso: {best_fix_month_export['taxa_sucesso']:.0f}%)
Retorno Médio (Fixação→Vencimento): {fixation_metrics['retorno_medio'].mean():.2f}%
"""

    insights_text += f"""
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
