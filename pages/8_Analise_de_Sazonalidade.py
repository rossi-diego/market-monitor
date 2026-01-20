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


def get_contract_expiration_month(month_code: str) -> int:
    """
    Get the expiration month for a contract.
    Most contracts expire in the month BEFORE the contract month code.
    Example: March (H) contract expires in February.
    """
    month_code_lower = month_code.lower()
    if month_code_lower in MONTH_CODES:
        _, contract_month = MONTH_CODES[month_code_lower]
        # Expiration is typically the month before
        expiration_month = contract_month - 1 if contract_month > 1 else 12
        return expiration_month
    return None


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

    # Get expiration month for this contract
    expiration_month = get_contract_expiration_month(month_code)

    # For each contract, calculate return from each calendar month to maturity
    results = []

    for contract_id in df_filtered['contract_id'].unique():
        df_contract = df_filtered[df_filtered['contract_id'] == contract_id].sort_values('date')

        # Get maturity price (last available price)
        maturity_price = df_contract['price'].iloc[-1]

        # For each calendar month in the contract's life
        for cal_month in range(1, 13):
            # Skip months that are after or equal to expiration month (can't fix after contract expires)
            # This ensures we don't recommend fixing in March for a contract that expires in February
            if expiration_month is not None:
                if cal_month >= expiration_month and expiration_month > 1:
                    continue
                # Handle December expiration (month 12) - skip months 12 and later
                if expiration_month == 12 and cal_month >= 12:
                    continue

            df_month = df_contract[df_contract['calendar_month'] == cal_month]

            if df_month.empty:
                continue

            # Get first price in this calendar month
            fixation_price = df_month['price'].iloc[0]

            # Calculate return to maturity
            ret_to_maturity = (maturity_price - fixation_price) / fixation_price * 100

            # Calculate drawdown during the period (worst case from fixation to maturity)
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

    # Calculate risk-adjusted return (Sharpe-like)
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
    st.error("Nenhum contrato futuro encontrado no dataset. Verifique o formato dos dados.")
    st.stop()

# Get available options
available_assets = sorted(df_contracts['asset'].unique())
available_months = sorted(
    df_contracts[['month_code', 'month_name', 'month_num']].drop_duplicates().values.tolist(),
    key=lambda x: x[2]
)
available_years = sorted(df_contracts['contract_year'].unique())

st.success(f"{len(df_contracts):,} observações de contratos futuros carregadas com sucesso!")

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
        st.error("Menos de 2 anos disponíveis para esta combinação. Selecione outro ativo/mês.")
        st.stop()

    selected_years = st.multiselect(
        "Selecione os anos",
        options=available_years_filtered,
        default=available_years_filtered,
        help="Anos dos contratos a serem incluídos na análise"
    )

    if len(selected_years) < 2:
        st.warning("Selecione pelo menos 2 anos para análise estatística robusta.")
        st.stop()

    st.markdown("---")
    st.caption(f"""
    **Resumo da Seleção:**
    - Ativo: {selected_asset}
    - Contrato: {selected_month_display}
    - Anos: {len(selected_years)} contratos
    """)

# ============================================================
# Calculate Fixation Metrics
# ============================================================
with st.spinner("Calculando métricas de fixação..."):
    fixation_metrics = calculate_fixation_metrics(
        df_contracts,
        selected_asset,
        selected_month_code,
        selected_years
    )

if fixation_metrics.empty:
    st.error("Dados insuficientes para análise. Tente selecionar mais anos ou outro contrato.")
    st.stop()

# Get expiration info for display
expiration_month = get_contract_expiration_month(selected_month_code)
month_names_dict = {
    1: 'Janeiro', 2: 'Fevereiro', 3: 'Março', 4: 'Abril',
    5: 'Maio', 6: 'Junho', 7: 'Julho', 8: 'Agosto',
    9: 'Setembro', 10: 'Outubro', 11: 'Novembro', 12: 'Dezembro'
}
expiration_month_name = month_names_dict.get(expiration_month, "N/A")

# ============================================================
# KPIs Summary - Indicadores de Fixação
# ============================================================
st.markdown("## 📊 Indicadores de Fixação")

with st.container(border=True):
    st.markdown(f"### Contrato **{selected_asset} - {selected_month_name} ({selected_month_code})**")
    st.caption(f"Baseado em {len(selected_years)} contratos históricos: {', '.join(map(str, selected_years))} | Vencimento: {expiration_month_name}")

    col1, col2, col3, col4 = st.columns(4)

    # Best month to fix (based on risk-adjusted return)
    best_month_idx = fixation_metrics['sharpe_like'].idxmax()
    best_month = fixation_metrics.loc[best_month_idx]

    with col1:
        st.metric(
            "Melhor Mês para Fixar",
            best_month['mes_nome'],
            f"+{best_month['retorno_medio']:.1f}% retorno médio",
            help=f"Mês com melhor relação retorno/risco histórica. Taxa de sucesso: {best_month['taxa_sucesso']:.0f}%"
        )

    # Highest win rate month
    highest_wr_idx = fixation_metrics['taxa_sucesso'].idxmax()
    highest_wr_month = fixation_metrics.loc[highest_wr_idx]

    with col2:
        wr_color = "🟢" if highest_wr_month['taxa_sucesso'] >= 70 else "🟡" if highest_wr_month['taxa_sucesso'] >= 60 else "🔴"
        st.metric(
            f"Maior Taxa de Sucesso",
            f"{highest_wr_month['mes_nome']}",
            f"{highest_wr_month['taxa_sucesso']:.0f}% {wr_color}",
            help=f"Mês em que fixar resultou em lucro na maior % dos casos. Em {highest_wr_month['taxa_sucesso']:.0f}% das vezes, o preço no vencimento foi maior que o preço de fixação."
        )

    # Month with lowest risk (lowest drawdown)
    lowest_dd_idx = fixation_metrics['drawdown_maximo'].idxmax()  # Less negative = better
    lowest_dd_month = fixation_metrics.loc[lowest_dd_idx]

    with col3:
        st.metric(
            "Menor Risco de Perda",
            lowest_dd_month['mes_nome'],
            f"{lowest_dd_month['drawdown_maximo']:.1f}% pior caso",
            help=f"Mês com menor queda máxima entre fixação e vencimento. Drawdown = maior queda observada após fixar."
        )

    # Number of contracts analyzed
    total_contracts = len(selected_years)
    total_fixations = fixation_metrics['total_obs'].sum()

    with col4:
        st.metric(
            "Contratos Analisados",
            f"{total_contracts}",
            f"{total_fixations} pontos de dados",
            help=f"Número de contratos históricos ({min(selected_years)}-{max(selected_years)}) e total de observações mensais analisadas."
        )

st.divider()

# ============================================================
# Fixation Analysis by Calendar Month
# ============================================================
st.markdown("## 🎯 Análise de Fixação por Mês")
st.markdown(f"**Quando fixar o contrato {selected_month_name} ({selected_month_code}) para obter o melhor resultado até o vencimento em {expiration_month_name}?**")

with st.container(border=True):
    # Create fixation table with clear explanations
    st.markdown("### 📊 Comparativo de Meses para Fixação")

    # Prepare display dataframe with clearer column names
    display_fix = fixation_metrics[[
        'mes_nome', 'retorno_medio', 'taxa_sucesso', 'drawdown_maximo',
        'volatilidade', 'pior_caso', 'melhor_caso', 'total_obs'
    ]].copy()

    display_fix.columns = [
        'Mês de Fixação',
        'Retorno Médio (%)',
        'Taxa de Sucesso (%)',
        'Pior Queda (%)',
        'Volatilidade (%)',
        'Pior Resultado (%)',
        'Melhor Resultado (%)',
        'Amostras'
    ]

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
        'Retorno Médio (%)': '{:.1f}',
        'Taxa de Sucesso (%)': '{:.0f}',
        'Pior Queda (%)': '{:.1f}',
        'Volatilidade (%)': '{:.1f}',
        'Pior Resultado (%)': '{:.1f}',
        'Melhor Resultado (%)': '{:.1f}',
        'Amostras': '{:.0f}'
    }).applymap(color_returns, subset=['Retorno Médio (%)']) \
      .applymap(color_success_rate, subset=['Taxa de Sucesso (%)'])

    st.dataframe(styled_df, use_container_width=True, hide_index=True)

    # Legend and explanation
    with st.expander("ℹ️ Como interpretar esta tabela"):
        st.markdown("""
        **Colunas explicadas:**

        | Coluna | Significado |
        |--------|-------------|
        | **Mês de Fixação** | Mês em que você fixa o preço do contrato frame |
        | **Retorno Médio (%)** | Ganho ou perda média entre o preço fixado e o preço no vencimento |
        | **Taxa de Sucesso (%)** | % de vezes que fixar nesse mês resultou em lucro (preço vencimento > preço fixação) |
        | **Pior Queda (%)** | Maior queda de preço observada após fixar (risco de marcação a mercado) |
        | **Volatilidade (%)** | Variabilidade dos retornos - quanto maior, mais incerto |
        | **Pior/Melhor Resultado** | Extremos históricos de retorno |
        | **Amostras** | Quantidade de observações (mais amostras = mais confiável) |

        **Cores:**
        - 🟢 **Verde escuro**: Excelente (>70% sucesso ou >2% retorno)
        - 🟢 **Verde claro**: Bom (60-70% sucesso ou 0-2% retorno)
        - 🟠 **Laranja**: Neutro (50-60% sucesso ou -2% a 0% retorno)
        - 🔴 **Vermelho**: Desfavorável (<50% sucesso ou <-2% retorno)
        """)

    # Recommendation box
    best_fix_month = fixation_metrics.loc[fixation_metrics['sharpe_like'].idxmax()]
    worst_fix_month = fixation_metrics.loc[fixation_metrics['sharpe_like'].idxmin()]

    col_rec1, col_rec2 = st.columns(2)

    with col_rec1:
        st.success(f"""
        **✅ Recomendação: Fixar em {best_fix_month['mes_nome']}**

        - Retorno médio até vencimento: **{best_fix_month['retorno_medio']:.1f}%**
        - Taxa de sucesso histórica: **{best_fix_month['taxa_sucesso']:.0f}%**
        - Pior queda observada: **{best_fix_month['drawdown_maximo']:.1f}%**
        """)

    with col_rec2:
        st.error(f"""
        **⚠️ Evitar: Fixar em {worst_fix_month['mes_nome']}**

        - Retorno médio até vencimento: **{worst_fix_month['retorno_medio']:.1f}%**
        - Taxa de sucesso histórica: **{worst_fix_month['taxa_sucesso']:.0f}%**
        - Pior queda observada: **{worst_fix_month['drawdown_maximo']:.1f}%**
        """)

st.divider()

# ============================================================
# Visualization: Bar chart of returns by month
# ============================================================
st.markdown("### 📈 Retorno Médio por Mês de Fixação")
st.markdown("*Visualização do retorno esperado desde a fixação até o vencimento do contrato*")

with st.container(border=True):
    fig_fix = go.Figure()

    # Add bars with color coding
    colors_bars = ['#00cc66' if x > 0 else '#ff6666' for x in fixation_metrics['retorno_medio']]

    fig_fix.add_trace(go.Bar(
        x=fixation_metrics['mes_nome'],
        y=fixation_metrics['retorno_medio'],
        marker_color=colors_bars,
        name='Retorno Médio',
        text=[f"{x:.1f}%" for x in fixation_metrics['retorno_medio']],
        textposition='outside',
        hovertemplate=(
            '<b>%{x}</b><br>'
            'Retorno médio: %{y:.1f}%<br>'
            'Taxa de sucesso: %{customdata[0]:.0f}%<br>'
            'Pior queda: %{customdata[1]:.1f}%<br>'
            'Amostras: %{customdata[2]:.0f}'
            '<extra></extra>'
        ),
        customdata=fixation_metrics[['taxa_sucesso', 'drawdown_maximo', 'total_obs']].values
    ))

    # Add error bars (volatility) to show uncertainty
    fig_fix.add_trace(go.Scatter(
        x=fixation_metrics['mes_nome'],
        y=fixation_metrics['retorno_medio'],
        error_y=dict(
            type='data',
            array=fixation_metrics['volatilidade'],
            visible=True,
            color='gray',
            thickness=1.5
        ),
        mode='markers',
        marker=dict(size=8, color='darkblue'),
        name='Volatilidade (±1σ)',
        hoverinfo='skip'
    ))

    fig_fix.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

    fig_fix.update_layout(
        xaxis_title="Mês de Fixação",
        yaxis_title="Retorno Médio até Vencimento (%)",
        height=450,
        template='plotly_white',
        hovermode='x unified',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig_fix, use_container_width=True)

    st.caption("""
    **Como ler este gráfico:**
    - **Barras verdes**: Meses onde fixar historicamente resultou em ganho médio
    - **Barras vermelhas**: Meses onde fixar historicamente resultou em perda média
    - **Linhas de erro (cinza)**: Indicam a volatilidade - barras maiores = maior incerteza no resultado
    - **Passe o mouse** sobre as barras para ver detalhes
    """)

st.divider()

# ============================================================
# Success Rate Visualization
# ============================================================
st.markdown("### 🎯 Taxa de Sucesso por Mês de Fixação")
st.markdown("*Em quantos % dos casos históricos a fixação resultou em lucro?*")

with st.container(border=True):
    fig_success = go.Figure()

    # Color based on success rate
    colors_success = [
        '#00cc66' if x >= 70 else '#90ee90' if x >= 60 else '#ffcc99' if x >= 50 else '#ff6666'
        for x in fixation_metrics['taxa_sucesso']
    ]

    fig_success.add_trace(go.Bar(
        x=fixation_metrics['mes_nome'],
        y=fixation_metrics['taxa_sucesso'],
        marker_color=colors_success,
        name='Taxa de Sucesso',
        text=[f"{x:.0f}%" for x in fixation_metrics['taxa_sucesso']],
        textposition='outside',
        hovertemplate=(
            '<b>%{x}</b><br>'
            'Taxa de sucesso: %{y:.0f}%<br>'
            'Baseado em %{customdata} observações'
            '<extra></extra>'
        ),
        customdata=fixation_metrics['total_obs']
    ))

    # Reference lines
    fig_success.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.5,
                          annotation_text="50% (aleatório)", annotation_position="right")
    fig_success.add_hline(y=70, line_dash="dot", line_color="green", opacity=0.5,
                          annotation_text="70% (favorável)", annotation_position="right")

    fig_success.update_layout(
        xaxis_title="Mês de Fixação",
        yaxis_title="Taxa de Sucesso (%)",
        yaxis_range=[0, 100],
        height=400,
        template='plotly_white',
        showlegend=False
    )

    st.plotly_chart(fig_success, use_container_width=True)

    st.caption("""
    **Interpretação:**
    - **Acima de 70%**: Padrão historicamente favorável para fixação
    - **Entre 50-70%**: Resultado incerto, considere outros fatores
    - **Abaixo de 50%**: Historicamente desfavorável - evite fixar neste período
    """)

st.divider()

# ============================================================
# Risk Analysis - Drawdown
# ============================================================
st.markdown("### ⚠️ Análise de Risco: Queda Máxima Após Fixação")
st.markdown("*Qual a maior perda temporária que você pode enfrentar após fixar em cada mês?*")

with st.container(border=True):
    st.info("""
    **O que é Drawdown (Queda Máxima)?**

    Após fixar o preço, o mercado pode se mover contra você antes do vencimento.
    O drawdown mostra a **maior queda observada** entre o momento da fixação e o vencimento.

    **Por que isso importa?**
    - Mesmo que a fixação resulte em lucro no vencimento, você pode ter que explicar perdas temporárias
    - Importante para gestão de margem e marcação a mercado
    - Ajuda a definir níveis de stop-loss e limites de risco
    """)

    fig_dd = go.Figure()

    fig_dd.add_trace(go.Bar(
        x=fixation_metrics['mes_nome'],
        y=fixation_metrics['drawdown_maximo'],
        marker_color='#ff6666',
        name='Pior Queda',
        text=[f"{x:.1f}%" for x in fixation_metrics['drawdown_maximo']],
        textposition='outside',
        hovertemplate=(
            '<b>%{x}</b><br>'
            'Pior queda observada: %{y:.1f}%<br>'
            'Queda média: %{customdata:.1f}%'
            '<extra></extra>'
        ),
        customdata=fixation_metrics['drawdown_medio']
    ))

    fig_dd.update_layout(
        xaxis_title="Mês de Fixação",
        yaxis_title="Pior Queda Observada (%)",
        height=400,
        template='plotly_white',
        showlegend=False
    )

    st.plotly_chart(fig_dd, use_container_width=True)

    # Summary
    col_dd1, col_dd2, col_dd3 = st.columns(3)

    safest_month = fixation_metrics.loc[fixation_metrics['drawdown_maximo'].idxmax()]
    riskiest_month = fixation_metrics.loc[fixation_metrics['drawdown_maximo'].idxmin()]

    with col_dd1:
        st.metric(
            "Mês Mais Seguro",
            safest_month['mes_nome'],
            f"{safest_month['drawdown_maximo']:.1f}% pior queda",
            help="Mês com menor risco de queda após fixação"
        )

    with col_dd2:
        st.metric(
            "Mês Mais Arriscado",
            riskiest_month['mes_nome'],
            f"{riskiest_month['drawdown_maximo']:.1f}% pior queda",
            delta_color="inverse",
            help="Mês com maior risco de queda após fixação"
        )

    with col_dd3:
        avg_dd = fixation_metrics['drawdown_maximo'].mean()
        st.metric(
            "Queda Média (todos os meses)",
            f"{avg_dd:.1f}%",
            help="Média da pior queda observada em todos os meses"
        )

st.divider()

# ============================================================
# Actionable Summary
# ============================================================
st.markdown("## 📋 Resumo de Recomendações")

with st.container(border=True):
    # Get top 3 best months and worst months
    best_months = fixation_metrics.nlargest(3, 'sharpe_like')
    worst_months = fixation_metrics.nsmallest(3, 'sharpe_like')

    col_summary1, col_summary2 = st.columns(2)

    with col_summary1:
        st.markdown("### ✅ Melhores Meses para Fixar")
        for i, (_, row) in enumerate(best_months.iterrows(), 1):
            confidence = "Alta" if row['taxa_sucesso'] >= 70 else "Média" if row['taxa_sucesso'] >= 60 else "Baixa"
            st.markdown(f"""
            **{i}. {row['mes_nome']}**
            - Retorno médio: {row['retorno_medio']:.1f}%
            - Taxa de sucesso: {row['taxa_sucesso']:.0f}%
            - Confiança: {confidence}
            """)

    with col_summary2:
        st.markdown("### ⚠️ Meses para Evitar")
        for i, (_, row) in enumerate(worst_months.iterrows(), 1):
            risk = "Alto" if row['drawdown_maximo'] < -10 else "Médio" if row['drawdown_maximo'] < -5 else "Baixo"
            st.markdown(f"""
            **{i}. {row['mes_nome']}**
            - Retorno médio: {row['retorno_medio']:.1f}%
            - Taxa de sucesso: {row['taxa_sucesso']:.0f}%
            - Risco: {risk}
            """)

    st.markdown("---")

    # Key insight
    overall_success = (fixation_metrics['retorno_medio'] > 0).sum() / len(fixation_metrics) * 100
    best_overall = fixation_metrics.loc[fixation_metrics['sharpe_like'].idxmax()]

    st.markdown(f"""
    ### 💡 Conclusão Principal

    Para o contrato **{selected_asset} {selected_month_name} ({selected_month_code})** com vencimento em **{expiration_month_name}**:

    - **{overall_success:.0f}%** dos meses analisados apresentam retorno médio positivo
    - O melhor momento para fixar é **{best_overall['mes_nome']}** com:
        - **{best_overall['taxa_sucesso']:.0f}%** de taxa de sucesso
        - **{best_overall['retorno_medio']:.1f}%** de retorno médio até o vencimento
    - Análise baseada em **{len(selected_years)} anos** de dados históricos

    ⚠️ *Lembre-se: resultados passados não garantem resultados futuros. Use esta análise como um dos fatores na sua decisão de fixação.*
    """)

st.divider()

# ============================================================
# Export
# ============================================================
st.markdown("## 📥 Exportar Análise")

col_exp1, col_exp2 = st.columns(2)

with col_exp1:
    # Export fixation metrics
    export_metrics = fixation_metrics.copy()
    export_metrics = export_metrics.rename(columns={
        'mes_calendario': 'Mês (número)',
        'mes_nome': 'Mês',
        'retorno_medio': 'Retorno Médio (%)',
        'retorno_mediano': 'Retorno Mediano (%)',
        'volatilidade': 'Volatilidade (%)',
        'pior_caso': 'Pior Resultado (%)',
        'melhor_caso': 'Melhor Resultado (%)',
        'drawdown_medio': 'Queda Média (%)',
        'drawdown_maximo': 'Pior Queda (%)',
        'vitorias': 'Casos de Sucesso',
        'total_obs': 'Total de Amostras',
        'taxa_sucesso': 'Taxa de Sucesso (%)',
        'sharpe_like': 'Índice Sharpe'
    })

    csv_data = export_metrics.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Baixar Métricas de Fixação (CSV)",
        data=csv_data,
        file_name=f"fixacao_{selected_asset}_{selected_month_code}_{min(selected_years)}_{max(selected_years)}.csv",
        mime="text/csv",
        key="download_metrics"
    )

with col_exp2:
    # Export summary report
    best_fix = fixation_metrics.loc[fixation_metrics['sharpe_like'].idxmax()]
    worst_fix = fixation_metrics.loc[fixation_metrics['sharpe_like'].idxmin()]

    report_text = f"""ANÁLISE DE FIXAÇÃO - {selected_asset} {selected_month_name} ({selected_month_code})
{'='*80}

CONFIGURAÇÃO
------------
Anos Analisados: {', '.join(map(str, selected_years))}
Número de Contratos: {len(selected_years)}
Vencimento do Contrato: {expiration_month_name}

{'='*80}
RECOMENDAÇÃO PRINCIPAL
{'='*80}

MELHOR MÊS PARA FIXAR: {best_fix['mes_nome']}
- Retorno médio até vencimento: {best_fix['retorno_medio']:.1f}%
- Taxa de sucesso histórica: {best_fix['taxa_sucesso']:.0f}%
- Pior queda observada: {best_fix['drawdown_maximo']:.1f}%
- Baseado em {best_fix['total_obs']:.0f} observações

MÊS PARA EVITAR: {worst_fix['mes_nome']}
- Retorno médio até vencimento: {worst_fix['retorno_medio']:.1f}%
- Taxa de sucesso histórica: {worst_fix['taxa_sucesso']:.0f}%
- Pior queda observada: {worst_fix['drawdown_maximo']:.1f}%

{'='*80}
MÉTRICAS POR MÊS
{'='*80}

"""

    for _, row in fixation_metrics.iterrows():
        report_text += f"""
{row['mes_nome']}:
  - Retorno Médio: {row['retorno_medio']:.1f}%
  - Taxa de Sucesso: {row['taxa_sucesso']:.0f}%
  - Pior Queda: {row['drawdown_maximo']:.1f}%
  - Volatilidade: {row['volatilidade']:.1f}%
  - Amostras: {row['total_obs']:.0f}
"""

    report_text += f"""
{'='*80}
NOTAS
{'='*80}

- Taxa de Sucesso: % de vezes que o preço no vencimento foi maior que o preço de fixação
- Retorno Médio: Média da variação percentual entre preço de fixação e preço de vencimento
- Pior Queda (Drawdown): Maior queda observada entre a fixação e o vencimento

AVISO: Resultados passados não garantem resultados futuros.
"""

    st.download_button(
        "📥 Baixar Relatório (TXT)",
        data=report_text.encode('utf-8'),
        file_name=f"relatorio_fixacao_{selected_asset}_{selected_month_code}_{min(selected_years)}_{max(selected_years)}.txt",
        mime="text/plain",
        key="download_report"
    )

# Footer
st.divider()
st.caption(f"""
📊 **Análise de Fixação:** {selected_asset} - {selected_month_name} ({selected_month_code}) |
Vencimento: {expiration_month_name} |
Contratos: {', '.join(map(str, selected_years))} |
Total de observações: {fixation_metrics['total_obs'].sum():.0f}
""")
