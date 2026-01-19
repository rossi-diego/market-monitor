# ============================================================
# Análise de Sazonalidade - Padrões em Contratos Futuros
# ============================================================
"""
Análise profissional de sazonalidade em contratos futuros de commodities.
Identifica padrões históricos e oportunidades de trading baseadas em estatísticas.
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
from src.utils import apply_theme, date_range_picker, section
from src.asset_config import categorized_asset_picker

# Apply theme
apply_theme()

# Page header
st.markdown("# 📅 Análise de Sazonalidade")
st.markdown("Identifique padrões sazonais históricos em contratos futuros e encontre oportunidades de trading baseadas em estatísticas")
st.divider()

# ============================================================
# Helper Functions
# ============================================================
@st.cache_data
def calculate_returns_by_period(df: pd.DataFrame, asset_col: str, period_type: str) -> pd.DataFrame:
    """
    Calcula retornos diários e agrupa por período (mês, dia do ano, etc).

    Args:
        df: DataFrame com coluna 'date' e ativo
        asset_col: Nome da coluna do ativo
        period_type: 'month', 'dayofyear', 'weekofyear'

    Returns:
        DataFrame com estatísticas por período
    """
    # Prepare data
    df_work = df[['date', asset_col]].copy()
    df_work['date'] = pd.to_datetime(df_work['date'])
    df_work = df_work.dropna().sort_values('date')

    # Calculate daily returns
    df_work['return'] = df_work[asset_col].pct_change() * 100
    df_work = df_work.dropna(subset=['return'])

    # Add period columns
    df_work['year'] = df_work['date'].dt.year
    df_work['month'] = df_work['date'].dt.month
    df_work['dayofyear'] = df_work['date'].dt.dayofyear
    df_work['weekofyear'] = df_work['date'].dt.isocalendar().week

    # Group by period
    grouped = df_work.groupby(period_type)['return']

    # Calculate statistics
    stats_df = pd.DataFrame({
        'periodo': grouped.apply(lambda x: x.name).index,
        'retorno_medio': grouped.mean(),
        'retorno_mediano': grouped.median(),
        'desvio_padrao': grouped.std(),
        'minimo': grouped.min(),
        'maximo': grouped.max(),
        'observacoes': grouped.count(),
    }).reset_index(drop=True)

    # Calculate hit rate (% of positive returns)
    stats_df['taxa_acerto'] = df_work.groupby(period_type)['return'].apply(
        lambda x: (x > 0).sum() / len(x) * 100 if len(x) > 0 else 0
    ).values

    # Calculate Sharpe-like ratio (mean / std)
    stats_df['sharpe_like'] = np.where(
        stats_df['desvio_padrao'] > 0,
        stats_df['retorno_medio'] / stats_df['desvio_padrao'],
        0
    )

    # Statistical significance (t-test against zero)
    if HAS_SCIPY:
        p_values = []
        for period_val in stats_df['periodo']:
            period_returns = df_work[df_work[period_type] == period_val]['return'].values
            if len(period_returns) > 2:
                _, p_val = stats.ttest_1samp(period_returns, 0)
                p_values.append(p_val)
            else:
                p_values.append(1.0)
        stats_df['p_valor'] = p_values
        stats_df['significante'] = stats_df['p_valor'] < 0.05

    # Filter periods with minimum observations
    stats_df = stats_df[stats_df['observacoes'] >= 10].copy()

    return stats_df


def get_period_label(period_type: str, period_val: int) -> str:
    """Retorna label amigável para o período."""
    if period_type == 'month':
        meses = ['', 'Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun',
                 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
        return meses[int(period_val)]
    elif period_type == 'weekofyear':
        return f"Sem {int(period_val)}"
    elif period_type == 'dayofyear':
        return f"Dia {int(period_val)}"
    return str(int(period_val))


# ============================================================
# Sidebar - Configuration
# ============================================================
with st.sidebar:
    section("Configuração", "Parâmetros da análise", "⚙️")

    # Asset selection using the standard picker
    st.markdown("### Ativo")
    asset_col, asset_label = categorized_asset_picker(
        BASE_DF,
        state_key="seasonality_asset",
        show_favorites=True
    )

    # Period selection
    st.markdown("### Período Histórico")
    start_date, end_date = date_range_picker(
        BASE_DF['date'],
        state_key="seasonality_range",
        default_days=365 * 5  # 5 years default
    )

    # Granularity
    st.markdown("### Granularidade")
    granularity = st.radio(
        "Agrupar por",
        options=['month', 'weekofyear', 'dayofyear'],
        format_func=lambda x: {
            'month': '📅 Mês do Ano',
            'weekofyear': '📆 Semana do Ano',
            'dayofyear': '🗓️ Dia do Ano'
        }[x],
        help="Como agrupar os dados históricos para análise de sazonalidade"
    )

    granularity_label = {
        'month': 'Mês',
        'weekofyear': 'Semana',
        'dayofyear': 'Dia'
    }[granularity]

# ============================================================
# Filter data
# ============================================================
BASE_DF['date'] = pd.to_datetime(BASE_DF['date'], errors='coerce')
mask = (BASE_DF['date'].dt.date >= start_date) & (BASE_DF['date'].dt.date <= end_date)
df_filtered = BASE_DF[mask].copy()

if df_filtered.empty or asset_col not in df_filtered.columns:
    st.error(f"❌ Sem dados disponíveis para {asset_label} no período selecionado.")
    st.stop()

# Remove NaN values
df_filtered = df_filtered[['date', asset_col]].dropna()

if len(df_filtered) < 30:
    st.warning("⚠️ Dados insuficientes para análise de sazonalidade. Selecione um período maior.")
    st.stop()

# ============================================================
# Calculate seasonality statistics
# ============================================================
with st.spinner("Calculando padrões de sazonalidade..."):
    seasonality_stats = calculate_returns_by_period(df_filtered, asset_col, granularity)

if seasonality_stats.empty:
    st.warning("⚠️ Não foi possível calcular estatísticas de sazonalidade. Tente outro período ou ativo.")
    st.stop()

# ============================================================
# KPIs Summary
# ============================================================
st.markdown("## 📊 Resumo da Análise")

with st.container(border=True):
    col1, col2, col3, col4, col5 = st.columns(5)

    # Best period
    best_idx = seasonality_stats['sharpe_like'].idxmax()
    best_period = seasonality_stats.loc[best_idx]

    with col1:
        st.metric(
            "Melhor Período",
            get_period_label(granularity, best_period['periodo']),
            f"{best_period['retorno_medio']:.2f}%",
            help=f"Período com melhor retorno ajustado por risco (Sharpe-like: {best_period['sharpe_like']:.2f})"
        )

    # Worst period
    worst_idx = seasonality_stats['sharpe_like'].idxmin()
    worst_period = seasonality_stats.loc[worst_idx]

    with col2:
        st.metric(
            "Pior Período",
            get_period_label(granularity, worst_period['periodo']),
            f"{worst_period['retorno_medio']:.2f}%",
            help=f"Período com pior retorno ajustado por risco (Sharpe-like: {worst_period['sharpe_like']:.2f})"
        )

    # Average hit rate
    avg_hit_rate = seasonality_stats['taxa_acerto'].mean()
    hit_color = "🟢" if avg_hit_rate > 55 else "🟡" if avg_hit_rate > 50 else "🔴"

    with col3:
        st.metric(
            "Taxa de Acerto Média",
            f"{avg_hit_rate:.1f}%",
            f"{hit_color}",
            help="% médio de dias com retorno positivo em cada período"
        )

    # Most consistent period (high hit rate, low std)
    seasonality_stats['consistencia'] = seasonality_stats['taxa_acerto'] / (seasonality_stats['desvio_padrao'] + 0.01)
    consistent_idx = seasonality_stats['consistencia'].idxmax()
    consistent_period = seasonality_stats.loc[consistent_idx]

    with col4:
        st.metric(
            "Período Mais Consistente",
            get_period_label(granularity, consistent_period['periodo']),
            f"{consistent_period['taxa_acerto']:.0f}% acerto",
            help=f"Período com maior taxa de acerto e menor volatilidade (std: {consistent_period['desvio_padrao']:.2f}%)"
        )

    # Total observations
    total_obs = seasonality_stats['observacoes'].sum()

    with col5:
        st.metric(
            "Total de Observações",
            f"{total_obs:,}",
            help=f"{len(seasonality_stats)} períodos analisados"
        )

st.divider()

# ============================================================
# Main Chart - Seasonality Pattern
# ============================================================
st.markdown(f"## 📈 Padrão de Sazonalidade - Retorno Médio por {granularity_label}")

with st.container(border=True):
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.65, 0.35],
        subplot_titles=(
            f"Retorno Médio Diário por {granularity_label}",
            "Taxa de Acerto (%)"
        ),
        vertical_spacing=0.12
    )

    # Upper panel - Mean return
    fig.add_trace(
        go.Scatter(
            x=seasonality_stats['periodo'],
            y=seasonality_stats['retorno_medio'],
            mode='lines+markers',
            name='Retorno Médio',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=6),
            hovertemplate='%{x}: %{y:.2f}%<extra></extra>'
        ),
        row=1, col=1
    )

    # Confidence bands (±1 std)
    upper_band = seasonality_stats['retorno_medio'] + seasonality_stats['desvio_padrao']
    lower_band = seasonality_stats['retorno_medio'] - seasonality_stats['desvio_padrao']

    fig.add_trace(
        go.Scatter(
            x=seasonality_stats['periodo'],
            y=upper_band,
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=seasonality_stats['periodo'],
            y=lower_band,
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

    # Highlight statistically significant periods
    if HAS_SCIPY and 'significante' in seasonality_stats.columns:
        sig_periods = seasonality_stats[seasonality_stats['significante']]
        if not sig_periods.empty:
            fig.add_trace(
                go.Scatter(
                    x=sig_periods['periodo'],
                    y=sig_periods['retorno_medio'],
                    mode='markers',
                    name='Estatisticamente Significante',
                    marker=dict(
                        size=10,
                        color='red',
                        symbol='star',
                        line=dict(color='darkred', width=1)
                    ),
                    hovertemplate='%{x}: %{y:.2f}% (p<0.05)<extra></extra>'
                ),
                row=1, col=1
            )

    # Lower panel - Hit rate
    colors = ['green' if x >= 60 else 'orange' if x >= 50 else 'red'
              for x in seasonality_stats['taxa_acerto']]

    fig.add_trace(
        go.Bar(
            x=seasonality_stats['periodo'],
            y=seasonality_stats['taxa_acerto'],
            marker_color=colors,
            name='Taxa de Acerto',
            hovertemplate='%{x}: %{y:.1f}%<extra></extra>',
            showlegend=False
        ),
        row=2, col=1
    )

    # 50% reference line
    fig.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.5, row=2, col=1)

    # Update layout
    fig.update_xaxes(title_text=granularity_label, row=2, col=1)
    fig.update_yaxes(title_text="Retorno (%)", row=1, col=1)
    fig.update_yaxes(title_text="Taxa de Acerto (%)", row=2, col=1)

    fig.update_layout(
        height=700,
        hovermode='x unified',
        template='plotly_white',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    # Explanation
    with st.expander("ℹ️ Como interpretar este gráfico"):
        st.markdown(f"""
        ### 📊 Componentes do Gráfico

        **Painel Superior - Retorno Médio:**
        - **Linha azul**: Retorno médio diário (%) para cada {granularity_label.lower()}
        - **Banda cinza**: Intervalo de ±1 desvio padrão (mostra a volatilidade)
        - **Estrelas vermelhas**: Períodos com retorno estatisticamente significante (p < 0.05)
        - **Linha pontilhada**: Retorno zero (referência)

        **Painel Inferior - Taxa de Acerto:**
        - **Verde**: Taxa de acerto ≥ 60% (períodos consistentemente positivos)
        - **Laranja**: Taxa de acerto 50-60% (períodos neutros/ligeiramente positivos)
        - **Vermelho**: Taxa de acerto < 50% (períodos frequentemente negativos)
        - **Linha pontilhada**: 50% (referência - equivalente a aleatório)

        ### 🎯 Como Usar para Trading

        **Buscar Oportunidades:**
        - Períodos com retorno médio positivo + alta taxa de acerto + baixa volatilidade = oportunidades de compra
        - Períodos com retorno médio negativo + baixa taxa de acerto = potenciais oportunidades de venda

        **Avaliar Risco:**
        - Banda larga (±1 std) = maior incerteza, retornos mais dispersos
        - Banda estreita = maior previsibilidade

        **Validar Padrões:**
        - Estrelas vermelhas indicam que o padrão é estatisticamente robusto (não é apenas sorte)
        - Mais observações = padrão mais confiável
        """)

st.divider()

# ============================================================
# Rankings - Top/Bottom Periods
# ============================================================
st.markdown("## 🏆 Rankings de Períodos")

with st.container(border=True):
    col_rank1, col_rank2 = st.columns(2)

    with col_rank1:
        st.markdown("### 🟢 Top 5 - Melhores Períodos")
        st.caption("Ordenado por retorno ajustado por risco (Sharpe-like)")

        top5 = seasonality_stats.nlargest(5, 'sharpe_like').copy()
        top5['periodo_label'] = top5['periodo'].apply(lambda x: get_period_label(granularity, x))

        display_top5 = top5[[
            'periodo_label', 'retorno_medio', 'taxa_acerto',
            'desvio_padrao', 'sharpe_like', 'observacoes'
        ]].copy()
        display_top5.columns = ['Período', 'Ret. Médio (%)', 'Tx. Acerto (%)',
                                'Volatilidade (%)', 'Sharpe-like', 'Obs.']

        st.dataframe(
            display_top5.style.format({
                'Ret. Médio (%)': '{:.2f}',
                'Tx. Acerto (%)': '{:.1f}',
                'Volatilidade (%)': '{:.2f}',
                'Sharpe-like': '{:.2f}',
                'Obs.': '{:.0f}'
            }).background_gradient(subset=['Sharpe-like'], cmap='RdYlGn', vmin=-1, vmax=1),
            use_container_width=True,
            hide_index=True
        )

        st.caption("💡 **Sharpe-like**: Quanto maior, melhor o retorno ajustado por risco. >0.5 é bom, >1.0 é excelente.")

    with col_rank2:
        st.markdown("### 🔴 Bottom 5 - Piores Períodos")
        st.caption("Ordenado por retorno ajustado por risco (Sharpe-like)")

        bottom5 = seasonality_stats.nsmallest(5, 'sharpe_like').copy()
        bottom5['periodo_label'] = bottom5['periodo'].apply(lambda x: get_period_label(granularity, x))

        display_bottom5 = bottom5[[
            'periodo_label', 'retorno_medio', 'taxa_acerto',
            'desvio_padrao', 'sharpe_like', 'observacoes'
        ]].copy()
        display_bottom5.columns = ['Período', 'Ret. Médio (%)', 'Tx. Acerto (%)',
                                   'Volatilidade (%)', 'Sharpe-like', 'Obs.']

        st.dataframe(
            display_bottom5.style.format({
                'Ret. Médio (%)': '{:.2f}',
                'Tx. Acerto (%)': '{:.1f}',
                'Volatilidade (%)': '{:.2f}',
                'Sharpe-like': '{:.2f}',
                'Obs.': '{:.0f}'
            }).background_gradient(subset=['Sharpe-like'], cmap='RdYlGn_r', vmin=-1, vmax=1),
            use_container_width=True,
            hide_index=True
        )

        st.caption("⚠️ Períodos com Sharpe-like negativo têm retorno médio negativo ou muito voláteis.")

st.divider()

# ============================================================
# Insights Section
# ============================================================
st.markdown("## 💡 Insights Acionáveis")

with st.container(border=True):
    # Generate insights
    insights = []

    # Best opportunity
    best_opp = seasonality_stats[
        (seasonality_stats['retorno_medio'] > 0) &
        (seasonality_stats['taxa_acerto'] > 55)
    ].nlargest(1, 'sharpe_like')

    if not best_opp.empty:
        opp = best_opp.iloc[0]
        insights.append(f"""
        **🎯 Melhor Oportunidade de Compra:**
        O {get_period_label(granularity, opp['periodo'])} apresenta retorno médio de **{opp['retorno_medio']:.2f}%**
        com taxa de acerto de **{opp['taxa_acerto']:.1f}%** (baseado em {int(opp['observacoes'])} observações).
        Sharpe-like: {opp['sharpe_like']:.2f}
        """)

    # Worst period
    worst_opp = seasonality_stats[
        (seasonality_stats['retorno_medio'] < 0) &
        (seasonality_stats['taxa_acerto'] < 45)
    ].nsmallest(1, 'sharpe_like')

    if not worst_opp.empty:
        opp = worst_opp.iloc[0]
        insights.append(f"""
        **⚠️ Período Mais Fraco:**
        O {get_period_label(granularity, opp['periodo'])} historicamente apresenta retorno médio de **{opp['retorno_medio']:.2f}%**
        com taxa de acerto de apenas **{opp['taxa_acerto']:.1f}%**.
        Considere evitar posições compradas ou avaliar hedge neste período.
        """)

    # Most consistent
    most_consistent = seasonality_stats.nlargest(1, 'consistencia').iloc[0]
    insights.append(f"""
    **✅ Período Mais Consistente:**
    O {get_period_label(granularity, most_consistent['periodo'])} combina alta taxa de acerto
    ({most_consistent['taxa_acerto']:.1f}%) com baixa volatilidade ({most_consistent['desvio_padrao']:.2f}%).
    Ideal para estratégias de menor risco.
    """)

    # Volatility insight
    high_vol = seasonality_stats.nlargest(1, 'desvio_padrao').iloc[0]
    insights.append(f"""
    **📊 Período Mais Volátil:**
    O {get_period_label(granularity, high_vol['periodo'])} tem a maior volatilidade
    ({high_vol['desvio_padrao']:.2f}%), com retornos variando entre {high_vol['minimo']:.2f}% e {high_vol['maximo']:.2f}%.
    Maior potencial de ganho, mas também maior risco.
    """)

    # Statistical significance
    if HAS_SCIPY and 'significante' in seasonality_stats.columns:
        sig_count = seasonality_stats['significante'].sum()
        total_count = len(seasonality_stats)
        insights.append(f"""
        **📈 Robustez Estatística:**
        {sig_count} de {total_count} períodos apresentam retorno estatisticamente significante (p < 0.05).
        Isso indica que **{sig_count/total_count*100:.0f}%** dos padrões observados são estatisticamente robustos.
        """)

    # Display insights
    for insight in insights:
        st.info(insight)

st.divider()

# ============================================================
# Detailed Statistics Table
# ============================================================
st.markdown("## 📋 Estatísticas Detalhadas")

with st.container(border=True):
    # Prepare display table
    display_stats = seasonality_stats.copy()
    display_stats['periodo_label'] = display_stats['periodo'].apply(
        lambda x: get_period_label(granularity, x)
    )

    display_cols = [
        'periodo_label', 'retorno_medio', 'retorno_mediano', 'desvio_padrao',
        'taxa_acerto', 'sharpe_like', 'minimo', 'maximo', 'observacoes'
    ]

    if HAS_SCIPY and 'p_valor' in display_stats.columns:
        display_cols.append('p_valor')

    display_table = display_stats[display_cols].copy()
    display_table.columns = [
        'Período', 'Ret. Médio (%)', 'Ret. Mediano (%)', 'Desvio Padrão (%)',
        'Taxa Acerto (%)', 'Sharpe-like', 'Mín. (%)', 'Máx. (%)', 'Obs.'
    ] + (['p-valor'] if HAS_SCIPY and 'p_valor' in display_stats.columns else [])

    st.dataframe(
        display_table.style.format({
            'Ret. Médio (%)': '{:.2f}',
            'Ret. Mediano (%)': '{:.2f}',
            'Desvio Padrão (%)': '{:.2f}',
            'Taxa Acerto (%)': '{:.1f}',
            'Sharpe-like': '{:.2f}',
            'Mín. (%)': '{:.2f}',
            'Máx. (%)': '{:.2f}',
            'Obs.': '{:.0f}',
            'p-valor': '{:.3f}' if HAS_SCIPY else None
        }).background_gradient(subset=['Ret. Médio (%)'], cmap='RdYlGn', vmin=-2, vmax=2),
        use_container_width=True,
        hide_index=True,
        height=400
    )

st.divider()

# ============================================================
# Export
# ============================================================
st.markdown("## 📥 Exportar Dados")

col_exp1, col_exp2 = st.columns(2)

with col_exp1:
    # Export statistics
    export_stats = seasonality_stats.copy()
    export_stats['periodo_label'] = export_stats['periodo'].apply(
        lambda x: get_period_label(granularity, x)
    )

    csv_data = export_stats.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Baixar Estatísticas (CSV)",
        data=csv_data,
        file_name=f"sazonalidade_{asset_label.replace(' ', '_')}_{granularity}_{start_date}_{end_date}.csv",
        mime="text/csv",
        key="download_stats"
    )

with col_exp2:
    # Export insights as text
    insights_text = f"""ANÁLISE DE SAZONALIDADE - {asset_label}
{'='*60}

Período: {start_date} a {end_date}
Granularidade: {granularity_label}
Total de Observações: {total_obs:,}

{'='*60}
INSIGHTS ACIONÁVEIS
{'='*60}

""" + "\n\n".join(insights)

    st.download_button(
        "📥 Baixar Insights (TXT)",
        data=insights_text.encode('utf-8'),
        file_name=f"insights_{asset_label.replace(' ', '_')}_{granularity}_{start_date}_{end_date}.txt",
        mime="text/plain",
        key="download_insights"
    )

# Footer
st.divider()
st.caption(f"""
📊 **Resumo da Análise:**
Ativo: {asset_label} | Período: {start_date} a {end_date} | Granularidade: {granularity_label} |
Total: {total_obs:,} observações em {len(seasonality_stats)} períodos
""")
