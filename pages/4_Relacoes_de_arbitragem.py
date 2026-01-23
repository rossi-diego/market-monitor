# ============================================================
# Imports & Config
# ============================================================
import pandas as pd
import numpy as np
import streamlit as st

from src.data_pipeline import oleo_farelo, oleo_palma, oleo_diesel, oil_share, gold_bitcoin
from src.visualization import plot_ratio_std_plotly
from src.utils import apply_theme, date_range_picker, rsi, section

# --- Theme
apply_theme()

# Page header
st.markdown("# ⚖️ Relações de Arbitragem")
st.markdown("Análise de spreads e ratios entre commodities para identificar oportunidades de arbitragem e hedging")
st.divider()

# ============================================================
# Ratios disponíveis (df, coluna_y, descrição, interpretação)
# ============================================================
RATIOS = {
    "Óleo/Farelo": {
        "data": (oleo_farelo, "oleo_farelo"),
        "desc": "Relação entre óleo e farelo de soja (USD/ton, C1)",
        "interpretation": "Ratio alto indica óleo caro vs farelo. Ratio baixo indica farelo caro vs óleo.",
        "range": "Historicamente entre 1.8 e 3.0"
    },
    "Óleo/Palma": {
        "data": (oleo_palma, "oleo_palma"),
        "desc": "Relação entre óleo de soja e óleo de palma (USD/ton, C1)",
        "interpretation": "Ratio alto indica óleo de soja premium sobre palma. Ratio baixo indica palma mais caro.",
        "range": "Historicamente entre 0.8 e 1.4"
    },
    "Óleo/Diesel": {
        "data": (oleo_diesel, "oleo_diesel"),
        "desc": "Relação entre óleo de soja e diesel (USD/ton, C1)",
        "interpretation": "Importante para análise de biodiesel. Ratio alto favorece produção de biodiesel.",
        "range": "Historicamente entre 0.6 e 1.2"
    },
    "Oil Share CME": {
        "data": (oil_share, "oil_share"),
        "desc": "Participação do óleo no valor total da soja (CME)",
        "interpretation": "% do valor da soja que vem do óleo. Média histórica ~19-22%.",
        "range": "Historicamente entre 16% e 25%"
    },
    "Ouro/Bitcoin": {
        "data": (gold_bitcoin, "gold_bitcoin"),
        "desc": "Relação entre Ouro (GCC1, USD/oz) e Bitcoin (BTC=, USD)",
        "interpretation": "Ratio alto indica ouro caro vs bitcoin. Ratio baixo indica bitcoin caro vs ouro. Útil para análise de ativos de reserva de valor.",
        "range": "Varia significativamente com a volatilidade do Bitcoin"
    },
}

# ============================================================
# Seleção do ratio
# ============================================================
with st.container(border=True):
    section("Seleção do Ratio", "Escolha a relação de arbitragem para análise", "📊")

    ratio_label = st.radio(
        "Ratio",
        options=list(RATIOS.keys()),
        horizontal=True,
        help="Todos em USD/ton (Future C1), exceto Oil Share que é percentual"
    )

    # Display ratio info
    ratio_info = RATIOS[ratio_label]
    st.info(f"**{ratio_label}**: {ratio_info['desc']}")

    with st.expander("ℹ️ Como interpretar este ratio"):
        st.markdown(f"""
        ### 📊 {ratio_label}

        **Descrição:** {ratio_info['desc']}

        **Interpretação:**
        {ratio_info['interpretation']}

        **Range Histórico:**
        {ratio_info['range']}

        ### 🎯 Oportunidades de Trading

        - **Ratio acima da média + RSI > 70**: Possível reversão para baixo
        - **Ratio abaixo da média + RSI < 30**: Possível reversão para cima
        - **Ratio rompe média móvel**: Sinal de mudança de tendência
        - **Volatilidade alta (STD)**: Maior incerteza, aguardar estabilização
        """)

df_sel, y_col = ratio_info["data"]

# Checagens iniciais
if df_sel is None or df_sel.empty:
    st.warning(f"Sem dados disponíveis para **{ratio_label}**.")
    st.stop()

if y_col not in df_sel.columns:
    st.error(f"A coluna **{y_col}** não existe na view do ratio **{ratio_label}**.")
    st.stop()

st.divider()

# ============================================================
# Período e configurações
# ============================================================
with st.container(border=True):
    section("Configurações", "Período e indicadores técnicos", "⚙️")

    # Período
    try:
        start_date, end_date = date_range_picker(
            df_sel["date"],
            state_key="arb_range",
            default_days=365
        )
    except Exception:
        df_sel["date"] = pd.to_datetime(df_sel["date"], errors="coerce")
        start_date, end_date = date_range_picker(
            df_sel["date"],
            state_key="arb_range",
            default_days=365
        )

    st.markdown("#### Indicadores Técnicos")

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        subplot_opt = st.radio(
            "Subplot inferior",
            ["Rolling STD", "RSI"],
            index=0,
            horizontal=True,
            help="STD: Volatilidade | RSI: Sobrecompra/sobrevenda"
        )
    with c2:
        rsi_len = st.slider(
            "RSI window",
            min_value=7,
            max_value=50,
            value=14,
            step=1,
            help="Período de cálculo do RSI (padrão: 14)"
        )
    with c3:
        ma_windows = st.multiselect(
            "Médias móveis",
            options=[20, 50, 90, 200],
            default=[90],
            help="Médias móveis para identificar tendências"
        )

subplot_key = "std" if subplot_opt == "Rolling STD" else "rsi"

st.divider()

# ============================================================
# Filtra dados
# ============================================================
df_sel["date"] = pd.to_datetime(df_sel["date"], errors="coerce")
mask = (df_sel["date"].dt.date >= start_date) & (df_sel["date"].dt.date <= end_date)
df_filtered = df_sel[mask]
view = df_filtered.loc[:, ["date", y_col]].dropna().sort_values("date")

if view.empty:
    st.warning("❌ Sem dados no período selecionado.")
    st.stop()

# ============================================================
# Estatísticas do Ratio
# ============================================================
st.markdown("### 📊 Estatísticas do Ratio")

with st.container(border=True):
    # Calculate statistics
    current_value = view[y_col].iloc[-1]
    first_value = view[y_col].iloc[0]
    mean_value = view[y_col].mean()
    std_value = view[y_col].std()
    min_value = view[y_col].min()
    max_value = view[y_col].max()

    # Z-score (distance from mean)
    z_score = (current_value - mean_value) / std_value if std_value > 0 else 0

    # Volatility (coefficient of variation)
    cv = (std_value / mean_value * 100) if mean_value > 0 else 0

    # Period change
    period_change = ((current_value - first_value) / first_value * 100) if first_value > 0 else 0

    # Display metrics
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "Valor Atual",
            f"{current_value:.3f}",
            f"{period_change:+.1f}%",
            help=f"Último valor do ratio e variação NO PERÍODO selecionado ({len(view)} dias)"
        )

    with col2:
        st.metric(
            "Média (período)",
            f"{mean_value:.3f}",
            help="Média do ratio NO PERÍODO selecionado"
        )

    with col3:
        # Z-Score interpretation
        z_interp = "Caro" if z_score > 1.5 else "Barato" if z_score < -1.5 else "Normal"
        z_color = "🔴" if z_score > 1.5 else "🟢" if z_score < -1.5 else "🟡"
        st.metric(
            "Z-Score (período)",
            f"{z_score:.2f}",
            f"{z_color} {z_interp}",
            help="Distância da média DO PERÍODO em desvios padrão. >1.5: caro, <-1.5: barato"
        )

    with col4:
        st.metric(
            "Min / Max (período)",
            f"{min_value:.3f}",
            f"Max: {max_value:.3f}",
            help="Range de valores NO PERÍODO selecionado"
        )

    with col5:
        volatility_level = "Alta" if cv > 15 else "Moderada" if cv > 8 else "Baixa"
        vol_color = "🔴" if cv > 15 else "🟡" if cv > 8 else "🟢"
        st.metric(
            "Volatilidade (período)",
            f"{cv:.1f}%",
            f"{vol_color} {volatility_level}",
            help="Coeficiente de variação DO PERÍODO (desvio/média)"
        )

    # Trading signal
    st.markdown("#### 🎯 Sinal de Trading")

    # Calculate simple signal
    if z_score > 1.5:
        signal = "🔴 **VENDA** - Ratio acima da média histórica (sobrecomprado)"
        signal_detail = "Ratio está caro. Considere vender o numerador ou comprar o denominador."
    elif z_score < -1.5:
        signal = "🟢 **COMPRA** - Ratio abaixo da média histórica (sobrevendido)"
        signal_detail = "Ratio está barato. Considere comprar o numerador ou vender o denominador."
    else:
        signal = "🟡 **NEUTRO** - Ratio próximo da média histórica"
        signal_detail = "Ratio está em zona neutra. Aguarde melhor oportunidade."

    st.info(f"{signal}\n\n{signal_detail}")

st.divider()

# ============================================================
# Gráfico
# ============================================================
st.markdown(f"### 📈 Evolução do Ratio - {ratio_label}")

fig = plot_ratio_std_plotly(
    x=view["date"],
    y=view[y_col],
    title=f"Relação {ratio_label}",
    ylabel=f"Relação {ratio_label}",
    rolling_window=90,
    label_series=ratio_label,
    subplot=subplot_key,
    rsi_len=rsi_len,
    rsi_fn=rsi,
    ma_windows=ma_windows,
)

fig.update_layout(
    title=dict(pad=dict(b=12), x=0.0, xanchor="left", y=0.98, yanchor="top"),
    margin=dict(t=80),
)

st.plotly_chart(fig, use_container_width=True)

# Explanatory notes
with st.expander("ℹ️ Como interpretar o gráfico", expanded=False):
    st.markdown(f"""
    ### 📊 Componentes do Gráfico

    **Painel Superior - Ratio {ratio_label}:**
    - **Linha Azul**: Valor do ratio ao longo do tempo
    - **Médias Móveis**: Tendências de longo prazo
      - Ratio acima da MA = tendência de alta
      - Ratio abaixo da MA = tendência de baixa
      - Cruzamentos indicam mudança de tendência

    **Painel Inferior:**
    - **Rolling STD**: Volatilidade do ratio nos últimos 90 dias
      - STD alto = maior incerteza e risco
      - STD baixo = maior estabilidade
    - **RSI**: Índice de força relativa (0-100)
      - RSI > 70 = Sobrecomprado (possível correção para baixo)
      - RSI < 30 = Sobrevendido (possível recuperação)

    ### 🎯 Estratégias de Arbitragem

    **Mean Reversion (Reversão à Média):**
    - Quando o ratio se distancia muito da média (|Z-Score| > 2)
    - Apostar na convergência de volta à média histórica
    - Usar médias móveis para confirmar reversão

    **Trend Following (Seguir Tendência):**
    - Ratio rompe média móvel de longo prazo (200 dias)
    - Entrar na direção da nova tendência
    - Aguardar confirmação com volume/volatilidade

    **Volatility Trading:**
    - Alta volatilidade (STD) = maior risco e oportunidade
    - Baixa volatilidade = aguardar breakout
    - Combinar STD com RSI para timing de entrada
    """)

st.divider()

# ============================================================
# Export functionality
# ============================================================
st.markdown("### 📥 Exportar Dados")

col_exp1, col_exp2 = st.columns(2)

with col_exp1:
    try:
        # Export ratio data
        export_df = view.copy()
        export_df["date"] = export_df["date"].dt.strftime("%Y-%m-%d")

        csv = export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Baixar dados do ratio (CSV)",
            data=csv,
            file_name=f"ratio_{ratio_label.replace('/', '_')}_{start_date}_{end_date}.csv",
            mime="text/csv",
            key="download_ratio_csv",
        )
    except Exception as e:
        st.warning(f"⚠️ Erro ao exportar dados: {str(e)}")

with col_exp2:
    try:
        # Export statistics summary
        stats_df = pd.DataFrame({
            "Métrica": ["Valor Atual", "Média", "Z-Score", "Mínimo", "Máximo", "Volatilidade (%)", "Variação Período (%)"],
            "Valor": [
                f"{current_value:.3f}",
                f"{mean_value:.3f}",
                f"{z_score:.2f}",
                f"{min_value:.3f}",
                f"{max_value:.3f}",
                f"{cv:.1f}",
                f"{period_change:.1f}"
            ]
        })

        csv_stats = stats_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Baixar estatísticas (CSV)",
            data=csv_stats,
            file_name=f"stats_{ratio_label.replace('/', '_')}_{start_date}_{end_date}.csv",
            mime="text/csv",
            key="download_stats_csv",
        )
    except Exception as e:
        st.warning(f"⚠️ Erro ao exportar estatísticas: {str(e)}")
