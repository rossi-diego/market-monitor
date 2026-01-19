# ============================================================
# Imports & Config
# ============================================================
import pandas as pd
import streamlit as st

from src.data_pipeline import df
from src.utils import apply_theme, section, compute_corr, date_range_picker, vspace
from src.visualization import plot_corr_heatmap
from src.asset_config import ASSETS_MAP, get_available_assets

# --- Theme
apply_theme()

# Page header
st.markdown("# 🔗 Análise de Correlações")
st.markdown("Análise de correlação entre ativos e commodities para identificar relações e oportunidades de trading")
st.divider()

st.markdown("""
<style>
/* Mais respiro entre as seções section()/sub */
.mm-sec { margin: 1.2rem 0 .55rem !important; }
.mm-sub { margin: .25rem 0 .75rem !important; }

/* Espaça blocos grandes (slider, multiselect, etc.) */
.block-gap { margin-top: .4rem; margin-bottom: 1.0rem; }

/* Opcional: aproxima um pouco as colunas de controles */
div[data-testid="stHorizontalBlock"] { gap: .75rem; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Get available assets from shared configuration
# ============================================================
AVAILABLE = get_available_assets(df)

# ============================================================
# Parâmetros
# ============================================================
with st.container(border=True):
    section("Parâmetros do heatmap", "Escolha período, método e variáveis", "🧰")

    # período (usa seu helper)
    start_date, end_date = date_range_picker(df["date"], state_key="corr_range", default_days=365)
    st.markdown('<div class="block-gap"></div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 1, 1], gap="small")
    with c1:
        method = st.selectbox(
            "Método",
            ["spearman", "pearson", "kendall"],
            index=0,
            help="Spearman: robusto a outliers | Pearson: relação linear | Kendall: concordância ordinal"
        )
    with c2:
        vspace(22)  # <- empurra para alinhar verticalmente com o selectbox
        show_annot = st.checkbox("Exibir rótulo de dados", value=True)
    with c3:
        vspace(22)  # <- idem
        mask_upper = st.checkbox("Exibir somente triângulo inferior", value=True)

    # Explanatory note about methods
    with st.expander("ℹ️ Entenda os métodos de correlação"):
        st.markdown("""
        ### 📊 Métodos de Correlação

        **Spearman (Recomendado):**
        - Mede relações **monotônicas** (mesma direção, mas não necessariamente linear)
        - **Mais robusto** a outliers e valores extremos
        - Ideal para dados de commodities que podem ter picos e variações extremas
        - Range: -1 (correlação negativa perfeita) a +1 (correlação positiva perfeita)

        **Pearson:**
        - Mede relações **lineares** entre variáveis
        - Sensível a outliers e valores extremos
        - Assume distribuição normal dos dados
        - Melhor quando a relação é estritamente linear

        **Kendall:**
        - Mede **concordância ordinal** entre variáveis
        - Similar ao Spearman, mas mais conservador
        - Útil para datasets menores
        - Menos comum em análise financeira

        ### 🎯 Interpretação dos Valores

        | Correlação | Interpretação | Aplicação Trading |
        |------------|---------------|-------------------|
        | 0.9 a 1.0  | Muito forte positiva | Movem-se juntos - não hedge |
        | 0.7 a 0.9  | Forte positiva | Alta relação - pair trading |
        | 0.4 a 0.7  | Moderada positiva | Relação moderada |
        | 0.0 a 0.4  | Fraca positiva | Pouca relação |
        | -0.4 a 0.0 | Fraca negativa | Pouca relação inversa |
        | -0.7 a -0.4| Moderada negativa | Relação inversa moderada |
        | -0.9 a -0.7| Forte negativa | Movem-se opostos - hedge |
        | -1.0 a -0.9| Muito forte negativa | Oposição perfeita - hedge ideal |
        """)

    st.markdown('<div class="block-gap"></div>', unsafe_allow_html=True)

# ===== Variáveis =====
with st.container(border=True):
    section("Variáveis", "Selecione as séries para a correlação", "🧩")

    labels_sorted = sorted(AVAILABLE.keys())
    # default: tenta priorizar alguns conhecidos; senão, os 8 primeiros disponíveis
    prefer = {"oleo_flat_brl", "fcpoc1", "boc1", "smc1", "hoc1", "sc1", "so-premp-c1", "lcoc1"}
    default_sel = [lbl for lbl in labels_sorted if AVAILABLE[lbl] in prefer] or labels_sorted[:8]

    labels_selected = st.multiselect(
        "Séries (mín. 2)",
        options=labels_sorted,
        default=default_sel,
        key="corr_vars",
    )

    if len(labels_selected) < 2:
        st.warning("Selecione pelo menos **duas** séries para calcular a correlação.")
        st.stop()

    cols_selected = [AVAILABLE[lbl] for lbl in labels_selected]

# (opcional) um pequeno respiro abaixo do container
st.markdown("<div class='block-gap'></div>", unsafe_allow_html=True)


# ============================================================
# Cálculo
# ============================================================
corr = compute_corr(
    df_base=df,
    cols_in=cols_selected,
    method=method,
    start_date=start_date,
    end_date=end_date,
)

# ============================================================
# Heatmap (com presets de tamanho)
# ============================================================
st.markdown("### 📊 Matriz de Correlação")
size_opt = st.radio("Tamanho do heatmap", ["Pequeno", "Médio", "Grande"], index=1, horizontal=True)

fig = plot_corr_heatmap(
    corr=corr,
    size_opt=size_opt,
    show_annot=show_annot,
    mask_upper=mask_upper,
    method=method,
    start_date=start_date,
    end_date=end_date,
)
st.pyplot(fig, use_container_width=False)

st.divider()

# ============================================================
# Insights: Top correlações
# ============================================================
st.markdown("### 💡 Insights e Destaques")

# Get correlation pairs (excluding diagonal and duplicates)
corr_pairs = []
for i in range(len(corr.index)):
    for j in range(i+1, len(corr.columns)):
        asset1 = corr.index[i]
        asset2 = corr.columns[j]
        value = corr.iloc[i, j]
        corr_pairs.append((asset1, asset2, value))

# Sort by absolute value
corr_pairs_sorted = sorted(corr_pairs, key=lambda x: abs(x[2]), reverse=True)

col_pos, col_neg = st.columns(2)

with col_pos:
    st.markdown("#### 🟢 Correlações Positivas Mais Fortes")
    positive_corrs = [c for c in corr_pairs_sorted if c[2] > 0][:5]

    if positive_corrs:
        for asset1, asset2, value in positive_corrs:
            # Find original labels
            label1 = next((k for k, v in AVAILABLE.items() if v == asset1), asset1)
            label2 = next((k for k, v in AVAILABLE.items() if v == asset2), asset2)

            strength = "Muito forte" if value > 0.9 else "Forte" if value > 0.7 else "Moderada"
            st.metric(
                f"{label1[:30]}... ↔ {label2[:30]}...",
                f"{value:.3f}",
                f"{strength}",
                help=f"Correlação {method}: {value:.4f}"
            )
    else:
        st.info("Nenhuma correlação positiva significativa encontrada")

with col_neg:
    st.markdown("#### 🔴 Correlações Negativas Mais Fortes")
    negative_corrs = [c for c in corr_pairs_sorted if c[2] < 0][:5]

    if negative_corrs:
        for asset1, asset2, value in negative_corrs:
            # Find original labels
            label1 = next((k for k, v in AVAILABLE.items() if v == asset1), asset1)
            label2 = next((k for k, v in AVAILABLE.items() if v == asset2), asset2)

            strength = "Muito forte" if value < -0.9 else "Forte" if value < -0.7 else "Moderada"
            st.metric(
                f"{label1[:30]}... ↔ {label2[:30]}...",
                f"{value:.3f}",
                f"{strength} (Hedge)",
                help=f"Correlação {method}: {value:.4f}"
            )
    else:
        st.info("Nenhuma correlação negativa significativa encontrada")

st.divider()

# ============================================================
# Extras: baixar CSV e ver a tabela
# ============================================================
st.markdown("### 📥 Exportar e Visualizar Dados")

col_dl, col_tbl = st.columns([1, 2])
with col_dl:
    csv = corr.round(4).to_csv().encode("utf-8")
    st.download_button(
        "📥 Baixar correlação (CSV)",
        data=csv,
        file_name=f"correlacao_{method}_{start_date}_{end_date}.csv",
        mime="text/csv",
        key="download_corr_csv",
    )
    st.caption(f"Método: {method.capitalize()} | Período: {start_date} a {end_date}")

with col_tbl:
    st.dataframe(
        corr.round(3),
        use_container_width=True,
        height=min(400, len(corr) * 35 + 38)
    )
