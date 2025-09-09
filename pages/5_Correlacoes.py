# ============================================================
# Imports & Config
# ============================================================
import pandas as pd
import streamlit as st

from src.data_pipeline import df                    # df já vem com flats pelo pipeline
from src.utils import apply_theme, section, available_assets, compute_corr, date_range_picker
from src.visualization import plot_corr_heatmap

# --- Theme
apply_theme()

# ============================================================
# Mapa de rótulos -> colunas (o usuário vê o rótulo)
# ============================================================
COL_MAP = {
    "Flat do óleo de soja (BRL - C1)": "oleo_flat_brl",
    "Flat do óleo de soja (USD - C1)": "oleo_flat_usd",
    "Óleo de soja (BOC1)": "boc1",
    "Flat do farelo de soja (BRL - C1)": "farelo_flat_brl",
    "Flat do farelo de soja (USD - C1)": "farelo_flat_usd",
    "Farelo de soja (SMC1)": "smc1",
    "Óleo – Prêmio C1": "so-premp-c1",
    "Farelo – Prêmio C1": "sm-premp-c1",
    "Soja (SC1)": "sc1",
    "Milho (CC1)": "cc1",
    "RIN D4": "rin-d4-us",
    "Óleo de palma (FCPOC1)": "fcpoc1",
    "Brent (LCOC1)": "lcoc1",
    "Heating Oil (HOC1)": "hoc1",
    "Dólar": "brl=",
}

# Somente colunas existentes no df
AVAILABLE = available_assets(df, COL_MAP)

# ============================================================
# Parâmetros
# ============================================================
section("Parâmetros do heatmap", "Escolha período, método e variáveis", "🧰")

# Período (presets + slider genérico)
start_date, end_date = date_range_picker(df["date"], state_key="corr_range", default_days=365)

# Método e opções
c1, c2, c3 = st.columns([1, 1, 1])
with c1:
    method = st.selectbox("Método", ["spearman", "pearson", "kendall"], index=0)
with c2:
    show_annot = st.checkbox("Exibir rótulo de dados", value=True)
with c3:
    mask_upper = st.checkbox("Exibir somente triângulo inferior", value=True)

section("Variáveis", "Selecione as séries para a correlação", "🧩")
labels_sorted = sorted(AVAILABLE.keys())
# default: tenta priorizar alguns conhecidos; senão, os 8 primeiros disponíveis
prefer = {"oleo_flat_brl", "fcpoc1", "boc1", "smc1"}
default_sel = [lbl for lbl in labels_sorted if AVAILABLE[lbl] in prefer] or labels_sorted[:8]

labels_selected = st.multiselect("Séries (mín. 2)", options=labels_sorted, default=default_sel)
if len(labels_selected) < 2:
    st.warning("Selecione pelo menos **duas** séries para calcular a correlação.")
    st.stop()

cols_selected = [AVAILABLE[lbl] for lbl in labels_selected]

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

# ============================================================
# Extras: baixar CSV e ver a tabela
# ============================================================
col_dl, col_tbl = st.columns([1, 2])
with col_dl:
    csv = corr.round(4).to_csv().encode("utf-8")
    st.download_button(
        "Baixar correlação (CSV)",
        data=csv,
        file_name=f"correlacao_{method}.csv",
        mime="text/csv",
    )
with col_tbl:
    st.dataframe(corr.round(3))
