# ============================================================
# Imports & Config
# ============================================================
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from src.data_pipeline import df
# from src.utils import plot_price_rsi, rsi  # (não usados aqui)

st.set_page_config(layout="wide", page_title="Correlações")

base="dark"
primaryColor="#7aa2f7"
backgroundColor="#0E1117"
secondaryBackgroundColor="#161a23"
textColor="#e6e6e6"

# ===== Estilo de títulos (dark-friendly) =====
st.markdown("""
<style>
.mm-sec { margin: .8rem 0 .35rem; }
.mm-sec .accent {
  display:inline-block; padding:.35rem .7rem;
  border-left:4px solid #7aa2f7; border-radius:8px;
  background: rgba(122,162,247,.10); color:#e6e6e6;
  font-weight:800; font-size:1.05rem; letter-spacing:.02em;
}
.mm-sub { color:#9aa0a6; font-size:.85rem; margin:.15rem 0 0; }
</style>
""", unsafe_allow_html=True)

def section(text, subtitle=None, icon=""):
    st.markdown(f'<div class="mm-sec"><span class="accent">{icon} {text}</span></div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="mm-sub">{subtitle}</div>', unsafe_allow_html=True)

# ============================================================
# Base / feature engineering
# ============================================================
df_heatmap = df.copy()
df_heatmap["date"] = pd.to_datetime(df_heatmap["date"], errors="coerce")

# cria colunas derivadas se os insumos existirem
if {"boc1","so-premp-c1"}.issubset(df_heatmap.columns):
    df_heatmap["oleo_flat_usd"] = (df_heatmap["boc1"] + (df_heatmap["so-premp-c1"] / 100)) * 22.0462
if {"boc1","so-premp-c1","brl="}.issubset(df_heatmap.columns):
    df_heatmap["oleo_flat_brl"] = (df_heatmap["boc1"] + (df_heatmap["so-premp-c1"] / 100)) * 22.0462 * df_heatmap["brl="]

# dicionário rótulo -> coluna (o usuário vê o rótulo)
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

# mantém só as colunas que existem na base
AVAILABLE = {label: col for label, col in COL_MAP.items() if col in df_heatmap.columns}

# ============================================================
# Controles
# ============================================================
section("Parâmetros do heatmap", "Escolha período, método e variáveis", "🧰")

c1, c2, c3 = st.columns([1.2, 1, 1])
with c1:
    # período
    dmin = df_heatmap["date"].min().date()
    dmax = df_heatmap["date"].max().date()
    default_start = dt.date(2022,1,1) if dmin <= dt.date(2022,1,1) <= dmax else dmin
    start_date, end_date = st.slider(
        "Período",
        min_value=dmin, max_value=dmax,
        value=(default_start, dmax),
        step=dt.timedelta(days=1)
    )
with c2:
    method = st.selectbox("Método", ["spearman", "pearson", "kendall"], index=0)
with c3:
    show_annot = st.checkbox("Exibir rótulo de dados", value=True)

# seleção de variáveis (multi)
section("Variáveis", "Selecione as séries para a correlação", "🧩")
labels_sorted = sorted(AVAILABLE.keys())
default_sel = [lbl for lbl in labels_sorted if AVAILABLE[lbl] in ["boc1","smc1","sc1","lcoc1","hoc1","brl="] and lbl in labels_sorted] or labels_sorted[:8]
labels_selected = st.multiselect(
    "Séries (mín. 2)",
    options=labels_sorted,
    default=default_sel
)

# máscara do triângulo superior
mask_upper = st.checkbox("Exibir somente triângulo inferior", value=True)

st.divider()

# ============================================================
# Preparação dos dados conforme seleção
# ============================================================
if len(labels_selected) < 2:
    st.warning("Selecione pelo menos **duas** séries para calcular a correlação.")
    st.stop()

cols_selected = [AVAILABLE[lbl] for lbl in labels_selected]
df_sel = df_heatmap.loc[
    (df_heatmap["date"].dt.date >= start_date) & (df_heatmap["date"].dt.date <= end_date),
    ["date"] + cols_selected
].copy()

# dropna só nas colunas escolhidas
df_sel = df_sel.dropna(subset=cols_selected)

if df_sel.empty or df_sel.shape[0] < 2:
    st.info("Sem dados suficientes no período selecionado após remover valores ausentes.")
    st.stop()

# correlação
corr = df_sel[cols_selected].corr(method=method)

# ============================================================
# Heatmap
# ============================================================
# --- controles de tamanho ---
size_opt = st.radio("Tamanho do heatmap", ["Pequeno", "Médio", "Grande"],
                    index=1, horizontal=True)
scale = {"Pequeno": 0.8, "Médio": 1.0, "Grande": 1.2}[size_opt]

# dimensões proporcionais ao nº de variáveis
n = len(cols_selected)
cell_w = 0.85 * scale   # largura por célula
cell_h = 0.60 * scale   # altura por célula (menor para ficar mais compacto)
fig_w = max(6, min(cell_w * n + 3, 18))
fig_h = max(4, min(cell_h * n + 3, 14))

fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=110)
sns.set_theme(style="white")

mask = np.triu(np.ones_like(corr, dtype=bool)) if mask_upper else None

sns.heatmap(
    corr,
    annot=show_annot,
    fmt=".2f",
    vmin=-1, vmax=1,
    cmap="RdYlBu_r",
    square=False,                 # <<< menos alto que square=True
    linewidths=0.5,
    mask=mask,
    annot_kws={"size": 8},        # <<< números menores
    cbar_kws={"shrink": 0.7},     # <<< barra de cor menor
    ax=ax
)

ax.set_title(f"Matriz de Correlação ({method.title()}) — {start_date} → {end_date}",
             fontsize=14, pad=10)
ax.set_xticklabels(labels_selected, rotation=45, ha="right")
ax.set_yticklabels(labels_selected, rotation=0)
ax.tick_params(labelsize=9)

# fundo transparente + textos claros (dark theme)
fig.patch.set_alpha(0); ax.set_facecolor((0,0,0,0))
for item in [ax.title, ax.xaxis.label, ax.yaxis.label]:
    item.set_color("#e6e6e6")
ax.tick_params(colors="#e6e6e6")

# NÃO estica para a largura total
st.pyplot(fig, use_container_width=False)

# ============================================================
# Extras: baixar CSV e ver a tabela
# ============================================================
col_dl, col_tbl = st.columns([1, 2])
with col_dl:
    csv = corr.round(4).to_csv().encode("utf-8")
    st.download_button("Baixar correlação (CSV)", data=csv, file_name=f"correlacao_{method}.csv", mime="text/csv")
with col_tbl:
    st.dataframe(corr.round(3))
