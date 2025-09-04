# ============================================================
# Imports & Config
# ============================================================
import numpy as np
import pandas as pd
import streamlit as st
import datetime as dt
from src.data_pipeline import df, oleo_quote
from src.utils import plot_price_rsi, rsi

st.set_page_config(layout="wide", page_title="Análise de ativos")

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
# Flats
# ============================================================
# Flat USD óleo
df['oleo_flat_usd'] = (df['boc1'] + (df['so-premp-c1']/100)) * 22.0462
df['oleo_flat_brl'] = (df['boc1'] + (df['so-premp-c1']/100)) * 22.0462 * df['brl=']
df['farelo_flat_usd'] = (df['smc1'] + (df['sm-premp-c1']/100)) * 1.1023
df['farelo_flat_brl'] = (df['smc1'] + (df['sm-premp-c1']/100)) * 1.1023 * df['brl=']

# ============================================================
# Base
# ============================================================
BASE = df.copy()  # ou oleo_quote
BASE["date"] = pd.to_datetime(BASE["date"], errors="coerce")

# ============================================================
# Seleção do ativo (botões)
# ============================================================
section("Selecione o ativo", "Clique para trocar o ativo rapidamente", "📊")

ASSETS_MAP = {
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

# estado inicial (coluna real)
if "close_col" not in st.session_state:
    st.session_state["close_col"] = "boc1"

# grade de chips (2 linhas)
labels = list(ASSETS_MAP.keys())
rows = [labels[:6], labels[6:]]
for row in rows:
    cols = st.columns(len(row))
    for label, col in zip(row, cols):
        code = ASSETS_MAP[label]                    # coluna real
        with col:
            clicked = st.button(
                label,                              # rótulo visível
                key=f"btn_{code}",
                use_container_width=True,
                type=("primary" if code == st.session_state["close_col"] else "secondary"),
                help=f"Coluna: {code}",            # opcional: tooltip
            )
            if clicked:
                st.session_state["close_col"] = code

# mostra o rótulo selecionado (não o código)
LABEL_BY_CODE = {v: k for k, v in ASSETS_MAP.items()}
CLOSE = st.session_state["close_col"]
st.caption(f"Ativo selecionado: **{LABEL_BY_CODE.get(CLOSE, CLOSE)}**")
st.divider()

# Proteção se a coluna não existir
if CLOSE not in BASE.columns:
    st.error(f"Coluna '{CLOSE}' não encontrada na base.")
    st.stop()

# ============================================================
# Pré-processamento do ativo e RSI (no dataset completo)
# ============================================================
# 1) força numérico na coluna escolhida
BASE[CLOSE] = pd.to_numeric(BASE[CLOSE], errors="coerce")

# 2) REMOVE somente as linhas onde esse ativo está NaN (após a seleção)
BASE_clean = BASE.dropna(subset=[CLOSE]).copy().sort_values("date")

# proteção: se ficou vazio, avisa e para
if BASE_clean.empty:
    st.warning("Não há valores válidos para o ativo selecionado após remover NaNs.")
    st.stop()

# 3) calcula o RSI na base já limpa
rsi_df = rsi(
    BASE_clean[["date", CLOSE]].copy(),
    ticker_col=CLOSE,
    date_col="date",
    window=14
)

# 4) junta o RSI (mantendo possíveis NaNs iniciais do RSI, que são normais)
df_full = BASE_clean.merge(rsi_df[["date", "RSI"]], on="date", how="left")

# ============================================================
# Presets + Slider de datas
# ============================================================
valid_dates = df_full["date"].dropna()
if valid_dates.empty:
    st.warning("Sem datas válidas na base.")
    st.stop()

global_min = valid_dates.min().date()
global_max = valid_dates.max().date()
default_start = max(global_min, (global_max - dt.timedelta(days=365)))  # último 1 ano

section("Selecione o período do gráfico", "Use presets ou ajuste no slider", "🗓️")

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    if st.button("1M", use_container_width=True):
        st.session_state["range"] = (global_max - dt.timedelta(days=30), global_max)
with c2:
    if st.button("3M", use_container_width=True):
        st.session_state["range"] = (global_max - dt.timedelta(days=90), global_max)
with c3:
    if st.button("6M", use_container_width=True):
        st.session_state["range"] = (global_max - dt.timedelta(days=180), global_max)
with c4:
    if st.button("YTD", use_container_width=True):
        st.session_state["range"] = (dt.date(global_max.year, 1, 1), global_max)
with c5:
    if st.button("Máx", use_container_width=True):
        st.session_state["range"] = (global_min, global_max)

if "range" in st.session_state:
    default_start, default_end = st.session_state["range"]
else:
    default_start, default_end = default_start, global_max

start_date, end_date = st.slider(
    label="Datas disponíveis",
    min_value=global_min,
    max_value=global_max,
    value=(default_start, default_end),
    step=dt.timedelta(days=1),
)

# ============================================================
# Médias móveis (botões 20 / 50 / 200) – abaixo do slider
# ============================================================
section("Selecione a média móvel para adicionar ao gráfico", None, "📈")

if "ma_window" not in st.session_state:
    st.session_state["ma_window"] = 90  # default atual

mw1, mw2, mw3 = st.columns(3)
with mw1:
    if st.button(f"MM 20{' ✓' if st.session_state['ma_window']==20 else ''}", key="mm20", use_container_width=True):
        st.session_state["ma_window"] = 20
with mw2:
    if st.button(f"MM 50{' ✓' if st.session_state['ma_window']==50 else ''}", key="mm50", use_container_width=True):
        st.session_state["ma_window"] = 50
with mw3:
    if st.button(f"MM 200{' ✓' if st.session_state['ma_window']==200 else ''}", key="mm200", use_container_width=True):
        st.session_state["ma_window"] = 200

# Mostra escolha atual
st.caption(f"Média móvel selecionada: **{st.session_state['ma_window']}** períodos")

# ============================================================
# Filtro por período e Plot
# ============================================================
mask = (df_full["date"].dt.date >= start_date) & (df_full["date"].dt.date <= end_date)
df_view = df_full.loc[mask].copy()

if df_view.empty:
    st.info("Sem dados no período selecionado.")
else:
    fig = plot_price_rsi(
        df_view,
        title=CLOSE.upper(),
        date_col="date",
        close_col=CLOSE,
        rsi_col="RSI",        # já calculado acima
        rsi_fn=None,          # não chamar função externa aqui
        rsi_len=14,
        ma_window=st.session_state["ma_window"],  # <<< aplica seleção
        show_bollinger=False,
        bands_window=20,
        bands_sigma=2.0,
        theme="transparent",
    )
    st.pyplot(fig, use_container_width=True)
