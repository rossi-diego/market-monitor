# ============================================================
# Imports & Config
# ============================================================
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import streamlit as st
from src.utils import rsi, plot_ratio_std
from src.data_pipeline import oleo_farelo, oleo_palma, oleo_diesel, oleo_quote, oleo_flat_usd

st.set_page_config(layout="wide", page_title="Relações de arbitragem")

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
# Data Prep (garante datetime e separa séries X/Y)
# ============================================================
for df in (oleo_farelo, oleo_palma, oleo_diesel):
    df['date'] = pd.to_datetime(df['date'])

# Bases
x_of = oleo_farelo['date']; y_of = oleo_farelo['oleo/farelo']
x_op = oleo_palma['date'];  y_op = oleo_palma['oleo/palma']
x_od = oleo_diesel['date']; y_od = oleo_diesel['oleo/diesel']

# ============================================================
# UI State (qual gráfico exibir)
# ============================================================
section("📊 Selecione o ratio")

if "plot" not in st.session_state:
    st.session_state["plot"] = None

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Relação Óleo/Farelo", use_container_width=True):
        st.session_state["plot"] = "of"
with col2:
    if st.button("Relação Óleo/Palma", use_container_width=True):
        st.session_state["plot"] = "op"
with col3:
    if st.button("Relação Óleo/Diesel", use_container_width=True):
        st.session_state["plot"] = "od"

# ============================================================
# Intervalo de Datas (min/max globais + presets + slider)
# ============================================================
# Descobre min/max globais para manter UX consistente
global_min = min(x_of.min(), x_op.min(), x_od.min()).date()
global_max = max(x_of.max(), x_op.max(), x_od.max()).date()
default_start = max(global_min, (global_max - dt.timedelta(days=365)))  # ex.: último 1 ano como padrão

section("Selecione o período do gráfico", "Use presets ou ajuste no slider", "🗓️")

# ---- Presets rápidos (atualizam st.session_state["range"])
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    if st.button("1M"):
        st.session_state["range"] = (global_max - dt.timedelta(days=30), global_max)
with c2:
    if st.button("3M"):
        st.session_state["range"] = (global_max - dt.timedelta(days=90), global_max)
with c3:
    if st.button("6M"):
        st.session_state["range"] = (global_max - dt.timedelta(days=180), global_max)
with c4:
    if st.button("YTD"):
        st.session_state["range"] = (dt.date(global_max.year, 1, 1), global_max)
with c5:
    if st.button("Máx"):
        st.session_state["range"] = (global_min, global_max)

# Decide qual valor inicial o slider usa
if "range" in st.session_state:
    default_start, default_end = st.session_state["range"]
else:
    default_start, default_end = default_start, global_max

# ---- Slider com “bolinhas” arrastáveis
start_date, end_date = st.slider(
    "Período do gráfico",
    min_value=global_min,
    max_value=global_max,
    value=(default_start, default_end),
    step=dt.timedelta(days=1),
)

# ============================================================
# Filtro por Data (helper)
# ============================================================
def filter_by_date(x, y, sd, ed):
    """Filtra as séries x/y pelo intervalo [sd, ed] (inclusive),
    mantendo índices originais para anotações por idx."""
    mask = (x.dt.date >= sd) & (x.dt.date <= ed)
    return x[mask], y[mask]

# Proteção caso o usuário limpe/inverta datas
if isinstance(start_date, tuple):  # compat com versões antigas do streamlit
    start_date, end_date = start_date

# ============================================================
# Renderização (plota conforme o botão selecionado)
# ============================================================
if start_date is None or end_date is None or start_date > end_date:
    st.warning("Selecione um intervalo de datas válido.")
else:
    if st.session_state["plot"] == "of":
        xf, yf = filter_by_date(x_of, y_of, start_date, end_date)
        if len(yf) == 0:
            st.info("Sem dados no período selecionado.")
        else:
            fig = plot_ratio_std(xf, yf, title="Relação Óleo/Farelo", ylabel="Relação Óleo/Farelo")
            st.pyplot(fig)

    elif st.session_state["plot"] == "op":
        xf, yf = filter_by_date(x_op, y_op, start_date, end_date)
        if len(yf) == 0:
            st.info("Sem dados no período selecionado.")
        else:
            fig = plot_ratio_std(xf, yf, title="Relação Óleo/Palma", ylabel="Relação Óleo/Palma")
            st.pyplot(fig)

    elif st.session_state["plot"] == "od":
        xf, yf = filter_by_date(x_od, y_od, start_date, end_date)
        if len(yf) == 0:
            st.info("Sem dados no período selecionado.")
        else:
            fig = plot_ratio_std(xf, yf, title="Relação Óleo/Diesel", ylabel="Óleo/Diesel")
            st.pyplot(fig)
    else:
        st.info("Clique em um dos botões para exibir o gráfico.")
