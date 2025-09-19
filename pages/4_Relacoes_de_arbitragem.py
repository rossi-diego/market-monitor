# ============================================================
# Imports & Config
# ============================================================
import pandas as pd
import streamlit as st

from src.data_pipeline import oleo_farelo, oleo_palma, oleo_diesel, oil_share
from src.visualization import plot_ratio_std_plotly
from src.utils import apply_theme, date_range_picker, rsi, section

# --- Theme
apply_theme()

# ============================================================
# Ratios disponíveis (df, coluna_y)
# ============================================================
RATIOS = {
    "Óleo/Farelo": (oleo_farelo, "oleo/farelo"),
    "Óleo/Palma":  (oleo_palma,  "oleo/palma"),
    "Óleo/Diesel": (oleo_diesel, "oleo/diesel"),
    "Oil Share CME": (oil_share, "oil_share"),    
}

# ============================================================
# Seleção do ratio
# ============================================================
section("Selecione o ratio", "Todos em USD/ton (Future C1), já convertidos no pipeline, com exceção do Oil Share", "📊")
ratio_label = st.radio("Ratio", options=list(RATIOS.keys()), horizontal=True)

df_src, y_col = RATIOS[ratio_label]

# Se for função, chama; se já for DataFrame, usa direto
try:
    df_sel = df_src() if callable(df_src) else df_src
except Exception as e:
    st.error(f"Falha ao carregar a fonte de dados para '{ratio_label}': {e}")
    st.stop()

# Valida tipo
if df_sel is None:
    st.error(f"A fonte '{ratio_label}' retornou None.")
    st.stop()
if not isinstance(df_sel, pd.DataFrame):
    # tenta converter de forma amigável; se não der, aborta com info útil
    try:
        df_sel = pd.DataFrame(df_sel)
    except Exception:
        st.error(f"Tipo inesperado para '{ratio_label}': {type(df_sel)}. Esperado: pandas.DataFrame.")
        st.stop()

# Normaliza e valida coluna de data
df_sel = df_sel.copy()
if "date" not in df_sel.columns and "Date" in df_sel.columns:
    df_sel = df_sel.rename(columns={"Date": "date"})

if "date" not in df_sel.columns:
    st.error(f"O dataset de '{ratio_label}' não possui coluna 'date'. Colunas: {list(df_sel.columns)}")
    st.stop()

df_sel["date"] = pd.to_datetime(df_sel["date"], errors="coerce")
df_sel = df_sel.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

# Garante que a coluna do y existe
if y_col not in df_sel.columns:
    st.error(
        f"A coluna '{y_col}' não existe em '{ratio_label}'. "
        f"Colunas disponíveis: {list(df_sel.columns)}"
    )
    st.stop()

# ============================================================
# ===== Opções de subplot e MMs =====
# ============================================================
c1, c2, c3 = st.columns([1, 1, 1])
with c1:
    subplot_opt = st.radio("Subplot inferior", ["Rolling STD", "RSI"], index=0, horizontal=True)
with c2:
    rsi_len = st.slider("RSI window", min_value=7, max_value=50, value=14, step=1)
with c3:
    ma_windows = st.multiselect(
        "Médias móveis",
        options=[20, 50, 90, 200],
        default=[90],
        help="Selecione 1 ou mais MMs para sobrepor no gráfico."
    )

subplot_key = "std" if subplot_opt == "Rolling STD" else "rsi"

# ============================================================
# Filtra e plota
# ============================================================
mask = (df_sel["date"].dt.date >= start_date) & (df_sel["date"].dt.date <= end_date)
view = df_sel.loc[mask, ["date", y_col]].dropna().sort_values("date")

if view.empty:
    st.info("Sem dados no período selecionado.")
else:
    fig = plot_ratio_std_plotly(
        x=view["date"],
        y=view[y_col],
        title=f"Relação {ratio_label}",
        ylabel=f"Relação {ratio_label}",
        rolling_window=90,            # segue sendo usado para o STD "default"
        label_series=ratio_label,
        subplot=subplot_key,          # <-- novo
        rsi_len=rsi_len,              # <-- novo
        rsi_fn=rsi,                   # <-- usa o mesmo RSI da outra página
        ma_windows=ma_windows,        # <-- novo: lista de MMs
    )
    fig.update_layout(
        title=dict(pad=dict(b=12), x=0.0, xanchor="left", y=0.98, yanchor="top"),
        margin=dict(t=80),
    )
    st.plotly_chart(fig, use_container_width=True)
