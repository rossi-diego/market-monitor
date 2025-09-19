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

df_sel, y_col = RATIOS[ratio_label]

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
