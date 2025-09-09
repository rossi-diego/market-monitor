# ============================================================
# Imports & Config
# ============================================================
import pandas as pd
import streamlit as st

from src.data_pipeline import oleo_farelo, oleo_palma, oleo_diesel, oil_share
from src.visualization import plot_ratio_std_plotly
from src.utils import apply_theme, section, date_range_picker

# --- Theme
apply_theme()

# ============================================================
# Ratios disponíveis (df, coluna_y)
# ============================================================
RATIOS = {
    "Óleo/Farelo": (oleo_farelo, "oleo/farelo"),
    "Óleo/Palma":  (oleo_palma,  "oleo/palma"),
    "Óleo/Diesel": (oleo_diesel, "oleo/diesel"),
    "Oil Share CME": (oil_share, "oil share"),    
}

# ============================================================
# Seleção do ratio
# ============================================================
section("Selecione o ratio", "Todos em USD/ton (Future C1), já convertidos no pipeline, com exceção do Oil Share", "📊")
ratio_label = st.radio("Ratio", options=list(RATIOS.keys()), horizontal=True)

df_sel, y_col = RATIOS[ratio_label]

# ============================================================
# Período (presets + slider)
# ============================================================
section("Selecione o período do gráfico", "Use presets ou ajuste no slider", "🗓️")

# usa o helper genérico (pega min/max automaticamente)
start_date, end_date = date_range_picker(df_sel["date"], state_key="ratio_range", default_days=365)

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
        rolling_window=90,
        label_series=ratio_label,
    )
    # pequeno respiro entre título e gráfico
    fig.update_layout(
        title=dict(
            pad=dict(b=12),
            x=0.0, xanchor="left",
            y=0.98, yanchor="top",
            ),
            margin=dict(t=80),
    )
    st.plotly_chart(fig, use_container_width=True)
