# ============================================================
# Imports & Config
# ============================================================
import pandas as pd
import streamlit as st

from src.data_pipeline import oleo_farelo, oleo_palma, oleo_diesel, oil_share
from src.visualization import plot_ratio_std_plotly
from src.utils import apply_theme, date_range_picker, rsi, section
from src.email_utils import send_email_with_chart_attachments, EmailConfig

# --- Theme
apply_theme()

# ============================================================
# Ratios disponíveis (df, coluna_y)
# ============================================================
RATIOS = {
    "Óleo/Farelo": (oleo_farelo, "oleo_farelo"),
    "Óleo/Palma":  (oleo_palma,  "oleo_palma"),
    "Óleo/Diesel": (oleo_diesel, "oleo_diesel"),
    "Oil Share CME": (oil_share, "oil_share"),    
}

# ============================================================
# Seleção do ratio
# ============================================================
section(
    "Selecione o ratio",
    "Todos em USD/ton (Future C1), já convertidos no pipeline, com exceção do Oil Share",
    "📊"
)
ratio_label = st.radio("Ratio", options=list(RATIOS.keys()), horizontal=True)
df_sel, y_col = RATIOS[ratio_label]

# Checagens iniciais
if df_sel is None or df_sel.empty:
    st.warning(f"Sem dados disponíveis para **{ratio_label}**.")
    st.stop()

if y_col not in df_sel.columns:
    st.error(f"A coluna **{y_col}** não existe na view do ratio **{ratio_label}**.")
    st.stop()

# ============================================================
# Período (usa seu helper)
# ============================================================
try:
    start_date, end_date = date_range_picker(df_sel["date"], state_key="arb_range", default_days=365)
except Exception:
    # garante datetime antes do helper (se 'date' vier como string)
    df_sel["date"] = pd.to_datetime(df_sel["date"], errors="coerce")
    start_date, end_date = date_range_picker(df_sel["date"], state_key="arb_range", default_days=365)

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
df_sel["date"] = pd.to_datetime(df_sel["date"], errors="coerce")
mask = (df_sel["date"].dt.date >= start_date) & (df_sel["date"].dt.date <= end_date)
df_sel = df_sel[mask]
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

    # ============================================================
    # Email sending section
    # ============================================================
    st.divider()

    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("### 📧 Enviar relatório por email")
        st.caption("Envia o gráfico atual por email para os destinatários configurados")

    with col2:
        if st.button("📨 Enviar Email", type="primary", use_container_width=True):
            with st.spinner("Enviando email..."):
                try:
                    # Create email config
                    email_config = EmailConfig(
                        recipients=[
                            "diego.santanna@oleoplan.com.br",
                            "otavio.kucharski@oleoplan.com.br"
                        ],
                        subject=f"Market Monitor - {ratio_label}",
                        body_text=(
                            f"Segue abaixo o gráfico da relação {ratio_label} "
                            f"no período de {start_date.strftime('%d/%m/%Y')} a {end_date.strftime('%d/%m/%Y')}.\n\n"
                            "Os dados consideram o último settlement price com conversão "
                            "das unidades dos ativos para toneladas, utilizando o continuation future 1."
                        ),
                        footer_text="Este email foi gerado automaticamente pelo Market Monitor Panel.",
                        show_timestamp=True,
                    )

                    # Send email with current chart
                    charts = {ratio_label: fig}
                    success = send_email_with_chart_attachments(
                        charts=charts,
                        config=email_config,
                    )

                    if success:
                        st.success("✅ Email enviado com sucesso!")
                    else:
                        st.error("❌ Falha ao enviar email. Verifique os logs no console.")

                except Exception as e:
                    st.error(f"❌ Erro ao enviar email: {str(e)}")

    # Optional: Add recipient configuration in expander
    with st.expander("⚙️ Configurar destinatários"):
        st.info(
            "Para alterar os destinatários padrão do email, edite o arquivo:\n\n"
            "`src/email_utils.py` (classe `EmailConfig`, linha 24-27)"
        )
