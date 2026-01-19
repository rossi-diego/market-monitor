import streamlit as st

st.set_page_config(page_title="Market Monitor", page_icon="📊", layout="wide")

pg_home     = st.Page("pages/1_Home.py",                 title="Home",                  icon="🏠")
pg_virtual  = st.Page("pages/2_Virtual.py",              title="Virtual",               icon="📊")
pg_analise  = st.Page("pages/3_Analise_de_ativos.py",    title="Análise de ativos",     icon="📈")
pg_rel      = st.Page("pages/4_Relacoes_de_arbitragem.py", title="Relações de arbitragem", icon="🔁")
pg_corr     = st.Page("pages/5_Correlacoes.py",          title="Correlações",           icon="🧩")
pg_wasde    = st.Page("pages/6_WASDE_reports.py",        title="WASDE reports",         icon="📄")
pg_ml       = st.Page("pages/7_Machine_Learning.py",     title="Machine Learning",      icon="🤖")
pg_season   = st.Page("pages/8_Analise_de_Sazonalidade.py", title="Análise de Sazonalidade", icon="📈")

nav = st.navigation({
    "Menu": [pg_home, pg_virtual, pg_analise, pg_rel, pg_corr, pg_wasde, pg_ml]
})
nav.run()
