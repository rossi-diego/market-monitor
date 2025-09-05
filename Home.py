import streamlit as st

base="dark"
primaryColor="#7aa2f7"
backgroundColor="#0E1117"
secondaryBackgroundColor="#161a23"
textColor="#e6e6e6"

st.set_page_config(page_title="Market Monitor", page_icon="📊", layout="wide")

pg_home     = st.Page("Home.py", title="Home", icon="🏠")
pg_analise  = st.Page("pages/2_Analise_de_ativos.py", title="Análise de ativos", icon="📈")
pg_rel      = st.Page("pages/3_Relacoes_de_arbitragem.py", title="Relações de arbitragem", icon="🔁")
pg_corr     = st.Page("pages/4_Correlacoes.py", title="Correlações", icon="🧩")
pg_wasde    = st.Page("pages/5_WASDE_reports.py", title="WASDE Reports", icon="📄")
pg_ml       = st.Page("pages/6_Machine_Learning.py", title="Machine Learning", icon="🤖")
pg_virtual  = st.Page("pages/1_Virtual.py", title="Virtual", icon="🧪")

nav = st.navigation({
    "Menu": [pg_home, pg_analise, pg_rel, pg_corr, pg_wasde, pg_ml, pg_virtual]
})
nav.run()

st.title('Monitor de mercado - Riscos e Inteligência de Mercado')