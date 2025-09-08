# ============================================================
# Imports & Config
# ============================================================
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from src.data_pipeline import df  # sua base

# Cores/estilo (dark)
base = "dark"
primaryColor = "#7aa2f7"
backgroundColor = "#0E1117"
secondaryBackgroundColor = "#161a23"
textColor = "#e6e6e6"

st.markdown(
    """
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
""",
    unsafe_allow_html=True,
)

def section(text, subtitle=None, icon=""):
    st.markdown(
        f'<div class="mm-sec"><span class="accent">{icon} {text}</span></div>',
        unsafe_allow_html=True,
    )
    if subtitle:
        st.markdown(f'<div class="mm-sub">{subtitle}</div>', unsafe_allow_html=True)

# ============================================================
# Base / Feature Engineering
# ============================================================
df_heatmap = df.copy()
df_heatmap["date"] = pd.to_datetime(df_heatmap["date"], errors="coerce")

# Derivadas (se insumos existirem)
if {"boc1", "so-premp-c1"}.issubset(df_heatmap.columns):
    df_heatmap["oleo_flat_usd"] = (
        df_heatmap["boc1"] + (df_heatmap["so-premp-c1"] / 100)
    ) * 22.0462

if {"boc1", "so-premp-c1", "brl="}.issubset(df_heatmap.columns):
    df_heatmap["oleo_flat_brl"] = (
        df_heatmap["boc1"] + (df_heatmap["so-premp-c1"] / 100)
    ) * 22.0462 * df_heatmap["brl="]

# Rotulo -> coluna (o usuário vê o rótulo)
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

# Apenas colunas existentes
AVAILABLE = {label: col for label, col in COL_MAP.items() if col in df_heatmap.columns}

# ============================================================
# Controles (UI)
# ============================================================
section("Parâmetros do heatmap", "Escolha período, método e variáveis", "🧰")

c1, c2, c3 = st.columns([1.2, 1, 1])
with c1:
    dmin = df_heatmap["date"].min().date()
    dmax = df_heatmap["date"].max().date()
    default_start = dt.date(2022, 1, 1) if dmin <= dt.date(2022, 1, 1) <= dmax else dmin
    start_date, end_date = st.slider(
        "Período",
        min_value=dmin,
        max_value=dmax,
        value=(default_start, dmax),
        step=dt.timedelta(days=1),
    )

with c2:
    method = st.selectbox("Método", ["spearman", "pearson", "kendall"], index=0)

with c3:
    show_annot = st.checkbox("Exibir rótulo de dados", value=True)

section("Variáveis", "Selecione as séries para a correlação", "🧩")
labels_sorted = sorted(AVAILABLE.keys())
default_sel = [
    lbl
    for lbl in labels_sorted
    if AVAILABLE[lbl] in ["oleo_flat_brl", "fcpoc1"] and lbl in labels_sorted
] or labels_sorted[:8]

labels_selected = st.multiselect(
    "Séries (mín. 2)", options=labels_sorted, default=default_sel
)

mask_upper = st.checkbox("Exibir somente triângulo inferior", value=True)

st.divider()

# Guardas rápidos
if len(labels_selected) < 2:
    st.warning("Selecione pelo menos **duas** séries para calcular a correlação.")
    st.stop()

cols_selected = [AVAILABLE[lbl] for lbl in labels_selected]

# ============================================================
# Helpers
# ============================================================
def adapt_font(base, n, step=0.18, floor=6):
    """Diminui a fonte quanto maior o n; nunca abaixo de 'floor'."""
    return max(floor, int(round(base - step * max(0, n - 10))))

def _rank_fallback(x: np.ndarray) -> np.ndarray:
    """Ranking com tratamento de empates (média das posições), sem SciPy."""
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(x) + 1, dtype=float)
    ux, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    for k, c in enumerate(counts):
        if c > 1:
            ranks[inv == k] = ranks[inv == k].mean()
    return ranks

def _kendall_tau_fallback(u: np.ndarray, v: np.ndarray) -> float:
    """Kendall tau simples (O(n^2)), suficiente para tamanhos típicos."""
    mask = np.isfinite(u) & np.isfinite(v)
    u, v = u[mask], v[mask]
    if u.size < 2:
        return np.nan
    ru, rv = _rank_fallback(u), _rank_fallback(v)
    conc = disc = 0
    # conta pares concordantes/discordantes
    for i in range(len(ru) - 1):
        du = ru[i + 1 :] - ru[i]
        dv = rv[i + 1 :] - rv[i]
        s = np.sign(du * dv)
        conc += np.sum(s > 0)
        disc += np.sum(s < 0)
    denom = len(ru) * (len(ru) - 1) / 2
    if denom == 0:
        return np.nan
    return (conc - disc) / denom

def compute_corr(
    df_base: pd.DataFrame,
    cols_in: list,
    method: str,
    start_date: dt.date,
    end_date: dt.date,
) -> pd.DataFrame:
    """Calcula correlação robusta (pearson/spearman/kendall), filtrando período dentro."""
    if "date" not in df_base.columns:
        st.error("Coluna 'date' não encontrada no DataFrame base.")
        st.stop()

    # Filtro de período
    df_period = df_base.loc[
        (df_base["date"].dt.date >= start_date)
        & (df_base["date"].dt.date <= end_date),
        cols_in,
    ].copy()

    # Saneia seleção
    cols_ok = [c for c in cols_in if c in df_period.columns]
    missing = [c for c in cols_in if c not in df_period.columns]
    if missing:
        st.warning(f"Colunas não encontradas e ignoradas: {', '.join(map(str, missing))}")
    if len(cols_ok) < 2:
        st.info("Selecione ao menos 2 colunas válidas para a correlação.")
        st.stop()

    # Numérico + dropna
    df_num = df_period[cols_ok].apply(pd.to_numeric, errors="coerce").dropna(how="any")
    if df_num.shape[0] < 2:
        st.info("Sem amostras suficientes no período após remover valores ausentes.")
        st.stop()

    # Remove colunas constantes (Kendall falha com variância zero)
    nun = df_num.nunique(dropna=True)
    if (nun <= 1).any():
        dropped = nun[nun <= 1].index.tolist()
        df_num = df_num.loc[:, nun > 1]
        st.warning(f"Colunas sem variabilidade removidas: {', '.join(dropped)}")
    if df_num.shape[1] < 2:
        st.info("Após remover colunas constantes, restaram menos de 2 colunas.")
        st.stop()

    m = method.lower()
    if m in ("pearson", "spearman"):
        return df_num.corr(method=m)

    # Kendall
    cols_ = df_num.columns.tolist()
    n = len(cols_)
    M = np.eye(n, dtype=float)

    # Tenta SciPy; se não houver, usa fallback
    try:
        from scipy.stats import kendalltau  # type: ignore
        use_scipy = True
    except Exception:
        use_scipy = False

    if use_scipy:
        for i in range(n):
            xi = df_num.iloc[:, i]
            for j in range(i + 1, n):
                xj = df_num.iloc[:, j]
                pair = pd.concat([xi, xj], axis=1).dropna()
                if pair.shape[0] < 2:
                    tau = np.nan
                else:
                    try:
                        tau, _ = kendalltau(
                            pair.iloc[:, 0], pair.iloc[:, 1], nan_policy="omit"
                        )
                    except Exception:
                        tau = np.nan
                M[i, j] = M[j, i] = tau if np.isfinite(tau) else np.nan
    else:
        for i in range(n):
            xi = df_num.iloc[:, i].to_numpy(dtype=float)
            for j in range(i + 1, n):
                xj = df_num.iloc[:, j].to_numpy(dtype=float)
                tau = _kendall_tau_fallback(xi, xj)
                M[i, j] = M[j, i] = tau if np.isfinite(tau) else np.nan

    return pd.DataFrame(M, index=cols_, columns=cols_)

# ============================================================
# Cálculo da correlação (único ponto!)
# ============================================================
corr = compute_corr(
    df_base=df_heatmap,
    cols_in=cols_selected,
    method=method,
    start_date=start_date,
    end_date=end_date,
)
labels_selected = list(corr.columns)
n = len(labels_selected)

# ============================================================
# Heatmap
# ============================================================
size_opt = st.radio(
    "Tamanho do heatmap", ["Pequeno", "Médio", "Grande"], index=1, horizontal=True
)

SIZE_PRESETS = {
    "Pequeno": dict(scale=0.28, annot=6, tick=7, title=12, cbar=0.55),
    "Médio": dict(scale=0.45, annot=8, tick=9, title=14, cbar=0.65),
    "Grande": dict(scale=0.70, annot=10, tick=11, title=16, cbar=0.75),
}
P = SIZE_PRESETS[size_opt]

if n <= 1:
    st.info("Seleção insuficiente para matriz de correlação.")
else:
    # Tamanhos proporcionais
    cell_w = 0.60 * P["scale"]
    cell_h = 0.45 * P["scale"]
    fig_w = max(4, min(cell_w * n + 2.2, 14))
    fig_h = max(3, min(cell_h * n + 2.0, 10))

    annot_size = adapt_font(P["annot"], n)
    tick_size = adapt_font(P["tick"], n)
    title_size = adapt_font(P["title"], n, step=0.12)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=110)
    sns.set_theme(style="white")

    mask = np.triu(np.ones_like(corr, dtype=bool)) if mask_upper else None

    sns.heatmap(
        corr,
        annot=show_annot,
        fmt=".2f",
        vmin=-1,
        vmax=1,
        cmap="RdYlBu_r",
        square=False,
        linewidths=0.4 if size_opt == "Pequeno" else 0.5,
        linecolor="white",
        mask=mask,
        annot_kws={"size": annot_size},
        cbar_kws={"shrink": P["cbar"]},
        ax=ax,
    )

    ax.set_title(
        f"Matriz de Correlação ({method.title()}) — {start_date} → {end_date}",
        fontsize=title_size,
        pad=8,
    )
    ax.set_xticklabels(labels_selected, rotation=45, ha="right", fontsize=tick_size)
    ax.set_yticklabels(labels_selected, rotation=0, fontsize=tick_size)

    # Dark theme
    fig.patch.set_alpha(0)
    ax.set_facecolor((0, 0, 0, 0))
    for item in [ax.title, ax.xaxis.label, ax.yaxis.label]:
        item.set_color("#e6e6e6")
    ax.tick_params(colors="#e6e6e6", labelsize=tick_size)

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
