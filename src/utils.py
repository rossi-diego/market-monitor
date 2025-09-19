import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st

# --- Theme (dark-friendly centralizado)
THEME = {
    "BASE": "dark",
    "PRIMARY_COLOR": "#7aa2f7",
    "BACKGROUND_COLOR": "#0E1117",
    "SECONDARY_BACKGROUND_COLOR": "#161a23",
    "TEXT_COLOR": "#e6e6e6",
}

def apply_theme():
    """Aplica o CSS e estilo de seções para todos os apps."""
    st.markdown(f"""
    <style>
    .mm-sec {{ margin: .8rem 0 .35rem; }}
    .mm-sec .accent {{
      display:inline-block; padding:.35rem .7rem;
      border-left:4px solid {THEME["PRIMARY_COLOR"]}; border-radius:8px;
      background: rgba(122,162,247,.10); color:{THEME["TEXT_COLOR"]};
      font-weight:800; font-size:1.05rem; letter-spacing:.02em;
    }}
    .mm-sub {{ color:#9aa0a6; font-size:.85rem; margin:.15rem 0 0; }}
    </style>
    """, unsafe_allow_html=True)

def section(text, subtitle=None, icon=""):
    """Renderiza título estilizado com opcional subtítulo e ícone."""
    st.markdown(f'<div class="mm-sec"><span class="accent">{icon} {text}</span></div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="mm-sub">{subtitle}</div>', unsafe_allow_html=True)

def rsi(df, ticker_col, date_col='date', window=14): 
    df = df.copy()
    df.sort_values(by=date_col, inplace=True) df['delta'] = df[ticker_col].diff()
    df['gain'] = df['delta'].where(df['delta'] > 0, 0)
    df['loss'] = -df['delta'].where(df['delta'] < 0, 0)
    df['avg_gain'] = df['gain'].rolling(window=window).mean()
    df['avg_loss'] = df['loss'].rolling(window=window).mean()
    df['rs'] = df['avg_gain'] / df['avg_loss']
    df['RSI'] = 100 - (100 / (1 + df['rs']))

    return df[[date_col, ticker_col, 'RSI']]

def available_assets(df, assets_map: dict[str, str]) -> dict[str, str]:
    """Filtra o mapa exibindo só colunas que existem no df."""
    return {label: col for label, col in assets_map.items() if col in df.columns}

def asset_picker(df, assets_map: dict[str, str], state_key: str = "close_col", cols_per_row: int = 6):
    """Renderiza botões em grade e retorna (code, assets_filtrados)."""
    assets = available_assets(df, assets_map)
    if not assets:
        st.error("Nenhuma coluna disponível no DataFrame para os ativos configurados.")
        st.stop()

    # Estado inicial (default pra 'Óleo de soja (BOC1)' se existir; senão, 1ª opção)
    if state_key not in st.session_state:
        st.session_state[state_key] = assets.get("Óleo de soja (BOC1)", next(iter(assets.values())))

    labels = list(assets.keys())
    for i in range(0, len(labels), cols_per_row):
        row = labels[i:i+cols_per_row]
        cols = st.columns(len(row))
        for label, col in zip(row, cols):
            code = assets[label]
            with col:
                if st.button(
                    label,
                    key=f"btn_{code}",
                    type=("primary" if code == st.session_state[state_key] else "secondary"),
                    use_container_width=True,
                ):
                    st.session_state[state_key] = code

    code = st.session_state[state_key]
    # Mostra o rótulo atual
    label_atual = next((k for k, v in assets.items() if v == code), code)
    st.caption(f"Ativo selecionado: **{label_atual}**")
    return code, assets

def asset_picker_dropdown(
    df,
    assets_map: dict[str, str],
    state_key: str = "close_col",
    favorites: list[str] | None = None,
    label_select: str = "Ativo",
):
    """
    Barra de favoritos (botões) + selectbox pesquisável com TODOS os ativos.
    Retorna (code, assets_filtrados).
    """
    import streamlit as st

    # 1) filtra só o que existe no df (mantém ordem do assets_map)
    available = {lbl: col for lbl, col in assets_map.items() if col in df.columns}
    if not available:
        st.error("Nenhum ativo disponível no DataFrame.")
        st.stop()

    # 2) favoritos (só os que existem)
    default_favs = [
        "Flat do óleo de soja (BRL - C1)",
        "Flat do óleo de soja (USD - C1)",
        "Óleo de soja (BOC1)",
        "Óleo de palma (FCPOC1)",
        "Heating Oil (HOC1)",
    ]
    favs = [f for f in (favorites or default_favs) if f in available]

    # 3) estado inicial (prefere BOC1 se existir)
    if state_key not in st.session_state:
        st.session_state[state_key] = available.get(
            "Óleo de soja (BOC1)", next(iter(available.values()))
        )

    # 4) barra de favoritos (opcional)
    if favs:
        cols = st.columns(len(favs))
        for label, col in zip(favs, cols):
            code = available[label]
            with col:
                pressed = st.button(
                    label,
                    key=f"fav_{code}",
                    type=("primary" if code == st.session_state[state_key] else "secondary"),
                    use_container_width=True,
                )
                if pressed:
                    st.session_state[state_key] = code

    # 5) selectbox com todos (é pesquisável por teclado)
    labels_all = list(available.keys())
    current_code = st.session_state[state_key]
    current_label = next((k for k, v in available.items() if v == current_code), labels_all[0])

    sel_label = st.selectbox(
        label_select,
        options=labels_all,
        index=labels_all.index(current_label),
    )

    # sincroniza
    new_code = available[sel_label]
    if new_code != st.session_state[state_key]:
        st.session_state[state_key] = new_code

    st.caption(f"Ativo selecionado: **{sel_label}**")
    return new_code, available

def date_range_picker(dates, state_key: str = "range", default_days: int = 365, label_slider: str = "Datas disponíveis"):
    """Presets + slider. Retorna (start_date, end_date) como date()."""
    dates = dates.dropna()
    if dates.empty:
        st.warning("Sem datas válidas na base.")
        st.stop()

    gmin = dates.min().date()
    gmax = dates.max().date()
    default_start = max(gmin, (gmax - dt.timedelta(days=default_days)))

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        if st.button("1M", use_container_width=True):
            st.session_state[state_key] = (gmax - dt.timedelta(days=30), gmax)
    with c2:
        if st.button("3M", use_container_width=True):
            st.session_state[state_key] = (gmax - dt.timedelta(days=90), gmax)
    with c3:
        if st.button("6M", use_container_width=True):
            st.session_state[state_key] = (gmax - dt.timedelta(days=180), gmax)
    with c4:
        if st.button("YTD", use_container_width=True):
            st.session_state[state_key] = (dt.date(gmax.year, 1, 1), gmax)
    with c5:
        if st.button("Máx", use_container_width=True):
            st.session_state[state_key] = (gmin, gmax)

    if state_key in st.session_state:
        default_start, default_end = st.session_state[state_key]
    else:
        default_start, default_end = default_start, gmax

    start_date, end_date = st.slider(
        label=label_slider,
        min_value=gmin,
        max_value=gmax,
        value=(default_start, default_end),
        step=dt.timedelta(days=1),
    )
    return start_date, end_date

def ma_picker(options=(20, 50, 200), default: int = 90, state_key: str = "ma_window"):
    """Escolha da média móvel via radio. Retorna o inteiro selecionado (1 clique)."""
    # 1) Gera lista ordenada e única, inserindo o default no lugar correto
    opts = list(options)
    if default not in opts:
        insert_at = next((i for i, v in enumerate(opts) if v > default), len(opts))
        opts = sorted(set(list(options) + [default]))
    # opcional: garantir ordem crescente (se preferir sempre crescente)
    # opts = sorted(set(opts))

    # 2) Inicializa o estado só uma vez ANTES do radio
    if state_key not in st.session_state:
        st.session_state[state_key] = default

    # 3) Radio com key controla o estado; não use `index` aqui
    mm = st.radio("Média móvel", options=opts, horizontal=True, key=state_key)
    return mm


def compute_corr(
    df_base: pd.DataFrame,
    cols_in: list,
    method: str,
    start_date: dt.date,
    end_date: dt.date,
) -> pd.DataFrame:
    """
    Calcula matriz de correlação (pearson/spearman/kendall) no período.
    Faz limpeza: numérico, dropna, remove colunas constantes, lida com SciPy ausente.
    """
    if "date" not in df_base.columns:
        st.error("Coluna 'date' não encontrada no DataFrame base.")
        st.stop()

    # Filtro de período (mantém só as colunas pedidas)
    df_period = df_base.loc[
        (df_base["date"].dt.date >= start_date) & (df_base["date"].dt.date <= end_date),
        ["date"] + cols_in
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

    # ---- Kendall (com fallback) ----
    cols_ = df_num.columns.tolist()
    n = len(cols_)
    M = np.eye(n, dtype=float)

    # tenta SciPy
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
                        tau, _ = kendalltau(pair.iloc[:, 0], pair.iloc[:, 1], nan_policy="omit")
                    except Exception:
                        tau = np.nan
                M[i, j] = M[j, i] = tau if np.isfinite(tau) else np.nan
    else:
        # fallback simples
        def _rank_fallback(x: np.ndarray) -> np.ndarray:
            order = np.argsort(x)
            ranks = np.empty_like(order, dtype=float)
            ranks[order] = np.arange(1, len(x) + 1, dtype=float)
            ux, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
            for k, c in enumerate(counts):
                if c > 1:
                    ranks[inv == k] = ranks[inv == k].mean()
            return ranks

        def _kendall_tau_fallback(u: np.ndarray, v: np.ndarray) -> float:
            mask = np.isfinite(u) & np.isfinite(v)
            u, v = u[mask], v[mask]
            if u.size < 2:
                return np.nan
            ru, rv = _rank_fallback(u), _rank_fallback(v)
            conc = disc = 0
            for i in range(len(ru) - 1):
                du = ru[i + 1:] - ru[i]
                dv = rv[i + 1:] - rv[i]
                s = np.sign(du * dv)
                conc += np.sum(s > 0)
                disc += np.sum(s < 0)
            denom = len(ru) * (len(ru) - 1) / 2
            if denom == 0:
                return np.nan
            return (conc - disc) / denom

        for i in range(n):
            xi = df_num.iloc[:, i].to_numpy(dtype=float)
            for j in range(i + 1, n):
                xj = df_num.iloc[:, j].to_numpy(dtype=float)
                tau = _kendall_tau_fallback(xi, xj)
                M[i, j] = M[j, i] = tau if np.isfinite(tau) else np.nan

    return pd.DataFrame(M, index=cols_, columns=cols_)


def vspace(px: int = 12):
    """Espaço vertical simples."""
    st.markdown(f"<div style='height:{int(px)}px'></div>", unsafe_allow_html=True)