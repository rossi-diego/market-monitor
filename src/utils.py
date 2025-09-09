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
    df.sort_values(by=date_col, inplace=True)

    df['delta'] = df[ticker_col].diff()

    # Separar ganhos e perdas
    df['gain'] = df['delta'].where(df['delta'] > 0, 0)
    df['loss'] = -df['delta'].where(df['delta'] < 0, 0)

    # Média dos ganhos e perdas
    df['avg_gain'] = df['gain'].rolling(window=window).mean()
    df['avg_loss'] = df['loss'].rolling(window=window).mean()

    # RS e RSI
    df['rs'] = df['avg_gain'] / df['avg_loss']
    df['RSI'] = 100 - (100 / (1 + df['rs']))

    # Retornar apenas as colunas úteis
    return df[[date_col, ticker_col, 'RSI']]