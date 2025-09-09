from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import BASE
from src.utils import rsi

# -----------------------------
# 1) Load base e padronização
# -----------------------------
def _load_base(path: str | pd.PathLike = BASE) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.rename(columns={"Date": "date"})
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        raise ValueError("Coluna de data não encontrada: espere 'Date' ou 'date'.")
    df = df.sort_values("date").reset_index(drop=True)
    return df

# -----------------------------
# 2) Colunas derivadas (flats) — IN PLACE
# -----------------------------
TON_OLEO   = 22.0462   # conversão BOc -> USD/ton
TON_FARELO = 1.1023    # conversão SMc -> USD/ton

def add_flats_inplace(
    df: pd.DataFrame,
    maturities = range(1, 7),   # C1..C6
    brl_col: str = "brl=",
) -> None:
    """Cria colunas flat (USD e BRL) para óleo e farelo, para C1..C6, modificando df IN PLACE."""
    # força numérico no que existir
    to_numeric = set()
    for n in maturities:
        to_numeric |= {
            f"boc{n}", f"so-premp-c{n}",
            f"smc{n}", f"sm-premp-c{n}",
        } & set(df.columns)
    if brl_col in df.columns:
        to_numeric.add(brl_col)
    for c in to_numeric:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # gera colunas para cada vencimento
    for n in maturities:
        boc = f"boc{n}"
        sop = f"so-premp-c{n}"
        smc = f"smc{n}"
        smp = f"sm-premp-c{n}"

        # Óleo
        if boc in df.columns and sop in df.columns:
            usd_col = f"oleo_flat_usd_c{n}"
            df.loc[:, usd_col] = (df[boc] + (df[sop] / 100.0)) * TON_OLEO
            if brl_col in df.columns:
                df.loc[:, f"oleo_flat_brl_c{n}"] = df[usd_col] * df[brl_col]

        # Farelo
        if smc in df.columns and smp in df.columns:
            usd_col = f"farelo_flat_usd_c{n}"
            df.loc[:, usd_col] = (df[smc] + (df[smp] / 100.0)) * TON_FARELO
            if brl_col in df.columns:
                df.loc[:, f"farelo_flat_brl_c{n}"] = df[usd_col] * df[brl_col]

    # aliases (compat com páginas antigas)
    alias_map = {
        "oleo_flat_usd":  "oleo_flat_usd_c1",
        "oleo_flat_brl":  "oleo_flat_brl_c1",
        "farelo_flat_usd":"farelo_flat_usd_c1",
        "farelo_flat_brl":"farelo_flat_brl_c1",
    }
    for alias, base_col in alias_map.items():
        if base_col in df.columns and alias not in df.columns:
            df.loc[:, alias] = df[base_col]

# -----------------------------
# 3) Views (ratios) — usam C1 por padrão
# -----------------------------
def build_views(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    views: dict[str, pd.DataFrame] = {}

    # Relação óleo/farelo (C1)
    if {"boc1", "smc1"}.issubset(df.columns):
        v = df[["date", "boc1", "smc1"]].dropna().copy()
        v["oleo_tons"]   = v["boc1"] * TON_OLEO
        v["farelo_tons"] = v["smc1"] * TON_FARELO
        v["oleo/farelo"] = v["oleo_tons"] / v["farelo_tons"]
        views["oleo_farelo"] = v[["date", "oleo/farelo"]].sort_values("date")

    # Relação óleo/palma (C1) — requer FCPO e MYR
    if {"boc1", "fcpoc1", "myr="}.issubset(df.columns):
        v = df[["date", "boc1", "fcpoc1", "myr="]].dropna().copy()
        v["oleo_tons"]  = v["boc1"] * TON_OLEO
        v["palma_tons"] = v["fcpoc1"] / v["myr="]
        v["oleo/palma"] = v["oleo_tons"] / v["palma_tons"]
        views["oleo_palma"] = v[["date", "oleo/palma"]].sort_values("date")

    # Relação óleo/diesel (C1) — Heating Oil
    if {"boc1", "hoc1"}.issubset(df.columns):
        v = df[["date", "boc1", "hoc1"]].dropna().copy()
        v["oleo_tons"]   = v["boc1"] * TON_OLEO
        v["diesel_tons"] = v["hoc1"] / 0.003785 / 0.782
        v["oleo/diesel"] = v["oleo_tons"] / v["diesel_tons"]
        views["oleo_diesel"] = v[["date", "oleo/diesel"]].sort_values("date")


    return views

# -----------------------------
# 4) Pipeline público
# -----------------------------
def load_all() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    base = _load_base(BASE)
    add_flats_inplace(base)      # <- MUTANDO o df original
    views = build_views(base)
    return base, views

# -----------------------------
# 5) Exports de conveniência
# -----------------------------
df, _views  = load_all()
oleo_farelo = _views.get("oleo_farelo")
oleo_palma  = _views.get("oleo_palma")
oleo_diesel = _views.get("oleo_diesel")
