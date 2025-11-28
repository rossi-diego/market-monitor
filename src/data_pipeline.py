from __future__ import annotations
from os import PathLike

import numpy as np
import pandas as pd

from src.config import DATA
from src.utils import rsi

# -----------------------------
# 1) Load base e padronização
# -----------------------------
def load_data_csv(
    path: PathLike = DATA,
    possible_date_columns: tuple[str, ...] = ("date", "Date"),
) -> pd.DataFrame:
    """
    Load a CSV file and standardize its date column.

    - Reads the CSV from `path` into a pandas DataFrame.
    - Looks for a date column using the names in `possible_date_columns`.
    - Casts the date column to datetime with `errors='coerce'`.
    - Renames the chosen date column to `"date"`.
    - Sorts the DataFrame by `"date"` and resets the index.

    Parameters
    ----------
    path : str | os.PathLike | Path, optional
        Path to the CSV file. Defaults to the global BASE variable.
    possible_date_columns : tuple[str, ...], optional
        Ordered list of column names to try as the date column. The
        first one found will be used and then renamed to `"date"`.

    Returns
    -------
    pd.DataFrame
        DataFrame with a standardized `"date"` column (datetime),
        sorted in ascending order.

    Raises
    ------
    ValueError
        If none of the `possible_date_columns` are found in the CSV.
    """
    # Load the CSV
    df = pd.read_csv(path)

    # Find which column will be used as the date column
    date_col_name = None
    for col in possible_date_columns:
        if col in df.columns:
            date_col_name = col
            break

    if date_col_name is None:
        raise ValueError(
            f"Date column not found. Expected one of: {possible_date_columns}. "
            f"Available columns: {list(df.columns)}"
        )

    # Convert to datetime
    df[date_col_name] = pd.to_datetime(df[date_col_name], errors="coerce")

    # Standardize column name to 'date' (only rename if needed)
    if date_col_name != "date":
        df = df.rename(columns={date_col_name: "date"})

    # Sort and reset index
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

    # Relação óleo/palma (C1) — robusta a lacunas curtas em FCPO e MYR
    if {"boc1", "fcpoc1", "myr="}.issubset(df.columns):
        v = df[["date", "boc1", "fcpoc1", "myr="]].copy()
        v["date"] = pd.to_datetime(v["date"], errors="coerce")
        v[["boc1", "fcpoc1", "myr="]] = v[["boc1", "fcpoc1", "myr="]].apply(pd.to_numeric, errors="coerce")
        v = v.dropna(subset=["date"]).sort_values("date")

        if not v.empty:
            # calendário de dias úteis
            idx = pd.bdate_range(v["date"].min(), v["date"].max(), name="date")

            # limites de preenchimento (ajuste conforme seu apetite)
            OIL_LIMIT  = 1   # óleo (CME) raramente “congela”; 1 dia já resolve feriados pontuais
            PALM_LIMIT = 3   # palma (BMD) pode “sumir” mais; 2–3 dias costuma ser ok
            MYR_LIMIT  = 3

            # reindexa e preenche CADA série separadamente (evita perder quando uma está fechada)
            bo   = (v[["date", "boc1"]].set_index("date")
                    .reindex(idx)
                    .ffill(limit=OIL_LIMIT)
                    .rename(columns={"boc1": "boc1"}))

            fcpo = (v[["date", "fcpoc1"]].set_index("date")
                    .reindex(idx)
                    .ffill(limit=PALM_LIMIT)
                    .rename(columns={"fcpoc1": "fcpoc1"}))

            myr  = (v[["date", "myr="]].set_index("date")
                    .reindex(idx)
                    .ffill(limit=MYR_LIMIT)
                    .rename(columns={"myr=": "myr"}))

            # junta e mantém só dias em que as três séries existem após o ffill limitado
            w = pd.concat([bo, fcpo, myr], axis=1).dropna(how="any").reset_index()
            w.rename(columns={"index": "date"}, inplace=True)

            # calcula em toneladas / USD
            w["oleo_tons"]  = w["boc1"] * TON_OLEO
            w["palma_tons"] = w["fcpoc1"] / w["myr"]
            w["oleo/palma"] = w["oleo_tons"] / w["palma_tons"]

            # salva a view se restou algo
            if not w.empty:
                views["oleo_palma"] = w[["date", "oleo/palma"]].sort_values("date")

    # Relação óleo/diesel (C1) — Heating Oil
    if {"boc1", "hoc1"}.issubset(df.columns):
        v = df[["date", "boc1", "hoc1"]].dropna().copy()
        v["oleo_tons"]   = v["boc1"] * TON_OLEO
        v["diesel_tons"] = v["hoc1"] / 0.003785 / 0.782
        v["oleo/diesel"] = v["oleo_tons"] / v["diesel_tons"]
        views["oleo_diesel"] = v[["date", "oleo/diesel"]].sort_values("date")

    # Oil Share CME
    if {"boc1", "smc1"}.issubset(df.columns):
        v = df[["date", "boc1", "smc1"]].dropna().copy()
        v["oleo_revenue"]   = v["boc1"] * 0.11
        v["farelo_revenue"] = v["smc1"] * 0.022
        v["crushing_revenue"] = v["oleo_revenue"] + v["farelo_revenue"]
        v["oil_share"] = v["oleo_revenue"] / v["crushing_revenue"]
        views["oil_share"] = v[["date", "oil_share"]].sort_values("date")


    return views

# -----------------------------
# 4) Pipeline público
# -----------------------------
def load_all() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    data = load_data_csv(DATA)
    add_flats_inplace(data)
    views = build_views(data)
    return data, views

# -----------------------------
# 5) Exports de conveniência
# -----------------------------
df, _views  = load_all()
oleo_farelo = _views.get("oleo_farelo")
oleo_palma  = _views.get("oleo_palma")
oleo_diesel = _views.get("oleo_diesel")
oil_share = _views.get("oil_share")