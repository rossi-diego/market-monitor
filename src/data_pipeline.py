from __future__ import annotations
from collections.abc import Iterable
from os import PathLike

import numpy as np
import pandas as pd

from src.config import DATA
from src.utils import rsi

TON_OLEO: float   = 22.0462  # BOc → USD/ton
TON_FARELO: float = 1.1023   # SMc → USD/ton

# -----------------------------
# 1) Load data e padronização
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
        Path to the CSV file. Defaults to the global DATA variable.
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
# 2) Flat calculation
# -----------------------------

def add_flats_inplace(
    df: pd.DataFrame,
    maturities: Iterable[int] = range(1, 7),  # C1..C6
    brl_col: str = "brl=",
) -> None:
    """
    Add flat price columns (USD and BRL) for soybean oil and soybean meal.

    For each maturity n in `maturities`, creates:
      - Oil:
          oleo_flat_usd_c{n}
          oleo_flat_brl_c{n}  (if BRL column exists)
      - Meal:
          farelo_flat_usd_c{n}
          farelo_flat_brl_c{n}

    The function modifies the DataFrame IN PLACE.
    """

    # ----------------------------------------
    # 2.1) Identify all columns that must be numeric
    # ----------------------------------------
    numeric_cols: set[str] = set()

    for n in maturities:
        candidate_cols = (
            f"boc{n}",
            f"so-premp-c{n}",
            f"smc{n}",
            f"sm-premp-c{n}",
        )
        for col in candidate_cols:
            if col in df.columns:
                numeric_cols.add(col)

    if brl_col in df.columns:
        numeric_cols.add(brl_col)

    # Convert selected columns to numeric
    if numeric_cols:
        df[list(numeric_cols)] = df[list(numeric_cols)].apply(
            pd.to_numeric, errors="coerce"
        )

    # ----------------------------------------
    # 2.2) Compute flat prices per maturity
    # ----------------------------------------
    has_brl = brl_col in df.columns

    for n in maturities:
        boc = f"boc{n}"
        sop = f"so-premp-c{n}"
        smc = f"smc{n}"
        smp = f"sm-premp-c{n}"

        # OIL
        if boc in df.columns and sop in df.columns:
            oil_usd = f"oleo_flat_usd_c{n}"
            df.loc[:, oil_usd] = (df[boc] + df[sop] / 100.0) * TON_OLEO

            if has_brl:
                df.loc[:, f"oleo_flat_brl_c{n}"] = df[oil_usd] * df[brl_col]

        # MEAL
        if smc in df.columns and smp in df.columns:
            meal_usd = f"farelo_flat_usd_c{n}"
            df.loc[:, meal_usd] = (df[smc] + df[smp] / 100.0) * TON_FARELO

            if has_brl:
                df.loc[:, f"farelo_flat_brl_c{n}"] = df[meal_usd] * df[brl_col]

    # ----------------------------------------
    # 2.3) Backward-compatibility aliases
    # ----------------------------------------
    alias_map = {
        "oleo_flat_usd":   "oleo_flat_usd_c1",
        "oleo_flat_brl":   "oleo_flat_brl_c1",
        "farelo_flat_usd": "farelo_flat_usd_c1",
        "farelo_flat_brl": "farelo_flat_brl_c1",
    }

    for alias, source in alias_map.items():
        if source in df.columns and alias not in df.columns:
            df.loc[:, alias] = df[source]



# -----------------------------
# 3) Pipeline
# -----------------------------
def load_all() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    data = load_data_csv(DATA)
    add_flats_inplace(data)
    views = build_views(data)
    return data, views

# -----------------------------
# 4) Views (ratios) — use C1 by default
# -----------------------------
def build_views(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Build standard price-ratio views (using C1 contracts) from the input data.

    Returns a dict with zero or more of the following keys,
    depending on which columns are available in `df`:

        - "oleo_farelo" : oil/meal ratio (C1)
        - "oleo_palma"  : oil/palm ratio (C1), with short-gap filling in FCPO and MYR
        - "oleo_diesel" : oil/diesel ratio (C1, Heating Oil)
        - "oil_share"   : CME-style oil share (C1)
    """
    views: dict[str, pd.DataFrame] = {}

    # --- Oil / Meal ratio (C1) -------------------------------------------
    if {"boc1", "smc1"}.issubset(df.columns):
        v = df[["date", "boc1", "smc1"]].dropna().copy()

        v["oleo_tons"] = v["boc1"] * TON_OLEO
        v["farelo_tons"] = v["smc1"] * TON_FARELO
        v["oleo_farelo"] = v["oleo_tons"] / v["farelo_tons"]

        views["oleo_farelo"] = (
            v[["date", "oleo_farelo"]]
            .sort_values("date")
            .reset_index(drop=True)
        )

    # --- Oil / Palm ratio (C1) — robust to short gaps in FCPO and MYR ----
    if {"boc1", "fcpoc1", "myr="}.issubset(df.columns):
        v = df[["date", "boc1", "fcpoc1", "myr="]].copy()

        # Ensure proper dtypes
        v["date"] = pd.to_datetime(v["date"], errors="coerce")
        v[["boc1", "fcpoc1", "myr="]] = v[["boc1", "fcpoc1", "myr="]].apply(
            pd.to_numeric, errors="coerce"
        )
        v = v.dropna(subset=["date"]).sort_values("date")

        if not v.empty:
            # Business-day calendar
            idx = pd.bdate_range(v["date"].min(), v["date"].max(), name="date")

            # Forward-fill limits (tune according to your risk appetite)
            OIL_LIMIT = 1   # CME oil rarely "freezes"; 1 day handles isolated holidays
            PALM_LIMIT = 3  # BMD palm can miss more days; 2–3 days usually ok
            MYR_LIMIT = 3

            # Reindex and forward-fill EACH series separately
            oil = (
                v[["date", "boc1"]]
                .set_index("date")
                .reindex(idx)
                .ffill(limit=OIL_LIMIT)
            )

            palm = (
                v[["date", "fcpoc1"]]
                .set_index("date")
                .reindex(idx)
                .ffill(limit=PALM_LIMIT)
            )

            myr = (
                v[["date", "myr="]]
                .set_index("date")
                .reindex(idx)
                .ffill(limit=MYR_LIMIT)
                .rename(columns={"myr=": "myr"})
            )

            # Join and keep only days where all three series exist
            w = (
                pd.concat([oil, palm, myr], axis=1)
                .dropna(how="any")
                .reset_index()
                .rename(columns={"index": "date"})
            )

            if not w.empty:
                # Convert to USD/ton
                w["oleo_tons"] = w["boc1"] * TON_OLEO
                w["palma_tons"] = w["fcpoc1"] / w["myr"]
                w["oleo_palma"] = w["oleo_tons"] / w["palma_tons"]

                views["oleo_palma"] = (
                    w[["date", "oleo_palma"]]
                    .sort_values("date")
                    .reset_index(drop=True)
                )

    # --- Oil / Diesel ratio (C1) — Heating Oil ---------------------------
    if {"boc1", "hoc1"}.issubset(df.columns):
        v = df[["date", "boc1", "hoc1"]].dropna().copy()

        v["oleo_tons"] = v["boc1"] * TON_OLEO
        # hoc1 in USD/gal → USD/ton (using your original factors)
        v["diesel_tons"] = v["hoc1"] / 0.003785 / 0.782
        v["oleo_diesel"] = v["oleo_tons"] / v["diesel_tons"]

        views["oleo_diesel"] = (
            v[["date", "oleo_diesel"]]
            .sort_values("date")
            .reset_index(drop=True)
        )

    # --- Oil Share (CME-style) -------------------------------------------
    if {"boc1", "smc1"}.issubset(df.columns):
        v = df[["date", "boc1", "smc1"]].dropna().copy()

        v["oleo_revenue"] = v["boc1"] * 0.11
        v["farelo_revenue"] = v["smc1"] * 0.022
        v["crushing_revenue"] = v["oleo_revenue"] + v["farelo_revenue"]

        # Avoid division by zero (edge cases)
        v = v[v["crushing_revenue"] != 0].copy()
        if not v.empty:
            v["oil_share"] = v["oleo_revenue"] / v["crushing_revenue"]

            views["oil_share"] = (
                v[["date", "oil_share"]]
                .sort_values("date")
                .reset_index(drop=True)
            )

    # --- Gold / Bitcoin ratio ---------------------------------------------
    if {"gcc1", "btc="}.issubset(df.columns):
        v = df[["date", "gcc1", "btc="]].copy()

        # Ensure proper dtypes
        v["date"] = pd.to_datetime(v["date"], errors="coerce")
        v[["gcc1", "btc="]] = v[["gcc1", "btc="]].apply(
            pd.to_numeric, errors="coerce"
        )
        v = v.dropna().copy()

        # Avoid division by zero
        v = v[v["btc="] != 0].copy()
        if not v.empty:
            # Gold (USD/oz) / Bitcoin (USD) ratio
            v["gold_bitcoin"] = v["gcc1"] / v["btc="]

            views["gold_bitcoin"] = (
                v[["date", "gold_bitcoin"]]
                .sort_values("date")
                .reset_index(drop=True)
            )

    return views

# -----------------------------
# 5) Exports
# -----------------------------
def load_convenience_views() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Load the full dataset and build all standard views, returning both.

    This is a thin wrapper around `load_all()` and exists only to make the
    module-level convenience variables below cleaner.

    Returns
    -------
    data : pd.DataFrame
        The enriched dataset returned by `load_all()`.

    views : dict[str, pd.DataFrame]
        Dictionary of all built views, typically containing:
            - "oleo_farelo"
            - "oleo_palma"
            - "oleo_diesel"
            - "oil_share"
    """
    return load_all()


# Load once at import time (convenience for notebooks and scripts)
df, _views = load_convenience_views()

# Expose the most common views directly as module-level variables
oleo_farelo  = _views.get("oleo_farelo")
oleo_palma   = _views.get("oleo_palma")
oleo_diesel  = _views.get("oleo_diesel")
oil_share    = _views.get("oil_share")
gold_bitcoin = _views.get("gold_bitcoin")
