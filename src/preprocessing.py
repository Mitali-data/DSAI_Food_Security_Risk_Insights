"""Load, audit and normalise the real Kaggle indicator panel."""
from pathlib import Path

import numpy as np
import pandas as pd

from config import (DEFAULT_YEAR, ID_COLS, INDICATORS, LEAKY_COLUMNS,
                    MIN_COVERAGE, WINSOR_LIMITS, direction)

DATA = Path(__file__).resolve().parents[1] / "data" / "global_food_security_intelligence.csv"


def load(path: Path = DATA) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in list(INDICATORS) + ID_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Columns missing from data: {missing}. Fix src/config.py.")
    # Guard: if a leaky column ever gets into INDICATORS, fail loudly.
    leaked = set(INDICATORS) & set(LEAKY_COLUMNS)
    if leaked:
        raise ValueError(f"LEAKAGE: {leaked} are derived composites. Remove from INDICATORS.")
    return df


# --------------------------------------------------------------- missingness
def missingness_audit(df: pd.DataFrame, ref: str = "undernourishment_pct") -> pd.DataFrame:
    """Is data missing at random, or missing where risk is highest?

    For each indicator: is mean undernourishment HIGHER in the rows where that
    indicator is missing? If yes -> MNAR, and imputation will optimistically bias
    exactly the countries you most need to flag.

    (undernourishment is used only as the diagnostic yardstick here. It is not a
    feature -- see config.LEAKY_COLUMNS.)
    """
    rows = []
    for col in INDICATORS:
        m = df[col].isna()
        present = df.loc[~m, ref].mean()
        absent = df.loc[m, ref].mean() if m.sum() else np.nan
        rows.append({
            "indicator": col,
            "pct_missing": round(100 * m.mean(), 1),
            "undernourishment_when_present": round(present, 1),
            "undernourishment_when_missing": round(absent, 1) if m.sum() else np.nan,
            "mnar_gap": round(absent - present, 1) if m.sum() else 0.0,
        })
    return (pd.DataFrame(rows)
            .sort_values("mnar_gap", ascending=False)
            .reset_index(drop=True))


# ---------------------------------------------------------------- gating
def apply_coverage_gate(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Refuse to score countries with too little data.

    A country with 3 of 15 indicators present would get a score that is 80% our
    imputation rule and 20% evidence. Publishing that as a risk rating is worse
    than publishing nothing: it launders a guess as a measurement. We drop them
    and REPORT them -- the excluded list is itself a finding, because the
    countries with no data are disproportionately the fragile ones.
    """
    cols = list(INDICATORS)
    cov = df[cols].notna().mean(axis=1)
    keep = cov >= MIN_COVERAGE
    dropped = df.loc[~keep, ["country_name", "region"]].copy()
    dropped["coverage"] = cov[~keep].round(2).values
    return df.loc[keep].reset_index(drop=True), dropped.sort_values("coverage")


def impute(df: pd.DataFrame) -> pd.DataFrame:
    """Median imputation by (region x income_group), then region, then global.

    Region alone is too coarse: 'Europe & Central Asia' holds both Switzerland
    and Tajikistan, so a regional median would hand Tajikistan a Swiss value.
    Income group alone ignores geography. The interaction is the right cell --
    with fallbacks, because some cells are tiny.
    """
    out = df.copy()
    cols = list(INDICATORS)
    out["n_imputed"] = df[cols].isna().sum(axis=1)

    out[cols] = out.groupby(["region", "income_group"])[cols].transform(
        lambda s: s.fillna(s.median()))
    out[cols] = out.groupby("region")[cols].transform(lambda s: s.fillna(s.median()))
    out[cols] = out[cols].fillna(out[cols].median())
    return out


# --------------------------------------------------------------- normalisation
def winsorise(s: pd.Series, limits=WINSOR_LIMITS) -> pd.Series:
    return s.clip(s.quantile(limits[0]), s.quantile(limits[1]))


def to_risk_units(df: pd.DataFrame) -> pd.DataFrame:
    """Every indicator -> [0, 1] where 1 = worst.

    Winsorise first. Zimbabwe's 2008-style hyperinflation in the inflation column
    would otherwise compress every other country into a sliver of the 0-1 range
    and make 200 countries look identical.
    """
    out = pd.DataFrame(index=df.index)
    for col in INDICATORS:
        s = winsorise(df[col].astype(float))
        span = s.max() - s.min()
        z = (s - s.min()) / span if span > 0 else pd.Series(0.5, index=s.index)
        out[col] = z if direction(col) == +1 else 1 - z
    return out


def prepare(path: Path = DATA, year: int = DEFAULT_YEAR):
    """Returns (meta, risk_units, audit, dropped)."""
    full = load(path)
    audit = missingness_audit(full)

    yr = full[full["year"] == year].reset_index(drop=True)
    if yr[list(INDICATORS)].notna().sum().sum() == 0:
        raise ValueError(f"Year {year} has no measurements at all (2022-23 are empty shells).")

    kept, dropped = apply_coverage_gate(yr)
    meta = impute(kept)
    return meta, to_risk_units(meta), audit, dropped
