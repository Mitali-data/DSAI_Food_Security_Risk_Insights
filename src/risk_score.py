"""Composite food-security risk score, built two independent ways.

Why two? Any weighting scheme is a value judgement. If a country's RANK is the
same under equal weights and under data-driven (PCA) weights, the ranking is
robust to that judgement -- and you can say so with a number instead of a shrug.
That agreement statistic is the honest core of this project.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from config import INDICATORS, PILLARS, RANDOM_STATE, indicators_for


# ------------------------------------------------------- 1. equal-weight index
def pillar_scores(risk_units: pd.DataFrame) -> pd.DataFrame:
    """Mean of the indicators inside each FAO pillar. 0-1, higher = worse."""
    return pd.DataFrame(
        {p: risk_units[indicators_for(p)].mean(axis=1) for p in PILLARS},
        index=risk_units.index,
    )


def equal_weight_score(risk_units: pd.DataFrame) -> pd.Series:
    """Unweighted mean of the four pillars.

    Note this is NOT the same as the unweighted mean of the 12 indicators:
    pillar-first stops 'utilisation' (4 indicators) from silently outvoting
    'stability' (2 indicators). That is a design choice -- defend it in the report.
    """
    return pillar_scores(risk_units).mean(axis=1).rename("score_equal")


# --------------------------------------------------------- 2. PCA-weight index
def pca_weight_score(risk_units: pd.DataFrame):
    """Weights = |PC1 loadings|, normalised. Returns (score, weights, var_explained).

    Interpretation: PC1 is the dominant axis of co-variation across indicators.
    In practice it is 'general development/fragility'. Using its loadings as
    weights lets the data decide what matters, rather than the analyst.
    Caveat: PCA weights reward indicators that CORRELATE with the majority, not
    indicators that MATTER. Report both, trust neither alone.
    """
    cols = list(INDICATORS)
    X = StandardScaler().fit_transform(risk_units[cols])
    pca = PCA(n_components=min(len(cols), 5), random_state=RANDOM_STATE).fit(X)

    load = pca.components_[0]
    if load.sum() < 0:                       # sign of PC1 is arbitrary; orient it
        load = -load                         # so that "more" = "more risk"
    w = pd.Series(np.abs(load), index=cols)
    w = w / w.sum()

    score = (risk_units[cols] * w).sum(axis=1).rename("score_pca")
    return score, w.sort_values(ascending=False), float(pca.explained_variance_ratio_[0])


# ------------------------------------------------------------------ 3. combine
def build_scores(meta: pd.DataFrame, risk_units: pd.DataFrame) -> dict:
    eq = equal_weight_score(risk_units)
    pc, weights, var1 = pca_weight_score(risk_units)

    out = meta[["country_name", "region", "income_group", "year", "n_imputed"]].copy()
    out = pd.concat([out, pillar_scores(risk_units), eq, pc], axis=1)
    out["score"] = out[["score_equal", "score_pca"]].mean(axis=1)   # headline
    out["rank_equal"] = out["score_equal"].rank(ascending=False).astype(int)
    out["rank_pca"] = out["score_pca"].rank(ascending=False).astype(int)
    out["rank"] = out["score"].rank(ascending=False).astype(int)
    out["risk_band"] = pd.qcut(
        out["score"], [0, .25, .50, .75, .90, 1.0],
        labels=["Low", "Moderate", "Elevated", "High", "Severe"],
    )

    rho, p = spearmanr(out["score_equal"], out["score_pca"])
    top_eq = set(out.nsmallest(10, "rank_equal")["country_name"])
    top_pc = set(out.nsmallest(10, "rank_pca")["country_name"])

    return {
        "scores": out,
        "pca_weights": weights,
        "pc1_variance_explained": var1,
        "agreement_spearman": float(rho),
        "agreement_p": float(p),
        "top10_overlap": len(top_eq & top_pc),
    }
