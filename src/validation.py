"""Three validations. This module is what separates the project from a notebook.

1. CONSTRUCT VALIDITY - does the score agree with an index it was never shown?
2. BOOTSTRAP STABILITY - if I had sampled different countries, would the same
   ones still be in the top quintile?
3. LEAVE-ONE-INDICATOR-OUT - is the top-10 an artefact of one column?

An unsupervised index with no ground truth CAN be validated. Most students
claim it can't, and skip this. Do not skip this.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.utils import resample

from config import EXTERNAL_LABEL, INDICATORS, RANDOM_STATE
from risk_score import equal_weight_score, pca_weight_score


# ------------------------------------------------------- 1. construct validity
def external_validity(scores: pd.DataFrame, raw: pd.DataFrame) -> dict:
    """Can the score identify crisis countries it was never shown?

    food_crisis_flag == (undernourishment_pct >= 25), exactly. Because the whole
    undernourishment family is excluded from the feature set (config.LEAKY_COLUMNS),
    this is a genuine held-out supervised test, not a circular one.

    AUC is the right metric, not accuracy: the base rate is ~9%, so a model that
    predicts "no crisis" for every country scores 91% accuracy and is useless.
    Reporting accuracy here would be the single most misleading number available.

    We also report precision@k, because that is what the user actually does --
    they can fund N missions, so what matters is how many of the top N are real.
    """
    if EXTERNAL_LABEL not in raw.columns:
        return {"available": False}
    y = raw[EXTERNAL_LABEL].values
    out = {"available": True, "n": int(len(scores)),
           "n_crisis": int(y.sum()), "base_rate": float(y.mean())}
    if y.sum() < 2:
        return {**out, "auc": float("nan")}

    out["auc"] = float(roc_auc_score(y, scores["score"]))
    out["auc_equal"] = float(roc_auc_score(y, scores["score_equal"]))
    out["auc_pca"] = float(roc_auc_score(y, scores["score_pca"]))
    out["ap"] = float(average_precision_score(y, scores["score"]))

    order = np.argsort(-scores["score"].values)
    for k in (10, 20, 30):
        k = min(k, len(y))
        out[f"precision_at_{k}"] = float(y[order[:k]].sum() / k)
        out[f"recall_at_{k}"] = float(y[order[:k]].sum() / y.sum())

    # accuracy of the naive baseline, shown ONLY to be argued against
    out["accuracy_if_predict_no_crisis"] = float(1 - y.mean())

    # per-pillar AUC: which pillar carries the signal on its own?
    for p in ["availability", "access", "utilisation", "stability"]:
        if p in scores.columns:
            out[f"auc_{p}"] = float(roc_auc_score(y, scores[p]))
    return out


# ------------------------------------------------------ 2. bootstrap stability
def bootstrap_stability(risk_units: pd.DataFrame, meta: pd.DataFrame,
                        n_boot: int = 200, top_frac: float = 0.20) -> pd.DataFrame:
    """Resample countries with replacement, recompute the score, and record how
    often each country lands in the top quintile of risk.

    A country flagged 'Severe' that only appears in the top quintile 60% of the
    time is a country you should NOT send a mission to on this evidence alone.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    idx = np.arange(len(risk_units))
    hits = pd.Series(0, index=meta["country_name"].values, dtype=float)
    draws = pd.Series(0, index=meta["country_name"].values, dtype=float)

    for _ in range(n_boot):
        b = resample(idx, replace=True, n_samples=len(idx),
                     random_state=int(rng.integers(1e9)))
        ru = risk_units.iloc[b].reset_index(drop=True)
        countries = meta["country_name"].iloc[b].values

        eq = equal_weight_score(ru)
        pc, _, _ = pca_weight_score(ru)
        s = (eq.values + pc.values) / 2

        cut = np.quantile(s, 1 - top_frac)
        top = countries[s >= cut]
        for c in np.unique(countries):
            draws[c] += 1
        for c in np.unique(top):
            hits[c] += 1

    out = pd.DataFrame({
        "country_name": hits.index,
        "top_quintile_frequency": (hits / draws.replace(0, np.nan)).round(3).values,
    })
    return out.sort_values("top_quintile_frequency", ascending=False).reset_index(drop=True)


# ------------------------------------------------- 3. leave-one-indicator-out
def loio_sensitivity(risk_units: pd.DataFrame, meta: pd.DataFrame,
                     top_n: int = 10) -> pd.DataFrame:
    """Drop each indicator in turn; how much does the top-10 change?"""
    def top_set(ru):
        eq = equal_weight_score(ru)
        pc, _, _ = pca_weight_score(ru)
        s = (eq + pc) / 2
        return set(meta["country_name"].iloc[s.nlargest(top_n).index])

    base = top_set(risk_units)
    rows = []
    for col in INDICATORS:
        kept = risk_units.drop(columns=[col])
        # equal_weight_score needs the full pillar structure; recompute inline
        from config import PILLARS, indicators_for
        piv = pd.DataFrame({
            p: kept[[c for c in indicators_for(p) if c in kept.columns]].mean(axis=1)
            for p in PILLARS if any(c in kept.columns for c in indicators_for(p))
        })
        eq = piv.mean(axis=1)
        pc, _, _ = pca_weight_score_subset(kept)
        s = (eq + pc) / 2
        t = set(meta["country_name"].iloc[s.nlargest(top_n).index])
        rows.append({"dropped_indicator": col,
                     "top10_retained": len(base & t),
                     "churn": top_n - len(base & t)})
    return pd.DataFrame(rows).sort_values("churn", ascending=False).reset_index(drop=True)


def pca_weight_score_subset(ru: pd.DataFrame):
    """PCA weights over whatever columns remain (used by LOIO)."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    cols = list(ru.columns)
    X = StandardScaler().fit_transform(ru[cols])
    pca = PCA(n_components=1, random_state=RANDOM_STATE).fit(X)
    load = pca.components_[0]
    if load.sum() < 0:
        load = -load
    w = pd.Series(np.abs(load), index=cols)
    w = w / w.sum()
    return (ru[cols] * w).sum(axis=1), w, float(pca.explained_variance_ratio_[0])


# --------------------------------------------------- 4. direction sanity check
def direction_check(risk_units: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """Does each indicator, ALONE, point the way we assumed?

    AUC < 0.5 means our assumed direction is backwards -- the indicator says
    'safe' about countries that are in crisis. This is a SMOKE ALARM, not an
    auto-fix: flipping signs to maximise AUC would fit the score to the label
    and destroy the held-out test. When this fires, go and read what the column
    actually MEANS, then decide on domain grounds.

    It caught four wrong-signed indicators in this project. See
    config.STABILITY_PILLAR_POSTMORTEM.
    """
    from sklearn.metrics import roc_auc_score
    y = raw[EXTERNAL_LABEL]
    rows = [{"indicator": c,
             "pillar": INDICATORS[c][1],
             "assumed_direction": INDICATORS[c][0],
             "auc_alone": round(roc_auc_score(y, risk_units[c]), 3)}
            for c in INDICATORS]
    out = pd.DataFrame(rows).sort_values("auc_alone").reset_index(drop=True)
    out["VERDICT"] = np.where(out["auc_alone"] < 0.5, "BACKWARDS -- REVIEW", "ok")
    return out
