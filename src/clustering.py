"""Risk segmentation.

A cluster number is worthless. A cluster NAME derived from which indicators are
elevated inside it ("calorie-adequate but utilisation-poor") is a policy insight,
because it implies a different intervention. This module does the naming.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from config import INDICATORS, PILLARS, RANDOM_STATE, indicators_for


def choose_k(risk_units: pd.DataFrame, k_range=range(2, 9)) -> pd.DataFrame:
    X = StandardScaler().fit_transform(risk_units[list(INDICATORS)])
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE).fit(X)
        rows.append({"k": k,
                     "silhouette": round(silhouette_score(X, km.labels_), 3),
                     "inertia": round(km.inertia_, 1)})
    return pd.DataFrame(rows)


# A pillar must deviate from the country's own mean by at least this much (in
# risk units, 0-1 scale) before we are willing to call it a "bottleneck".
# Below it, the profile is flat and naming it after its largest rounding error
# would be a lie dressed as an insight.
BOTTLENECK_MIN = 0.05


def _name_cluster(profile: pd.Series, mean_risk: float, mode: str) -> str:
    """Name a cluster from its pillar composition.

    TWO RULES, both learned the hard way:

    1. NO SEVERITY IN A SHAPE-CLUSTER NAME. Shape clustering deliberately REMOVES
       severity. Putting "moderate severity" back into the label re-imports the
       thing we just stripped out, and produces absurdities -- DR Congo, rank 1 of
       178 and banded Severe, sitting in a cluster labelled "moderate severity"
       because that is the cluster's AVERAGE. The name must describe shape only.

    2. NAME NOTHING THAT ISN'T THERE. argmax always returns something. A cluster
       whose deviations are all under BOTTLENECK_MIN has no bottleneck, and saying
       otherwise is noise with a job title.
    """
    if mode != "shape":
        sev = ("Severe" if mean_risk > .70 else "High" if mean_risk > .55
               else "Moderate" if mean_risk > .40 else "Low")
        top = profile.sort_values(ascending=False)
        drivers = [p for p in top.index[:2] if top[p] > 0.25]
        if not drivers:
            return f"{sev} risk - no dominant driver"
        return (f"{sev} risk - "
                f"{' & '.join(d.capitalize() for d in drivers)}-driven")

    top = profile.sort_values(ascending=False)
    lead, lead_val = top.index[0], top.iloc[0]
    lag, lag_val = top.index[-1], top.iloc[-1]

    if lead_val < BOTTLENECK_MIN:
        return "Balanced cohort - no dominant bottleneck"

    return (f"{lead.capitalize()}-bottlenecked cohort "
            f"({lead_val:+.2f}), relatively strong on {lag} ({lag_val:+.2f})")


def _design_matrix(risk_units: pd.DataFrame, pillars: pd.DataFrame, mode: str):
    """mode='level'  -> cluster on raw standardised indicators.
       mode='shape'  -> cluster on the SHAPE of the risk profile.

    Why 'shape' matters:
    Food-security indicators share a huge common factor (national income).
    Cluster on the raw values and KMeans will simply rediscover rich-vs-poor at
    every k -- the pillar profiles come out flat, every pillar elevated together,
    and you learn nothing you didn't already know.

    'shape' applies IPSATIVE CENTERING: subtract each country's own mean risk
    from its pillars, leaving only the RELATIVE composition of its risk. Two
    countries with identical severity but different bottlenecks now land in
    different clusters. That is the segmentation a policy user actually needs,
    because availability failures and utilisation failures require different money.

    Use both. 'level' gives you triage tiers; 'shape' gives you intervention type.
    """
    if mode == "level":
        return StandardScaler().fit_transform(risk_units[list(INDICATORS)])
    if mode == "shape":
        centred = pillars.sub(pillars.mean(axis=1), axis=0)   # remove severity
        return StandardScaler().fit_transform(centred[PILLARS])
    raise ValueError("mode must be 'level' or 'shape'")


def choose_k_shape(risk_units, pillars, k_range=range(2, 9)) -> pd.DataFrame:
    X = _design_matrix(risk_units, pillars, "shape")
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE).fit(X)
        rows.append({"k": k, "silhouette": round(silhouette_score(X, km.labels_), 3)})
    return pd.DataFrame(rows)


def fit(risk_units: pd.DataFrame, pillars: pd.DataFrame, k: int,
        mode: str = "shape") -> dict:
    cols = list(INDICATORS)
    X = _design_matrix(risk_units, pillars, mode)

    km = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE).fit(X)
    labels = km.labels_

    # stability: does a different algorithm find the same partition?
    ag = AgglomerativeClustering(n_clusters=k, linkage="ward").fit(X)
    ari = adjusted_rand_score(labels, ag.labels_)   # 1.0 = identical, 0 = chance

    # indicator profile: mean z-score of each indicator per cluster
    Xi = StandardScaler().fit_transform(risk_units[cols])
    zi = pd.DataFrame(Xi, columns=cols, index=risk_units.index)
    zi["cluster"] = labels
    indicator_profile = zi.groupby("cluster").mean()

    # pillar profile drives the naming
    if mode == "shape":
        zp = pillars.sub(pillars.mean(axis=1), axis=0)   # composition, severity removed
    else:
        zp = (pillars - pillars.mean()) / pillars.std()
    zp = zp.copy()
    zp["cluster"] = labels
    pillar_profile = zp.groupby("cluster")[PILLARS].mean()

    mean_risk = pd.Series(
        pillars.mean(axis=1).values, index=risk_units.index
    ).groupby(labels).mean()

    names = {c: _name_cluster(pillar_profile.loc[c], mean_risk[c], mode) for c in range(k)}

    return {
        "labels": labels,
        "names": names,
        "indicator_profile": indicator_profile,
        "pillar_profile": pillar_profile,
        "silhouette": float(silhouette_score(X, labels)),
        "stability_ari": float(ari),
        "sizes": pd.Series(labels).value_counts().sort_index().to_dict(),
        "mode": mode,
        "mean_severity": mean_risk.round(3).to_dict(),
    }
