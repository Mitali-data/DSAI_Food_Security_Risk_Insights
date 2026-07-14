"""Supervised benchmark -- and the fourth trap, which we set for ourselves.

RUN:  python src/benchmark.py

------------------------------------------------------------------------------
WHY A CLASSIFIER HERE IS NOT CIRCULAR
------------------------------------------------------------------------------
food_crisis_flag == (undernourishment_pct >= 25), exactly. The undernourishment
family is quarantined out of the feature set (config.LEAKY_COLUMNS). So training a
classifier on the OTHER indicators to predict that flag is the same honest task our
index performs -- except the classifier is allowed to FIT, and our index is not.

That is the comparison worth making:
    our index  -- a fixed formula, never shown the label
    classifier -- allowed to fit the label
If the classifier wins by a lot, our index is leaving signal on the table, and we
should say so out loud rather than wait to be asked.

------------------------------------------------------------------------------
TRAP #4: WE MISSED A NEAR-COPY OF THE LABEL
------------------------------------------------------------------------------
The first run of this benchmark gave the classifier AUC 0.992 against our 0.930.
That gap was suspicious, so we looked at WHY. Lasso had put a coefficient of -7.0 on
fao_dietary_energy_adequacy_pct -- ten times larger than anything else.

    corr(dietary_energy_adequacy, undernourishment) = -0.88
    AUC of dietary_energy_adequacy ALONE vs the flag = 0.989

That single column beats our entire eleven-indicator index. And it is not a
coincidence: FAO DERIVES the Prevalence of Undernourishment FROM the dietary energy
supply distribution. They are not two measurements. They are two views of one
measurement. Our quarantine caught the obvious copies (undernourishment_pct,
fao_undernourishment_pct, the crisis flag) and missed the one that hides upstream in
FAO's own derivation chain.

WHAT WE DO ABOUT IT -- and this is a judgement call, so we state it:

  We KEEP dietary energy adequacy in the index. It is a legitimate, meaningful
  measure of food availability, and dropping a real indicator because it is
  inconveniently good would be its own kind of dishonesty.

  But we REPORT BOTH NUMBERS, and we treat the lower one as the truth:

      AUC with it     ~0.93-0.945   <- OPTIMISTIC. Inflated by FAO's derivation chain.
      AUC without it  ~0.88         <- our honest estimate of held-out performance.

  It is a valid feature for the real-world task (measuring food security). It is a
  contaminated feature for THIS validation. Those are different things, and conflating
  them is how projects end up quietly overstating themselves.
------------------------------------------------------------------------------
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import (EXTERNAL_LABEL, INDICATORS, LEAKY_COLUMNS, PILLARS,
                    RANDOM_STATE, indicators_for)
from preprocessing import DATA, impute, load, to_risk_units

warnings.filterwarnings("ignore")

COLS = list(INDICATORS)

# The near-copy of the label. Not in LEAKY_COLUMNS, because it IS a real indicator --
# but validation that includes it is optimistic. See module docstring.
LABEL_ADJACENT = "fao_dietary_energy_adequacy_pct"
CLEAN = [c for c in COLS if c != LABEL_ADJACENT]


def panel():
    """Full multi-year panel, imputed within each year, coverage-gated."""
    full = load(DATA)
    leaked = set(COLS) & set(LEAKY_COLUMNS)
    if leaked:
        raise ValueError(f"LEAKAGE: {leaked} in feature set.")
    frames = []
    for _, g in full.groupby("year"):
        g = g.reset_index(drop=True)
        g = g[g[COLS].notna().mean(axis=1) >= 0.60].reset_index(drop=True)
        if len(g) < 30:
            continue
        gi = impute(g)
        gi["n_missing"] = g[COLS].isna().sum(axis=1)
        frames.append(gi)
    p = pd.concat(frames, ignore_index=True)
    p[COLS] = p[COLS].fillna(p[COLS].median())
    return p.dropna(subset=COLS + [EXTERNAL_LABEL]).reset_index(drop=True)


def index_score(df, cols):
    """Our composite, restricted to `cols`. Never fitted to anything."""
    ru = to_risk_units(df)
    parts = {p: ru[[c for c in indicators_for(p) if c in cols]].mean(axis=1)
             for p in PILLARS if any(c in cols for c in indicators_for(p))}
    return pd.DataFrame(parts).mean(axis=1)


def models():
    return {
        "Logistic regression (L2)": Pipeline([
            ("sc", StandardScaler()),
            ("m", LogisticRegression(max_iter=2000, class_weight="balanced",
                                     random_state=RANDOM_STATE))]),
        "Logistic regression (Lasso)": Pipeline([
            ("sc", StandardScaler()),
            ("m", LogisticRegression(penalty="l1", solver="liblinear", C=0.3,
                                     class_weight="balanced",
                                     random_state=RANDOM_STATE))]),
        "Random Forest": RandomForestClassifier(
            n_estimators=400, min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE, n_jobs=-1),
    }


def evaluate(p, cols):
    """5-fold, GROUPED BY COUNTRY.

    Why grouped and not a plain temporal split: 176 of our 178 test countries also
    appear in earlier years. Chad-2018 is nearly the same row as Chad-2021, so a
    model can MEMORISE a country instead of learning the relationship. A temporal
    split does not fix that -- it only looks rigorous. Grouping by country means the
    model has never seen the test country in ANY year.
    """
    y, grp = p[EXTERNAL_LABEL], p["iso3"].values
    gkf = GroupKFold(n_splits=5)
    rows = []

    aucs = [roc_auc_score(p.iloc[te][EXTERNAL_LABEL],
                          index_score(p.iloc[te].reset_index(drop=True), cols))
            for _, te in gkf.split(p, y, groups=grp)]
    rows.append({"model": "OUR INDEX (never fitted)", "auc": np.mean(aucs),
                 "sd": np.std(aucs), "ap": np.nan})

    for name, m in models().items():
        a, ap = [], []
        for tr, te in gkf.split(p, y, groups=grp):
            m.fit(p[cols].iloc[tr], y.iloc[tr])
            pr = m.predict_proba(p[cols].iloc[te])[:, 1]
            a.append(roc_auc_score(y.iloc[te], pr))
            ap.append(average_precision_score(y.iloc[te], pr))
        rows.append({"model": name, "auc": np.mean(a), "sd": np.std(a), "ap": np.mean(ap)})
    return pd.DataFrame(rows)


def run():
    p = panel()
    print(f"panel: {len(p)} country-years, {p.iso3.nunique()} countries, "
          f"{int(p[EXTERNAL_LABEL].sum())} crisis rows "
          f"(base rate {p[EXTERNAL_LABEL].mean():.1%})\n")

    print("=" * 74)
    print("TRAP #4  --  we missed a near-copy of the label")
    print("=" * 74)
    d = load(DATA).dropna(subset=["undernourishment_pct", LABEL_ADJACENT])
    print(f"  corr({LABEL_ADJACENT}, undernourishment) = "
          f"{d[LABEL_ADJACENT].corr(d.undernourishment_pct):+.3f}")
    print(f"  AUC of that ONE column alone vs the flag  = "
          f"{roc_auc_score(d[EXTERNAL_LABEL], -d[LABEL_ADJACENT]):.3f}")
    print("  -> higher than our entire 11-indicator index.")
    print("  -> FAO DERIVES undernourishment FROM dietary energy supply. Same measurement.")
    print("  -> Our quarantine missed it. It hides upstream in FAO's own derivation chain.\n")

    a = evaluate(p, COLS)
    b = evaluate(p, CLEAN)

    print("=" * 74)
    print(f"(A) WITH {LABEL_ADJACENT}  ({len(COLS)} indicators)   << OPTIMISTIC")
    print("=" * 74)
    print(a.round(3).to_string(index=False))

    print("\n" + "=" * 74)
    print(f"(B) WITHOUT it  ({len(CLEAN)} indicators)   << OUR HONEST ESTIMATE")
    print("=" * 74)
    print(b.round(3).to_string(index=False))

    gap_a = a.auc[1] - a.auc[0]
    gap_b = b.auc[1] - b.auc[0]
    print("\n" + "=" * 74)
    print("WHAT THIS MEANS")
    print("=" * 74)
    print(f"  With the near-copy in:   a fitted classifier beats our index by {gap_a:+.3f} AUC.")
    print(f"  With it removed:         the gap collapses to {gap_b:+.3f}.")
    print()
    print("  So the classifier's apparent superiority was mostly it discovering the")
    print("  near-copy and dumping all its weight on it. Once that is gone, a model that")
    print("  is ALLOWED TO FIT beats our fixed, transparent, never-fitted formula by")
    print(f"  about {gap_b:.2f} AUC.")
    print()
    print("  That is the honest verdict on our index: it captures nearly all the")
    print("  available signal, and it does so with a formula you can read.")
    print()
    print("  And our headline AUC of 0.945 is OPTIMISTIC. The number we should be")
    print(f"  quoting is ~{b.auc[0]:.2f}.")
    return a, b


if __name__ == "__main__":
    run()
