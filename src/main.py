"""End-to-end pipeline. Run:  python src/main.py

Writes everything the report/PPT needs into outputs/.
"""
from pathlib import Path

import pandas as pd

import clustering
import validation
from config import DEFAULT_YEAR, EXTERNAL_LABEL
from explain import explain_country
from preprocessing import prepare
from risk_score import build_scores

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

YEAR = DEFAULT_YEAR   # 2021: the most recent year with real measurements
pd.set_option("display.width", 120)


def rule(t):
    print(f"\n{'=' * 72}\n{t}\n{'=' * 72}")


def main():
    # ---------------------------------------------------------------- 1. data
    rule("1. DATA + MISSINGNESS AUDIT")
    meta, ru, audit, dropped = prepare(year=YEAR)
    print(f"{len(meta)} countries scored, year {YEAR}")
    print(f"{len(dropped)} countries EXCLUDED for <60% indicator coverage.")
    print()
    print(">> THE EXCLUSION LIST IS BIMODAL, AND THIS IS THE HEADLINE FINDING.")
    print("   Data is missing for two OPPOSITE reasons, and the gate cannot tell")
    print("   them apart:")
    print("     (a) micro-territories nobody BOTHERS to measure (Monaco, Cayman,")
    print("         Aruba, Gibraltar) -- harmless to drop, genuinely low risk.")
    print("     (b) collapsed states nobody CAN measure -- Somalia, Eritrea,")
    print("         DPR Korea, West Bank & Gaza. These are among the most")
    print("         food-insecure places on earth AND THEY GET NO SCORE.")
    print()
    print("   A risk index that is silent precisely where risk is greatest is not")
    print("   a neutral gap. Any user must be told this explicitly, or the")
    print("   absence of a country from the priority list reads as safety.")
    print()
    fragile = dropped[dropped["region"].isin(
        ["Sub-Saharan Africa", "South Asia", "East Asia & Pacific",
         "Middle East, North Africa, Afghanistan & Pakistan"])]
    print(f"   Excluded countries in food-insecure regions ({len(fragile)}):")
    print(fragile.to_string(index=False))
    print()
    print(audit.head(6).to_string(index=False))
    mnar = audit[audit["mnar_gap"] > 2]
    if len(mnar):
        print(f"\n>> MNAR WARNING: {len(mnar)} indicator(s) are missing "
              f"disproportionately in high-undernourishment countries.")
        print(">> Imputation therefore biases the score DOWNWARD exactly where "
              "risk is highest. -> limitations section.")
    audit.to_csv(OUT / "missingness_audit.csv", index=False)

    # -------------------------------------------------------------- 2. score
    rule("2. COMPOSITE RISK SCORE (two weighting schemes)")
    res = build_scores(meta, ru)
    scores = res["scores"]
    print(f"PC1 explains {res['pc1_variance_explained']:.1%} of indicator variance")
    print("\nTop PCA weights:")
    print(res["pca_weights"].head(5).round(3).to_string())
    print(f"\n>> ROBUSTNESS: Spearman(equal-weight, PCA-weight) = "
          f"{res['agreement_spearman']:.3f}  (p={res['agreement_p']:.1e})")
    print(f">> Top-10 overlap between the two schemes: {res['top10_overlap']}/10")
    print("\nHighest-risk countries:")
    print(scores.nsmallest(10, "rank")[
        ["rank", "country_name", "region", "score", "risk_band", "n_imputed"]
    ].to_string(index=False))
    scores.to_csv(OUT / "risk_scores.csv", index=False)

    # ----------------------------------------------------------- 3. clusters
    rule("3. RISK SEGMENTATION")
    from risk_score import pillar_scores
    pil = pillar_scores(ru)

    print("-- (a) LEVEL clustering: how bad is it? --")
    sweep = clustering.choose_k(ru)
    print(sweep.to_string(index=False))

    # Silhouette almost always peaks at k=2 on development data, because the
    # dominant signal is simply "rich vs poor". That partition is statistically
    # clean and operationally useless: it prescribes one intervention for half
    # the world. We therefore constrain k >= 3 and pick the best silhouette
    # within the range that can express DIFFERENT DRIVERS of risk.
    # This is a defensible analyst choice -- but you must state it, not bury it.
    MIN_K = 3
    k_unconstrained = int(sweep.loc[sweep["silhouette"].idxmax(), "k"])
    cand = sweep[sweep["k"] >= MIN_K]
    k = int(cand.loc[cand["silhouette"].idxmax(), "k"])
    print(f"\nUnconstrained silhouette optimum: k={k_unconstrained} "
          f"(a trivial high/low split).")
    print(f"Selected k={k} (best silhouette with k>={MIN_K}), so that segments can "
          f"distinguish DRIVERS of risk, not just its level.")

    lvl = clustering.fit(ru, pil, k, mode="level")
    print(f"silhouette={lvl['silhouette']:.3f}  ARI vs Ward={lvl['stability_ari']:.3f}")
    for c, name in lvl["names"].items():
        print(f"  [{c}] n={lvl['sizes'][c]:>2}  {name}")
    print("Pillar profile (z, + = worse):")
    print(lvl["pillar_profile"].round(2).to_string())
    # Do NOT hardcode the interpretation. MEASURE it and let the data speak.
    spread = lvl["pillar_profile"].max(axis=1) - lvl["pillar_profile"].min(axis=1)
    print(f"\nWithin-cluster pillar spread (max - min z): "
          f"{spread.round(2).to_dict()}")
    if spread.max() < 0.6:
        print(">> Profiles are FLAT: every pillar elevated together. Level clustering "
              "recovered severity only, and cannot tell you what to DO.")
    else:
        print(">> Profiles are NOT flat -- unlike the synthetic pilot, the real data "
              "DOES show driver structure even under level clustering.")
        print(">> Shape clustering below isolates that structure cleanly by removing "
              "severity, but the honest statement is that level clustering already "
              "hints at it. Report this, do not overclaim the shape method.")

    print("\n-- (b) SHAPE clustering: what KIND of failure is it? --")
    sweep_s = clustering.choose_k_shape(ru, pil)
    print(sweep_s.to_string(index=False))
    # Cap at 5: beyond that the segments start splitting hairs and you cannot
    # write a distinct policy recommendation for each one. Interpretability is a
    # legitimate model-selection criterion when the output is a policy brief.
    cand_s = sweep_s[(sweep_s["k"] >= 3) & (sweep_s["k"] <= 5)]
    ks = int(cand_s.loc[cand_s["silhouette"].idxmax(), "k"])
    cl = clustering.fit(ru, pil, ks, mode="shape")
    print(f"\nk={ks}, silhouette={cl['silhouette']:.3f}, "
          f"ARI vs Ward={cl['stability_ari']:.3f}")
    scores["cluster"] = cl["labels"]
    scores["cluster_name"] = [cl["names"][c] for c in cl["labels"]]
    print("\nSegments (severity removed -- these are BOTTLENECKS):")
    for c, name in cl["names"].items():
        print(f"  [{c}] n={cl['sizes'][c]:>2}  avg severity "
              f"{cl['mean_severity'][c]:.2f}  |  {name}")
    print("\nPillar composition (deviation from the country's OWN mean risk):")
    print(cl["pillar_profile"].round(3).to_string())
    print("\nSegment x region:")
    print(pd.crosstab(scores["cluster"], scores["region"]).to_string())
    cl["pillar_profile"].to_csv(OUT / "cluster_pillar_profile.csv")
    cl["indicator_profile"].to_csv(OUT / "cluster_indicator_profile.csv")

    # --------------------------------------------------------- 4. validation
    rule("4. VALIDATION")
    dirchk = validation.direction_check(ru, meta)
    dirchk.to_csv(OUT / "direction_check.csv", index=False)
    print("0) DIRECTION SANITY CHECK (does each indicator alone point the right way?)")
    print(dirchk.to_string(index=False))
    if (dirchk["auc_alone"] < 0.5).any():
        raise SystemExit("\nFATAL: an indicator is anti-predictive. Read what the "
                         "column MEANS and fix config.py on domain grounds -- do NOT "
                         "flip the sign to chase AUC.")
    print("   All indicators point the assumed way. (Four did not, and were removed:")
    print("    see config.STABILITY_PILLAR_POSTMORTEM.)\n")

    ext = validation.external_validity(scores, meta)
    if ext["available"]:
        print(f"a) HELD-OUT SUPERVISED TEST vs {EXTERNAL_LABEL}")
        print(f"   The flag == (undernourishment >= 25%). The undernourishment family")
        print(f"   is excluded from the score, so this is NOT circular.")
        print(f"   n={ext['n']}, crisis countries={ext['n_crisis']} "
              f"(base rate {ext['base_rate']:.1%})")
        print(f"   ROC-AUC          = {ext['auc']:.3f}   "
              f"(equal {ext['auc_equal']:.3f} | PCA {ext['auc_pca']:.3f})")
        print(f"   Avg precision    = {ext['ap']:.3f}")
        for k in (10, 20, 30):
            if f"precision_at_{k}" in ext:
                print(f"   Top-{k:<2}: precision {ext[f'precision_at_{k}']:.2f}  "
                      f"recall {ext[f'recall_at_{k}']:.2f}")
        print(f"   >> A model that predicts 'no crisis' for everyone scores "
              f"{ext['accuracy_if_predict_no_crisis']:.1%} ACCURACY and is useless.")
        print(f"   >> Never report accuracy on a {ext['base_rate']:.0%} base rate.")
        print("   Per-pillar AUC (which pillar carries the signal alone?):")
        for p_ in ["availability", "access", "utilisation", "stability"]:
            if f"auc_{p_}" in ext:
                print(f"      {p_:<13} {ext[f'auc_{p_}']:.3f}")

    boot = validation.bootstrap_stability(ru, meta, n_boot=200)
    boot.to_csv(OUT / "bootstrap_stability.csv", index=False)
    flagged = set(scores[scores["risk_band"].isin(["Severe", "High"])]["country_name"])
    shaky = boot[(boot["country_name"].isin(flagged)) &
                 (boot["top_quintile_frequency"] < 0.75)]
    print(f"\nb) BOOTSTRAP (200 resamples)")
    print(f"   {len(flagged)} countries flagged High/Severe; "
          f"{len(shaky)} of them appear in the top quintile <75% of the time:")
    if len(shaky):
        print(shaky.to_string(index=False))
    print("   -> these are the flags you should NOT act on without more data.")

    loio = validation.loio_sensitivity(ru, meta)
    loio.to_csv(OUT / "loio_sensitivity.csv", index=False)
    print(f"\nc) LEAVE-ONE-INDICATOR-OUT (churn in top-10)")
    print(loio.head(4).to_string(index=False))
    worst = loio.iloc[0]
    print(f"   -> most influential single indicator: {worst['dropped_indicator']} "
          f"(dropping it changes {worst['churn']}/10 of the top list)")

    # ------------------------------------------------------- 5. explanation
    rule("5. DECISION-SUPPORT OUTPUT (sample)")
    worst_country = scores.nsmallest(1, "rank")["country_name"].iloc[0]
    print(explain_country(worst_country, scores, ru,
                          cl["names"], cl["labels"]))

    scores.to_csv(OUT / "risk_scores.csv", index=False)
    print(f"\n\nArtefacts written to {OUT}/")


if __name__ == "__main__":
    main()
