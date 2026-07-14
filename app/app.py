"""Streamlit decision-support app.  Run:  streamlit run app/app.py

Design intent: this is a TRIAGE tool for an analyst deciding where to send the
next needs-assessment mission. Every screen therefore answers one of:
    1. Where is risk highest?          (Triage)
    2. Why is it high there?           (Country brief)
    3. What kind of response fits?     (Segments)
    4. How much should I trust this?   (Confidence)
A dashboard that only answers (1) is a chart, not a decision-support system.
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import clustering                                    # noqa: E402
import validation                                    # noqa: E402
from config import INDICATORS, PILLARS               # noqa: E402
from explain import explain_country                  # noqa: E402
from preprocessing import prepare                    # noqa: E402
from risk_score import build_scores, pillar_scores   # noqa: E402

st.set_page_config(page_title="Food Security Risk Insights", layout="wide")


@st.cache_data(show_spinner="Building risk index...")
def load(year: int):
    meta, ru, audit, dropped = prepare(year=year)
    res = build_scores(meta, ru)
    scores = res["scores"]
    pil = pillar_scores(ru)
    # Choose k the SAME way main.py does, or the app and the report disagree --
    # and a demo that contradicts your own deck is worse than no demo.
    sweep = clustering.choose_k_shape(ru, pil)
    cand = sweep[(sweep["k"] >= 3) & (sweep["k"] <= 5)]
    k = int(cand.loc[cand["silhouette"].idxmax(), "k"])
    cl = clustering.fit(ru, pil, k, mode="shape")
    scores["cluster"] = cl["labels"]
    scores["segment"] = [cl["names"][c] for c in cl["labels"]]
    boot = validation.bootstrap_stability(ru, meta, n_boot=100)
    scores = scores.merge(boot, on="country_name", how="left")
    ext = validation.external_validity(scores, meta)
    return meta, ru, audit, dropped, scores, res, cl, ext


st.title("Food Security & Nutrition Risk Insights")
st.caption("Screening tool for prioritising needs assessments. "
           "Not a famine declaration. Read the Confidence tab before acting.")

year = st.sidebar.selectbox("Year", [2021, 2020, 2019, 2018, 2017])
meta, ru, audit, dropped, scores, res, cl, ext = load(year)

st.sidebar.metric("Countries", len(scores))
st.sidebar.metric("Weighting agreement (Spearman)",
                  f"{res['agreement_spearman']:.3f}")
if ext["available"]:
    st.sidebar.metric("Held-out ROC-AUC", f"{ext['auc']:.3f}")
st.sidebar.metric("Countries excluded (no data)", len(dropped))
st.sidebar.caption("Somalia, Eritrea and DPR Korea are among the excluded. "
                   "Absence from this tool is NOT safety.")

t1, t2, t3, t4 = st.tabs(["Triage", "Country brief", "Segments", "Confidence"])

# ------------------------------------------------------------------- 1 triage
with t1:
    c1, c2 = st.columns([2, 1])
    with c1:
        BANDS = ["Low", "Moderate", "Elevated", "High", "Severe"]
        fig = px.scatter(
            scores, x="score_equal", y="score_pca", color="risk_band",
            hover_name="country_name", size="score",
            category_orders={"risk_band": BANDS},
            labels={"score_equal": "Equal-weight score",
                    "score_pca": "PCA-weight score"},
            title="Do the two weighting schemes agree? (points on the diagonal = robust)",
            color_discrete_map=dict(zip(BANDS, ["#fee5d9", "#fcae91", "#fb6a4a",
                                                "#de2d26", "#a50f15"])),
        )
        fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                      line=dict(dash="dash", color="grey"))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("Priority list")
        st.dataframe(
            scores.nsmallest(15, "rank")[
                ["rank", "country_name", "score", "risk_band",
                 "top_quintile_frequency", "n_imputed"]
            ].rename(columns={"top_quintile_frequency": "confidence",
                              "n_imputed": "imputed"}),
            hide_index=True, use_container_width=True,
        )
        st.caption("`confidence` = share of 100 bootstrap resamples in which the "
                   "country still lands in the top risk quintile. Below 0.75, "
                   "treat the flag as provisional.")

    st.plotly_chart(
        px.bar(scores.groupby("region")["score"].mean().sort_values(),
               orientation="h", title="Mean risk score by region",
               labels={"value": "risk score", "region": ""}),
        use_container_width=True,
    )

# ------------------------------------------------------------ 2 country brief
with t2:
    country = st.selectbox("Country", sorted(scores["country_name"]),
                           index=int(scores.nsmallest(1, "rank").index[0]))
    i = scores.index[scores["country_name"] == country][0]
    left, right = st.columns([1, 1])

    with left:
        st.markdown(explain_country(country, scores, ru, cl["names"], cl["labels"]))

    with right:
        row = scores.loc[i]
        reg_med = scores[scores["region"] == row["region"]][PILLARS].median()
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=[row[p] for p in PILLARS] + [row[PILLARS[0]]],
            theta=[p.capitalize() for p in PILLARS] + [PILLARS[0].capitalize()],
            fill="toself", name=country))
        fig.add_trace(go.Scatterpolar(
            r=[reg_med[p] for p in PILLARS] + [reg_med[PILLARS[0]]],
            theta=[p.capitalize() for p in PILLARS] + [PILLARS[0].capitalize()],
            name=f"{row['region']} median", line=dict(dash="dot")))
        fig.update_layout(
            polar=dict(radialaxis=dict(range=[0, 1])),
            title="Risk by pillar vs regional median (outward = worse)")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Indicator detail")
    det = pd.DataFrame({
        "indicator": list(INDICATORS),
        "pillar": [INDICATORS[c][1] for c in INDICATORS],
        "raw_value": [meta.loc[i, c] for c in INDICATORS],
        "risk_units_0_1": [ru.loc[i, c] for c in INDICATORS],
        "vs_global_mean": [ru.loc[i, c] - ru[c].mean() for c in INDICATORS],
    }).sort_values("vs_global_mean", ascending=False)
    st.dataframe(det.round(3), hide_index=True, use_container_width=True)

# ---------------------------------------------------------------- 3 segments
with t3:
    st.subheader("Bottleneck segments (severity removed)")
    st.caption("Clustered on the SHAPE of each country's risk profile, not its "
               "level. Two countries can be equally at risk for opposite reasons "
               "and need opposite interventions.")
    prof = cl["pillar_profile"].copy()
    prof.index = [f"[{c}] {cl['names'][c]}" for c in prof.index]
    st.plotly_chart(
        px.imshow(prof, color_continuous_scale="RdBu_r", origin="lower",
                  aspect="auto", zmin=-0.3, zmax=0.3,
                  title="Pillar composition per segment "
                        "(red = this pillar is the country's relative weak point)"),
        use_container_width=True)

    seg = st.selectbox("Inspect segment", sorted(cl["names"]),
                       format_func=lambda c: f"[{c}] {cl['names'][c]}")
    st.dataframe(
        scores[scores["cluster"] == seg][
            ["country_name", "region", "score", "risk_band"] + PILLARS
        ].sort_values("score", ascending=False).round(3),
        hide_index=True, use_container_width=True)

    st.plotly_chart(
        px.imshow(pd.crosstab(scores["segment"], scores["region"]),
                  text_auto=True, aspect="auto", color_continuous_scale="Blues",
                  title="Segment x region"),
        use_container_width=True)

# -------------------------------------------------------------- 4 confidence
with t4:
    st.subheader("How much should you trust this?")

    st.markdown(f"""
**Held-out supervised test.** `food_crisis_flag` is exactly
`undernourishment >= 25%`. The entire undernourishment family is excluded from the
score, so this is a real test, not a circular one.

- **ROC-AUC = {ext.get('auc', float('nan')):.3f}** on {ext.get('n', 0)} countries,
  {ext.get('n_crisis', 0)} of them in crisis (base rate {ext.get('base_rate', 0):.1%}).
- **Precision in the top 10 = {ext.get('precision_at_10', 0):.0%}** — of the ten
  countries this tool would send you to first, six are in genuine crisis.
- A model predicting "no crisis" for everyone scores
  **{ext.get('accuracy_if_predict_no_crisis', 0):.1%} accuracy** and is worthless.
  Never quote accuracy at an 8% base rate.

**Weighting robustness.** Equal-weight and PCA-weight rankings agree at Spearman
**{res['agreement_spearman']:.3f}**, with **{res['top10_overlap']}/10** of the top-10
shared. The ranking is therefore not an artefact of one weighting choice.

**PC1 explains {res['pc1_variance_explained']:.0%}** of indicator variance — the
indicators are largely one underlying factor, which is exactly why the *shape*
clustering in the Segments tab is necessary.
""")

    st.divider()
    st.subheader("Missingness is not random")
    st.dataframe(audit, hide_index=True, use_container_width=True)
    st.warning(
        "Indicators with a positive `mnar_gap` are missing MORE OFTEN in "
        "high-undernourishment countries. Region-median imputation therefore "
        "pulls those countries' scores toward the regional average — biasing the "
        "index **downward exactly where risk is highest**. Countries with a high "
        "`imputed` count in the Triage tab are under-flagged, not safe."
    )

    st.divider()
    st.subheader("Hard limitations")
    st.error(
        f"**The tool is blind where it matters most.** {len(dropped)} countries were "
        "excluded for having under 60% of indicators. That list is bimodal: tax havens and "
        "micro-territories nobody bothers to measure (Monaco, Cayman) sit alongside "
        "collapsed states nobody CAN measure — **Somalia, Eritrea, DPR Korea, "
        "West Bank & Gaza**. These are among the most food-insecure places on earth "
        "and they receive no score at all. A country's absence from the priority "
        "list must never be read as safety."
    )
    st.markdown("""
- **Only three of the four FAO pillars are measured.** The stability pillar was
  built, tested and removed: every candidate indicator was anti-predictive
  (AUC < 0.5). Import dependency is not risk — Hong Kong and Singapore import
  nearly all their food and are highly food-secure. The tool is blind to supply
  reliability and shock exposure.
- **National averages hide the vulnerable.** A country can look safe while a
  region, caste, or income decile inside it is in crisis. This tool cannot see
  people, only countries. It is the single most important caveat here.
- **No conflict or displacement variable.** The largest driver of modern acute
  food insecurity is armed conflict, and it is absent from these indicators.
- **Annual data, slow signal.** A price spike or failed harvest will not appear
  for months. Do not use this for acute emergency detection.
- **Correlation, not causation.** The segments describe co-occurring weaknesses.
  They do not establish that fixing a pillar would reduce risk.
- **Imputed values are guesses.** See the missingness warning above.
""")
