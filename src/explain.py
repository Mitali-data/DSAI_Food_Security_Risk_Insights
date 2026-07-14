"""Per-country explanation of the risk score.

Deliberately NOT an LLM. The score is a weighted sum, so its drivers are exactly
recoverable by decomposition -- a template over that decomposition is faithful by
construction, whereas a generated paragraph can hallucinate a driver that isn't
in the arithmetic. In a policy-adjacent tool, faithfulness beats fluency.
Say this out loud on your 'guardrails' slide.
"""
import pandas as pd

from clustering import BOTTLENECK_MIN
from config import INDICATORS, PILLARS

PRETTY = {
    "cereal_yield_kg_per_ha": "cereal yield",
    "food_production_index": "food production",
    "fao_dietary_energy_adequacy_pct": "dietary energy adequacy",
    "gdp_per_capita_usd": "income per capita",
    "inflation_consumer_prices_pct": "consumer price inflation",
    "employment_in_agriculture_pct": "reliance on farm employment",
    "fao_child_stunting_pct": "child stunting",
    "child_mortality_per_1000": "child mortality",
    "fao_protein_supply_g_per_day": "protein supply",
    "fao_animal_protein_g_per_day": "animal protein supply",
    "fao_basic_water_access_pct": "basic water access",
    "fao_basic_sanitation_pct": "basic sanitation",
    "fao_cereal_import_dependency_pct": "cereal import dependency",
    "food_imports_pct_merchandise": "food import share of trade",
    "freshwater_withdrawal_pct": "freshwater withdrawal stress",
}

ACTIONS = {
    "availability": "supply-side support (production inputs, strategic reserves)",
    "access":       "demand-side support (cash transfers, price stabilisation)",
    "utilisation":  "nutrition & WASH programming (fortification, water, sanitation)",
    "stability":    "resilience measures (import diversification, drought insurance)",
}


def explain_country(country: str, scores: pd.DataFrame, risk_units: pd.DataFrame,
                    cluster_names: dict | None = None,
                    cluster_labels=None, top_k: int = 3) -> str:
    """Per-country brief.

    A LESSON PAID FOR IN CONFUSION -- read before editing.

    The first version reported the country's highest-scoring pillar as its
    "dominant pillar" and prescribed a response from it. That is wrong, and it
    produced a self-contradicting screen: DR Congo showed "dominant pillar:
    availability, response: supply-side support" while its cluster read "balanced,
    no distinctive bottleneck".

    Both statements were defensible; together they were incoherent. The argmax of
    the RAW pillars just finds whichever pillar is highest -- and for a country in
    severe crisis every pillar is high, so the argmax is close to arbitrary. The
    bottleneck is the pillar that is high RELATIVE TO THAT COUNTRY'S OWN MEAN.
    That is the same ipsative logic the clustering uses, and the brief must use it
    too or the tool argues with itself.

    We now report:
      - severity          (how bad, from the raw score)
      - own bottleneck    (what KIND of failure, from the centred profile)
      - cohort            (which segment it clusters into -- an AVERAGE, flagged
                           as such, because a country can sit in a cohort whose
                           mean shape is not its own)
    """
    i = scores.index[scores["country_name"] == country][0]
    row = scores.loc[i]

    pil = row[PILLARS].astype(float)
    own_mean = pil.mean()
    dev = (pil - own_mean).sort_values(ascending=False)      # ipsative: the shape
    lead, lead_val = dev.index[0], dev.iloc[0]
    lag, lag_val = dev.index[-1], dev.iloc[-1]

    lines = [
        f"**{country}** ({row['region'].strip()}, {row['year']})",
        "",
        f"**Severity** — risk score {row['score']:.2f}, rank {int(row['rank'])} "
        f"of {len(scores)}, band **{row['risk_band']}**.",
        "",
        "Pillar scores (0–1, higher = worse): "
        + ", ".join(f"{p} {pil[p]:.2f}" for p in PILLARS),
        "",
    ]

    if lead_val >= BOTTLENECK_MIN:
        lines += [
            f"**Bottleneck — {lead}.** Relative to this country's own average risk "
            f"({own_mean:.2f}), {lead} is its weakest pillar ({lead_val:+.2f}) and "
            f"{lag} its relatively strongest ({lag_val:+.2f}).",
            "",
            f"**Indicated response:** {ACTIONS[lead]}",
        ]
    else:
        lines += [
            f"**No distinct bottleneck.** Every pillar sits within {BOTTLENECK_MIN:.2f} "
            f"of this country's own mean ({own_mean:.2f}) — the failure is broad, not "
            "concentrated. A single-pillar intervention is not indicated; this needs a "
            "full assessment, not a targeted one.",
        ]

    dv = (risk_units.loc[i] - risk_units.mean()).sort_values(ascending=False)
    drivers = [d for d in dv.index[:top_k] if dv[d] > 0.05]
    if drivers:
        lines += ["", "Indicators furthest above the global average:"]
        for d in drivers:
            lines.append(f"- {PRETTY[d]} ({dv[d]:+.2f} risk units, {INDICATORS[d][1]})")

    if cluster_names is not None and cluster_labels is not None:
        c = int(cluster_labels[i])
        lines += ["", f"**Cohort:** {cluster_names[c]}",
                  "_(This describes the cohort's average shape, not necessarily this "
                  "country's. Trust the bottleneck line above for this country.)_"]

    if row["n_imputed"] > 0:
        lines += ["", f"⚠️ **{int(row['n_imputed'])} of {len(INDICATORS)} "
                      "indicators were imputed** from the regional median. This score "
                      "is less reliable than it looks."]

    return "\n".join(lines)
