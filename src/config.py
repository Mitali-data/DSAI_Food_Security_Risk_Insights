"""Indicator definitions for the REAL Kaggle 'Global Food Security Intelligence' data.

Direction: +1 = higher value means higher risk.  -1 = higher value means lower risk.

------------------------------------------------------------------------------
READ THIS BEFORE ADDING A COLUMN.

The dataset ships with columns that have 0% missing values while every genuine
indicator has 15-84% missing. Measured data cannot be complete when its inputs
are not. They are DERIVED composites built by the dataset author:

    hunger_severity_index          r=0.93 with child stunting
    economic_vulnerability_index   r=0.90 with poverty headcount
    protein_adequacy_score         built from protein supply
    agriculture_dependence_index   built from agricultural land / employment
    food_crop_production_gap       built from crop production index
    food_crisis_flag               EXACTLY (undernourishment_pct >= 25) -- 100.0% match

Feeding these into the score and then "predicting" them is circular: it yields
~100% accuracy and zero knowledge. They are quarantined in LEAKY_COLUMNS and
never enter the feature set.

We exploit the circularity instead of falling into it. undernourishment is
EXCLUDED from the score, which turns food_crisis_flag into a genuinely held-out
label. AUC against it then measures something real: can the other 15 indicators
find the crisis countries without being told the answer?
------------------------------------------------------------------------------
"""

# Never features. Never inputs.
LEAKY_COLUMNS = [
    "hunger_severity_index",
    "economic_vulnerability_index",
    "protein_adequacy_score",
    "agriculture_dependence_index",
    "food_crop_production_gap",
    "food_crisis_flag",
    # undernourishment family held out so the flag stays an honest label
    "undernourishment_pct",
    "fao_undernourishment_pct",
    "undernourished_pop_millions",
    "undernourishment_yoy_change_pp",
    "fao_severe_food_insecurity_pct",
    "fao_mod_severe_food_insecure_millions",
    "fao_severe_food_insecure_millions",
]

# indicator -> (direction, FAO pillar)
INDICATORS = {
    # --- Availability: does enough food exist?
    "cereal_yield_kg_per_ha":            (-1, "availability"),
    # WARNING -- see LABEL_ADJACENT below and src/benchmark.py.
    # This is a legitimate availability indicator, BUT it is a near-copy of the
    # validation label: FAO derives undernourishment FROM the dietary energy supply
    # distribution (r = -0.88; this column alone scores AUC 0.989 against the flag).
    # We keep it as a FEATURE, but any AUC that includes it is OPTIMISTIC.
    "fao_dietary_energy_adequacy_pct":   (-1, "availability"),

    # --- Access: can households obtain it?
    "gdp_per_capita_usd":                (-1, "access"),
    "inflation_consumer_prices_pct":     (+1, "access"),
    "employment_in_agriculture_pct":     (+1, "access"),

    # --- Utilisation: does food become nutrition?
    "fao_child_stunting_pct":            (+1, "utilisation"),
    "child_mortality_per_1000":          (+1, "utilisation"),
    "fao_protein_supply_g_per_day":      (-1, "utilisation"),
    "fao_animal_protein_g_per_day":      (-1, "utilisation"),
    "fao_basic_water_access_pct":        (-1, "utilisation"),
    "fao_basic_sanitation_pct":          (-1, "utilisation"),

    # --- Stability: DELIBERATELY EMPTY. See STABILITY_PILLAR_POSTMORTEM below.
}

STABILITY_PILLAR_POSTMORTEM = """
The stability pillar was BUILT, TESTED, AND REMOVED. Report this; do not hide it.

Its three candidate indicators all scored AUC < 0.5 against the held-out crisis
flag -- i.e. they were ANTI-predictive. The pillar as a whole scored 0.340.
That prompted a review, and the domain reasoning (not the AUC) is what condemned
them:

  fao_cereal_import_dependency_pct  (AUC 0.451)
  food_imports_pct_merchandise      (AUC 0.334)
      The countries at 100% cereal import dependency are Hong Kong, Barbados,
      Antigua, Grenada -- rich, food-SECURE places that import nearly everything.
      Singapore imports ~90% of its food and is among the most food-secure
      nations on earth. Import dependency is not risk. It is risk ONLY IF you
      cannot afford to import; unconditioned, it is a wealth proxy with the
      sign reversed. Correctly modelling it needs an income interaction, which
      is beyond a simple additive index.

  freshwater_withdrawal_pct         (AUC 0.336)
      High withdrawal means you HAVE irrigation infrastructure. It measures
      agricultural development, not water scarcity. The variable we wanted is a
      water-stress RATIO (withdrawal / renewable supply). It is not in this
      dataset.

Also removed from availability:
  food_production_index             (AUC 0.365)
      This is an index rebased to each country's OWN base year. It measures
      growth relative to its own past, not the level of production relative to
      other countries. It is not comparable cross-sectionally and should never
      have been in a cross-sectional index. This was our error in reading the
      column, and the AUC caught it.

METHODOLOGICAL NOTE, and be honest about this in the viva:
We did NOT flip these signs to match the label. Flipping signs to maximise AUC
would fit the score to the very thing we validate against, destroying the
held-out test and reintroducing the circularity we set out to avoid. The AUC
diagnostic was used ONLY as a smoke alarm that triggered a domain review; the
decision to remove is justified by what the columns actually MEAN, and would
stand even if the flag had never existed. Residual risk: having seen the label,
we cannot claim perfect independence. That caveat is honest and should be stated.

CONSEQUENCE: this dataset cannot measure the stability pillar of the FAO
framework. The index therefore covers three of four pillars, and is blind to
supply reliability, price volatility over time, and shock exposure. That is a
real limitation of the DATA, not a shortcut in the method.
"""

# DROPPED, and why -- this list belongs in the report:
#   poverty_headcount_pct    65% missing. Imputing 2 of every 3 values means the
#                            column would mostly measure our own imputation rule.
#   stunting_prevalence_pct  84% missing. fao_child_stunting_pct (30%) replaces it.
#   fao_safe_water_pct       47% missing. fao_basic_water_access_pct (19%) replaces it.
#   fao_child_wasting_*      93% missing. Unusable.

# Three pillars, not four. This is a finding, not an omission.
PILLARS = ["availability", "access", "utilisation"]

ID_COLS = ["iso3", "country_name", "region", "income_group", "year",
           "latitude", "longitude"]

# Held-out supervised label. Legitimate ONLY because undernourishment is not a feature.
EXTERNAL_LABEL = "food_crisis_flag"

# ---------------------------------------------------------------- TRAP #4
# A near-copy of the label that our quarantine MISSED. It is not in LEAKY_COLUMNS,
# because it is a real, meaningful indicator of food availability and we keep it in
# the score. But FAO DERIVES undernourishment FROM the dietary energy supply
# distribution, so it is contaminated with respect to THIS validation label.
#
#     corr(this, undernourishment)          = -0.88
#     AUC of this column ALONE vs the flag  =  0.989   (higher than our whole index)
#
# Consequence, stated plainly rather than buried:
#     AUC that INCLUDES it  ~0.93-0.945   OPTIMISTIC. Do not quote this.
#     AUC that EXCLUDES it  ~0.88         Our honest estimate. Quote this.
#
# A valid feature for the real task; a contaminated feature for this test. Those are
# different things, and conflating them is how a project quietly overstates itself.
# See src/benchmark.py, which reports both.
LABEL_ADJACENT = "fao_dietary_energy_adequacy_pct"

# Most recent year with usable measurements. 2022-2023 are metadata-only shells.
DEFAULT_YEAR = 2021

# A country needs at least this share of indicators present to be scored at all.
MIN_COVERAGE = 0.60

WINSOR_LIMITS = (0.05, 0.95)
RANDOM_STATE = 42


def indicators_for(pillar):
    return [k for k, (_, p) in INDICATORS.items() if p == pillar]


def direction(indicator):
    return INDICATORS[indicator][0]
