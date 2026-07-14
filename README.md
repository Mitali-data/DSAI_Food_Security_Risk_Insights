# Food Security and Nutrition Risk Insights System

Ranks 178 countries by food-security risk, explains *why* each is at risk,
segments them by the **kind** of failure they face, and reports how far the
ranking can be trusted.

**Held-out ROC-AUC = 0.88** (grouped by country, 5-fold). Six of the top ten flagged
countries are in genuine crisis. The score never sees undernourishment data — see §5.

> **We revised this number down ourselves.** We originally reported 0.945. Then we
> benchmarked our index against fitted classifiers, lost, and investigated why — and
> found that `fao_dietary_energy_adequacy_pct` is a near-copy of the label, because FAO
> *derives* undernourishment from the dietary energy supply distribution. That single
> column alone scores 0.989. Our 0.945 was inflated by it. **0.88 is the honest number,
> and it is the one we quote.** See §7.1 and `src/benchmark.py`.

---

## 1. Problem statement

**User:** an analyst at a food-security agency choosing where to send the next
needs-assessment mission, and what kind of team to send.

A single ranked list of "worst countries" is not actionable, because two countries
at the same risk level can need opposite interventions. So the system outputs a
risk score **and** a bottleneck type **and** a confidence figure — per country.

## 2. Data

`global_food_security_intelligence.csv` (Kaggle) — 217 countries, 2000–2023,
World Bank + FAO indicators.

Two things must be said about it up front:

**(a) 2022 and 2023 are empty shells.** Every metadata row exists; no measurements
do. The analysis uses **2021**, the most recent year with real data.

**(b) The dataset ships with pre-built answer keys, and they are circular.**
Five columns have **0% missing** while every genuine indicator has 15–84% missing.
Measured data cannot be complete when its inputs are not — they are derived.
Most damningly:

> `food_crisis_flag` == `(undernourishment_pct >= 25)`, at **100.0%** agreement.

Use undernourishment as a feature and that flag as a target and you will score
~100% accuracy having learned nothing but how to re-derive a threshold. These
columns are quarantined in `config.LEAKY_COLUMNS` and never enter the feature set.

## 3. Tools

Python, pandas, NumPy, scikit-learn, SciPy, Plotly, Streamlit.

## 4. How to run

```bash
pip install -r requirements.txt
python src/main.py            # full pipeline -> outputs/
streamlit run app/app.py      # Triage / Country brief / Segments / Confidence
```

## 5. The AI/ML component, and why it earns its place

**We never train a model to predict our own composite score.** That is circular.
Instead:

**(i) Turning the leak into a test.** Because the whole undernourishment family is
excluded from the score, `food_crisis_flag` becomes a genuinely **held-out
supervised label**. AUC against it answers a real question: *can the other 11
indicators find the crisis countries without being told the answer?*

**(ii) PCA-derived weights** (`risk_score.py`). Equal weighting is a value judgement
in disguise. PCA supplies a second, data-driven weighting; the **Spearman
correlation between the two rankings (0.989)** shows the ranking is robust to that
judgement. A number, not a shrug.

**(iii) Shape clustering** (`clustering.py`). Clustering raw indicators only
rediscovers rich-vs-poor — the silhouette optimum is k=2 at every attempt. We apply
**ipsative centering**: subtract each country's own mean risk from its pillars and
cluster the *composition* of risk. This separates countries that need food from
countries that need water and sanitation.

**(iv) Faithful explanation** (`explain.py`). A template over the score
decomposition, deliberately **not** an LLM. The score is a weighted sum, so its
drivers are exactly recoverable; a template is faithful by construction, while a
generated paragraph can name a driver that isn't in the arithmetic. In a
policy-adjacent tool, faithfulness beats fluency.

## 6. Validation

| Check | Result |
|---|---|
| **Held-out ROC-AUC** (honest, near-copy removed) | **0.880** |
| Held-out ROC-AUC (with near-copy — optimistic, do not quote) | 0.930–0.945 |
| Precision @ top-10 / top-20 / top-30 | 0.60 / 0.50 / 0.43 |
| Recall @ top-30 | 0.87 |
| Weighting robustness (Spearman equal vs PCA) | 0.989, 7/10 top-10 overlap |
| Bootstrap (200×): High/Severe flags holding <75% of the time | 11 of 45 |
| Leave-one-indicator-out: max top-10 churn | 3/10 (`child_mortality`) |

**Do not report accuracy.** The base rate is 8.4%, so predicting "no crisis" for
every country scores **91.6% accuracy** and is useless. AUC and precision@k are the
honest metrics.

The bootstrap result is the most *useful*: it names the 11 flags an analyst should
**not** act on without more evidence (Pakistan, Syria, Myanmar, Mali…).


## 7.1 We benchmarked ourselves against real models — and lost

Because the undernourishment family is quarantined, training a classifier on the other
indicators to predict `food_crisis_flag` is **not** circular. It is the same honest task
our index performs — except the classifier is allowed to **fit**, and our index is not.

So we ran it. 5-fold, **grouped by country**, so no test country is ever seen in *any*
year. (A plain temporal split would not have been enough: 176 of our 178 countries appear
in earlier years, and a model can simply memorise them.)

| Model | AUC (with near-copy) | AUC (without) |
|---|---|---|
| **Our index — never fitted** | 0.930 | **0.880** |
| Logistic regression | 0.992 | 0.909 |
| Random Forest | 0.985 | 0.888 |

**The classifiers beat us by 0.062.** That gap was suspicious, so we looked at what they
were leaning on. Lasso had put a coefficient of **−7.0** on one column —
`fao_dietary_energy_adequacy_pct` — ten times larger than anything else. That column
**alone** scores **AUC 0.989**, higher than our entire eleven-indicator index.

**Trap #4, and this one was ours.** FAO *derives* the Prevalence of Undernourishment
**from** the dietary energy supply distribution (r = −0.88). These are not two
measurements; they are two views of one measurement. Our quarantine caught the obvious
copies and missed the one hiding upstream in FAO's own arithmetic.

**Two consequences, both stated plainly:**

1. **Our headline was inflated.** Strip the near-copy out and our index scores **0.880**,
   not 0.945. That is the number we now quote.
2. **But so was theirs.** Without it, the classifier's advantage collapses from 0.062 to
   **0.029** — and Random Forest (0.888) essentially *ties* with our unfitted index. A
   model *allowed to fit* barely beats a fixed formula you can read off a page.

**We kept the indicator.** Calorie adequacy is a legitimate measure of food availability,
and dropping a real indicator because it is inconveniently predictive would be its own
kind of dishonesty. But it is a *valid feature* for the real-world task and a
*contaminated feature* for this particular validation. Those are different things, and
conflating them is how projects quietly overstate themselves.

Run it yourself: `python src/benchmark.py`.

## 7.2 Key findings

**Four of our fifteen indicators were pointing the wrong way.** A per-indicator
AUC check (`validation.direction_check`) exposed them:

| Indicator | We assumed | AUC alone |
|---|---|---|
| `food_imports_pct_merchandise` | imports = risk | 0.334 |
| `freshwater_withdrawal_pct` | withdrawal = stress | 0.336 |
| `food_production_index` | production = safety | 0.365 |
| `fao_cereal_import_dependency_pct` | imports = risk | 0.451 |

The countries at **100% cereal import dependency** are Hong Kong, Barbados,
Antigua, Grenada — rich, food-**secure** places that import nearly everything.
Singapore imports ~90% of its food and is among the most food-secure nations on
earth. **Import dependency is not risk**; it is risk only *conditional on being
unable to afford imports*. Unconditioned, it is a wealth proxy with the sign
reversed. Likewise, high freshwater withdrawal means you *have* irrigation, not
that you lack water; and `food_production_index` is rebased to each country's own
base year, so it is a growth index and not comparable across countries at all.

**We did not flip the signs to match the label.** Fitting directions to maximise
AUC would fit the score to the very thing we validate against, reintroducing the
circularity we set out to avoid. The AUC was used only as a *smoke alarm* that
triggered a domain review; the removals are justified by what the columns
**mean**. Residual caveat, stated honestly: having seen the label, we cannot claim
perfect independence.

**Consequence: this dataset cannot measure the FAO stability pillar.** The index
covers three of four pillars. That is a limitation of the data, not a shortcut in
the method — and removing the bad indicators *improved* every metric (AUC
0.939→0.945, top-10 precision 0.50→0.60).

**Missingness is the crisis signal.** Countries **missing** GDP data have mean
undernourishment of **37.4%**, versus **10.7%** for those that report it. Absence
of data is not absence of risk.

## 8. Limitations and responsible use

- **The tool is blind exactly where it matters most.** 39 countries were excluded
  for <60% indicator coverage. That list is **bimodal**: tax havens nobody bothers
  to measure (Monaco, Cayman, Aruba) sit alongside collapsed states nobody *can*
  measure — **Somalia, Eritrea, DPR Korea, West Bank & Gaza**. These are among the
  most food-insecure places on earth and they receive **no score at all**. A
  country's absence from the priority list must never be read as safety. This is
  the single most important caveat in the project.
- **National averages hide vulnerable people.** A country can look safe while a
  district or income decile inside it is in crisis. This tool sees countries, not
  people.
- **No conflict or displacement variable**, although conflict is the largest driver
  of modern acute food insecurity.
- **Only 3 of 4 FAO pillars measured** (see §7).
- **Annual, lagging data** — unusable for acute emergency detection.
- **Correlational.** Segments describe co-occurring weaknesses; they do not prove
  that fixing a pillar reduces risk.
- Intended as a **screening and prioritisation aid only**. It must not be cited as
  evidence of a food-security emergency, nor used to justify withholding aid.

## 9. Future improvements

Sub-national data; conflict and displacement indicators; a proper water-stress
ratio to rebuild the stability pillar; an income-interacted import-dependency term;
time-series early warning on price series.

## 10. Repository structure

```
├── data/global_food_security_intelligence.csv
├── src/
│   ├── config.py            # indicators, directions, pillars, LEAKAGE QUARANTINE
│   ├── preprocessing.py     # MNAR audit, coverage gate, imputation, normalisation
│   ├── risk_score.py        # equal-weight + PCA-weight composite
│   ├── clustering.py        # level vs shape (ipsative) segmentation
│   ├── validation.py        # direction check, AUC, bootstrap, leave-one-out
│   ├── benchmark.py         # supervised benchmark + the trap-4 finding
│   ├── explain.py           # faithful per-country brief
│   └── main.py              # end-to-end
├── app/app.py               # Streamlit decision-support UI
├── outputs/                 # generated CSVs + full pipeline log
└── requirements.txt
```

## 11. Team

_(add names)_
