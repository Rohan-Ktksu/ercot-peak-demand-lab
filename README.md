# ERCOT Peak Demand Lab

**Research and development: Rohan Sai · REVA University · Machine Learning**

A reproducible project that predicts the daily peak-demand hour in the ERCOT
electricity system and presents a **three-hour window centered on that hour**.
This is a separate companion to the [Michigan Outage Lab](https://rohan-ktksu.github.io/michigan-powercut-model/).
It predicts electricity **demand timing**, not power cuts, causes or restoration.

**Public website:** [Open ERCOT Peak Demand Lab](https://rohan-ktksu.github.io/ercot-peak-demand-lab/)

**Repository:** [Rohan-Ktksu/ercot-peak-demand-lab](https://github.com/Rohan-Ktksu/ercot-peak-demand-lab)

The website is deployed with GitHub Pages from `main` → `/docs`. The public link
does not require ChatGPT sign-in. A separate owner-private development preview
also exists on Sites; the GitHub Pages link above is the public project website.

## What the website does

The guided browser workflow is now **Dataset → Train & compare → Test & compare
→ Predict**. Prediction remains disabled until training and testing complete.

### Interactive browser experiment

The first training action loads Python/scikit-learn through Pyodide in a background
worker. A fixed whole-day sample contains 14,401 fit rows (600 days through 2021),
5,761 validation rows (240 days in 2022–2023), and all 17,496 reused test rows
(729 days in 2024–2025), seed 42. Training-only Extra Trees importance and correlation
pruning select 24 of 146 pre-engineered candidate inputs. The browser fits Logistic
Regression, Random Forest, Extra Trees and Histogram Gradient Boosting. Required
imputation/scaling fit only on training data. Validation window coverage selects
the winner; testing compares every candidate without changing the winner.

Date prediction then uses that **newly fitted browser model**, not the preloaded
research XGBoost model. Model state stays in the tab and resets on reload. The
initial runtime/package download and fitting can take several minutes. Browser
sample scores may differ from the full experiment; this is a chronological
train/validation/test experiment, not the full notebook's five-fold CV.

### Full Python research reference

The original fitted XGBoost model, five-fold CV results, model-comparison tables,
and independent reference archive evaluation remain separately labeled on the site.
The following original-model description and historical figures refer to that
reference, not the newly trained browser sample. The notebook remains the route
for reproducing the complete Python experiment.

1. Enter a complete historical 2024/2025 date, such as `2025-07-15`.
2. Retrieve its archived, selected predictor values.
3. After Train/Test, evaluate the validation-selected browser model (or separately verify the full XGBoost reference).
4. Return the central predicted hour and surrounding three-hour window.
5. Explore hourly model rankings, inspect inputs, optionally view recorded demand,
   and download the prediction as CSV.

For July 15, 2025, the predicted central interval is **4–5 PM CDT**, with a displayed
window of **3–6 PM CDT**. All times use America/Chicago with daylight-saving offsets.
The website also compares models, runs a complete 729-day historical evaluation,
and supplies the lab notebook, Python code and source datasets for download.

The browser uses actual fitted-tree inference, not hardcoded date-to-window lookup
or simulated training. Recorded demand is stored separately for retrospective
charts/evaluation and is never passed to the classifier as an input.

## Research progression and objective

The earlier work investigated Bengaluru/BESCOM outage prediction, then Michigan
county outage burden. This follow-up asks a different question: **which part of
the day is likely to have the highest electricity demand?** Estimated Wayne County
profiles were tested first. ERCOT-reported system load was then used for a separate
benchmark with additional historical years and archived forecast weather.

The initial ambition of 70% exact-hour accuracy has **not** been achieved. The
chosen presentation is a three-hour window, with its coverage reported separately
from exact accuracy. Changing the display does not make exact-hour accuracy higher.

## Dataset and provenance

| Data | Contents | Scope |
| --- | --- | --- |
| `data/demand.csv` | 99,481 source hourly records, including reserved 2026 records | Official ERCOT (ERCO) reported demand from EIA-930, July 2015 onward |
| `data/weather.csv` | 280,512 location-hour forecasts | JMA GSM nominal 48-hour-lead temperature, humidity and wind for four cities, 2018–2025 |
| `data/selected_feature_dataset.csv` | 91,465 eligible hourly modeling rows, 31 CSV columns | Timestamp, labels/partition metadata and 24 selected inputs, 2015–2025 |

Demand comes from the [official EIA-930 ERCOT workbook](https://www.eia.gov/electricity/gridmonitor/knownissues/xls/ERCO.xlsx).
[EIA data notes](https://www.eia.gov/electricity/930-content/about.html) describe
reporting, revisions and quality limitations. Source timestamps mark UTC hour
endings; the pipeline subtracts one hour for interval starts. The target is
original reported demand, not an imputed/adjusted replacement. Demand is served
ERCOT system load, not every part of Texas, a county estimate or a household meter.

Weather comes from [Open-Meteo's Previous Runs API](https://open-meteo.com/en/docs/previous-runs-api),
explicit JMA GSM nominal 48-hour forecasts. Attribute **Open-Meteo and JMA,
CC BY 4.0**. Dallas, Houston, Austin and San Antonio are four weather proxies, not
comprehensive load-weighted ERCOT weather. Earlier missing weather is preserved.

## Machine learning pipeline

The human-readable `ERCOT_Lab.ipynb` and equivalent `ercot_ml_lab.py` include:

1. Dataset import, source/hash verification, shapes, data types and missing counts.
2. Timestamp/duplicate/range checks, cleaning, EDA and complete-day target validation.
3. Calendar extraction, lagged demand, past profile/peak features and forecast weather.
4. Chronological splits before fitting learned transformations.
5. Training-only median imputation/scaling where needed. HGB/XGBoost use native
   missing handling without scaling; Logistic Regression uses both.
6. HGB-specific permutation importance and training correlation pruning above 0.98.
7. Feature budgets of 12/24/40, multiple model families and five-fold temporal CV.
8. A frozen selection rule, final refit, reused-year diagnostic evaluation and plots.
9. Model saving/reloading, input validation and a readable date-based prediction.

Each eligible day has exactly one peak label at its earliest highest-demand
interval. Missing targets are not replaced with zero. Incomplete actual-demand
days are excluded, including December 4–5, 2025. Initial history warm-up and
23/25-hour daylight-saving days are handled explicitly. A target-day intervention
checks that changing future actual demand does not change predictor inputs.

## Cross-validation and feature selection

Ordinary shuffled k-fold can train on future observations and evaluate earlier
ones. Instead, **five sklearn TimeSeriesSplit folds** use whole calendar days,
365-day validation periods and two-day gaps. All preprocessing/selection is fitted
inside each training fold. Five folds × twelve configurations = **60 comparisons**.

Compared configurations include two baselines; full HGB; 12/24/40-feature HGB;
24-feature Logistic Regression, Random Forest, Extra Trees, XGBoost classifier,
XGBoost ranker and HGB regressor. The selector uses HGB permutation importance
on sampled training days, not the later diagnostic labels. Importance describes
training reliance and may reflect overfitting/correlations; outer temporal folds
evaluate the resulting feature reduction. It is not a causal explanation.

Choose the smallest compact feature budget within one percentage point of the
best pooled CV three-hour coverage, then use coverage/exact accuracy as tie-breakers.
The selected model is **XGBoost Classifier, 24 inputs, 300 trees**: 83.56% fewer
inputs than 146. Final fitting uses **73,921 rows** before December 30, 2023,
preserving a two-day gap before 2024. No 2026 outcome is evaluated.

## Results and correct interpretation

| Selected 24-feature XGBoost model | Exact peak hour | Three-hour window coverage |
| --- | ---: | ---: |
| Five-fold development CV | 57.92% | 84.55% |
| Reused 2024 diagnostics, 366 days | 54.64% | 83.33% |
| Reused 2025 diagnostics, 363 days | 49.86% | 82.37% |

CV scores select the model; they are not independent test-performance claims.
2024/2025 were inspected earlier and are **reused diagnostics**. Full model and
feature-comparison records are in `model/research_results.json` and the notebook.

The original 146-input model achieved 57.10% exact / 84.43% window coverage in
2024 and 52.07% / 83.47% in 2025. The compact model saves inputs; it did not establish
a more accurate replacement on these examined years. Extra Trees with 24 inputs
looked promising in diagnostics, but the CV-frozen winner is not changed afterward.

Exact accuracy requires the correct one-hour interval. Window coverage allows
one hour either side of the predicted interval. For a predicted 4–5 PM center,
actual peak intervals 3–4, 4–5 or 5–6 PM count as covered. These metrics cannot be
added. Historical coverage is not a calibrated probability for an individual date.

## Verification

`node verify.mjs` compares browser inference with native Python predictions across
**729 dates and 17,496 hourly intervals**. All central peak choices match; maximum
score difference is about **5.96 × 10⁻⁸**. It also checks metrics, feature count,
checksums, missing dates and DST grids. Desktop/mobile browser controls and the
interactive full-archive evaluation were checked separately.

## Run locally

The browser website needs Node.js for local serving, with no npm dependencies:

```powershell
node preview.mjs
node verify.mjs
```

Open http://127.0.0.1:4173. To run/retrain the Python lab from this directory:

```powershell
python -m pip install -r requirements.txt
python ercot_ml_lab.py
```

Or open `ERCOT_Lab.ipynb` in Jupyter/Colab and follow its data-location instructions.
`export_model.py` rebuilds browser assets from this repository's fitted artifact,
selected dataset, audit and source CSVs. Re-export and rerun verification together;
do not manually reorder predictor columns.

## GitHub Pages deployment

Publish from **branch `main`, folder `/docs`**, using Settings → Pages → Deploy
from a branch. `docs/.nojekyll` makes this a buildless static site. Paths are relative
so both Pages' repository subpath and local preview work. This GitHub repository
contains no Sites account configuration, hosting token or private service secret.

## Limitations and next work

- Historical illustration only: future dates need fresh as-of load/weather inputs.
- Historical source revisions are not publication-time vintages.
- A two-hour load delay and six-hour weather allowance are research assumptions.
- Served demand can fall during load shedding; this is not unconstrained demand.
- Four-city forecast proxies do not fully represent the ERCOT service area.
- Predictor importance is associational, not causal evidence for an individual peak.
- Reused diagnostics and changing patterns limit future-performance claims.
- 70% exact-hour accuracy remains unmet; window coverage is a different metric.

Future work should compare frozen candidates on genuinely untouched later dates
with validated feed availability. This independent student project is not affiliated
with ERCOT or EIA. The Michigan website and repository remain separate.

## Future-date approximation: 2026–2030

Future dates produce a three-hour approximate peak-demand window using historical month/weekday patterns. This separate calendar baseline uses training days through December 29, 2023. It selects the three-hour window containing the most training-day peaks in each group. It does not invent future load or weather, and can run before browser training.

Reused historical coverage: 71.58% in 2024 (366 days), 73.28% in 2025 (363 days). These are historical diagnostics, not confidence percentages for future dates. Weather, holidays and structural changes can move the actual peak outside the window. No future MW curve is fabricated. Historical model predictions still require browser training/testing.

### Synthetic scenario option
Check the synthetic-input option for 2026–2030 to run the full exported XGBoost reference model on an explicitly simulated scenario. Each month/weekday profile borrows all 24 selected input features from a complete 2023 historical analog day nearest mid-month. This includes engineered previous-day, 2/7/14-day load and archived forecast-weather inputs. Calendar inputs are updated for the requested date. Historical target MW is not copied or presented as future demand. DST days contain 23 or 25 actual hourly intervals.

This is historical-analog resampling, not newly measured 2026 data, a weather forecast, or a validated operational forecast. No synthetic outcomes are added to training, validation or test scores. Future scenario accuracy is unknown and the interface labels the result **Approximate peak-demand window**, identifies the synthetic inputs, and shows future accuracy as **Not yet measured**. The chart shows only uncalibrated model ranking; it does not fabricate a future demand curve. Run `node verify-synthetic.mjs` for winter/summer, DST, input-width and inference checks.
