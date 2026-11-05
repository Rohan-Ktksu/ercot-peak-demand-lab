# %% [markdown]
# # Machine Learning Lab: Predicting a Three-Hour Electricity Peak-Demand Window
# **Subject:** Machine Learning | **Institution:** REVA University
# **Dataset:** Official ERCOT-reported hourly electricity demand, published by EIA-930.
#
# ## Aim
# Build and explain a complete Python machine learning pipeline. Given a supported
# historical date, retrieve its available past demand and archived weather forecasts,
# predict the central peak-demand hour, and display a surrounding **three-hour window**.
#
# This predicts **ERCOT system electricity demand**, not power cuts or household use.
# ERCOT does not cover every part of Texas. The target is reported served demand in MW.
#
# ## Experiment flow
# Import libraries → import datasets → profile and visualize → clean and validate →
# extract/engineer features → chronological split → fit missing-value handling and
# scaling on training data → select features inside each fold → train several models
# → compare cross-validation metrics → select a compact model → refit → evaluate
# already-examined historical years → save/reload → accept a date → print a window.
#
# **Important:** calendar/lag construction is deterministic and respects issue time.
# Learned transformations and selection never fit on validation or diagnostic labels.
# The 2024–2025 years were inspected earlier; they are **reused diagnostics**, not a
# new independent test. We do not use 2026 for fitting or accuracy reporting.

# %% [markdown]
# ## Upload this folder and run
# This folder contains only `ERCOT_Lab.ipynb`, `demand.csv`, and `weather.csv`.
#
# **Google Drive route:** upload the entire `ERCOT_Colab_Upload` folder to My Drive.
# Open `ERCOT_Lab.ipynb` with Google Colab. Before section 2, insert and run:
#
# ```python
# from google.colab import drive
# drive.mount('/content/drive')
# ```
#
# Then set this in section 2:
#
# ```python
# PROJECT_FOLDER = "/content/drive/MyDrive/ERCOT_Colab_Upload"
# ```
#
# Run the remaining cells in order. Approve Drive access yourself when Google asks.
# If the folder is inside another Drive folder, adjust that path accordingly.
#
# **Direct Colab route:** upload both CSVs into the runtime Files panel and open the
# notebook. Leave `PROJECT_FOLDER = None`. No Drive mount is needed for that route.
#
# The notebook creates an `outputs/` folder only when run. Do not upload JSON source
# notes, old result files, or the combined engineered dataset. Training calculates
# features and metrics again. Enter a supported historical date at the final prompt.

# %% [markdown]
# ## 1. Import Python and machine learning libraries
# Install missing packages in the active notebook kernel. NumPy and pandas handle
# data; scikit-learn supplies preprocessing, models and validation; XGBoost supplies
# boosted trees and ranking; Matplotlib plots results; joblib saves the fitted model.

# %%
import importlib.util
import subprocess
import sys

packages = {'numpy': 'numpy', 'pandas': 'pandas', 'sklearn': 'scikit-learn',
            'xgboost': 'xgboost', 'matplotlib': 'matplotlib', 'joblib': 'joblib', 'tzdata': 'tzdata'}
missing = [package for module, package in packages.items() if importlib.util.find_spec(module) is None]
if missing:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *missing])

# %%
import hashlib
import json
from pathlib import Path
from functools import lru_cache
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from threadpoolctl import threadpool_limits
THREAD_LIMIT = threadpool_limits(limits=2)
try:
    from IPython.display import display
except ImportError:
    display = print


import os
import re
from sklearn.model_selection import TimeSeriesSplit
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# %% [markdown]
# ## 2. Locate the two uploaded CSVs
# Leave `PROJECT_FOLDER = None` for Colab uploads to `/content`. The loader looks
# beside the notebook and in `data/`. If files are missing in Colab, it opens a
# chooser for both CSVs. Nothing is mounted from Drive. All times are interpreted
# in **America/Chicago**, including daylight-saving changes.

# %%
PROJECT_FOLDER = None  # Colab default: /content. Leave this unchanged for manual uploads.
ROOT = Path(PROJECT_FOLDER).expanduser().resolve() if PROJECT_FOLDER else Path.cwd().resolve()
if PROJECT_FOLDER is None and (ROOT / "ERCOT_Colab_Upload").is_dir():
    ROOT = ROOT / "ERCOT_Colab_Upload"
required_csvs = ["demand.csv", "weather.csv"]
candidate_folders = [ROOT, ROOT / "data", ROOT / "ERCOT_ML_Lab" / "data"]
DATA = next((folder for folder in candidate_folders
             if all((folder / name).is_file() for name in required_csvs)), ROOT)

if not all((DATA / name).is_file() for name in required_csvs):
    if "google.colab" in sys.modules:
        from google.colab import files
        print("Upload both raw CSV files using their original names:", required_csvs)
        files.upload()
        DATA = ROOT
    missing_files = [name for name in required_csvs if not (DATA / name).is_file()]
    if missing_files:
        raise FileNotFoundError("Upload these CSVs beside the notebook and rerun this cell: " + ", ".join(missing_files))

OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
LOCAL_TZ = "America/Chicago"
AVAILABILITY_DELAY_HOURS = 2
SEED = 42
pd.set_option("display.max_columns", 12)
pd.set_option("display.max_rows", 40)
print("CSV folder:", DATA)
print("Output folder:", OUT)
print("Time zone:", LOCAL_TZ)


# %% [markdown]
# ## 3. Import the datasets and verify their sources
# **Demand:** EIA's official ERCOT (ERCO) workbook extraction, July 2015 onward. Its
# CSV retains original reported demand and audit fields. Operator forecasts and EIA
# adjusted/imputed demand are excluded from our predictor list and target replacement.
#
# **Weather:** archived 48-hour-lead JMA forecasts via Open-Meteo for Dallas, Houston,
# Austin and San Antonio, 2018–2025. These are city proxies, not load-weighted ERCOT
# weather. Forecasts, rather than observed target-day weather, are used.
#
# File hashes detect changes to the supplied datasets. Attribution, source URLs and verification hashes
# are embedded in the import cell below; no JSON upload is needed. Open-Meteo/JMA weather: CC BY 4.0.

# %%
def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

# Embedded source identities remove the need to upload JSON files.
source = {'source_url': 'https://www.eia.gov/electricity/gridmonitor/knownissues/xls/ERCO.xlsx', 'csv_sha256': '482c0af5314bfa64e264826c4202ec879461f3695b93f3fec0f3c430a24f2bbb'}
weather_source = {'source': 'Open-Meteo Previous Runs API / JMA GSM', 'docs': 'https://open-meteo.com/en/docs/previous-runs-api', 'csv_sha256': '50a26879434f59f32e34c39f0950f6b4e0f9ced43cbc162025b9a23525252f9d'}
assert sha256(DATA / "demand.csv") == source["csv_sha256"]
assert sha256(DATA / "weather.csv") == weather_source["csv_sha256"]
raw = pd.read_csv(DATA / "demand.csv")
weather_raw = pd.read_csv(DATA / "weather.csv")
print("Demand dataset shape (rows, columns):", raw.shape)
print("Weather dataset shape (rows, columns):", weather_raw.shape)
print("Demand source:", source["source_url"])
print("Source file validation: PASSED")
display(raw.head())
display(weather_raw.head())

# %% [markdown]
# ## 4. Data profiling: shape, types, missing values and duplicates
# Profiling shows what we actually received before cleaning. Missing targets cannot
# be invented: a day with unknown actual demand cannot have a verified daily peak.
# Missing predictor values are handled differently from missing targets.

# %%
profile = pd.DataFrame({"dtype": raw.dtypes.astype(str),
                        "missing_values": raw.isna().sum(),
                        "missing_percent": (raw.isna().mean() * 100).round(2)})
print("Raw demand rows:", len(raw))
print("Raw demand columns:", raw.shape[1])
print("Total raw demand missing cells:", int(raw.isna().sum().sum()))
print("Missing reported-demand values:", int(raw["demand_mw"].isna().sum()))
print("Duplicate demand timestamps:", int(raw["timestamp_end_utc"].duplicated().sum()))
print("Total raw weather missing cells:", int(weather_raw.isna().sum().sum()))
print("Duplicate weather keys:", int(weather_raw.duplicated(["location", "valid_time_utc"]).sum()))
display(profile)
display(raw.select_dtypes(include="number").describe().round(2))

# %% [markdown]
# ## 5. Data cleaning and validation
# Parse UTC timestamps, validate the ERCOT identifier, require hour alignment,
# reject duplicate keys, convert demand to numeric and mark invalid values missing.
# Shift hour-ending timestamps back one hour to represent interval starts, sort and
# reindex the hourly grid. Conflicting duplicates require review rather than averaging.
#
# We do not blindly delete high demand as an outlier: genuine extremes are important.
# The 500,000 MW check is only a generous plausibility guard. Missing actual targets
# remain missing; complete-day filtering happens after creating the local-day grid.

# %%
assert set(raw.balancing_authority) == {'ERCO'}
raw['timestamp_end_utc'] = pd.to_datetime(raw.timestamp_end_utc, utc=True, errors='raise')
assert not raw.timestamp_end_utc.duplicated().any()
assert raw.timestamp_end_utc.dt.minute.eq(0).all()
values = pd.to_numeric(raw.demand_mw, errors='coerce').to_numpy(copy=True)
invalid = ~np.isfinite(values) | (values <= 0) | (values > 500000)
values[invalid] = np.nan
index = pd.DatetimeIndex(raw.timestamp_end_utc) - pd.Timedelta(hours=1)
load = pd.Series(values, index=index).sort_index()
load = load.reindex(pd.date_range(load.index.min(), load.index.max(), freq='h', tz='UTC'))
load.index.name = 'timestamp_start_utc'
load.name = 'reported_demand_mw'
assert len(load) >= 5000

print("Hourly grid shape:", load.shape)
print("Invalid/missing reported targets:", int(invalid.sum()))
print("Missing demand after reindexing:", int(load.isna().sum()))
print("UTC interval-start range:", load.index.min(), "to", load.index.max())
print("Data cleaning and key validation: PASSED")
print("Downloaded 2026 rows remain outside the modeling and evaluation periods.")

# %% [markdown]
# ## 6. Exploratory data analysis
# Plot daily mean demand over the modeling years. This is descriptive exploration,
# not a way to select a model using diagnostic-year performance. Demand patterns
# can shift across seasons and years, which is why chronological validation matters.

# %%
historical_load = load.loc[load.index.tz_convert(LOCAL_TZ).year <= 2025]
display(historical_load.groupby(historical_load.index.tz_convert(LOCAL_TZ).year)
        .agg(["size", "count", "min", "max", "mean"]).round(2))
fig, ax = plt.subplots(figsize=(12, 3))
historical_load.resample("D").mean().plot(ax=ax, color="teal")
ax.set(title="ERCOT reported daily mean demand", xlabel="UTC date", ylabel="Demand (MW)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7. Calendar feature extraction and demand feature engineering
# **Extraction** obtains hour, weekday, month, holidays and cyclic encodings from
# timestamps. **Engineering** constructs lagged demand, historical summaries, prior
# peak timing and standardized historical load shapes.
#
# Each target day is issued at local midnight. Source load intervals must end before
# the assumed two-hour reporting cutoff. The previous day is partial; 2/7/14-day
# profiles are historical. Unknown hours stay missing. DST days have 23 or 25 intervals.
# The following helpers contain no model fitting and never read target-day load as an input.

# %%
def local_day_grid(date):
    day = pd.Timestamp(date)
    if day.tzinfo is not None or day != day.normalize():
        raise ValueError('Enter a date only, YYYY-MM-DD.')
    a = day.tz_localize(LOCAL_TZ)
    b = (day + pd.Timedelta(days=1)).tz_localize(LOCAL_TZ)
    return pd.date_range(a, b, freq='h', inclusive='left').tz_convert('UTC')

def holiday_dates(first_year, last_year):
    return USFederalHolidayCalendar().holidays(start=f'{first_year - 1}-01-01', end=f'{last_year + 1}-12-31')

def calendar_features(index):
    local = index.tz_convert(LOCAL_TZ)
    holidays = holiday_dates(local.min().year, local.max().year)
    values = pd.DataFrame(index=index)
    values['hour'] = local.hour
    values['weekday'] = local.day_of_week
    values['month'] = local.month
    values['day_of_year'] = local.day_of_year
    values['year_from_2016'] = local.year - 2016
    values['weekend'] = (local.day_of_week >= 5).astype(int)
    values['federal_holiday'] = local.normalize().tz_localize(None).isin(holidays).astype(int)
    values['utc_offset_hours'] = [t.utcoffset().total_seconds() / 3600 for t in local]
    for name, period in [('hour', 24), ('weekday', 7), ('day_of_year', 365.2425)]:
        values[name + '_sin'] = np.sin(2 * np.pi * values[name] / period)
        values[name + '_cos'] = np.cos(2 * np.pi * values[name] / period)
    return values.astype(float)

# %%
def vectorized_history_features(history, first='2018-01-01', last='2023-12-31'):
    local = history.index.tz_convert(LOCAL_TZ)
    mask = (local.tz_localize(None) >= first) & (local.tz_localize(None) < pd.Timestamp(last) + pd.Timedelta(days=1))
    index = history.index[mask]
    local = index.tz_convert(LOCAL_TZ)
    dates = pd.Index(local.date)
    issue = local.normalize().tz_convert('UTC')
    cutoff = issue - pd.Timedelta(hours=AVAILABILITY_DELAY_HOURS)
    feature = calendar_features(index)
    for lag in (48, 168):
        source_times = index - pd.Timedelta(hours=lag)
        assert (source_times + pd.Timedelta(hours=1) <= cutoff).all()
        feature[f'lag_{lag}h_mw'] = history.reindex(source_times).to_numpy()
    for stat in ('mean', 'min', 'max', 'std'):
        rolling = getattr(history.rolling(24, min_periods=24), stat)()
        feature[f'previous_24h_{stat}_mw'] = rolling.reindex(cutoff - pd.Timedelta(hours=1)).to_numpy()
    extras = {}
    all_local = history.index.tz_convert(LOCAL_TZ)
    for days in (1, 2, 7, 14):
        eligible = history.loc[all_local.hour < 22] if days == 1 else history
        eligible = eligible.dropna()
        eligible_local = eligible.index.tz_convert(LOCAL_TZ)
        groups = eligible.groupby(eligible_local.date)
        stats = groups.agg(['count', 'mean', 'std'])
        peak_times = groups.idxmax()
        stats['peak_hour'] = pd.DatetimeIndex(peak_times).tz_convert(LOCAL_TZ).hour
        profile = eligible.groupby([eligible_local.date, eligible_local.hour]).mean().unstack().reindex(columns=range(24))
        source_dates = dates - pd.Timedelta(days=days)
        source_dates = pd.Index([v.date() if hasattr(v, 'date') else v for v in source_dates])
        daily = stats.reindex(source_dates)
        matrix = profile.reindex(source_dates).to_numpy()
        values = matrix[np.arange(len(index)), local.hour]
        prefix = f'past_{days}d_'
        extras[prefix + 'known_hours'] = daily['count'].fillna(0).to_numpy()
        extras[prefix + 'peak_hour'] = daily.peak_hour.to_numpy()
        extras[prefix + 'mean_mw'] = daily['mean'].to_numpy()
        extras[prefix + 'std_mw'] = daily['std'].clip(lower=1).to_numpy()
        extras[prefix + 'candidate_load_mw'] = values
        extras[prefix + 'candidate_shape'] = (values - daily['mean'].to_numpy()) / daily['std'].clip(lower=1).to_numpy()
        extras[prefix + 'distance_to_peak'] = np.abs(local.hour - daily.peak_hour.to_numpy())
        if days in (2, 7, 14):
            for hour in range(24):
                extras[prefix + f'profile_h{hour:02d}'] = matrix[:, hour]
    feature = pd.concat([feature, pd.DataFrame(extras, index=index)], axis=1)
    feature['local_date'] = [str(d) for d in dates]
    feature['issue_time_utc'] = issue
    feature['target_mw'] = history.reindex(index).to_numpy()
    complete = feature.groupby('local_date').target_mw.transform(lambda s: s.notna().all())
    # Require the complete actual local-day grid, including both sides of source boundaries.
    size = feature.groupby('local_date').size()
    expected = {date: len(local_day_grid(date)) for date in size.index}
    complete &= feature.local_date.map(size).eq(feature.local_date.map(expected))
    return feature.loc[complete].copy()

# %% [markdown]
# ### Create the target and the initial 120 load/calendar inputs
# `target_mw` is actual reported hourly demand. `peak_label` is 1 only at the earliest
# maximum-demand interval of each complete day. These are **labels**, never inputs.
# Fourteen initial days warm up history; incomplete target days are excluded.
# Missing historical predictor values do not automatically discard an otherwise
# verifiable target day.

# %%
CAL_FEATURES = list(calendar_features(load.index[:24]).columns)
HISTORY_FEATURES = ['lag_48h_mw', 'lag_168h_mw', 'previous_24h_mean_mw',
                    'previous_24h_min_mw', 'previous_24h_max_mw', 'previous_24h_std_mw']
frame = vectorized_history_features(load, first='2015-07-15', last='2025-12-31')
frame['local_year'] = pd.to_datetime(frame.local_date).dt.year.to_numpy()
LOAD_FEATURES = CAL_FEATURES + HISTORY_FEATURES + [c for c in frame if c.startswith('past_')]
assert len(LOAD_FEATURES) == 120
first_peak = frame.groupby('local_date').target_mw.idxmax()
frame['peak_label'] = frame.index.isin(first_peak).astype(int)
assert frame.groupby('local_date').peak_label.sum().eq(1).all()


print("Load/calendar candidate features:", len(LOAD_FEATURES))
print("Eligible hourly rows:", len(frame))
print("Complete local days:", frame.local_date.nunique())
display(frame[["local_date", "target_mw", "peak_label"] + CAL_FEATURES[:5]].head())

# %% [markdown]
# ## 8. Engineer weather forecast features and check availability
# Join archived forecast values by UTC valid time. Create four-city summaries,
# forecast daily temperature statistics, forecast curve shape and heating/cooling
# proxies. Daily summaries use predictions available before issue time, not later
# weather observations.
#
# Require valid time minus 48 hours plus a six-hour allowance to precede issue time.
# This is a documented availability assumption, not a verified publication log.
# Keep pre-2018 weather missing instead of backfilling it with future observations.

# %%
weather = weather_raw.copy()
weather['valid_time_utc'] = pd.to_datetime(weather.valid_time_utc, utc=True)
assert not weather.duplicated(['location', 'valid_time_utc']).any()
for column, lower, upper in [('temperature_c', -80, 65), ('humidity_pct', 0, 100), ('wind_kmh', 0, 350)]:
    weather[column] = pd.to_numeric(weather[column], errors='coerce')
    weather.loc[~weather[column].between(lower, upper), column] = np.nan
wide = weather.pivot(index='valid_time_utc', columns='location', values=['temperature_c', 'humidity_pct', 'wind_kmh'])
wide.columns = ['forecast_' + variable + '_' + city.lower() for variable, city in wide.columns]
frame = frame.join(wide, how='left')
for variable in ['temperature_c', 'humidity_pct', 'wind_kmh']:
    columns = [c for c in wide if c.startswith('forecast_' + variable + '_')]
    frame['forecast_' + variable + '_city_mean'] = frame[columns].mean(axis=1)
    if variable == 'temperature_c':
        frame['forecast_temperature_city_spread'] = frame[columns].max(axis=1) - frame[columns].min(axis=1)
groups = frame.groupby('local_date', sort=False)
for stat in ['mean', 'min', 'max', 'std']:
    frame['forecast_temp_day_' + stat] = groups.forecast_temperature_c_city_mean.transform(stat)
frame['forecast_temp_shape'] = (frame.forecast_temperature_c_city_mean - frame.forecast_temp_day_mean) / frame.forecast_temp_day_std.clip(lower=0.1)
frame['forecast_temp_step_previous'] = groups.forecast_temperature_c_city_mean.diff().fillna(0)
frame['forecast_temp_step_next'] = groups.forecast_temperature_c_city_mean.shift(-1).sub(frame.forecast_temperature_c_city_mean).fillna(0)
frame['forecast_heating_degree'] = (18 - frame.forecast_temperature_c_city_mean).clip(lower=0)
frame['forecast_cooling_degree'] = (frame.forecast_temperature_c_city_mean - 18).clip(lower=0)
frame['forecast_temperature_rank'] = groups.forecast_temperature_c_city_mean.rank(pct=True)
FORECAST_FEATURES = [c for c in frame if c.startswith('forecast_')]
for date, group in frame.groupby('local_date', sort=False):
    known = group.forecast_temperature_c_city_mean.dropna()
    if len(known):
        available = known.index - pd.Timedelta(hours=48) + pd.Timedelta(hours=6)
        assert (available <= group.issue_time_utc.iloc[0]).all()

# %%
ALL_FEATURES = LOAD_FEATURES + FORECAST_FEATURES
assert len(ALL_FEATURES) == 146
assert not set(ALL_FEATURES).intersection({"target_mw", "peak_label", "operator_forecast_mw", "local_year"})
missing_inputs = frame[ALL_FEATURES].isna().sum().sort_values(ascending=False)
print("Engineered dataset shape (including labels/metadata):", frame.shape)
print("Candidate input matrix shape:", frame[ALL_FEATURES].shape)
print("Calendar/load features:", len(LOAD_FEATURES))
print("Forecast-weather features:", len(FORECAST_FEATURES))
print("TOTAL CANDIDATE FEATURES:", len(ALL_FEATURES))
print("Total missing predictor cells:", int(missing_inputs.sum()))
print("Features containing missing values:", int(missing_inputs.gt(0).sum()))
display(missing_inputs.loc[missing_inputs.gt(0)].rename("missing_values").to_frame())
display(frame.groupby("local_year").agg(hourly_rows=("target_mw", "size"), days=("local_date", "nunique")))

# %% [markdown]
# ## 9. Additional data validation and a leakage intervention
# Check that each target day has exactly one peak label. Artificially change every
# target-day load value and confirm that its predictor inputs remain unchanged.
# This checks that target-day observations cannot silently become forecasting inputs.
# Retain unknown targets and exclude incomplete days rather than pretending zero load.

# %%
probe = local_day_grid('2022-07-15')
modified = load.copy()
modified.loc[probe] += 100000
altered = vectorized_history_features(modified, first='2022-07-15', last='2022-07-15')
pd.testing.assert_frame_equal(altered.loc[probe, LOAD_FEATURES], frame.loc[probe, LOAD_FEATURES], check_dtype=False, check_freq=False)
print('Target-day load intervention check passed: no change to predictor inputs.')


assert frame.groupby("local_date").peak_label.sum().eq(1).all()
assert frame.index.is_unique
excluded_days = {}
for year in range(2016, 2026):
    calendar_count = len(pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D"))
    usable_count = frame.loc[frame.local_year.eq(year)].local_date.nunique()
    excluded_days[year] = calendar_count - usable_count
print("Missing-target days excluded by year (complete years only):")
display(pd.Series(excluded_days, name="excluded_days").to_frame())
print("Timestamp, target and leakage validation: PASSED")

# %% [markdown]
# ## 10. Chronological splitting before learned preprocessing
# The final training period ends before December 30, 2023. December 30–31 form a
# two-day gap before the 2024 diagnostic period. For cross-validation, five expanding
# training blocks validate the next 365 calendar days, also with two-day gaps.
#
# The diagnostic years 2024 and 2025 were examined in earlier work. Their results
# will be labeled reused diagnostics and **cannot select the winner**. Ordinary
# random train/test splitting or shuffled KFold would mix past and future.

# %%
final_training = frame.loc[frame.local_date.lt("2023-12-30")].copy()
gap_rows = frame.loc[frame.local_date.between("2023-12-30", "2023-12-31")]
diagnostic_2024 = frame.loc[frame.local_year.eq(2024)]
diagnostic_2025 = frame.loc[frame.local_year.eq(2025)]
partition_summary = pd.DataFrame([
    {"partition": name, "rows": len(part), "days": part.local_date.nunique(),
     "first_date": part.local_date.min(), "last_date": part.local_date.max()}
    for name, part in [("Final training", final_training), ("Gap", gap_rows),
                       ("Reused 2024 diagnostics", diagnostic_2024), ("Reused 2025 diagnostics", diagnostic_2025)]
])
display(partition_summary)
print("Training input shape:", final_training[ALL_FEATURES].shape)
print("Training label shape:", final_training.peak_label.shape)
print("2024 input shape:", diagnostic_2024[ALL_FEATURES].shape)
print("2025 input shape:", diagnostic_2025[ALL_FEATURES].shape)

calendar_days = pd.date_range(frame.local_date.min(), "2023-12-31", freq="D")
splitter = TimeSeriesSplit(n_splits=5, test_size=365, gap=2)
fold_specification = list(splitter.split(calendar_days))
split_preview = pd.DataFrame([
    {"fold": number, "train_end": str(calendar_days[tr[-1]].date()),
     "validation_start": str(calendar_days[va[0]].date()), "validation_end": str(calendar_days[va[-1]].date())}
    for number, (tr, va) in enumerate(fold_specification, 1)
])
print("Five chronological validation folds:")
display(split_preview)

# %% [markdown]
# ## 11. Handle missing predictor values and demonstrate scaling
# **Logistic Regression:** training-fitted median imputation followed by StandardScaler.
# **Random Forest / Extra Trees:** training-fitted median imputation; no scaling needed.
# **HGB / XGBoost:** retain missing inputs for native missing-value handling; no scaling needed.
#
# The illustration below uses only the first fold's training rows. It is not fed to
# the other folds: each actual model pipeline refits its transformations inside its
# own training fold. Fitting an imputer/scaler once on the whole dataset before
# cross-validation would leak information.
#
# Scaling makes training columns roughly zero mean and unit variance. It changes
# units, not which future observations are available. Labels are never imputed or scaled.

# %%
first_train_end = str(calendar_days[fold_specification[0][0][-1]].date())
first_fold_training = frame.loc[frame.local_date.le(first_train_end)]
preprocessing_demo = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler())
scaled_training = preprocessing_demo.fit_transform(first_fold_training[ALL_FEATURES])
assert np.isfinite(scaled_training).all()
print("Demonstration fitted only through:", first_train_end)
print("Missing training input cells before imputation:", int(first_fold_training[ALL_FEATURES].isna().sum().sum()))
print("Missing input cells after imputation/scaling:", int(np.isnan(scaled_training).sum()))
print("Scaled training shape:", scaled_training.shape)
example_columns = ["hour", "lag_48h_mw", "forecast_temperature_c_dallas"]
print("Before preprocessing:")
display(first_fold_training[example_columns].head().round(3))
print("After training-fitted imputation and scaling:")
display(pd.DataFrame(scaled_training[:5], columns=ALL_FEATURES)[example_columns].round(3))
del scaled_training  # Keep the notebook's memory use modest.

# %% [markdown]
# ## 12. Plan the models, feature budgets and selection rule
# Compare twelve configurations covering seven estimator types plus two baselines.
# The full 146-input HGB is a reference; compact candidates must use at most 40 inputs.
#
# Primary metric: **three-hour window coverage**, equivalent to the actual peak
# being within one hour of the predicted interval. Secondary metrics: exact hour
# and average absolute timing error. Choose the smallest compact feature budget
# within one percentage point of the best compact CV window coverage, then use
# coverage and exact accuracy as tie-breakers. Fix this rule before reading diagnostics.

# %%
CANDIDATES = ['calendar_mode', 'previous_week_peak', 'hgb_full_reference',
    'hgb_12', 'hgb_24', 'hgb_40', 'logistic_24', 'random_forest_24',
    'extra_trees_24', 'xgb_classifier_24', 'xgb_ranker_24', 'hgb_regressor_24']
plan = dict(primary_metric='Daily three-hour window coverage (within one hour of predicted interval)',
    secondary_metrics=['exact_peak_hour_accuracy', 'peak_time_mae_hours', 'mean_demand_regret_pct'],
    cross_validation='sklearn TimeSeriesSplit, five folds, 365 calendar days validation, two calendar days gap',
    selection='Smallest compact configuration within 1 percentage point of best compact CV window coverage; then coverage and exact accuracy',
    feature_budgets=[12, 24, 40], maximum_deployed_features=40, candidates=CANDIDATES,
    selector='Permutation negative-log-loss importance of full HGB classifier on 96 evenly spaced training days from last 730; two repeats; training correlation pruning >0.98',
    selector_limitation='Importance is in-sample model reliance, not unbiased importance or causation; outer temporal CV evaluates the reduced models.',
    final_training='Eligible days before 2023-12-30, preserving two-day gap before 2024',
    diagnostic_years=[2024, 2025], evaluation_status='Already inspected years reused for diagnostics; no fresh independent claim',
    untouched_year=2026, source_demand_sha256=source['csv_sha256'], weather_sha256=weather_source['csv_sha256'])
(OUT / 'selected_cv_plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
print("Configurations to compare:", len(CANDIDATES))
print("Feature budgets:", plan["feature_budgets"])
print("Maximum deployed inputs:", plan["maximum_deployed_features"])
print("Primary metric:", plan["primary_metric"])
print("Selection rule:", plan["selection"])
display(pd.DataFrame({"configuration": CANDIDATES}))

# %% [markdown]
# ## 13. Feature selection using the actual HGB classifier
# Within each training fold, fit the full HGB classifier. Measure permutation
# importance on 96 evenly spaced training days from its last 730 available days,
# using two permutations and negative log loss. Remove constant/all-missing inputs
# and lower-ranked features whose absolute training correlation exceeds 0.98.
#
# Then refit HGB with 12/24/40 retained features. The other compact families use the
# same 24-feature ranking for a controlled comparison. Importance describes
# in-sample model reliance, not causation or unbiased effect size. Outer temporal
# validation evaluates whether the resulting feature reduction generalizes.

# %%
def new_hgb():
    return HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=15,
        min_samples_leaf=30, learning_rate=0.05, l2_regularization=5,
        early_stopping=False, random_state=SEED)

# %%
def classifier_selector(training):
    full = new_hgb().fit(training[ALL_FEATURES], training.peak_label)
    available_days = np.array(sorted(training.local_date.unique()))[-730:]
    sampled_days = available_days[np.linspace(0, len(available_days) - 1, min(96, len(available_days)), dtype=int)]
    sample = training.loc[training.local_date.isin(sampled_days)]
    importance = permutation_importance(full, sample[ALL_FEATURES], sample.peak_label,
        scoring='neg_log_loss', n_repeats=2, random_state=SEED, n_jobs=1)
    mean = pd.Series(importance.importances_mean, index=ALL_FEATURES)
    std = pd.Series(importance.importances_std, index=ALL_FEATURES)
    variable = training[ALL_FEATURES].nunique(dropna=True).gt(1)
    ordering = mean.loc[variable].sort_values(ascending=False, kind='stable').index.tolist()
    corr = training[ordering].corr().abs()
    kept, records = [], []
    for column in ordering:
        redundant = next((previous for previous in kept if corr.loc[column, previous] > 0.98), None)
        if redundant is None:
            kept.append(column)
        records.append(dict(feature=column, permutation_log_loss_increase=float(mean[column]),
            importance_std=float(std[column]), eligible=redundant is None,
            reason='eligible' if redundant is None else 'training_correlation_above_0.98', redundant_with=redundant or ''))
    for column in mean.index[~variable]:
        records.append(dict(feature=column, permutation_log_loss_increase=float(mean[column]),
            importance_std=float(std[column]), eligible=False, reason='constant_or_all_missing', redundant_with=''))
    assert len(kept) >= 40
    metadata = dict(training_end_utc=str(training.index.max()), importance_sample_start=str(sample.index.min()),
        importance_sample_end=str(sample.index.max()), importance_sample_days=len(sampled_days), ranking=kept)
    return full, kept, pd.DataFrame(records), metadata

# %% [markdown]
# ## 14. Define the models and baseline predictions
# Logistic Regression models a linear decision boundary after imputation/scaling.
# Random Forest and Extra Trees combine many trees. HGB and XGBoost use boosted
# trees. XGBoost ranking directly orders intervals within each day. HGB regression
# predicts hourly MW and chooses its daily maximum. The two simple baselines test
# whether learned models add value beyond calendar or previous-week peak timing.

# %%
def fit_reduced(name, training, columns):
    if name.startswith('hgb_') and name != 'hgb_regressor_24':
        estimator = new_hgb()
    elif name == 'logistic_24':
        estimator = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(),
            LogisticRegression(C=1.0, max_iter=2000, class_weight='balanced', random_state=SEED))
    elif name == 'random_forest_24':
        estimator = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True),
            RandomForestClassifier(n_estimators=200, max_depth=14, min_samples_leaf=8,
                max_features='sqrt', class_weight='balanced_subsample', n_jobs=2, random_state=SEED))
    elif name == 'extra_trees_24':
        estimator = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True),
            ExtraTreesClassifier(n_estimators=200, max_depth=18, min_samples_leaf=5,
                class_weight='balanced', n_jobs=2, random_state=SEED))
    elif name == 'xgb_classifier_24':
        estimator = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
            min_child_weight=5, reg_lambda=5, subsample=1, n_jobs=2, random_state=SEED, tree_method='hist')
    elif name == 'xgb_ranker_24':
        estimator = xgb.XGBRanker(objective='rank:ndcg', n_estimators=300, max_depth=4, learning_rate=0.05,
            min_child_weight=5, reg_lambda=5, subsample=1, n_jobs=2, random_state=SEED, tree_method='hist')
        estimator.fit(training[columns], training.peak_label,
            group=training.groupby('local_date', sort=False).size().to_numpy(), verbose=False)
        return dict(kind='ranker', estimator=estimator, columns=list(columns))
    elif name == 'hgb_regressor_24':
        estimator = HistGradientBoostingRegressor(max_iter=200, max_leaf_nodes=15, min_samples_leaf=30,
            learning_rate=0.05, l2_regularization=5, early_stopping=False, random_state=SEED)
        estimator.fit(training[columns], training.target_mw)
        return dict(kind='regressor', estimator=estimator, columns=list(columns))
    else:
        raise ValueError(name)
    estimator.fit(training[columns], training.peak_label)
    return dict(kind='classifier', estimator=estimator, columns=list(columns))

# %%
def baseline_models(training):
    peaks = training.loc[training.peak_label.eq(1)]
    modes = peaks.groupby(['month', 'weekday']).hour.agg(lambda s: s.mode().iloc[0]).to_dict()
    return {'calendar_mode': dict(kind='mode', lookup=modes, default=float(peaks.hour.mode().iloc[0]), columns=['month', 'weekday', 'hour']),
            'previous_week_peak': dict(kind='weekly', columns=['hour', 'past_7d_peak_hour'])}

def scores_for(model, evaluation):
    if model['kind'] == 'mode':
        expected = np.array([model['lookup'].get((m, w), model['default']) for m, w in zip(evaluation.month, evaluation.weekday)])
        return -np.abs(evaluation.hour.to_numpy() - expected)
    if model['kind'] == 'weekly':
        return -np.abs(evaluation.hour - evaluation.past_7d_peak_hour).fillna(24).to_numpy()
    if model['kind'] == 'classifier':
        return model['estimator'].predict_proba(evaluation[model['columns']])[:, 1]
    return model['estimator'].predict(evaluation[model['columns']])

def candidates_with_selection(training, full, ordering):
    yield from baseline_models(training).items()
    yield 'hgb_full_reference', dict(kind='classifier', estimator=full, columns=list(ALL_FEATURES))
    for name in CANDIDATES[3:]:
        count = int(name.rsplit('_', 1)[1])
        yield name, fit_reduced(name, training, ordering[:count])

# %% [markdown]
# ## 15. Define meaningful metrics
# For each complete local day, select one predicted interval and compare it with
# the earliest actual maximum. Example: predicted 4–5 PM → displayed window 3–6 PM.
#
# | Metric | Meaning |
# | --- | --- |
# | Exact-hour accuracy | Predicted interval exactly equals the actual peak interval |
# | Three-hour window coverage | Actual peak interval lies in the surrounding three hourly intervals |
# | Timing MAE in hours | Mean absolute difference between predicted and actual peak starts |
# | Demand regret (%) | How far below the day's actual maximum the chosen hour's actual demand was |
#
# Exact and within-one-hour scores are not added together. Ordinary hourly binary
# accuracy is misleading because most hours are not the daily peak. With exactly
# one predicted and one actual peak hour per day, top-one precision, recall and F1
# equal exact-hour accuracy; reporting all three would repeat the same information.
# Ranking scores are not calibrated probabilities.

# %%
def evaluate_peak(frame, scores):
    work = frame[['local_date', 'target_mw']].copy()
    work['score'] = np.asarray(scores)
    records = []
    for day, g in work.groupby('local_date', sort=True):
        actual = g.target_mw.idxmax()
        predicted = g.score.idxmax()
        error = abs((predicted - actual).total_seconds()) / 3600
        top = g.sort_values('score', ascending=False, kind='stable').head(2).index
        actual_load = float(g.target_mw.max())
        chosen_load = float(g.loc[predicted, 'target_mw'])
        records.append(dict(date=day, actual_peak_start_utc=actual, predicted_peak_start_utc=predicted,
                            exact=int(error == 0), within_one_hour=int(error <= 1), top_two=int(actual in top),
                            time_error_hours=error, demand_regret_pct=100 * (actual_load - chosen_load) / actual_load))
    days = pd.DataFrame(records)
    return dict(days=len(days), exact_peak_hour_accuracy=float(days.exact.mean()),
                within_one_hour_accuracy=float(days.within_one_hour.mean()),
                top_two_hour_coverage=float(days.top_two.mean()),
                peak_time_mae_hours=float(days.time_error_hours.mean()),
                mean_demand_regret_pct=float(days.demand_regret_pct.mean())), days

# %% [markdown]
# ## 16. Train, validate and compare five chronological folds
# This is the longest section. Each fold independently fits its classifier-based
# feature selector, preprocessing and candidate models. No previously fitted
# full-dataset scaler or selector is reused. The progress lines show each completed
# comparison. Feature audits and split boundaries are saved for checking.

# %%
calendar_days = pd.date_range(frame.local_date.min(), '2023-12-31', freq='D')
splitter = TimeSeriesSplit(n_splits=5, test_size=365, gap=2)
fold_records, split_records, feature_records = [], [], []
for number, (train_days, validation_days) in enumerate(fold_specification, start=1):
    train_end = str(calendar_days[train_days[-1]].date())
    validation_start = str(calendar_days[validation_days[0]].date())
    validation_end = str(calendar_days[validation_days[-1]].date())
    training = frame.loc[frame.local_date.le(train_end)]
    evaluation = frame.loc[frame.local_date.between(validation_start, validation_end)]
    assert training.index.max() + pd.Timedelta(hours=1 + AVAILABILITY_DELAY_HOURS) <= evaluation.issue_time_utc.min()
    assert not set(training.local_date).intersection(evaluation.local_date)
    split_records.append(dict(fold=number, training_start=training.local_date.min(), training_end=train_end,
        validation_start=validation_start, validation_end=validation_end, gap_calendar_days=2,
        training_rows=len(training), validation_rows=len(evaluation)))
    print('Selecting HGB features inside fold', number, 'training through', train_end, flush=True)
    full, ordering, audit, metadata = classifier_selector(training)
    audit.to_csv(OUT / f'hgb_feature_audit_fold_{number}.csv', index=False)
    feature_records.append(dict(fold=number, **metadata))
    for name, model in candidates_with_selection(training, full, ordering):
        metrics, daily = evaluate_peak(evaluation, scores_for(model, evaluation))
        fold_records.append(dict(fold=number, model=name, feature_count=len(model['columns']), **metrics))
        print('Fold', number, name, 'window coverage', round(metrics['within_one_hour_accuracy'], 4), flush=True)
        pd.DataFrame(fold_records).to_csv(OUT / 'cv_fold_metrics.csv', index=False)
    pd.DataFrame(split_records).to_csv(OUT / 'cv_splits.csv', index=False)
    (OUT / 'selector_fold_metadata.json').write_text(json.dumps(feature_records, indent=2), encoding='utf-8')
folds = pd.DataFrame(fold_records)
comparison = []
for name, group in folds.groupby('model', sort=False):
    comparison.append(dict(model=name, feature_count=int(group.feature_count.max()),
        days=int(group.days.sum()), exact_peak_hour_accuracy=float(np.average(group.exact_peak_hour_accuracy, weights=group.days)),
        three_hour_window_coverage=float(np.average(group.within_one_hour_accuracy, weights=group.days)),
        fold_window_std=float(group.within_one_hour_accuracy.std()),
        peak_time_mae_hours=float(np.average(group.peak_time_mae_hours, weights=group.days))))
comparison = pd.DataFrame(comparison).sort_values('three_hour_window_coverage', ascending=False)

# %%
cv_display = comparison.copy()
cv_display["Window coverage (%)"] = cv_display.three_hour_window_coverage * 100
cv_display["Exact hour (%)"] = cv_display.exact_peak_hour_accuracy * 100
cv_display["Fold coverage std (percentage points)"] = cv_display.fold_window_std * 100
print("CROSS-VALIDATION COMPARISON: selection scores, not final diagnostic accuracy")
display(cv_display[["model", "feature_count", "Window coverage (%)", "Exact hour (%)",
                    "peak_time_mae_hours", "Fold coverage std (percentage points)"]].round(2))
print("Completed folds:", folds.fold.nunique())
print("Configurations per fold:", folds.groupby("fold").size().tolist())
print("Total CV comparisons:", len(folds))

# %% [markdown]
# ## 17. Select and freeze the best compact model using CV only
# Select using the predeclared rule, not the later 2024/2025 table. The full reference
# and baselines remain visible but are not eligible compact replacements. The best
# algorithm may differ from the original full model. Reduction is a size/performance
# tradeoff and is not guaranteed to improve every metric.

# %%
compact = comparison.loc[~comparison.model.isin(['calendar_mode', 'previous_week_peak', 'hgb_full_reference'])]
eligible = compact.loc[compact.three_hour_window_coverage >= compact.three_hour_window_coverage.max() - 0.01]
WINNER = eligible.sort_values(['feature_count', 'three_hour_window_coverage', 'exact_peak_hour_accuracy', 'peak_time_mae_hours'],
                             ascending=[True, False, False, True]).iloc[0].model
comparison.to_csv(OUT / 'cv_model_comparison.csv', index=False)
(OUT / 'frozen_selected_choice.json').write_text(json.dumps(dict(model=WINNER, plan=plan), indent=2), encoding='utf-8')


print('Frozen reduced-feature choice:', WINNER)


chosen_cv = comparison.loc[comparison.model.eq(WINNER)].iloc[0]
print("SELECTED MODEL:", WINNER)
print("Selected feature budget:", int(chosen_cv.feature_count))
print("CV three-hour coverage:", f"{chosen_cv.three_hour_window_coverage:.2%}")
print("CV exact-hour accuracy:", f"{chosen_cv.exact_peak_hour_accuracy:.2%}")
print("Choice is now frozen before reused-year diagnostic evaluation.")

# %% [markdown]
# ## 18. Refit feature selection and train the final models
# Refit the selector using final training rows only. The feature budget is already
# fixed; feature identities are learned again from training. Print the selected names,
# count reduction, training shape and remaining missing values. The final compact
# model uses only its selected columns during inference.

# %%
training = final_training
full, ordering, final_audit, selector_metadata = classifier_selector(training)
trained_models = {}
for name, model in candidates_with_selection(training, full, ordering):
    trained_models[name] = model
    print("Fitted final model:", name, "with", len(model["columns"]), "inputs")
selected = trained_models[WINNER]
assert len(selected["columns"]) <= 40
final_audit["selected"] = final_audit.feature.isin(selected["columns"])
final_audit.to_csv(OUT / "final_selected_feature_audit.csv", index=False)
(OUT / "final_selector_metadata.json").write_text(json.dumps(selector_metadata, indent=2), encoding="utf-8")
print("Candidate features before selection:", len(ALL_FEATURES))
print("Features selected for prediction:", len(selected["columns"]))
print("Feature reduction:", f"{1 - len(selected['columns']) / len(ALL_FEATURES):.2%}")
print("Final selected training shape:", training[selected["columns"]].shape)
print("Missing selected training cells:", int(training[selected["columns"]].isna().sum().sum()))
print("Tree models handle these natively; imputation occurs inside relevant fitted pipelines.")
display(final_audit.loc[final_audit.selected, ["feature", "permutation_log_loss_increase", "importance_std"]])

# %% [markdown]
# ## 19. Diagnostic model comparison on already-examined 2024/2025
# These scores describe historical generalization on reused years. Do not change the
# CV-frozen choice just because another model looks better here. Missing actual
# targets exclude two 2025 dates; no labels are fabricated. We keep reporting exact
# accuracy separately from three-hour coverage.

# %%
diagnostic_records = []
for name, model in trained_models.items():
    for year in [2024, 2025]:
        evaluation = frame.loc[frame.local_year.eq(year)]
        assert training.index.max() + pd.Timedelta(hours=1 + AVAILABILITY_DELAY_HOURS) <= evaluation.issue_time_utc.min()
        metrics, daily = evaluate_peak(evaluation, scores_for(model, evaluation))
        daily["window_start_utc"] = daily.predicted_peak_start_utc - pd.Timedelta(hours=1)
        daily["window_end_utc"] = daily.predicted_peak_start_utc + pd.Timedelta(hours=2)
        diagnostic_records.append(dict(year=year, model=name, feature_count=len(model["columns"]), **metrics))
        daily.to_csv(OUT / f"diagnostic_predictions_{year}_{name}.csv", index=False)
diagnostics = pd.DataFrame(diagnostic_records)
diagnostics.to_csv(OUT / "reused_year_model_comparison.csv", index=False)
diagnostic_display = diagnostics.copy()
diagnostic_display["Exact hour (%)"] = diagnostics.exact_peak_hour_accuracy * 100
diagnostic_display["Three-hour coverage (%)"] = diagnostics.within_one_hour_accuracy * 100
print("REUSED HISTORICAL DIAGNOSTICS: these rows do not select the winner")
display(diagnostic_display[["year", "model", "feature_count", "days", "Exact hour (%)",
                            "Three-hour coverage (%)", "peak_time_mae_hours", "mean_demand_regret_pct"]].round(2))
print("Selected model's diagnostic results:")
display(diagnostic_display.loc[diagnostic_display.model.eq(WINNER),
    ["year", "model", "days", "Exact hour (%)", "Three-hour coverage (%)", "peak_time_mae_hours"]].round(2))

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
ordered = comparison.sort_values("three_hour_window_coverage")
axes[0].barh(ordered.model, ordered.three_hour_window_coverage * 100, color="teal")
axes[0].set(title="Five-fold CV model comparison", xlabel="Three-hour coverage (%)", xlim=(0, 100))
selected_diagnostics = diagnostics.loc[diagnostics.model.eq(WINNER)]
x = np.arange(len(selected_diagnostics))
axes[1].bar(x - 0.18, selected_diagnostics.exact_peak_hour_accuracy * 100, width=0.36, label="Exact hour")
axes[1].bar(x + 0.18, selected_diagnostics.within_one_hour_accuracy * 100, width=0.36, label="Three-hour window")
axes[1].set_xticks(x, selected_diagnostics.year.astype(str))
axes[1].set(title="Selected model: reused-year diagnostics", ylabel="Percent", ylim=(0, 100))
axes[1].legend()
plt.tight_layout()
fig.savefig(OUT / "lab_model_comparison.png", dpi=150)
plt.show()

# %% [markdown]
# ## 20. Save the model, selected dataset and reproducible results
# Save the fitted estimator together with its exact input column order. The combined
# CSV includes labels and split metadata for training and diagnostic rows. Its
# metadata columns are not predictor inputs. Reload only artifacts you trust; the
# joblib file below was produced by this notebook.

# %%
frame["final_partition"] = np.select(
    [frame.local_date.lt("2023-12-30"), frame.local_date.between("2023-12-30", "2023-12-31"), frame.local_year.eq(2024)],
    ["training", "gap", "reused_diagnostic_2024"], default="reused_diagnostic_2025")
report = dict(plan=plan, model=WINNER, candidate_features=len(ALL_FEATURES), selected_feature_count=len(selected['columns']),
    selected_features=selected['columns'], selected_weather_features=[c for c in selected['columns'] if c.startswith('forecast_')],
    training_rows=len(training), modeling_rows=len(frame), cv_comparison=comparison.to_dict('records'),
    reused_year_comparison=diagnostics.to_dict('records'), final_selector_metadata=selector_metadata,
    limitations=['Feature importance sampled training days is descriptive and can reflect overfitting/correlations; outer temporal CV evaluates selection.',
        '2024 and 2025 were examined earlier and are not fresh independent tests.',
        'Scores are historical; source revisions, assumed feed delays and four-city weather proxies limit operational claims.',
        'Three-hour coverage is distinct from exact-hour accuracy; model scores are not calibrated probabilities.'])
(OUT / 'lab_results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
frame[['local_date', 'local_year', 'issue_time_utc', 'target_mw', 'peak_label', 'final_partition'] + selected['columns']].to_csv(OUT / 'lab_selected_dataset.csv')
joblib.dump(dict(model=selected, name=WINNER, report=report), OUT / 'lab_peak_model.joblib', compress=3)

reloaded = joblib.load(OUT / "lab_peak_model.joblib")
check_day = frame.loc[frame.local_date.eq("2025-07-15")]
assert np.allclose(scores_for(selected, check_day), scores_for(reloaded["model"], check_day))
print("Model save/reload prediction check: PASSED")
print("Saved model:", OUT / "lab_peak_model.joblib")
print("Saved selected dataset:", OUT / "lab_selected_dataset.csv")
print("Saved selected CSV shape:", (len(frame), 7 + len(selected["columns"])))
print("The CSV has one timestamp, six metadata/label fields and the selected input fields.")

# %% [markdown]
# ## 21. Define date-input prediction and a readable output
# Only the date is typed by the user. The notebook automatically retrieves the
# historical features for that date, passes **only selected input columns** to the
# fitted model, chooses its highest-scoring interval, then expands it by one hour
# on each side. Actual target values do not enter the model call.
#
# Supported illustrations are complete 2024/2025 dates. New/future dates need current
# as-of demand and forecast weather; the notebook rejects them rather than inventing
# observations. Missing-target dates `2025-12-04` and `2025-12-05` are unavailable.
# The window stays three elapsed hours through DST; offsets make this explicit.

# %%
def predict_window(date):
    if not isinstance(date, str) or re.fullmatch(r"\d{4}-\d{2}-\d{2}", date.strip()) is None:
        raise ValueError("Enter the date as YYYY-MM-DD, for example 2025-07-15.")
    date = str(pd.Timestamp(date.strip()).date())
    if int(date[:4]) not in [2024, 2025]:
        raise ValueError('Use a supported 2024/2025 historical date. New dates need fresh as-of load and forecast inputs.')
    evaluation = frame.loc[frame.local_date.eq(date)]
    if evaluation.empty:
        raise ValueError('Incomplete historical target day.')
    scores = scores_for(selected, evaluation)
    peak = evaluation.index[int(np.argmax(scores))]
    return dict(date=date, area='ERCOT system', model=WINNER, input_features=len(selected['columns']),
        predicted_peak_start=peak.tz_convert(LOCAL_TZ).isoformat(),
        predicted_peak_end=(peak + pd.Timedelta(hours=1)).tz_convert(LOCAL_TZ).isoformat(),
        window_start=(peak - pd.Timedelta(hours=1)).tz_convert(LOCAL_TZ).isoformat(),
        window_end=(peak + pd.Timedelta(hours=2)).tz_convert(LOCAL_TZ).isoformat(),
        interpretation='Historical three-hour peak-demand window; not a live forecast or calibrated probability')

# %%
def print_prediction(date):
    result = predict_window(date)
    center_start = pd.Timestamp(result["predicted_peak_start"])
    center_end = pd.Timestamp(result["predicted_peak_end"])
    window_start = pd.Timestamp(result["window_start"])
    window_end = pd.Timestamp(result["window_end"])
    print("=" * 58)
    print("ELECTRICITY PEAK-DEMAND WINDOW")
    print("Date:", result["date"])
    print("Area:", result["area"])
    print("Predicted 3-hour window:", window_start.strftime("%d %b %Y, %I:%M %p"),
          "to", window_end.strftime("%d %b %Y, %I:%M %p"))
    print("Central predicted hour:", center_start.strftime("%I:%M %p"), "to", center_end.strftime("%I:%M %p"))
    print("Time zone: America/Chicago; UTC offset", window_start.strftime("%z"))
    print("Model:", result["model"])
    print("Number of selected input features:", result["input_features"])
    print("Historical illustration; coverage is not a guaranteed probability.")
    print("=" * 58)
    return result

example = print_prediction("2025-07-15")
examples = [predict_window(date) for date in ["2024-01-15", "2024-07-15", "2025-01-15", "2025-07-15", "2025-11-02"]]
(OUT / "window_examples.json").write_text(json.dumps(examples, indent=2), encoding="utf-8")

# %% [markdown]
# ## 22. Enter your date here
# When running interactively, type a date and press Enter. Press Enter without a
# date to use the example. If the date is unavailable, read the message and rerun
# this cell with another date; **you do not need to retrain the models**.
#
# Automated verification disables the prompt through an environment variable and
# prints the default example. In a normal Jupyter/Colab session the prompt is enabled.

# %%
DEFAULT_DATE = "2025-07-15"
ASK_FOR_DATE = os.environ.get("ERCOT_LAB_NONINTERACTIVE", "0") != "1"
if ASK_FOR_DATE:
    user_date = input(f"Enter a date YYYY-MM-DD [{DEFAULT_DATE}]: ").strip() or DEFAULT_DATE
else:
    user_date = DEFAULT_DATE
    print("Automated notebook run: using example date", user_date)

try:
    user_prediction = print_prediction(user_date)
except (ValueError, TypeError) as error:
    print("Prediction unavailable:", error)
    print("Rerun this cell with a supported historical date; no retraining is required.")

# %% [markdown]
# ## 23. Conclusion and items added to complete the lab
# The requested steps are all included. Extra checks added for a sound experiment:
# source verification; timestamp/duplicate/range checks; descriptive exploration;
# complete-day and DST validation; target-day leakage intervention; whole-day
# chronological CV with a gap; training-only transformations/selection; baseline
# comparison; fixed selection rule; feature-importance caveats; model save/reload;
# input validation; and the limits of historical versus live predictions.
#
# The most important correction to the usual lab order is: **split first, then fit
# imputers, scalers and selectors on training data only**, repeating this within CV.
# Deterministic calendar and as-of historical feature construction may precede this.
#
# Use the printed tables for your lab observations and conclusion. A smaller model
# does not guarantee higher accuracy. Exact-hour accuracy and three-hour coverage
# measure different things; neither establishes prospective operating performance.
# 2024/2025 remain reused diagnostics, and 2026 is not evaluated. Future operational
# validation needs genuinely new outcomes and verified feed availability.
#
# Sources: [EIA demand and quality notes](https://www.eia.gov/electricity/930-content/about.html),
# [Open-Meteo archived forecasts](https://open-meteo.com/en/docs/previous-runs-api),
# [sklearn temporal CV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit).
# Weather attribution: Open-Meteo and JMA, CC BY 4.0.

# %%
print("LAB SUMMARY")
print("Raw demand shape:", raw.shape)
print("Raw weather shape:", weather_raw.shape)
print("Eligible modeling rows:", len(frame))
print("Candidate inputs:", len(ALL_FEATURES))
print("Selected inputs:", len(selected["columns"]))
print("Selected compact model:", WINNER)
print("Cross-validation:", len(fold_specification), "chronological folds;", len(folds), "comparisons")
print("Final training rows:", len(training))
print("All output files:", OUT)
