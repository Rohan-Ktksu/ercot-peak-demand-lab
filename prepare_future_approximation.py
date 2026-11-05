"""Build a calendar-only approximation without fitting to diagnostic years."""
import json
from pathlib import Path
import pandas as pd
import numpy as np
ROOT = Path(__file__).resolve().parent
df = pd.read_csv(ROOT / 'data/selected_feature_dataset.csv')
peaks = df.loc[df.groupby('local_date').target_mw.idxmax()].copy()
peaks['hour'] = pd.to_datetime(peaks.timestamp_start_utc, utc=True).dt.tz_convert('America/Chicago').dt.hour
dates = pd.to_datetime(peaks.local_date)
peaks['month'], peaks['weekday'] = dates.dt.month, dates.dt.weekday
train = peaks[peaks.local_date < '2023-12-30']
groups = {}
for month in range(1, 13):
    for weekday in range(7):
        subset = train[(train.month == month) & (train.weekday == weekday)]
        counts = np.bincount(subset.hour, minlength=24)
        hour = max(range(1, 23), key=lambda h: (sum(counts[h-1:h+2]), -abs(h-16)))
        groups[f'{month}-{weekday}'] = dict(hour=hour, examples=len(subset))
diagnostics = []
for year in (2024, 2025):
    subset = peaks[peaks.local_date.str.startswith(str(year))]
    hits = [abs(row.hour-groups[f'{row.month}-{row.weekday}']['hour']) <= 1 for row in subset.itertuples()]
    diagnostics.append(dict(year=year, days=len(subset), coverage=float(np.mean(hits))))
result = dict(groups=groups, diagnostics=diagnostics, training_end='2023-12-29', method='Seasonal weekday historical three-hour peak window; no future observed demand or weather')
assert len(groups) == 84 and all(g['examples'] > 0 for g in groups.values())
(ROOT / 'docs/data/future-approximation.json').write_text(json.dumps(result, separators=(',', ':')), encoding='utf-8')
print(json.dumps(diagnostics, indent=2))
