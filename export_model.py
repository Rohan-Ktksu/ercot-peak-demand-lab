"""Export the fitted ERCOT XGBoost model and original test-date features for JS."""
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import nbformat
from threadpoolctl import threadpool_limits

threadpool_limits(2)
ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'model'
DIST = ROOT / 'docs'
(DIST / 'data').mkdir(parents=True, exist_ok=True)
(DIST / 'downloads').mkdir(exist_ok=True)
artifact = joblib.load(SOURCE / 'fitted_model.joblib')
report = artifact['report']
estimator = artifact['model']['estimator']
booster = estimator.get_booster()
original = json.loads(booster.save_raw(raw_format='json'))['learner']
base = json.loads(original['learner_model_param']['base_score'])[0]
trees = original['gradient_booster']['model']['trees']
assert all(not any(tree['split_type']) for tree in trees)
model = dict(version=1, algorithm='XGBoost Classifier', features=report['selected_features'],
    base_probability=base, trees=[{key: tree[key] for key in
        ['left_children', 'right_children', 'split_conditions', 'split_indices', 'default_left']} for tree in trees])
(DIST / 'data/model.json').write_text(json.dumps(model, separators=(',', ':')), encoding='utf-8')
frame = pd.read_csv(ROOT / 'data/selected_feature_dataset.csv')
frame['timestamp_start_utc'] = pd.to_datetime(frame.timestamp_start_utc, utc=True)
reference = {}
for year in [2024, 2025]:
    test = frame.loc[frame.local_year.eq(year)].copy()
    x = test[model['features']].to_numpy(dtype=np.float32)
    predicted = estimator.predict_proba(test[model['features']])[:, 1]
    test['expected_score'] = predicted
    days = {}
    for date, group in test.groupby('local_date', sort=True):
        matrix = group[model['features']].to_numpy(dtype=np.float32).astype(float)
        rows = [[int(stamp.timestamp() * 1000), round(float(mw), 2),
                 [None if not np.isfinite(value) else float(value) for value in values]]
                for stamp, mw, values in zip(group.timestamp_start_utc, group.target_mw, matrix)]
        days[date] = rows
        reference[date] = dict(scores=group.expected_score.tolist(),
            winner=int(np.argmax(group.expected_score)), actual=int(np.argmax(group.target_mw)))
    (DIST / f'data/days-{year}.json').write_text(json.dumps(days, separators=(',', ':')), encoding='utf-8')
audit = pd.read_csv(SOURCE / 'feature_audit.csv')
selected = audit.loc[audit.selected]
original_report = json.loads((SOURCE / 'original_baseline_results.json').read_text())
summary = dict(owner='Rohan Sai', university='REVA University', timezone='America/Chicago',
    area='ERCOT system', model='XGBoost Classifier', model_key=report['model'],
    selected_features=model['features'], training_rows=report['training_rows'],
    modeling_rows=report['modeling_rows'], candidate_features=report['candidate_features'],
    folds=5, configurations=12, selected_feature_count=len(model['features']),
    cv=report['cv_comparison'], diagnostics=report['reused_year_comparison'],
    original_model_diagnostics=original_report['historical_test_metrics'],
    importance=selected[['feature', 'permutation_log_loss_increase']].to_dict('records'),
    limitations=report['limitations'], plan=report['plan'],
    sources=[dict(label='EIA-930 / ERCOT hourly demand', url='https://www.eia.gov/electricity/930-content/about.html'),
             dict(label='Open-Meteo / JMA archived forecasts', url='https://open-meteo.com/en/docs/previous-runs-api')])
(DIST / 'data/research.json').write_text(json.dumps(summary, separators=(',', ':')), encoding='utf-8')
(ROOT / 'verification-reference.json').write_text(json.dumps(reference, separators=(',', ':')), encoding='utf-8')
shutil.copyfile(ROOT / 'ERCOT_Lab.ipynb', DIST / 'downloads/ERCOT_Lab.ipynb')
book = nbformat.read(DIST / 'downloads/ERCOT_Lab.ipynb', as_version=4)
python_parts = []
for cell in book.cells:
    if cell.cell_type == 'code':
        python_parts.append('# %%\n' + cell.source)
    else:
        python_parts.append('# %% [markdown]\n' + '\n'.join('# ' + line for line in cell.source.splitlines()))
(DIST / 'downloads/ercot_ml_lab.py').write_text('\n'.join(line.rstrip() for line in '\n\n'.join(python_parts).splitlines()) + '\n', encoding='utf-8')
with zipfile.ZipFile(DIST / 'downloads/ERCOT_raw_datasets.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
    for name in ['demand.csv', 'weather.csv']:
        archive.write(ROOT / 'data' / name, name)
manifest = {str(p.relative_to(DIST)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (DIST / 'data').glob('*.json')}
(ROOT / 'export-manifest.json').write_text(json.dumps(manifest, indent=2))
print(json.dumps(dict(days=len(reference), features=len(model['features']), trees=len(trees),
    train_rows=report['training_rows'], files=list(manifest)), indent=2))
