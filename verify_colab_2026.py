"""Verify the Colab 2026 extension against the saved model, without retraining."""
from pathlib import Path
import ast
from contextlib import redirect_stdout
from functools import lru_cache
import hashlib
import io
import json
import os
import re
import sys
import tempfile
import joblib
import nbformat
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

ROOT = Path(__file__).resolve().parent
os.environ['ERCOT_LAB_NONINTERACTIVE'] = '1'
notebook = nbformat.read(ROOT / 'ERCOT_Lab.ipynb', as_version=4)
nbformat.validate(notebook)
for cell in notebook.cells:
    if cell.cell_type == 'code':
        compile(cell.source, '<notebook cell>', 'exec')

# Notebook helpers are defined here without running the training pipeline.
for node in ast.parse((ROOT / 'ercot_ml_lab.py').read_text(encoding='utf-8')).body:
    if isinstance(node, ast.FunctionDef) and node.name in {
        'holiday_dates', 'calendar_features', 'scores_for',
        'predict_window', 'print_prediction'
    }:
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<notebook functions>', 'exec'))

def display(value):
    pass

with redirect_stdout(io.StringIO()):
    for cell in notebook.cells:
        if cell.cell_type == 'code' and (
            'required_csvs =' in cell.source or 'def sha256(path):' in cell.source
        ):
            exec(cell.source)
    assert DATA == ROOT / 'data'
    loaded = joblib.load(ROOT / 'model/fitted_model.joblib')
    selected, WINNER = loaded['model'], loaded['name']
    ALL_FEATURES = json.loads((ROOT / 'docs/data/browser_lab.json').read_text())['features']
    frame = pd.read_csv(ROOT / 'data/selected_feature_dataset.csv', index_col='timestamp_start_utc', parse_dates=True)
    for cell in notebook.cells:
        if cell.cell_type == 'code' and (
            'synthetic_2026 = pd.read_csv' in cell.source or 'def predict_2026_approximation' in cell.source
        ):
            exec(cell.source)
    for date, hours in [('2026-01-15', 24), ('2026-07-15', 24), ('2026-03-08', 23), ('2026-11-01', 25)]:
        assert len(synthetic_2026[synthetic_2026.local_date.eq(date)]) == hours
        result = predict_date(date)
        assert result['future_accuracy'] == 'Not yet measured'
        assert (pd.Timestamp(result['window_end']) - pd.Timestamp(result['window_start'])).total_seconds() == 10800
    assert predict_date('2025-07-15')['date'] == '2025-07-15'
    for date in ['2026-02-30', '2027-01-01', 'invalid']:
        try:
            predict_date(date)
        except ValueError:
            pass
        else:
            raise AssertionError('Accepted invalid date: ' + date)

scores = np.asarray(scores_for(selected, synthetic_2026))
assert len(scores) == 8760 and np.isfinite(scores).all()
with tempfile.TemporaryDirectory() as temp:
    lf = (DATA / 'demand.csv').read_bytes().replace(b'\r\n', b'\n')
    path = Path(temp) / 'line_endings.csv'
    path.write_bytes(lf)
    assert sha256(path) == source['csv_sha256']
    path.write_bytes(lf.replace(b'\n', b'\r\n'))
    assert sha256(path) == source['csv_sha256']
    path.write_bytes(lf + b'changed')
    assert sha256(path) != source['csv_sha256']
assert (ROOT / 'ERCOT_Lab.ipynb').read_bytes() == (ROOT / 'docs/downloads/ERCOT_Lab.ipynb').read_bytes()
assert (ROOT / 'ercot_ml_lab.py').read_bytes() == (ROOT / 'docs/downloads/ercot_ml_lab.py').read_bytes()
print(json.dumps(dict(status='passed', synthetic_rows=len(synthetic_2026),
    synthetic_features=len(ALL_FEATURES), model_features=len(selected['columns']),
    notebook_schema_and_syntax=True, historical_and_synthetic_routing=True,
    dst_grids=True, finite_inference=True, cross_platform_source_hashes=True,
    changed_source_rejected=True, website_downloads_match=True), indent=2))
