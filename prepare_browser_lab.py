"""Build a fixed whole-day browser sample; no labels affect sampling."""
from pathlib import Path
import hashlib,json
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parent
source=ROOT.parent/'ercot_demand_project/outputs/selected_feature_dataset.csv'
df=pd.read_csv(source)
metadata={'timestamp_start_utc','local_date','local_year','issue_time_utc','target_mw','peak_label'}
features=[c for c in df if c not in metadata]
assert len(features)==146
rng=np.random.default_rng(42)
parts=[]
for name,mask,count in [('train',df.local_year.le(2021),600),('validation',df.local_year.between(2022,2023),240),('test',df.local_year.between(2024,2025),None)]:
    part=df.loc[mask].copy(); days=np.array(sorted(part.local_date.unique()))
    chosen=days if count is None else rng.choice(days,size=count,replace=False)
    part=part.loc[part.local_date.isin(chosen)].sort_values('timestamp_start_utc')
    part['partition']=name;parts.append(part)
data=pd.concat(parts)
path=ROOT/'docs/data/browser_lab.csv.gz'
data.to_csv(path,index=False,compression={'method':'gzip','mtime':0})
manifest=dict(features=features,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),seed=42,
    rows=data.groupby('partition').size().to_dict(),days=data.groupby('partition').local_date.nunique().to_dict(),
    note='Whole-day sample: fit through 2021, validation 2022–2023, reused diagnostics 2024–2025. Different from the full research experiment.')
(ROOT/'docs/data/browser_lab.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps({k:v for k,v in manifest.items() if k!='features'},indent=2))
