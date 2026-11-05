"""Real browser-side fitting, chronological validation, frozen testing and inference."""
import json,hashlib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier,HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
try:
    from js import report_progress
except ImportError:
    def report_progress(fraction,text):pass
state={}
def metrics(frame,scores):
    work=frame[['local_date','timestamp_start_utc','target_mw']].copy();work['score']=scores
    exact=covered=error=0
    for _,day in work.groupby('local_date',sort=True):
        actual=pd.Timestamp(day.loc[day.target_mw.idxmax(),'timestamp_start_utc'])
        predicted=pd.Timestamp(day.loc[day.score.idxmax(),'timestamp_start_utc'])
        delta=abs((actual-predicted).total_seconds())/3600
        exact+=delta==0;covered+=delta<=1;error+=delta
    n=work.local_date.nunique()
    return dict(days=int(n),exact=float(exact/n),coverage=float(covered/n),timing_mae=float(error/n))
def train():
    manifest=json.load(open('browser_lab.json'))
    assert hashlib.sha256(open('browser_lab.csv.gz','rb').read()).hexdigest()==manifest['sha256']
    data=pd.read_csv('browser_lab.csv.gz');features=manifest['features']
    tr=data.loc[data.partition.eq('train')];va=data.loc[data.partition.eq('validation')]
    assert tr.local_date.max()<va.local_date.min()
    report_progress(.15,'Checking data and selecting features on training rows only')
    imputer=SimpleImputer(strategy='median',keep_empty_features=True)
    x=imputer.fit_transform(tr[features])
    selector=ExtraTreesClassifier(n_estimators=40,max_depth=12,min_samples_leaf=8,n_jobs=1,random_state=42)
    selector.fit(x,tr.peak_label)
    order=np.argsort(-selector.feature_importances_,kind='stable');corr=tr[features].corr().abs();selected=[]
    for idx in order:
        column=features[idx]
        if tr[column].nunique(dropna=True)<=1:continue
        if any(corr.loc[column,previous]>.98 for previous in selected):continue
        selected.append(column)
        if len(selected)==24:break
    assert len(selected)==24
    candidates={
      'Logistic Regression':make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),LogisticRegression(max_iter=1500,class_weight='balanced',random_state=42)),
      'Random Forest':make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),RandomForestClassifier(n_estimators=80,max_depth=12,min_samples_leaf=8,n_jobs=1,random_state=42)),
      'Extra Trees':make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),ExtraTreesClassifier(n_estimators=80,max_depth=14,min_samples_leaf=5,n_jobs=1,random_state=42)),
      'Histogram Gradient Boosting':HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=15,min_samples_leaf=30,early_stopping=False,random_state=42)}
    rows=[]
    for i,(name,model) in enumerate(candidates.items()):
        report_progress(.25+i*.17,'Training '+name+' and evaluating 2022–2023 validation days')
        model.fit(tr[selected],tr.peak_label)
        rows.append(dict(model=name,**metrics(va,model.predict_proba(va[selected])[:,1])))
    ranked=sorted(rows,key=lambda r:(-r['coverage'],-r['exact'],r['timing_mae']))
    winner=ranked[0]['model']
    state.update(data=data,features=selected,model=candidates[winner],winner=winner,models=candidates,tested=False)
    return json.dumps(dict(winner=winner,features=selected,comparison=ranked,rows=manifest['rows'],days=manifest['days'],missing_training=int(tr[features].isna().sum().sum()),training_end=tr.local_date.max(),validation_start=va.local_date.min(),validation_end=va.local_date.max()))
def test():
    if not state:raise ValueError('Train and compare first.')
    if state['tested']:raise ValueError('Testing already completed. The winner remains fixed.')
    data=state['data'];rows=[]
    for year in [2024,2025]:
        part=data.loc[data.local_year.eq(year)]
        for name,model in state['models'].items():
            rows.append(dict(year=year,model=name,**metrics(part,model.predict_proba(part[state['features']])[:,1])))
    state['tested']=True
    return json.dumps(dict(winner=state['winner'],comparison=rows,note='All models compared on reused historical diagnostics; winner remains frozen from validation.'))
def predict(date):
    if not state.get('tested'):raise ValueError('Train, compare and test first.')
    part=state['data'].loc[state['data'].local_date.eq(date)&state['data'].partition.eq('test')]
    if part.empty:raise ValueError('Choose a complete 2024/2025 historical date; December 4–5, 2025 are unavailable.')
    scores=state['model'].predict_proba(part[state['features']])[:,1]
    values=part[state['features']].to_numpy(dtype=float)
    rows=[[int(pd.Timestamp(stamp).timestamp()*1000),float(mw),[None if not np.isfinite(v) else float(v) for v in row]] for stamp,mw,row in zip(part.timestamp_start_utc,part.target_mw,values)]
    return json.dumps(dict(scores=scores.tolist(),rows=rows,winner=int(np.argmax(scores)),features=state['features'],model=state['winner']))
