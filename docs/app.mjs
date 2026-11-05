import {inferDay} from './inference.mjs';
import {scenarioRows} from './synthetic.mjs';

const $ = id => document.getElementById(id);
const zone = 'America/Chicago';
const hourFormat = new Intl.DateTimeFormat('en-US',{timeZone:zone,hour:'numeric',minute:'2-digit'});
const shortHour = new Intl.DateTimeFormat('en-US',{timeZone:zone,hour:'numeric'});
const dateFormat = new Intl.DateTimeFormat('en-US',{timeZone:zone,month:'long',day:'numeric',year:'numeric'});
const zoneFormat = new Intl.DateTimeFormat('en-US',{timeZone:zone,timeZoneName:'short'});
const labels = {calendar_mode:'Calendar mode',previous_week_peak:'Previous-week peak',hgb_full_reference:'HGB · full reference',hgb_12:'HGB · 12 inputs',hgb_24:'HGB · 24 inputs',hgb_40:'HGB · 40 inputs',logistic_24:'Logistic Regression',random_forest_24:'Random Forest',extra_trees_24:'Extra Trees',xgb_classifier_24:'XGBoost Classifier',xgb_ranker_24:'XGBoost Ranker',hgb_regressor_24:'HGB Regressor'};
let model, research, current, chartView='ranking', requestSequence=0;
const years = new Map();
const percent = value => (value*100).toFixed(2)+'%';
const escape = value => String(value).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

async function getJSON(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error('The archived model or data could not be loaded. Please refresh to retry.');
  return response.json();
}
async function loadYear(year) {
  if (!years.has(year)) years.set(year,getJSON(`data/days-${year}.json`).catch(error=>{years.delete(year);throw error;}));
  return years.get(year);
}
function setBusy(busy) {
  $('run-button').disabled=busy||!model;
  $('run-button').innerHTML=busy?'Running model…':'Run prediction <span aria-hidden="true">↗</span>';
  document.querySelectorAll('.example').forEach(button=>{button.disabled=busy||!model;});
  $('result').setAttribute('aria-busy',String(busy));
}
function showError(message) { $('error').hidden=false;$('error').textContent=message; }
function clearError() { $('error').hidden=true;$('error').textContent=''; }
function zoneName(time) {return zoneFormat.formatToParts(time).find(part=>part.type==='timeZoneName')?.value ?? 'Central Time';}

async function predict(date, updateURL=true) {
  const sequence = ++requestSequence;
  clearError();
  if (/^20(2[6-9]|30)-\d{2}-\d{2}$/.test(date)) { await approximateFuture(date, sequence, updateURL); return; }
  if (!/^202[45]-\d{2}-\d{2}$/.test(date)) {
    showError('Choose 2024–2025 for historical prediction, or 2026–2030 for a historical-pattern approximation.');return;
  }
  setBusy(true);$('loading').hidden=false;$('loading').textContent='Reading archived inputs and running the 300 fitted trees…';
  try {
    const data = await loadYear(Number(date.slice(0,4)));
    if (sequence!==requestSequence) return;
    const rows=data[date];
    if(!rows)throw Error('This date has no complete historical inputs. December 4–5, 2025 are unavailable.');
    const prediction=inferDay(model,rows);
    current={date,rows,features:model.features,modelName:research.model,...prediction};
    document.querySelector('.result-note').textContent='Fitted XGBoost Classifier · 24 selected inputs. Historical illustration; scores are not calibrated probabilities.';
    document.querySelector('.input-details').hidden=false;
    $('date').value=date;
    $('result-date').textContent=dateFormat.format(prediction.peakStart).toUpperCase()+' / PEAK-DEMAND WINDOW';
    $('window-title').replaceChildren(document.createTextNode(hourFormat.format(prediction.windowStart)+' – '+hourFormat.format(prediction.windowEnd)));
    const arrow=document.createElement('span');arrow.className='result-arrow';arrow.setAttribute('aria-hidden','true');arrow.textContent='↗';$('window-title').append(arrow);
    $('center-hour').textContent=hourFormat.format(prediction.peakStart)+' – '+hourFormat.format(prediction.peakEnd);
    $('zone').textContent=zoneName(prediction.peakStart)+' / CENTRAL TIME';
    const metrics=research.diagnostics.find(row=>row.year===Number(date.slice(0,4))&&row.model===research.model_key);
    document.querySelector('.coverage-card>p').textContent='Historical coverage of this fitted XGBoost model, not a confidence percentage for this date.';
    $('coverage-label').textContent='PYTHON REFERENCE · '+date.slice(0,4);
    $('coverage').textContent=(metrics.within_one_hour_accuracy*100).toFixed(1)+'%';
    $('exact').textContent=percent(metrics.exact_peak_hour_accuracy);
    $('download-prediction').disabled=false;
    renderChart();renderInputs();
    if(updateURL){const url=new URL(location.href);url.searchParams.set('date',date);history.replaceState(null,'',url);}
  } catch(error) {if(sequence===requestSequence)showError(error.message);}
  finally {if(sequence===requestSequence){setBusy(false);$('loading').hidden=true;}}
}

function svgElement(name, attributes={}, text='') {
  const element=document.createElementNS('http://www.w3.org/2000/svg',name);
  Object.entries(attributes).forEach(([key,value])=>element.setAttribute(key,String(value)));
  if(text)element.textContent=text;
  return element;
}
function renderChart() {
  if(!current)return;
  if(current.modelName?.startsWith('SYNTHETIC')&&chartView==='demand'){ $('chart-note').textContent='Future actual demand is unknown. No demand curve is fabricated.';return;}
  const {rows,scores,winner,windowStart,windowEnd}=current;
  const demand=chartView==='demand';
  const values=demand?rows.map(row=>row[1]):scores;
  const width=760,height=232,left=45,right=8,top=22,bottom=35,plotHeight=height-top-bottom;
  const max=Math.max(...values)*(demand?1.12:1.1);
  const plotWidth=width-left-right,step=plotWidth/rows.length;
  const svg=svgElement('svg',{viewBox:`0 0 ${width} ${height}`,role:'img','aria-label':`${demand?'Recorded demand':'Relative model ranking'} by hour for ${current.date}, Texas Central Time`});
  svg.append(svgElement('title',{},`${current.date}: ${rows.length} hourly intervals; gold marks the predicted peak hour`));
  for(let tick=0;tick<=4;tick++){
    const y=top+plotHeight-tick*plotHeight/4;
    svg.append(svgElement('line',{x1:left,y1:y,x2:width-right,y2:y,stroke:'#e9ebdf','stroke-dasharray':'3 4'}));
    const label=demand?Math.round(max*tick/4/1000)+'k':tick===0?'Low':tick===4?'High':'';
    svg.append(svgElement('text',{x:left-8,y:y+3,'text-anchor':'end',fill:'#8a9280','font-size':9,'font-family':'DM Mono,monospace'},label));
  }
  rows.forEach((row,index)=>{
    const inWindow=row[0]>=windowStart&&row[0]<windowEnd;
    const color=index===winner?'#d2a64f':inWindow?'#a6b994':'#dfe4d5';
    const barHeight=Math.max(3,values[index]/max*plotHeight);
    const bar=svgElement('rect',{x:left+index*step+3,y:top+plotHeight-barHeight,width:Math.max(2,step-6),height:barHeight,rx:3,fill:color,tabindex:0});
    const description=`${hourFormat.format(row[0])} ${zoneName(row[0])}: ${demand?Math.round(row[1]).toLocaleString('en-US')+' MW':'uncalibrated model score '+scores[index].toFixed(5)}${index===winner?'; predicted peak hour':''}`;
    bar.setAttribute('aria-label',description);bar.append(svgElement('title',{},description));svg.append(bar);
    if(index%3===0||index===rows.length-1)svg.append(svgElement('text',{x:left+(index+.5)*step,y:height-12,'text-anchor':'middle',fill:'#818c75','font-size':9,'font-family':'DM Mono,monospace'},shortHour.format(row[0]).replace(' ','').toLowerCase()));
  });
  if(demand)svg.append(svgElement('text',{x:left-8,y:10,'text-anchor':'end',fill:'#8a9280','font-size':8},'MW'));
  $('chart').replaceChildren(svg);
  $('chart-note').textContent=demand?'Recorded demand is shown for retrospective context only. It does not enter the model call.':`Relative model ranking across ${rows.length} hourly intervals. Scores rank hours within a day; they are not calibrated probabilities.`;
}
function renderInputs() {
  const headings=['Hour / Central Time',...(current.features??model.features)];
  const header='<thead><tr>'+headings.map(name=>'<th scope="col">'+escape(name)+'</th>').join('')+'</tr></thead>';
  const body=current.rows.map((row,index)=>'<tr'+(index===current.winner?' class="chosen-row"':'')+'><td>'+escape(hourFormat.format(row[0])+' '+zoneName(row[0]))+'</td>'+row[2].map(value=>'<td>'+(value===null?'Missing':Number(value).toLocaleString('en-US',{maximumFractionDigits:3}))+'</td>').join('')+'</tr>').join('');
  $('input-table').innerHTML='<table><caption class="sr-only">Selected model input values, not target demand</caption>'+header+'<tbody>'+body+'</tbody></table>';
}
function renderComparison() {
  if(!research)return;
  const period=$('metric-period').value;
  const rows=period==='cv'?research.cv:research.diagnostics.filter(row=>row.year===Number(period));
  const body=rows.map(row=>'<tr'+(row.model===research.model_key?' class="chosen-row"':'')+'><td>'+escape(labels[row.model]??row.model)+(row.model===research.model_key?'<span class="chosen-tag">CV CHOICE</span>':'')+'</td><td>'+row.feature_count+'</td><td>'+percent(period==='cv'?row.three_hour_window_coverage:row.within_one_hour_accuracy)+'</td><td>'+percent(row.exact_peak_hour_accuracy)+'</td></tr>').join('');
  $('comparison-table').querySelector('tbody').innerHTML=body;
  $('comparison-caption').textContent=period==='cv'?'Five chronological folds, 2019–2023. Select the smallest compact model within 1 percentage point of the best compact window coverage.':`Previously inspected ${period} dates, reused for diagnostics. The model choice and feature budget remain frozen from cross-validation.`;
}

function saveFile(name,text,type='text/csv;charset=utf-8') {
  const url=URL.createObjectURL(new Blob([text],{type}));
  const anchor=document.createElement('a');anchor.href=url;anchor.download=name;document.body.append(anchor);anchor.click();anchor.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
function downloadPrediction() {
  if(!current)return;
  const columns=['date','area','timezone','model','features','predicted_peak_start_utc','predicted_peak_end_utc','window_start_utc','window_end_utc','window_start_central','window_end_central','scope'];
  const values=[current.date,'ERCOT system',zone,current.modelName,24,new Date(current.peakStart).toISOString(),new Date(current.peakEnd).toISOString(),new Date(current.windowStart).toISOString(),new Date(current.windowEnd).toISOString(),hourFormat.format(current.windowStart)+' '+zoneName(current.windowStart),hourFormat.format(current.windowEnd)+' '+zoneName(current.windowEnd),'Historical illustration; not a live forecast or calibrated probability'];
  const quote=value=>'"'+String(value).replace(/"/g,'""')+'"';
  saveFile(`ERCOT-window-${current.date}.csv`,columns.join(',')+'\n'+values.map(quote).join(',')+'\n');
}

async function evaluateArchive() {
  const button=$('evaluate'),progress=$('evaluation-progress');button.disabled=true;progress.hidden=false;$('evaluation-results').hidden=true;
  try {
    const results=[];let completed=0;
    for(const year of [2024,2025]){
      const data=await loadYear(year);let exact=0,covered=0,error=0;const entries=Object.entries(data);
      for(const [date,rows] of entries){
        const result=inferDay(model,rows);let actualIndex=0;
        for(let i=1;i<rows.length;i++)if(rows[i][1]>rows[actualIndex][1])actualIndex=i;
        const difference=Math.abs(result.peakStart-rows[actualIndex][0])/3600000;
        exact+=difference===0;covered+=difference<=1;error+=difference;completed++;
        if(completed%25===0){progress.textContent=`Evaluating historical dates: ${completed} / 729…`;await new Promise(resolve=>requestAnimationFrame(resolve));}
      }
      const count=entries.length;const recorded=research.diagnostics.find(row=>row.model===research.model_key&&row.year===year);
      const agrees=Math.abs(covered/count-recorded.within_one_hour_accuracy)<1e-12&&Math.abs(exact/count-recorded.exact_peak_hour_accuracy)<1e-12;
      if(!agrees)throw new Error('The browser evaluation differs from the exported research record. Please report this result.');
      results.push({year,days:count,exact:exact/count,coverage:covered/count,timingMAE:error/count});
    }
    progress.textContent='Evaluation complete · 729 dates · 17,496 hourly intervals';
    $('evaluation-results').innerHTML='<table><thead><tr><th scope="col">Year</th><th scope="col">Days</th><th scope="col">3h coverage</th><th scope="col">Exact hour</th><th scope="col">Timing MAE</th></tr></thead><tbody>'+results.map(row=>`<tr><td>${row.year}</td><td>${row.days}</td><td>${percent(row.coverage)}</td><td>${percent(row.exact)}</td><td>${row.timingMAE.toFixed(2)} hours</td></tr>`).join('')+'</tbody></table><p class="pass">✓ Browser results match the frozen Python research results. These remain reused historical diagnostics.</p><button class="button outline" type="button" id="save-evaluation">Download evaluation JSON ↓</button>';
    $('evaluation-results').hidden=false;
    $('save-evaluation').addEventListener('click',()=>saveFile('ERCOT-browser-evaluation.json',JSON.stringify({model:research.model_key,scope:'Reused historical diagnostics; not prospective accuracy',results},null,2),'application/json'));
  }catch(error){progress.textContent='Evaluation could not complete: '+error.message;}
  finally{button.disabled=false;}
}

$('date').addEventListener('change',()=>setBusy(false));
$('prediction-form').noValidate=true;
$('prediction-form').addEventListener('submit',event=>{event.preventDefault();predict($('date').value);});
document.querySelectorAll('.example').forEach(button=>button.addEventListener('click',()=>predict(button.dataset.date)));
document.querySelectorAll('[data-view]').forEach(button=>button.addEventListener('click',()=>{
  chartView=button.dataset.view;document.querySelectorAll('[data-view]').forEach(other=>{other.classList.toggle('active',other===button);other.setAttribute('aria-pressed',String(other===button));});renderChart();
}));
$('metric-period').addEventListener('change',renderComparison);
$('download-prediction').addEventListener('click',downloadPrediction);
$('evaluate').addEventListener('click',evaluateArchive);

async function initialize() {
  try {
    let summary;
    [model,research,summary]=await Promise.all([getJSON('data/model.json'),getJSON('data/research.json'),getJSON('data/pipeline-summary.json')]);
    renderPipeline(summary);
    $('model-state').textContent='Ready';$('evaluate').disabled=false;
    $('feature-list').replaceChildren(...model.features.map(name=>{const node=document.createElement('code');node.textContent=name;return node;}));
    renderComparison();setBusy(false);
    const requested=new URL(location.href).searchParams.get('date')??$('date').value;
    await predict(requested,false);
  }catch(error){showError(error.message);$('loading').hidden=true;$('model-state').textContent='Unavailable';$('run-button').textContent='Refresh to retry';}
}
initialize();

async function approximateFuture(date, sequence, updateURL=true){
 setBusy(true);
 try{
  const parsed=new Date(date+'T12:00:00Z');if(Number.isNaN(parsed.valueOf())||parsed.toISOString().slice(0,10)!==date)throw Error('Enter a valid calendar date.');
  const data=await getJSON('data/future-approximation.json');if(sequence!==requestSequence)return;const month=parsed.getUTCMonth()+1,weekday=(parsed.getUTCDay()+6)%7;
  const group=data.groups[month+'-'+weekday];
  $('date').value=date;
  if(updateURL){const url=new URL(location.href);url.searchParams.set('date',date);history.replaceState(null,'',url);}
  if($('synthetic-mode').checked){await simulateFuture(date,sequence);return;}
  // Resolve the chosen local civil hour via Intl, including future DST offsets.
  let peak=Date.parse(date+'T'+String(group.hour).padStart(2,'0')+':00:00Z');
  for(let i=0;i<2;i++){const parts=new Intl.DateTimeFormat('en-CA',{timeZone:zone,year:'numeric',month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',hourCycle:'h23'}).formatToParts(peak);const v=Object.fromEntries(parts.map(p=>[p.type,p.value]));const represented=Date.parse(`${v.year}-${v.month}-${v.day}T${v.hour}:${v.minute}:00Z`);peak+=Date.parse(date+'T'+String(group.hour).padStart(2,'0')+':00:00Z')-represented;}
  current=null;$('download-prediction').disabled=true;
  $('result-date').textContent=date+' / FUTURE APPROXIMATION';$('zone').textContent=zoneName(peak)+' / CENTRAL TIME';
  $('window-title').textContent=hourFormat.format(peak-3600000)+' – '+hourFormat.format(peak+7200000);
  $('center-hour').textContent='Historical-pattern center: '+hourFormat.format(peak);
  document.querySelector('.result-note').textContent=`APPROXIMATION ONLY. Based on ${group.examples} historical days with the same month and weekday, trained through ${data.training_end}. No future actual demand, recent load or forecast weather is supplied. This may be substantially wrong.`;
  $('chart').textContent='Calendar-based approximation: no hourly demand curve is assumed.';$('input-table').replaceChildren();document.querySelector('.input-details').hidden=true;$('chart-note').textContent='No hourly future demand curve is generated. Historical-pattern estimate only.';
  $('coverage-label').textContent='PATTERN BASELINE · REUSED DIAGNOSTICS';
  const n=data.diagnostics.reduce((a,r)=>a+r.days,0);const rate=data.diagnostics.reduce((a,r)=>a+r.coverage*r.days,0)/n;
  $('coverage').textContent=(100*rate).toFixed(1)+'%';$('exact').textContent='Not estimated for this date';
  document.querySelector('.coverage-card>p').textContent='Historical three-hour coverage of this separate seasonal/weekday approximation on reused 2024–2025 days. This is not the probability or guaranteed accuracy of your future prediction.';
  $('loading').hidden=true;$('error').hidden=true;$('result').setAttribute('aria-busy','false');
 }catch(error){if(sequence===requestSequence)showError(error.message);}
 finally{if(sequence===requestSequence)setBusy(false);}
}

async function simulateFuture(date,sequence){
 const profiles=await getJSON('data/synthetic-profiles.json');
 if(sequence!==requestSequence)return;
 const {rows,source}=scenarioRows(date,profiles,model.features);
 const result=inferDay(model,rows);
 current={date,rows,features:model.features,modelName:'SYNTHETIC SCENARIO / full reference XGBoost',...result};
 $('result-date').textContent=date+' / APPROXIMATE PEAK-DEMAND WINDOW';
 $('window-title').textContent=hourFormat.format(result.windowStart)+' – '+hourFormat.format(result.windowEnd);
 $('center-hour').textContent='Approximate peak hour: '+hourFormat.format(result.peakStart);
 $('zone').textContent=zoneName(result.peakStart)+' / CENTRAL TIME';
 document.querySelector('.result-note').textContent='APPROXIMATION using synthetic inputs based on a 2023 historical day ('+source+'), including previous-day, 2/7/14-day load features and weather. Calendar features are updated for your date. Actual peak timing may differ as demand and weather change. Future accuracy has not been measured.';
 $('coverage-label').textContent='APPROXIMATION ACCURACY';$('coverage').textContent='Not yet measured';$('exact').textContent='Not available';
 document.querySelector('.coverage-card>p').textContent='This approximation uses synthetic historical-pattern inputs. Its accuracy on future dates has not been measured. These estimates are excluded from training and testing metrics.';
 document.querySelector('.input-details').hidden=false;chartView='ranking';renderChart();renderInputs();
 $('chart-note').textContent='Synthetic-input model ranking, not a real demand curve or calibrated probability. The demand view is unavailable for synthetic scenarios.';
 $('download-prediction').disabled=true;$('loading').hidden=true;
}

function renderPipeline(summary){
 const n=value=>Number(value).toLocaleString('en-US');
 $('pipeline-profile').textContent=`Demand: ${n(summary.raw_demand_rows)} rows × ${summary.raw_demand_columns} columns. Weather: ${n(summary.raw_weather_rows)} city-hour rows × ${summary.raw_weather_columns} columns. Weather rows are not extra independent demand observations.`;
 $('pipeline-cleaning').textContent=`Duplicate demand timestamps: ${summary.duplicate_demand_timestamps}; duplicate weather keys: ${summary.duplicate_weather_keys}. Eligible hourly modeling rows: ${n(summary.modeling_rows)}. Missing target dates December 4–5, 2025 excluded. Missing predictor cells: ${n(summary.missing_candidate_inputs)} (preserved for model-specific handling).`;
 $('pipeline-engineering').textContent=`Output: ${summary.candidate_features} candidate features. Target: the earliest maximum-demand interval of each complete local day. Actual target demand is excluded from all predictor inputs.`;
 $('pipeline-splits').textContent=`Final training through Dec 29, 2023: ${n(summary.training_rows)} rows. Two-day gap: ${summary.gap_rows} rows. Reused diagnostics: 2024 = ${n(summary.test_2024_rows)} rows; 2025 = ${n(summary.test_2025_rows)} rows. Cross-validation validates 2019–2023 before the final refit.`;
 $('pipeline-selection').textContent=`Output: ${summary.candidate_features} → ${summary.selected_features} selected inputs (${((1-summary.selected_features/summary.candidate_features)*100).toFixed(1)}% fewer). Selected-input missing cells across all modeling years: ${n(summary.missing_selected_inputs)}. No refitting on diagnostic or future rows.`;
 $('pipeline-features').replaceChildren(...research.selected_features.map(name=>{const el=document.createElement('code');el.textContent=name;return el;}));
 const cv=research.cv.find(r=>r.model===research.model_key);
 $('pipeline-winner').textContent=`Selected: ${research.model} · ${model.features.length} inputs · ${model.trees.length} trees. Five-fold CV: ${percent(cv.three_hour_window_coverage)} three-hour coverage, ${percent(cv.exact_peak_hour_accuracy)} exact-hour accuracy. ${research.configurations} configurations × ${research.folds} folds = ${research.configurations*research.folds} recorded comparisons.`;
 $('pipeline-cv').innerHTML='<table><caption>Recorded five-fold cross-validation outputs</caption><thead><tr><th>Model / configuration</th><th>Inputs</th><th>3h coverage</th><th>Exact hour</th><th>Timing MAE</th></tr></thead><tbody>'+research.cv.map(r=>`<tr${r.model===research.model_key?' class="chosen-row"':''}><td>${escape(labels[r.model]??r.model)}${r.model===research.model_key?' · SELECTED':''}</td><td>${r.feature_count}</td><td>${percent(r.three_hour_window_coverage)}</td><td>${percent(r.exact_peak_hour_accuracy)}</td><td>${r.peak_time_mae_hours.toFixed(2)} h</td></tr>`).join('')+'</tbody></table>';
 $('pipeline-tests').innerHTML='<table><caption>Selected XGBoost model · reused historical diagnostics</caption><thead><tr><th>Year</th><th>Days</th><th>3h coverage</th><th>Exact hour</th><th>Timing MAE</th></tr></thead><tbody>'+research.diagnostics.filter(r=>r.model===research.model_key).map(r=>`<tr><td>${r.year}</td><td>${r.days}</td><td>${percent(r.within_one_hour_accuracy)}</td><td>${percent(r.exact_peak_hour_accuracy)}</td><td>${r.peak_time_mae_hours.toFixed(2)} h</td></tr>`).join('')+'</tbody></table>';
}
