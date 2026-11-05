const $=id=>document.getElementById(id);
let worker,pending,trained=false;
window.ercotLab={tested:false,features:[],predict:date=>request('predict',{date})};
function table(rows){
 const host=document.createElement('table');
 const head=host.createTHead().insertRow();['Model','Year / days','3h coverage','Exact hour','Timing MAE'].forEach(s=>{const th=document.createElement('th');th.textContent=s;head.append(th);});
 for(const row of rows){const tr=host.createTBody().insertRow();[row.model,(row.year??'Validation')+' / '+row.days,(row.coverage*100).toFixed(2)+'%',(row.exact*100).toFixed(2)+'%',row.timing_mae.toFixed(2)+'h'].forEach(v=>{tr.insertCell().textContent=v;});}
 return host;
}
function request(action,input={}){
 if(pending)return Promise.reject(Error('Please wait for the current operation.'));
 if(!worker){worker=new Worker('lab-worker.mjs',{type:'module'});worker.onmessage=({data})=>{
  if(data.kind==='progress'){$('lab-status').textContent=data.text;$('lab-progress').value=data.fraction;return;}
  const task=pending;pending=null;
  if(data.kind==='error'){task?.reject(Error(data.text));return;}
  task?.resolve(data.result);
 };worker.onerror=e=>{const task=pending;pending=null;task?.reject(Error('Browser Python worker failed: '+e.message));};}
 return new Promise((resolve,reject)=>{pending={resolve,reject};worker.postMessage({action,...input});});
}
$('lab-train').addEventListener('click',async()=>{
 $('lab-train').disabled=true;$('lab-progress').hidden=false;
 try{const result=await request('train');trained=true;window.ercotLab.features=result.features;
  $('lab-validation').replaceChildren(table(result.comparison));$('lab-validation').hidden=false;
  $('lab-selected').textContent='Validation-selected model: '+result.winner+' · 146 → '+result.features.length+' features. Choice frozen before test.';
  $('lab-status').textContent='Training and validation complete. Now test all four models.';$('lab-progress').value=1;$('lab-test').disabled=false;
 }catch(e){$('lab-status').textContent=e.message;$('lab-train').disabled=false;}
});
$('lab-test').addEventListener('click',async()=>{
 if(!trained)return;$('lab-test').disabled=true;$('lab-status').textContent='Testing the four fitted models on all 729 complete dates…';
 try{const result=await request('test');$('lab-test-results').replaceChildren(table(result.comparison));$('lab-test-results').hidden=false;
  window.ercotLab.tested=true;window.ercotLab.model=result.winner;
  $('lab-status').textContent='Testing complete. Predict is unlocked using the validation-selected browser model. Test scores do not change the winner.';
  window.dispatchEvent(new Event('ercot-lab-tested'));
 }catch(e){$('lab-status').textContent=e.message;$('lab-test').disabled=false;}
});
fetch('data/browser_lab.json').then(r=>{if(!r.ok)throw Error('Dataset manifest unavailable');return r.json();}).then(data=>{
 $('lab-data-summary').textContent=`Browser experiment: ${data.rows.train.toLocaleString()} fit rows / ${data.days.train} days through 2021; ${data.rows.validation.toLocaleString()} validation rows / ${data.days.validation} days in 2022–2023; ${data.rows.test.toLocaleString()} reused test rows / 729 days in 2024–2025. Fixed whole-day sample, seed 42.`;
}).catch(e=>{$('lab-status').textContent=e.message;});
