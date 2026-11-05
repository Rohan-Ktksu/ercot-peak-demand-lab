let runtime,busy=false;
self.report_progress=(fraction,text)=>self.postMessage({kind:'progress',fraction,text});
async function init(){
 if(runtime)return;
 report_progress(.01,'Downloading Python and scientific libraries; first load may take a minute');
 const url='https://cdn.jsdelivr.net/pyodide/v314.0.7/full/';
 const {loadPyodide}=await import(url+'pyodide.mjs');runtime=await loadPyodide({indexURL:url,stdout:()=>{},stderr:()=>{}});
 await runtime.loadPackage(['numpy','pandas','scikit-learn']);
 for(const [target,path] of [['browser_lab.py','browser_lab.py'],['browser_lab.json','data/browser_lab.json'],['browser_lab.csv.gz','data/browser_lab.csv.gz']]){
  const r=await fetch(new URL(path,self.location.href));if(!r.ok)throw Error('Unable to fetch '+path);runtime.FS.writeFile(target,new Uint8Array(await r.arrayBuffer()));
 }
 await runtime.runPythonAsync('import browser_lab');
}
self.onmessage=async({data})=>{
 if(busy)return;busy=true;
 try{
  await init();let code;
  if(data.action==='train')code='browser_lab.train()';
  else if(data.action==='test')code='browser_lab.test()';
  else if(data.action==='predict'){runtime.globals.set('requested_date',data.date);code='browser_lab.predict(requested_date)';}
  else throw Error('Unknown action');
  self.postMessage({kind:'result',action:data.action,result:JSON.parse(await runtime.runPythonAsync(code))});
 }catch(error){self.postMessage({kind:'error',action:data.action,text:String(error)});}finally{busy=false;}
};
