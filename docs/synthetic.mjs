// Historical analogs become explicitly synthetic inputs when assigned to a future day.
const zone='America/Chicago';
export function civilTime(date,hour=0){
 const target=Date.parse(`${date}T${String(hour).padStart(2,'0')}:00:00Z`);let time=target;
 for(let i=0;i<3;i++){
  const p=Object.fromEntries(new Intl.DateTimeFormat('en-CA',{timeZone:zone,year:'numeric',month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',hourCycle:'h23'}).formatToParts(time).map(p=>[p.type,p.value]));
  time+=target-Date.parse(`${p.year}-${p.month}-${p.day}T${p.hour}:${p.minute}:00Z`);
 }
 return time;
}
export function scenarioRows(date,profiles,features){
 const d=new Date(date+'T12:00:00Z');
 if(!Number.isFinite(+d)||d.toISOString().slice(0,10)!==date)throw Error('Invalid date');
 const profile=profiles[`${d.getUTCMonth()+1}-${(d.getUTCDay()+6)%7}`];
 const next=new Date(+d+86400000).toISOString().slice(0,10),rows=[];
 const doy=Math.floor((Date.UTC(d.getUTCFullYear(),d.getUTCMonth(),d.getUTCDate())-Date.UTC(d.getUTCFullYear(),0,1))/86400000)+1;
 for(let t=civilTime(date);t<civilTime(next);t+=3600000){
  const hour=Number(new Intl.DateTimeFormat('en-GB',{timeZone:zone,hour:'2-digit',hourCycle:'h23'}).format(t));
  const values=[...profile.hours[hour]];
  for(const [name,value] of Object.entries({hour,hour_cos:Math.cos(2*Math.PI*hour/24),hour_sin:Math.sin(2*Math.PI*hour/24),weekday:(d.getUTCDay()+6)%7,day_of_year_cos:Math.cos(2*Math.PI*doy/365.25)})){
   const index=features.indexOf(name);if(index>=0)values[index]=value;
  }
  rows.push([t,null,values]);
 }
 return {rows,source:profile.source_date};
}
