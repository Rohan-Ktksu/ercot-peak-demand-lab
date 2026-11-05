import fs from 'node:fs';
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import {inferDay} from './docs/inference.mjs';
const read = name => JSON.parse(fs.readFileSync(name,'utf8'));
const model = read('docs/data/model.json');
const research = read('docs/data/research.json');
const reference = read('verification-reference.json');
let dates = 0, hours = 0, maxError = 0;
for (const year of [2024,2025]) {
  const data = read(`docs/data/days-${year}.json`);
  let exact = 0, covered = 0;
  for (const [date,rows] of Object.entries(data)) {
    const result = inferDay(model,rows);
    assert.equal(result.winner, reference[date].winner, `Native peak mismatch on ${date}`);
    for (let i=0;i<rows.length;i++) maxError = Math.max(maxError, Math.abs(result.scores[i] - reference[date].scores[i]));
    const actual = rows[reference[date].actual][0];
    exact += actual === result.peakStart;
    covered += Math.abs(actual-result.peakStart) <= 3600000;
    dates++;hours += rows.length;
  }
  const count = Object.keys(data).length;
  const recorded = research.diagnostics.find(row=>row.year===year && row.model==='xgb_classifier_24');
  assert.ok(Math.abs(exact/count-recorded.exact_peak_hour_accuracy)<1e-12);
  assert.ok(Math.abs(covered/count-recorded.within_one_hour_accuracy)<1e-12);
}
assert.ok(maxError<0.00001,`Score conversion error ${maxError}`);
const data = read('docs/data/days-2025.json');
assert.equal(data['2025-03-09'].length,23);
assert.equal(data['2025-11-02'].length,25);
assert.equal(data['2025-12-04'],undefined);
assert.equal(data['2025-12-05'],undefined);
assert.equal(inferDay(model,data['2025-07-15']).peakStart,Date.parse('2025-07-15T16:00:00-05:00'));
assert.throws(()=>inferDay(model,undefined));
assert.equal(model.features.length,24);
assert.ok(!model.features.includes('target_mw'));
for(const [name,sha] of Object.entries(read('export-manifest.json'))) assert.equal(crypto.createHash('sha256').update(fs.readFileSync('docs/'+name)).digest('hex'),sha);
const result = {status:'passed',dates,hours,features:24,trees:300,max_native_score_error:maxError,all_daily_peaks_match_native:true,metrics_match_research:true};
fs.writeFileSync('verification-results.json',JSON.stringify(result,null,2));
console.log(JSON.stringify(result,null,2));
