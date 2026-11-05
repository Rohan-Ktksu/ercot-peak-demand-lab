import fs from 'node:fs';
import assert from 'node:assert/strict';
import {scenarioRows} from './docs/synthetic.mjs';
import {inferDay} from './docs/inference.mjs';
const model=JSON.parse(fs.readFileSync('docs/data/model.json'));
const profiles=JSON.parse(fs.readFileSync('docs/data/synthetic-profiles.json'));
for(const [date,count] of [['2026-01-15',24],['2026-07-15',24],['2026-03-08',23],['2026-11-01',25],['2030-12-31',24]]){
 const {rows,source}=scenarioRows(date,profiles,model.features);assert.equal(rows.length,count);
 assert(source.startsWith('2023-'));assert(rows.every(r=>r[1]===null&&r[2].length===24));
 const result=inferDay(model,rows);assert(result.scores.every(Number.isFinite));
 console.log(date,rows.length,'hours', 'source',source,'window',new Date(result.windowStart).toISOString(),new Date(result.windowEnd).toISOString());
}
assert.throws(()=>scenarioRows('2026-02-30',profiles,model.features));
