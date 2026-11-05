// Numerical tree traversal of the exported, fitted XGBoost model.
// Input values and accumulated margins use float32 to match native inference.
export function scoreRow(model, values) {
  if (values.length !== model.features.length) throw new Error('The model requires 24 selected inputs.');
  const probability = Math.fround(model.base_probability);
  let margin = Math.fround(Math.log(probability / (1 - probability)));
  for (const tree of model.trees) {
    let node = 0;
    while (tree.left_children[node] !== -1) {
      const value = values[tree.split_indices[node]];
      const left = value === null || !Number.isFinite(value)
        ? Boolean(tree.default_left[node])
        : Math.fround(value) < Math.fround(tree.split_conditions[node]);
      node = left ? tree.left_children[node] : tree.right_children[node];
    }
    margin = Math.fround(margin + Math.fround(tree.split_conditions[node]));
  }
  return Math.fround(1 / (1 + Math.exp(-margin)));
}

export function inferDay(model, rows) {
  if (!rows || rows.length < 23 || rows.length > 25) throw new Error('This date has no complete archived inputs.');
  const scores = rows.map(row => scoreRow(model, row[2]));
  let winner = 0;
  for (let i = 1; i < scores.length; i++) if (scores[i] > scores[winner]) winner = i;
  const peakStart = rows[winner][0];
  return {scores, winner, peakStart, peakEnd: peakStart + 3600000,
    windowStart: peakStart - 3600000, windowEnd: peakStart + 7200000};
}
