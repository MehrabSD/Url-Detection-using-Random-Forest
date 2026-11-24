// "Try to sneak it past the model" (live red team) and the robustness report.
//
// Like the rest of the page, nothing here is hard-coded: the menu of attacker
// edits and their prices come from /api/state (src/adversary.py), each attack is
// run by the server against the real model, and the report numbers come from
// reports/red_team.json (src/red_team.py).

import { $, el, fill, fmt, getJSON } from './util.js';
import { barTable, splitTable, lineChart } from './charts.js';

const DEFENCE_LABELS = { plain: 'Model as trained', no_query: 'Query-stripped view', multi_view: 'Hardened (multi-view)' };
const LAB_DEFENCES = ['plain', 'multi_view'];
const TOP_SINGLES = 8;

const lab = { defence: 'plain', result: null, busy: false, url: null, error: null, runId: 0 };
let ctx = null;   // { info, mode, url, load } from app.js, refreshed on every render

const prices = () => ctx.info.attack?.prices ?? {};
const priceName = (p) => prices()[String(p)]?.label ?? `Price ${p}`;
const reduceMotion = () => matchMedia('(prefers-reduced-motion: reduce)').matches;

/* ----------------------------------------------------------------- diff */

// Show what an edit changed: shared start, removed text struck through, added text marked.
function diffText(before, after) {
  let a = 0;
  while (a < before.length && a < after.length && before[a] === after[a]) a += 1;
  let b = 0;
  while (b < before.length - a && b < after.length - a
         && before[before.length - 1 - b] === after[after.length - 1 - b]) b += 1;
  const removed = before.slice(a, before.length - b);
  const added = after.slice(a, after.length - b);
  return [after.slice(0, a),
    removed && el('del', { text: removed }),
    added && el('ins', { text: added }),
    after.slice(after.length - b)];
}

/* ---------------------------------------------------------------- meter */

function meter({ score, previous, cut }) {
  const pos = (v) => `${(Math.max(0, Math.min(1, v)) * 100).toFixed(2)}%`;
  const passes = score < cut;
  return el('div', { class: 'meter-row' }, el('div', { class: 'meter', role: 'img',
    'aria-label': `Score ${fmt.score(score)}${previous != null ? `, down from ${fmt.score(previous)}` : ''}; cut-off ${fmt.score(cut)}. ${passes ? 'Passes.' : 'Still flagged.'}` },
    el('span', { class: 'meter-track' }),
    el('span', { class: 'meter-cut', style: { left: pos(cut) } }),
    previous != null && el('span', { class: 'meter-drop', style: {
      left: pos(Math.min(score, previous)), width: `${Math.abs(previous - score) * 100}%` } }),
    previous != null && el('span', { class: 'meter-ghost', style: { left: pos(previous) } }),
    el('span', { class: `meter-dot ${passes ? 'passes' : 'flagged'}`,
      style: { left: pos(score), '--from': pos(previous ?? score) } })),
    el('span', { class: `meter-value ${passes ? 'passes' : 'flagged'}`, text: fmt.score(score) }));
}

/* ------------------------------------------------------------- controls */

function renderControls() {
  const box = $('lab-controls');
  const url = ctx.url();
  const defences = ctx.info.defences ?? [];
  const ready = (id) => defences.find((d) => d.id === id)?.ready;
  if (!ready(lab.defence)) lab.defence = 'plain';

  fill(box,
    el('div', { class: 'lab-target' },
      el('p', { class: 'entry-label', text: 'Link under attack (the one in the check above)' }),
      el('p', { class: 'lab-url', text: url || 'Check a URL above first.' })),
    el('div', { class: 'toolbar' },
      el('fieldset', { class: 'modes' },
        el('legend', { text: 'Defending' }),
        el('div', { class: 'segments' }, LAB_DEFENCES.map((id) => [
          el('input', { type: 'radio', name: 'lab-defence', id: `lab-defence-${id}`, value: id,
            checked: id === lab.defence, disabled: !ready(id),
            onchange: () => { lab.defence = id; lab.result = null; renderLab(ctx); } }),
          el('label', { for: `lab-defence-${id}`, text: DEFENCE_LABELS[id] ?? id,
            title: ready(id) ? null : 'Run src/red_team.py to calibrate it' }),
        ]))),
      el('button', { class: 'button', type: 'button', disabled: !url || lab.busy, onclick: run,
        text: lab.busy ? 'Attacking…' : 'Run the attack' })),
    el('p', { class: 'mode-note', text:
      `${defences.find((d) => d.id === lab.defence)?.description ?? ''} Using the ${ctx.mode().label} cut-off.` }));
}

async function run() {
  const url = ctx.url();
  if (!url) return;
  const id = ++lab.runId;
  lab.busy = true; lab.error = null; lab.url = url;
  renderLab(ctx);
  try {
    const result = await getJSON('/api/attack', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url, mode: ctx.mode().id, defence: lab.defence }),
    });
    if (id !== lab.runId) return;
    lab.result = result;
    lab.busy = false;
    renderLab(ctx, { reveal: true });
  } catch (problem) {
    if (id !== lab.runId) return;
    lab.busy = false; lab.result = null; lab.error = problem.message;
    renderLab(ctx);
  }
}

/* --------------------------------------------------------------- result */

function headline(r) {
  if (r.already_passes) {
    return [el('p', { class: 'lab-verdict passes', text: 'Nothing to attack' }),
      el('p', { class: 'verdict-line', text: `It already scores ${fmt.score(r.score)}, under the cut-off of ${fmt.score(r.threshold)}, so it passes as it is.` })];
  }
  if (!r.found) {
    return [el('p', { class: 'lab-verdict holds', text: 'The detector held' }),
      el('p', { class: 'verdict-line', text: `No combination of edits tried, up to a total price of 8, took the score from ${fmt.score(r.score)} under the cut-off of ${fmt.score(r.threshold)}.` })];
  }
  const n = r.steps.length;
  const counts = {};
  for (const s of r.steps) counts[s.price] = (counts[s.price] ?? 0) + 1;
  const priceText = Object.entries(counts).map(([p, k]) => `${k} ${priceName(p).toLowerCase()} ${k === 1 ? 'edit' : 'edits'}`).join(', ');
  return [el('p', { class: 'lab-verdict gets-through', text: `Gets through after ${n} ${n === 1 ? 'edit' : 'edits'}` }),
    el('p', { class: 'verdict-line' },
      'Score ', el('strong', { text: fmt.score(r.score) }), ' to ', el('strong', { text: fmt.score(r.final_score) }),
      `, under the cut-off of ${fmt.score(r.threshold)}. Price to the attacker: ${priceText}.`)];
}

function pathList(r, reveal) {
  let previous = { url: r.url, score: r.score };
  const rows = [el('li', { class: 'step start' },
    el('div', { class: 'step-head' }, el('b', { text: 'The link as checked' })),
    el('p', { class: 'step-url', text: r.url }),
    meter({ score: r.score, cut: r.threshold }))];
  r.steps.forEach((s, i) => {
    rows.push(el('li', { class: 'step', style: { '--n': String(i + 1) } },
      el('div', { class: 'step-head' },
        el('b', { text: s.label }),
        el('span', { class: `price p${s.price}`, 'data-tip': prices()[String(s.price)]?.description, text: priceName(s.price) })),
      el('p', { class: 'step-url' }, diffText(previous.url, s.url)),
      meter({ score: s.score, previous: previous.score, cut: r.threshold })));
    previous = s;
  });
  return el('ol', { class: `steps${reveal && !reduceMotion() ? ' reveal' : ''}` }, rows);
}

function singlesTable(r) {
  const rows = r.single_moves.slice(0, TOP_SINGLES).map((m) => ({
    label: m.label, note: m.detail ? `${priceName(m.price)}: ${m.detail}` : priceName(m.price),
    amount: m.change, display: fmt.signed(m.change, 2),
  }));
  const scale = Math.max(...r.single_moves.map((m) => Math.abs(m.change)), 0.05);
  return splitTable({ caption: 'How far each single edit moves the score on its own', rows, scale, head: 'Edit' });
}

function viewsBlock(r) {
  if (!r.views) return null;
  const block = (title, views) => {
    const top = Math.max(...views.map((v) => v.score));
    return el('div', {},
      el('p', { class: 'split-title', text: title }),
      el('ul', { class: 'views' }, views.map((v) => el('li', { class: v.score === top ? 'top' : null },
        el('span', { class: 'case-score', text: fmt.score(v.score) }),
        el('span', { class: 'case-url', text: v.text })))));
  };
  return el('div', { class: 'block' },
    el('h3', { class: 'minor', text: 'What the hardened scorer looked at' }),
    el('p', { class: 'explain', text: 'It scores several views of the link and keeps the highest (in bold). Padding can dilute one view, but the view without the padding still shows the payload.' }),
    el('div', { class: 'pair views-pair' },
      block('The link as checked', r.views[0]),
      r.views[1] && block('After the attack', r.views[1])));
}

function renderResult({ reveal = false } = {}) {
  const box = $('lab-result');
  if (lab.error) return fill(box, el('p', { class: 'entry-error', text: lab.error }));
  const r = lab.result;
  if (!r || r.url == null) return fill(box);
  fill(box,
    el('div', { class: 'lab-outcome' }, headline(r)),
    (r.steps.length > 0 || !r.already_passes) && pathList(r, reveal),
    el('p', { class: 'lab-stats', text: `${fmt.int(r.evaluated)} edited versions scored by the real model in ${fmt.int(r.milliseconds)} ms. Every edit changes only the text of the link; nothing is fetched.` }),
    r.single_moves?.length > 0 && el('div', { class: 'block' },
      el('h3', { class: 'minor', text: 'Every first edit, on its own' }),
      el('p', { class: 'explain', text: 'The search starts from these. Bars to the left lower the score; the attacker keeps the strongest and builds on them.' }),
      singlesTable(r)),
    viewsBlock(r));
}

export function renderLab(context, { reveal = false } = {}) {
  ctx = context;
  if (lab.result && lab.url !== ctx.url()) { lab.result = null; lab.error = null; }
  if (lab.result && lab.result.mode !== ctx.mode().id) lab.result = null;
  renderControls();
  renderResult({ reveal });
}

/* ------------------------------------------------------------ report */

export function renderRobustness(context) {
  ctx = context;
  const body = $('robustness-body');
  const report = ctx.info.reports?.red_team;
  const defences = report?.defences;
  if (!defences?.plain?.attack) {
    return fill(body, el('p', { class: 'missing', text: 'No red-team report found. Run src/red_team.py to create reports/red_team.json.' }));
  }
  const ids = Object.keys(defences);
  const plain = defences.plain;
  const best = defences.multi_view ?? defences[ids.at(-1)];
  const free = (d) => d.attack.evaded_within_price['1'];
  const maxPrice = String(report.config.max_price);
  const sample = plain.attack.sample;
  const ext = report.external;

  const series = ids.map((id) => ({
    label: DEFENCE_LABELS[id] ?? id,
    strong: id === 'multi_view',
    dashed: id === 'no_query',
    points: Object.entries(defences[id].attack.evaded_within_price).map(([p, v]) => [Number(p), v]),
  }));

  const lk = report.lookalike;
  // small shares keep enough digits not to round to zero (0.03 %, not 0.0 %)
  const small = (x) => fmt.pct(x, x > 0 && x < 0.001 ? 2 : 1);
  fill(body,
    el('p', { class: 'explain', text:
      `The live attack above was run on ${fmt.int(sample)} phishing URLs drawn at random from ${ext?.dataset ?? 'the external dataset'}. `
      + `With the model as trained, ${fmt.pct(free(plain))} of the ones it caught could be slipped past with a single free edit, and ${fmt.pct(plain.attack.evaded_within_price[maxPrice], 1)} within a total price of ${maxPrice}.` }),
    el('p', { class: 'explain', text:
      'The cause is the model’s strongest input, which averages the evidence of every short run of characters: harmless padding outnumbers the suspicious part. '
      + `Scoring several views of the link and keeping the highest cuts free-edit evasion to ${fmt.pct(free(best))}, with no retraining. Each scorer’s cut-off was re-chosen on the validation sites.` }),
    el('div', { class: 'block', style: { 'margin-top': '2rem' } },
        el('h3', { class: 'minor', text: 'Share of caught phishing the attacker gets through' }),
        el('p', { class: 'explain', text: 'Against the total price the attacker is willing to pay. Lower is better.' }),
        lineChart({ series, xLabel: 'Attacker’s budget (total price)', yLabel: 'Gets through',
          tip: (s, x, y) => `${s.label}: ${fmt.pct(y, 1)} get through within a price of ${x}` })),
    el('div', { class: 'block', style: { 'margin-top': '2.5rem' } },
        el('h3', { class: 'minor', text: 'What each defence costs on clean links, and what it buys' }),
        el('p', { class: 'explain', text: 'Balanced cut-offs. Clean measures are before any attack; the last two rows are the attack on the same sample.' }),
        el('table', { class: 'figures defences' },
          el('caption', { class: 'visually-hidden', text: 'Accuracy of each scorer before any attack' }),
          el('thead', {}, el('tr', {},
            el('th', { scope: 'col' }, el('span', { class: 'visually-hidden', text: 'Measure' })),
            ids.map((id) => el('td', { text: DEFENCE_LABELS[id] ?? id })))),
          el('tbody', {}, [
            ['F1, validation sites', (d) => fmt.pct(d.validation.balanced.f1, 1)],
            ['F1, outside dataset', (d) => fmt.pct(d.external_unseen.balanced.f1, 1)],
            ['False alarms, outside dataset', (d) => fmt.pct(d.external_unseen.balanced.false_positive_rate, 1)],
            ['Caught before attack', (d) => `${d.attack.caught_before_attack} of ${d.attack.sample}`],
            ['Get through, one free edit', (d) => fmt.pct(free(d), 1)],
            [`Get through, price ${maxPrice}`, (d) => fmt.pct(d.attack.evaded_within_price[maxPrice], 1)],
          ].map(([name, f]) => el('tr', {}, el('th', { scope: 'row', text: name }),
            ids.map((id) => el('td', { text: f(defences[id]) }))))))),
    el('div', { class: 'block', style: { 'margin-top': '2.5rem' } },
      el('h3', { class: 'minor', text: 'The edits that hurt the model as trained most' }),
      el('p', { class: 'explain', text: 'Average change in score when each edit is made on its own to a caught phishing URL.' }),
      (() => {
        const changes = Object.entries(plain.attack.mean_single_move_change);
        const moves = Object.fromEntries((report.config.moves ?? []).map((m) => [m.id, m]));
        const widest = Math.max(...changes.map(([, v]) => Math.abs(v)), 0.01);
        return barTable({
          caption: 'Average score change of each single edit',
          head: ['Edit', 'Change', 'Price'],
          rows: changes.slice(0, 8).map(([id, v]) => ({
            label: moves[id]?.label ?? id, note: moves[id]?.description, share: Math.abs(v) / widest,
            display: fmt.signed(v, 2), aside: priceName(moves[id]?.price ?? 1),
          })),
        });
      })()),
    lk && el('div', { class: 'block', style: { 'margin-top': '2.5rem' } },
      el('h3', { class: 'minor', text: 'The lookalike check, measured' }),
      el('p', { class: 'explain', text:
        `On the outside dataset it flags ${small(lk.external_unseen.malicious_flagged)} of phishing URLs and ${small(lk.external_unseen.benign_flagged)} of safe ones. `
        + `Of the ${fmt.int(lk.external_unseen.malicious_missed_by_model)} phishing URLs the model misses there, it catches ${fmt.int(lk.external_unseen.of_those_flagged_by_lookalike)}. `
        + `On the ${fmt.int(lk.validation.rows)} validation URLs it flags ${small(lk.validation.benign_flagged)} of those labelled safe, and reading them shows why: many are phishing the source dataset mislabelled.` }),
      lk.validation.benign_labelled_examples?.length > 0 && el('ol', { class: 'cases' },
        lk.validation.benign_labelled_examples.slice(0, 5).map((u) => el('li', { class: 'case-plain' },
          el('span', { class: 'case-url' }, u, el('small', { text: 'labelled safe in the source data' })),
          el('button', { class: 'link', type: 'button', text: 'Check this URL', onclick: () => ctx.load(u, true) }))))));
}
