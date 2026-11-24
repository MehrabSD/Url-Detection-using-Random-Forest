// Demo page for the malicious-URL model.
//
// Nothing about the model is written into this file. The feature list, the
// decision settings, every number and every example come from the server
// (/api/state), which reads them from the saved model and the JSON reports.
// Retrain the model and the page shows the new one; sections whose data is
// missing from a report are left out with a note.

import { $, el, fill, fmt, getJSON, startTips } from './util.js';
import { barTable, splitTable, voteChart, halvesChart, tradeoffChart } from './charts.js';
import { renderLab, renderRobustness } from './lab.js';
import { setupInbox, renderInbox } from './inbox.js';
import { setupQr } from './qr.js';

const TOP_MOVES = 8;        // features listed before "show all"
const TOP_FEATURES = 10;
const TOP_CASES = 6;
const CHECK_EVERY_MS = 4000;

const view = {
  info: null,        // /api/state
  mode: null,        // chosen decision setting
  result: null,      // last /api/score answer
  metric: 'f1',
  allMoves: false,
  allFeatures: false,
  allCases: false,
};

const PART_NAMES = { userinfo: 'User name', host: 'Host', port: 'Port', path: 'Path', query: 'Query', fragment: 'Fragment' };

// Metrics the reports carry, in the words the page uses for them.
const METRICS = [
  ['accuracy', 'Accuracy', 'URLs it got right'],
  ['precision', 'Precision', 'flagged URLs that really were malicious'],
  ['recall', 'Recall', 'malicious URLs it caught'],
  ['false_positive_rate', 'False alarms', 'safe URLs it flagged'],
  ['f1', 'F1', 'one figure balancing precision and recall'],
  ['roc_auc', 'ROC-AUC', 'ranking quality over every cut-off, from 0.5 for guessing to 1'],
];

const report = (name) => view.info.reports?.[name] ?? null;
const featureInfo = (name) => view.info.features.find((f) => f.name === name)
  ?? { name, label: name, kind: 'number', description: '', group: 'other' };
// Reports key their results by threshold name (metrics.json) or by mode name (external_eval.json).
const forMode = (table, mode) => table?.[mode.key] ?? table?.[mode.id] ?? null;

// What the lab, inbox and QR modules may read or do.
const ctx = {
  get info() { return view.info; },
  mode: () => view.mode,
  url: () => view.result?.input ?? null,
  load: (url, scroll = false) => {
    load(url);
    if (scroll) $('check').scrollIntoView({ behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth' });
  },
};

function renderModeDependents() {
  renderReading();
  renderResults();
  renderInbox(ctx);
}

/* ------------------------------------------------------------ controls */

function modeControl(name) {
  return el('fieldset', { class: 'modes' },
    el('legend', { text: 'Decision setting' }),
    el('div', { class: 'segments' }, view.info.modes.map((m) => [
      el('input', {
        type: 'radio', name, id: `${name}-${m.id}`, value: m.id, checked: m.id === view.mode.id,
        onchange: () => { view.mode = m; renderModeDependents(); },
      }),
      el('label', { for: `${name}-${m.id}`, text: m.label }),
    ])));
}

function showAll(open, total, noun, toggle) {
  return el('button', {
    class: 'more', type: 'button', 'aria-expanded': String(open), onclick: toggle,
    text: open ? `Show fewer ${noun}` : `Show all ${total} ${noun}`,
  });
}

/* -------------------------------------------------------- live reading */

function renderSpecimen(result, reveal) {
  const { normalized, parts, site_span: site, heat } = result;
  const kindAt = new Array(normalized.length).fill(null);
  for (const p of parts) for (let i = p.start; i < p.end; i += 1) kindAt[i] = p.kind;

  const box = $('specimen');
  const cells = [...normalized].map((ch, i) => {
    const s = heat ? heat.strength[i] : 0;
    const lean = Math.abs(s) < 0.12 ? 'neutral' : `${Math.abs(s) > 0.5 ? 'strongly ' : ''}${s > 0 ? 'malicious' : 'benign'}`;
    const classes = ['cell'];
    if (!kindAt[i]) classes.push('sep');
    if (i > 0 && kindAt[i] !== kindAt[i - 1] && kindAt[i - 1]) classes.push('gap');
    if (site && i >= site.start && i < site.end) classes.push('site');
    return el('span', { class: classes.join(' '), 'data-tip': heat ? `“${ch}” sits in a stretch that reads ${lean}` : null },
      el('span', { class: 'glyph', text: ch }),
      heat && el('span', { class: 'strip' },
        el('i', { class: s >= 0 ? 'up' : 'down', style: { '--s': Math.abs(s).toFixed(3), '--i': String(i) } })));
  });
  const n = normalized.length;
  box.className = `specimen${n > 100 ? ' longest' : n > 76 ? ' longer' : n > 52 ? ' long' : ''}`;
  box.setAttribute('aria-label', `The address as the model reads it: ${normalized}`);
  fill(box, cells);
  if (reveal) { void box.offsetWidth; box.classList.add('reveal'); }

  const notes = [];
  if (heat) {
    const top = report('metrics')?.feature_importance?.[0]?.feature === view.info.heat_feature;
    notes.push(
      el('i', { class: 'key malicious' }), 'Bars rise where a stretch of characters resembles malicious URLs from training. ',
      el('i', { class: 'key benign' }), 'They dip where it resembles safe ones. ',
      top ? `This is the model's strongest single input; the model weighs it with ${view.info.model.features - 1} others.`
          : `This is one of the model's ${view.info.model.features} inputs.`);
  }
  if (result.normalized !== result.input) {
    notes.push(' The scheme and a leading www. are removed before scoring, because the model ignores them.');
  }
  fill($('specimen-note'), notes);

  const text = (p) => normalized.slice(p.start, p.end);
  const host = parts.find((p) => p.kind === 'host');
  const rows = [];
  if (result.site) rows.push(['Site it belongs to (underlined)', result.site]);
  for (const p of parts) {
    if (p.kind === 'host' && host && text(host) === result.site) continue;
    rows.push([PART_NAMES[p.kind] ?? p.kind, text(p)]);
  }
  fill($('parts'), rows.map(([name, value]) => el('div', {}, el('dt', { text: name }), el('dd', { text: value }))));
}

// The rule-based lookalike check (src/lookalike.py). It is shown beside the
// model's verdict and never changes the score.
const OP_TEXT = { change: 'changed', extra: 'added', missing: 'left out', swap: 'swapped' };

function alignment(compare) {
  const cells = compare.ops.map(([op, seen, real]) => el('span', { class: `pair-cell ${op}`,
    'data-tip': op === 'same' ? null : `${OP_TEXT[op]}: “${seen || '–'}” where the real name has “${real || '–'}”` },
    el('span', { class: 'seen', text: seen || '·' }),
    el('span', { class: 'real', text: real || '·' })));
  return el('div', { class: 'align', role: 'img',
    'aria-label': `This address spells ${compare.seen}; the real name is ${compare.real}.` },
    el('span', { class: 'align-labels' }, el('span', { text: 'This link' }), el('span', { text: 'Real name' })),
    el('span', { class: 'align-cells' }, cells));
}

function renderLookalike(result) {
  const box = $('lookalike');
  const look = result.lookalike;
  if (!look) return fill(box);
  const [main, ...more] = look.findings;
  const shown = look.display_host !== look.host;
  if (!main) {
    return fill(box, el('p', { class: 'look-clear' }, el('i', { class: 'key benign' }),
      'Lookalike check: this address does not imitate any of the well-known brands it knows.'));
  }
  fill(box,
    el('div', { class: `look-hit ${main.severity}` },
      el('h2', { class: 'look-title' }, `Imitates ${main.official}`),
      el('p', { class: 'look-kind', text: `${main.label}. ${main.detail}` }),
      main.compare && alignment(main.compare),
      shown && el('p', { class: 'explain' }, 'A browser shows this host as ',
        el('b', { class: 'mono', text: look.display_host }),
        look.scripts.length > 1 ? `, which mixes ${look.scripts.join(' and ')} letters.` : '.'),
      more.length > 0 && el('p', { class: 'explain', text: `Also: ${more.map((f) => `${f.label.toLowerCase()} (${f.brand})`).join('; ')}.` }),
      el('p', { class: 'mode-note', text: 'This is a separate, rule-based check. It does not change the model’s score.' })));
}

function renderVerdict(result) {
  const mode = view.mode;
  const flagged = result.score >= mode.threshold;
  const tested = forMode(report('metrics')?.test, mode);
  fill($('verdict'),
    el('p', { class: 'verdict-word' },
      el('span', { class: flagged ? 'verdict-flag malicious' : 'verdict-flag' }),
      flagged ? 'Malicious' : 'Benign'),
    el('p', { class: 'verdict-line' },
      'Score ', el('strong', { text: fmt.score(result.score) }),
      flagged ? `, at or above the ${mode.label} cut-off of ${fmt.score(mode.threshold)}, so it is flagged.`
              : `, below the ${mode.label} cut-off of ${fmt.score(mode.threshold)}, so it passes.`),
    !flagged && result.lookalike?.findings?.some((f) => f.severity === 'high') && el('p', { class: 'verdict-warn', text:
      'The model passes it, but the lookalike check found a brand imitation. Treat it as suspicious.' }),
    modeControl('mode-live'),
    el('p', { class: 'mode-note' }, mode.description, tested
      ? ` On the test sites it caught ${fmt.pct(tested.recall)} of malicious URLs and wrongly flagged ${fmt.pctAuto(tested.false_positive_rate)} of safe ones.`
      : null));
}

function renderVotes(result) {
  const votes = result.tree_votes;
  const leaning = votes.filter((v) => v >= 0.5).length;
  const halves = result.halves;
  if (!halves) {   // a model saved before the booster was added
    $('votes-title').textContent = `How the ${votes.length} trees voted`;
    return fill($('votes'),
      el('p', { class: 'explain', text: `${leaning} of ${votes.length} trees lean malicious for this URL. The score is the average of all of them.` }),
      voteChart({ votes, modes: view.info.modes, chosen: view.mode, score: result.score }));
  }
  $('votes-title').textContent = 'Two models, one score';
  fill($('votes'),
    el('p', { class: 'explain', text:
      'A random forest and a gradient-boosted model each give a probability, and the score is their average. '
      + `The forest's ${halves.trees} trees are grown independently and vote; the booster's ${halves.rounds} small trees are grown one after another, each correcting the ones before it.` }),
    halvesChart({ halves, score: result.score, mode: view.mode }),
    el('h3', { class: 'minor votes-sub', text: `How the forest's ${votes.length} trees voted` }),
    el('p', { class: 'explain', text: `${leaning} of ${votes.length} lean malicious for this URL. The forest's half of the score is their average.` }),
    voteChart({ votes, modes: [], chosen: view.mode, score: halves.forest }));
}

function renderMoves(result) {
  const all = result.contributions;
  const shown = view.allMoves ? all : all.slice(0, TOP_MOVES);
  const rest = all.slice(shown.length);
  const scale = Math.max(...all.map((c) => Math.abs(c.contribution)), 0.01);

  const rows = shown.map((c) => {
    const info = featureInfo(c.name);
    return {
      label: info.label,
      note: `This URL: ${fmt.feature(info.kind, c.value)}.${info.description ? ` ${info.description}` : ''}`,
      amount: c.contribution, display: fmt.signed(c.contribution),
    };
  });
  if (rest.length) {
    const sum = rest.reduce((total, c) => total + c.contribution, 0);
    rows.push({ label: `The other ${rest.length} features together`, amount: sum, display: fmt.signed(sum), summary: true });
  }
  $('why-intro').textContent = result.halves
    ? `Every URL starts at ${fmt.score(result.baseline)}, the two models' average starting point before they read anything. `
      + `Each feature then moves the score up or down through the splits of both models, and the ${all.length} changes add up to exactly ${fmt.score(result.score)}.`
    : `Every URL starts at ${fmt.score(result.baseline)}, the average share of malicious URLs the trees were trained on. `
      + `Each feature then moves the score up or down, and the ${all.length} changes add up to exactly ${fmt.score(result.score)}.`;
  fill($('why'),
    splitTable({ caption: 'How much each feature moved the score for this URL', rows, scale }),
    all.length > TOP_MOVES && showAll(view.allMoves, all.length, 'features',
      () => { view.allMoves = !view.allMoves; renderMoves(view.result); }));
}

function renderReading({ reveal = false } = {}) {
  const result = view.result;
  $('reading').hidden = !result;
  $('waiting').hidden = Boolean(result);
  renderLab(ctx);   // the lab attacks whatever is checked here
  if (!result) return;
  renderSpecimen(result, reveal);
  renderLookalike(result);
  renderVerdict(result);
  renderVotes(result);
  renderMoves(result);
}

/* ------------------------------------------------------------- checking */

let pending = null;

async function check(url, { reveal = false } = {}) {
  pending?.abort();
  const error = $('entry-error');
  if (!url.trim()) {
    view.result = null;
    error.hidden = true;
    $('waiting').textContent = 'Results appear here as you type.';
    return renderReading();
  }
  pending = new AbortController();
  try {
    view.result = await getJSON('/api/score', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url }), signal: pending.signal,
    });
    error.hidden = true;
    renderReading({ reveal });
  } catch (problem) {
    if (problem.name === 'AbortError') return;
    if (!problem.status) return serverLost();
    view.result = null;
    // While typing, a half-finished address is normal: say so quietly. A press of
    // the button (or an example) is a real request, so the problem is announced.
    error.textContent = reveal ? problem.message : '';
    error.hidden = !reveal;
    $('waiting').textContent = reveal ? '' : problem.message;
    renderReading();
  }
}

function load(url) {
  $('url').value = url;
  check(url, { reveal: true });
}

/* --------------------------------------------------------------- report */

function splitGroup(title, total, parts) {
  return el('div', { class: 'split-group' },
    el('p', { class: 'split-title' }, `${title} `, el('small', { text: `${fmt.int(total)} in the test` })),
    el('div', { class: 'split-bar' }, parts.map((p) => el('span', {
      class: p.said, style: { flex: `${p.count} 1 0` },
      'data-tip': `${p.name}: ${fmt.int(p.count)} (${fmt.pctAuto(p.count / total)})`,
    }))),
    el('div', { class: 'split-key' }, parts.map((p) => el('span', {},
      el('i', { class: `key ${p.said === 'said-malicious' ? 'malicious' : 'benign'}` }),
      `${p.name} ${fmt.int(p.count)} (${fmt.pctAuto(p.count / total)})`))));
}

function renderResults() {
  const metrics = report('metrics');
  const body = $('results-body');
  const test = forMode(metrics?.test, view.mode);
  if (!test) return fill(body, el('p', { class: 'missing', text: 'No test report found. Run train.py to create reports/metrics.json.' }));

  const size = metrics.split?.test;
  const external = report('external');
  const outside = forMode(external?.results?.unseen_sites, view.mode);
  const curves = metrics.test.curves;
  const points = view.info.modes
    .map((m) => ({ m, t: forMode(metrics.test, m) }))
    .filter((p) => p.t)
    .map(({ m, t }) => ({ id: m.id, label: m.label, fpr: t.false_positive_rate, tpr: t.recall }));

  fill(body,
    el('p', { class: 'explain', text: size
      ? `Tested once on ${fmt.int(size.rows)} URLs from ${fmt.int(size.sites)} sites that were kept out of training, so every one was new to the model.`
      : 'Tested once on sites that were kept out of training.' }),
    el('div', { class: 'toolbar' }, modeControl('mode-report')),
    el('div', { class: 'pair' },
      el('div', {},
        el('div', { class: 'block' },
          splitGroup('Malicious URLs', test.tp + test.fn, [
            { name: 'Caught', count: test.tp, said: 'said-malicious' },
            { name: 'Missed', count: test.fn, said: 'said-benign' }]),
          splitGroup('Safe URLs', test.tn + test.fp, [
            { name: 'Passed', count: test.tn, said: 'said-benign' },
            { name: 'Wrongly flagged', count: test.fp, said: 'said-malicious' }])),
        el('div', { class: 'block' },
          el('table', { class: 'figures' },
            el('caption', { class: 'visually-hidden', text: `Results with the ${view.mode.label} setting` }),
            outside && el('thead', {}, el('tr', {},
              el('th', { scope: 'col' }, el('span', { class: 'visually-hidden', text: 'Measure' })),
              el('td', { text: 'Test sites' }), el('td', { text: 'Outside dataset' }))),
            el('tbody', {}, METRICS.filter(([key]) => key in test).map(([key, name, meaning]) => el('tr', {},
              el('th', { scope: 'row' }, name, el('small', { text: meaning })),
              el('td', { text: key === 'roc_auc' ? test[key].toFixed(3) : fmt.pct(test[key], 1) }),
              outside && el('td', { text: key in outside ? (key === 'roc_auc' ? outside[key].toFixed(3) : fmt.pct(outside[key], 1)) : '' }))))),
          outside && el('p', { class: 'mode-note', text:
            `The outside dataset is ${external.dataset?.name ?? 'a separate collection'}: ${fmt.int(outside.n)} URLs from sites that never appeared in training, scored by the same unchanged model.` }))),
      curves?.fpr && el('div', { class: 'block' },
        el('h3', { class: 'minor', text: 'The trade-off behind each setting' }),
        el('p', { class: 'explain', text: 'Moving the cut-off catches more malicious URLs only by flagging more safe ones. Each dot is one setting.' }),
        tradeoffChart({ fpr: curves.fpr, tpr: curves.tpr, points, chosenId: view.mode.id }))));
}

function renderCompare() {
  const metrics = report('metrics');
  const body = $('compare-body');
  const models = Object.entries(metrics?.comparison ?? {});
  if (!models.length) return fill(body, el('p', { class: 'missing', text: 'No comparison found in reports/metrics.json.' }));

  const choices = METRICS.filter(([key]) => models.every(([, m]) => key in m));
  if (!choices.some(([key]) => key === view.metric)) view.metric = choices[0][0];
  const [key, name] = choices.find(([k]) => k === view.metric);
  const lowerIsBetter = key === 'false_positive_rate';
  const show = (v) => (key === 'roc_auc' ? v.toFixed(3) : fmt.pct(v, 1));
  const final = (models.find(([label]) => /final/i.test(label)) ?? models.at(-1))[0];

  const leak = metrics.leakage;
  const pairs = leak && [['Random split, the usual method', leak.random_split], ['Unseen sites, used on this page', leak.site_disjoint_split]]
    .filter(([, m]) => m && key in m);
  const quick = metrics.config?.quick_rf?.n_estimators;

  fill(body,
    el('p', { class: 'explain', text: 'Every model here was trained on the same URLs and scored on the same unseen test sites.' }),
    el('div', { class: 'toolbar' }, el('div', { class: 'field' },
      el('label', { for: 'metric', text: 'Measure' }),
      el('select', { id: 'metric', onchange: (e) => { view.metric = e.target.value; renderCompare(); } },
        choices.map(([k, label]) => el('option', { value: k, selected: k === key, text: label }))))),
    barTable({
      caption: `${name} of each model on the test sites`,
      head: ['Model', lowerIsBetter ? `${name} (lower is better)` : name],
      rows: models.map(([label, m]) => ({ label, share: m[key], display: show(m[key]), strong: label === final })),
    }),
    pairs?.length === 2 && el('div', { class: 'block', style: { 'margin-top': '3rem' } },
      el('h3', { class: 'minor', text: 'Why the test uses unseen sites' }),
      el('p', { class: 'explain', text:
        `With an ordinary random split, ${fmt.pct(leak.random_split.test_urls_whose_site_is_in_train)} of test URLs came from a site the model had already met in training. `
        + 'It can then recognise the site instead of the pattern, which flatters every score.'
        + (quick ? ` Measured in a separate experiment with a quicker ${quick}-tree forest.` : '') }),
      barTable({
        caption: `${name} under each way of splitting the data`,
        head: ['Split', lowerIsBetter ? `${name} (lower is better)` : name],
        rows: pairs.map(([label, m], i) => ({ label, share: m[key], display: show(m[key]), strong: i === 1 })),
      })));
}

function caseList(cases) {
  const shown = view.allCases ? cases : cases.slice(0, TOP_CASES);
  return el('ol', { class: 'cases' }, shown.map((c) => el('li', {},
    el('span', { class: 'case-score', text: fmt.score(c.score) }),
    el('span', { class: 'case-url' }, c.url,
      el('small', { text: [c.source && `${c.source} dataset`, c.kind].filter(Boolean).join(', ') })),
    el('button', { class: 'link', type: 'button', text: 'Check this URL', onclick: () => ctx.load(c.url, true) }))));
}

function renderMistakes() {
  const test = report('metrics')?.test;
  const body = $('mistakes-body');
  const kinds = Object.entries(test?.detection_rate_by_kind ?? {});
  const errors = test?.confident_errors;
  if (!kinds.length && !errors) return fill(body, el('p', { class: 'missing', text: 'No error analysis found in reports/metrics.json.' }));

  const balanced = view.info.modes.find((m) => m.id === 'balanced');
  const longest = Math.max(errors?.false_negatives?.length ?? 0, errors?.false_positives?.length ?? 0);
  fill(body,
    el('p', { class: 'explain', text: 'It sees only the text of an address, so it cannot tell a hacked site with an ordinary URL from the clean original, and real sign-in pages can look like the phishing that copies them.' }),
    kinds.length > 0 && el('div', { class: 'block', style: { 'margin-top': '2rem' } },
      el('h3', { class: 'minor', text: 'Some kinds of attack are caught far more often than others' }),
      balanced && el('p', { class: 'explain', text: `Share of malicious test URLs flagged with the ${balanced.label} setting, by kind and source dataset.` }),
      barTable({
        caption: 'Share of malicious test URLs caught, by kind of attack',
        head: ['Kind of attack', 'Caught', 'Test URLs'],
        rows: kinds.sort(([, a], [, b]) => b.detected - a.detected).map(([label, k]) => {
          const [source, kind] = label.split(' / ');
          return { label: kind ?? label, note: kind ? `${source} dataset` : null, share: k.detected,
                   display: fmt.pct(k.detected), aside: fmt.int(k.n) };
        }),
      })),
    errors && el('div', { class: 'pair' },
      errors.false_negatives?.length > 0 && el('div', {},
        el('h3', { class: 'minor', text: 'Malicious URLs it was sure were safe' }),
        caseList(errors.false_negatives)),
      errors.false_positives?.length > 0 && el('div', {},
        el('h3', { class: 'minor', text: 'Safe URLs it was sure were malicious' }),
        caseList(errors.false_positives))),
    longest > TOP_CASES && showAll(view.allCases, longest, 'examples in each list',
      () => { view.allCases = !view.allCases; renderMistakes(); }));
}

function renderFeatures() {
  const ranked = report('metrics')?.feature_importance ?? [];
  const body = $('features-body');
  if (!ranked.length) return fill(body, el('p', { class: 'missing', text: 'No feature importance found in reports/metrics.json.' }));

  const shuffled = 'permutation' in ranked[0];
  const key = shuffled ? 'permutation' : 'impurity';
  const sorted = [...ranked].sort((a, b) => b[key] - a[key]);
  const widest = Math.max(sorted[0][key], 1e-9);
  const shown = view.allFeatures ? sorted : sorted.slice(0, TOP_FEATURES);

  fill(body,
    el('p', { class: 'explain', text: `The model uses ${sorted.length} features, all worked out from the text of the URL. `
      + (shuffled ? "A feature's importance is how far the test score falls when its values are shuffled, so it stops carrying information."
                  : "A feature's importance is its share of the splits' total effect inside the trees.") }),
    barTable({
      caption: 'Importance of each feature',
      head: ['Feature', 'Importance', 'Comes from'],
      rows: shown.map((f) => {
        const info = featureInfo(f.feature);
        return { label: info.label, note: info.description, share: f[key] / widest,
                 display: f[key].toFixed(3), aside: view.info.groups[info.group] ?? info.group };
      }),
    }),
    sorted.length > TOP_FEATURES && showAll(view.allFeatures, sorted.length, 'features',
      () => { view.allFeatures = !view.allFeatures; renderFeatures(); }));
}

function renderFrame() {
  const { model, examples } = view.info;
  $('facts').textContent = model.boosting_rounds
    ? `${model.trees}-tree forest + ${model.boosting_rounds}-round booster, ${model.features} features`
    : `${model.trees} trees, ${model.features} features`;

  fill($('examples'), 'Try one: ',
    examples.map((x) => el('button', { class: 'link', type: 'button', text: x.label, onclick: () => load(x.url) })));

  const prepared = report('prepare');
  const sources = Object.keys(prepared?.raw_rows_by_source ?? {});
  fill($('foot'),
    prepared?.final_rows && el('p', { text:
      `Trained and tested on ${fmt.int(prepared.final_rows)} URLs from ${sources.length} public datasets (${sources.join(', ')}), split so that no site appears in both training and test.` }),
    el('p', { text: 'The score ranks URLs by how suspicious they look. It is not a calibrated probability, and a URL scored here is never opened.' }));
}

function renderAll() {
  renderFrame();
  renderReading();
  renderResults();
  renderCompare();
  renderMistakes();
  renderFeatures();
  renderRobustness(ctx);
  renderInbox(ctx);
}

/* ------------------------------------------------ staying in step with the model */

function notice(text, problem = false) {
  const box = $('notice');
  box.hidden = !text;
  box.textContent = text ?? '';
  box.classList.toggle('problem', problem);
}

let lost = false;
function serverLost() {
  lost = true;
  const local = ['localhost', '127.0.0.1', '[::1]'].includes(location.hostname);
  notice(local
    ? 'Cannot reach the scoring server. Start it again with: .venv\\Scripts\\python.exe src\\serve.py'
    : 'Cannot reach the scoring server. It may be restarting; this page reconnects by itself.', true);
}

async function loadState() {
  const info = await getJSON('/api/state');
  const keep = view.mode && info.modes.find((m) => m.id === view.mode.id);
  view.info = info;
  view.mode = keep ?? info.modes.find((m) => m.id === info.default_mode) ?? info.modes[0];
}

async function watch() {
  if (document.hidden || !view.info) return;
  try {
    const { version } = await getJSON('/api/version');
    if (lost) { lost = false; notice(null); }
    if (version === view.info.version) return;
    await loadState();
    renderAll();
    if ($('url').value.trim()) await check($('url').value);
    notice(`The model or its reports changed at ${fmt.time()}. This page now shows the new version.`);
  } catch {
    serverLost();
  }
}

async function start() {
  startTips($('tip'));
  setupQr({ area: $('check'), onText: (text) => ctx.load(text) });
  const input = $('url');
  let timer;
  input.addEventListener('input', () => {
    clearTimeout(timer);
    timer = setTimeout(() => check(input.value), 160);
  });
  $('entry').addEventListener('submit', (event) => {
    event.preventDefault();
    clearTimeout(timer);
    check(input.value, { reveal: true });
  });

  try {
    await loadState();
  } catch {
    $('waiting').textContent = '';
    return serverLost();
  }
  setupInbox(ctx);
  renderAll();
  if (view.info.examples.length) load(view.info.examples[0].url);
  else $('waiting').textContent = 'Results appear here as you type.';
  setInterval(watch, CHECK_EVERY_MS);
}

start();
