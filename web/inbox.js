// "Check a whole message": every link in a pasted email, scored and ranked.
// The server (src/triage.py) finds the links, refangs hxxp://site[.]com, spots
// HTML links whose text names another site, and scores them all in one batch.

import { $, el, fill, fmt, getJSON } from './util.js';

const state = { result: null };
let ctx = null;   // { info, mode, load } from app.js

const csvCell = (v) => {
  const s = String(v ?? '');
  // a leading = + - @ would run as a formula in a spreadsheet
  const safe = /^[=+\-@\t\r]/.test(s) ? `'${s}` : s;
  return /[",\n]/.test(safe) ? `"${safe.replace(/"/g, '""')}"` : safe;
};

function download(rows) {
  const mode = ctx.mode();
  const head = ['rank', 'url', 'site', 'score', `verdict (${mode.label})`, 'lookalike', 'hidden behind other link text', 'times in message'];
  const lines = [head, ...rows.map((r, i) => [i + 1, r.url, r.site, r.score.toFixed(4), r.verdicts[mode.id],
    r.lookalike ? `${r.lookalike.label}: ${r.lookalike.brand}` : '', r.hidden_target ? 'yes' : '', r.count])];
  const blob = new Blob([lines.map((l) => l.map(csvCell).join(',')).join('\r\n')], { type: 'text/csv' });
  const a = el('a', { href: URL.createObjectURL(blob), download: 'message-links.csv' });
  document.body.append(a);
  a.click();
  setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 1000);
}

function scoreBar(score, cut) {
  return el('span', { class: 'mini', role: 'img', 'aria-label': `score ${fmt.score(score)}` },
    el('span', { class: `mini-fill ${score >= cut ? 'malicious' : 'benign'}`, style: { width: `${score * 100}%` } }),
    el('span', { class: 'mini-cut', style: { left: `${cut * 100}%` } }));
}

function render() {
  const box = $('inbox-result');
  const r = state.result;
  if (!r) return fill(box);
  const mode = ctx.mode();
  const rows = r.urls;
  const flagged = rows.filter((x) => x.verdicts[mode.id] === 'malicious').length;
  const lookalikes = rows.filter((x) => x.lookalike).length;
  const hidden = r.mismatches.length;

  const facts = [
    `${rows.length} ${rows.length === 1 ? 'link' : 'links'} found`,
    `${flagged} flagged by the model (${mode.label} cut-off)`,
    lookalikes && `${lookalikes} imitating a brand`,
    hidden && `${hidden} hidden behind other link text`,
  ].filter(Boolean);

  fill(box,
    el('div', { class: 'inbox-summary' },
      el('p', { class: 'lab-verdict' + (flagged || lookalikes || hidden ? ' gets-through' : ' passes'),
        text: flagged || lookalikes || hidden ? 'Do not click before checking' : 'Nothing flagged' }),
      el('p', { class: 'verdict-line', text: facts.join('; ') + '.' }),
      r.truncated && el('p', { class: 'mode-note', text: 'The message was long, so only the first part was checked.' })),
    r.mismatches.length > 0 && el('div', { class: 'mismatches' },
      el('h3', { class: 'minor', text: 'Links that show one site and open another' }),
      el('ul', {}, r.mismatches.map((m) => el('li', {},
        el('span', { class: 'mm-label', text: 'Shows' }), el('span', { class: 'mm-url', text: m.shows }),
        el('span', { class: 'mm-label', text: 'Opens' }), el('span', { class: 'mm-url opens', text: m.opens }))))),
    el('table', { class: 'triage' },
      el('caption', { class: 'visually-hidden', text: 'Links in the message, riskiest first' }),
      el('thead', {}, el('tr', {},
        el('th', { scope: 'col', text: 'Score' }),
        el('th', { scope: 'col', text: 'Link' }),
        el('th', { scope: 'col' }, el('span', { class: 'visually-hidden', text: 'Action' })))),
      el('tbody', {}, rows.map((row) => {
        const bad = row.verdicts[mode.id] === 'malicious';
        return el('tr', { class: bad ? 'bad' : null },
          el('td', { class: 'triage-score' },
            el('b', { text: fmt.score(row.score) }), scoreBar(row.score, mode.threshold),
            el('small', { text: bad ? 'flagged' : 'passes' })),
          el('td', { class: 'triage-url' },
            el('span', { class: 'case-url', text: row.url }),
            el('span', { class: 'tags' },
              row.lookalike && el('span', { class: `tag ${row.lookalike.severity}`, 'data-tip': row.lookalike.detail,
                text: `${row.lookalike.label}: ${row.lookalike.brand}` }),
              row.hidden_target && el('span', { class: 'tag high', text: 'Hidden behind other link text' }),
              row.count > 1 && el('span', { class: 'tag', text: `Appears ${row.count} times` }))),
          el('td', { class: 'triage-act' },
            el('button', { class: 'link', type: 'button', text: 'Inspect', onclick: () => ctx.load(row.url, true) })));
      }))),
    el('div', { class: 'inbox-actions' },
      el('button', { class: 'more', type: 'button', text: 'Download as CSV', onclick: () => download(rows) }),
      el('span', { class: 'lab-stats', text: `Scored in ${fmt.int(r.milliseconds)} ms.` })));
}

async function submit(event) {
  event?.preventDefault();
  const text = $('inbox-text').value;
  const error = $('inbox-error');
  try {
    state.result = await getJSON('/api/triage', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text }),
    });
    error.hidden = true;
  } catch (problem) {
    state.result = null;
    error.textContent = problem.message;
    error.hidden = false;
  }
  render();
}

export function setupInbox(context) {
  ctx = context;
  $('inbox-form').addEventListener('submit', submit);
  $('inbox-sample').addEventListener('click', () => {
    $('inbox-text').value = ctx.info.example_message ?? '';
    submit();
  });
  $('inbox-clear').addEventListener('click', () => {
    $('inbox-text').value = '';
    state.result = null;
    $('inbox-error').hidden = true;
    render();
    $('inbox-text').focus();
  });
}

export function renderInbox(context) {
  ctx = context;
  render();
}
