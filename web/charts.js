// The page's charts. Bar charts are real tables, so every value is readable
// as text and nothing depends on colour alone.

import { el, svg, fmt } from './util.js';

// rows: [{ label, note, share (0..1 of the widest bar), display, aside, strong, tip }]
export function barTable({ caption, head, rows }) {
  const hasAside = rows.some((r) => r.aside);
  return el('table', { class: 'rows' },
    el('caption', { class: 'visually-hidden', text: caption }),
    el('thead', {}, el('tr', {},
      el('th', { scope: 'col', class: 'label', text: head[0] }),
      el('th', { scope: 'col' }, el('span', { class: 'visually-hidden', text: 'Bar' })),
      hasAside && el('th', { scope: 'col', class: 'aside', text: head[2] ?? '' }),
      el('th', { scope: 'col', class: 'value', text: head[1] }))),
    el('tbody', {}, rows.map((r) => el('tr', { class: r.strong ? 'strong' : null },
      el('th', { scope: 'row', class: 'label' }, el('b', { text: r.label }), r.note && el('small', { text: r.note })),
      el('td', {}, el('span', { class: 'track', 'data-tip': r.tip },
        el('span', { class: 'fill', style: { width: `${Math.max(0, Math.min(1, r.share)) * 100}%` } }))),
      hasAside && el('td', { class: 'aside', text: r.aside ?? '' }),
      el('td', { class: 'value', text: r.display })))));
}

// rows: [{ label, note, amount (signed), display, summary }]; bars grow from the middle.
export function splitTable({ caption, rows, scale, head = 'Feature' }) {
  return el('table', { class: 'rows' },
    el('caption', { class: 'visually-hidden', text: caption }),
    el('thead', {}, el('tr', {},
      el('th', { scope: 'col', class: 'label', text: head }),
      el('th', { scope: 'col' }, el('span', { class: 'split-head' },
        el('span', {}, el('i', { class: 'key benign' }), 'toward benign'),
        el('span', {}, 'toward malicious ', el('i', { class: 'key malicious' })))),
      el('th', { scope: 'col', class: 'value', text: 'Change' }))),
    el('tbody', {}, rows.map((r) => el('tr', { class: r.summary ? 'sum' : null },
      el('th', { scope: 'row', class: 'label' }, el('b', { text: r.label }), r.note && el('small', { text: r.note })),
      el('td', {}, el('span', { class: 'track split' },
        el('span', {
          class: `fill ${r.amount < 0 ? 'toward-benign' : 'toward-malicious'}`,
          style: { width: `${Math.min(1, Math.abs(r.amount) / scale) * 50}%` },
        }))),
      el('td', { class: 'value', text: r.display })))));
}

// Each tree's own answer for this URL, with the decision cut-offs and the average.
export function voteChart({ votes, modes, chosen, score }) {
  const BINS = 25;
  const counts = new Array(BINS).fill(0);
  for (const v of votes) counts[Math.min(BINS - 1, Math.floor(v * BINS))] += 1;
  const tallest = Math.max(...counts, 1);
  const side = (x) => (x < 0.14 ? ' left' : x > 0.86 ? ' right' : '');

  const bars = counts.map((n, i) => el('span', {
    class: n ? 'vbar' : 'vbar empty',
    style: { height: `${(n / tallest) * 100}%` },
    'data-tip': `${n} ${n === 1 ? 'tree' : 'trees'} between ${(i / BINS).toFixed(2)} and ${((i + 1) / BINS).toFixed(2)}`,
  }));
  const cuts = modes.map((m) => el('span', {
    class: m.id === chosen.id ? 'cut chosen' : 'cut',
    style: { left: `${m.threshold * 100}%` },
    'data-tip': `${m.label} cut-off: ${fmt.score(m.threshold)}`,
  }, m.id === chosen.id && el('span', {
    class: `cut-label${side(m.threshold)}`, text: `${m.label} cut-off ${fmt.score(m.threshold)}`,
  })));

  return el('div', { class: 'votes', role: 'img',
    'aria-label': `Histogram of the ${votes.length} tree votes from 0, benign, to 1, malicious. Their average is ${fmt.score(score)}.` },
    el('div', { class: 'votes-plot' }, bars, cuts),
    el('div', { class: 'votes-axis' },
      el('span', { text: '0 benign' }),
      el('span', { class: 'end', text: 'malicious 1' }),
      el('span', { class: `pointer${side(score)}`, style: { left: `${score * 100}%` } },
        el('b', { text: `average ${fmt.score(score)}` }))));
}

// The trade-off curve of the final model, with one marked point per decision setting.
export function tradeoffChart({ fpr, tpr, points, chosenId }) {
  const SIZE = 300, LEFT = 44, TOP = 34, BOTTOM = 44, RIGHT = 16;
  const x = (v) => LEFT + v * SIZE;
  const y = (v) => TOP + (1 - v) * SIZE;
  const ticks = [0, 0.25, 0.5, 0.75, 1];
  const path = fpr.map((f, i) => `${i ? 'L' : 'M'}${x(f).toFixed(1)} ${y(tpr[i]).toFixed(1)}`).join('');

  const probeLine = svg('line', { class: 'probe-line', y1: TOP, y2: TOP + SIZE, visibility: 'hidden' });
  const probe = svg('circle', { class: 'probe', r: 5, visibility: 'hidden' });
  const hit = svg('rect', { class: 'hit', x: LEFT, y: TOP, width: SIZE, height: SIZE });
  hit.addEventListener('pointermove', (event) => {
    const box = hit.getBoundingClientRect();
    const at = (event.clientX - box.left) / box.width;
    let best = 0;
    fpr.forEach((f, i) => { if (Math.abs(f - at) < Math.abs(fpr[best] - at)) best = i; });
    for (const node of [probe, probeLine]) node.setAttribute('visibility', 'visible');
    probe.setAttribute('cx', x(fpr[best])); probe.setAttribute('cy', y(tpr[best]));
    probeLine.setAttribute('x1', x(fpr[best])); probeLine.setAttribute('x2', x(fpr[best]));
    hit.dataset.tip = `Flagging ${fmt.pctAuto(fpr[best])} of safe URLs catches ${fmt.pct(tpr[best])} of malicious ones`;
  });
  hit.addEventListener('pointerleave', () => {
    for (const node of [probe, probeLine]) node.setAttribute('visibility', 'hidden');
  });

  const marks = points.map((p) => {
    const chosen = p.id === chosenId;
    return [
      svg('circle', { class: chosen ? 'point chosen' : 'point', cx: x(p.fpr), cy: y(p.tpr), r: chosen ? 6 : 5 }),
      chosen && svg('text', { class: 'point-label', x: x(p.fpr) + 12, y: y(p.tpr) + 16, text: p.label }),
    ];
  });

  return el('div', { class: 'curve' }, svg('svg', {
    viewBox: `0 0 ${LEFT + SIZE + RIGHT} ${TOP + SIZE + BOTTOM}`, role: 'img',
    'aria-label': 'Trade-off curve: share of malicious URLs caught against share of safe URLs wrongly flagged, for every possible cut-off.',
  },
    ticks.map((t) => [
      svg('line', { class: t ? 'grid' : 'axis', x1: LEFT, x2: LEFT + SIZE, y1: y(t), y2: y(t) }),
      svg('line', { class: t ? 'grid' : 'axis', x1: x(t), x2: x(t), y1: TOP, y2: TOP + SIZE }),
      svg('text', { x: LEFT - 8, y: y(t) + 4, 'text-anchor': 'end', text: fmt.pct(t) }),
      svg('text', { x: x(t), y: TOP + SIZE + 18, 'text-anchor': 'middle', text: fmt.pct(t) }),
    ]),
    svg('line', { class: 'chance', x1: x(0), y1: y(0), x2: x(1), y2: y(1) }),
    svg('text', { x: x(0.62), y: y(0.55), text: 'guessing' }),
    svg('text', { x: 0, y: 14, text: 'Malicious URLs caught' }),
    svg('text', { x: LEFT + SIZE / 2, y: TOP + SIZE + 38, 'text-anchor': 'middle', text: 'Safe URLs wrongly flagged' }),
    svg('path', { class: 'path', d: path }),
    probeLine, marks, probe, hit));
}

// A few lines over a small whole-number x axis, labelled at their ends (no legend).
// series: [{ label, points: [[x, y 0..1]], strong, dashed }]
export function lineChart({ series, xLabel, yLabel, tip }) {
  const W = 340, H = 230, LEFT = 44, TOP = 30, BOTTOM = 46, RIGHT = 150;
  const xs = [...new Set(series.flatMap((s) => s.points.map((p) => p[0])))].sort((a, b) => a - b);
  const [x0, x1] = [xs[0], xs.at(-1)];
  const x = (v) => LEFT + ((v - x0) / Math.max(1, x1 - x0)) * W;
  const y = (v) => TOP + (1 - v) * H;
  const ticks = [0, 0.25, 0.5, 0.75, 1];

  // end labels, nudged apart so they never overlap
  const ends = series.map((s) => ({ s, at: y(s.points.at(-1)[1]) })).sort((a, b) => a.at - b.at);
  for (let i = 1; i < ends.length; i += 1) ends[i].at = Math.max(ends[i].at, ends[i - 1].at + 16);

  return el('div', { class: 'curve lines' }, svg('svg', {
    viewBox: `0 0 ${LEFT + W + RIGHT} ${TOP + H + BOTTOM}`, role: 'img',
    'aria-label': series.map((s) => `${s.label}: ${s.points.map(([px, py]) => `${fmt.pct(py)} at ${px}`).join(', ')}`).join('. '),
  },
    ticks.map((t) => [
      svg('line', { class: t ? 'grid' : 'axis', x1: LEFT, x2: LEFT + W, y1: y(t), y2: y(t) }),
      svg('text', { x: LEFT - 8, y: y(t) + 4, 'text-anchor': 'end', text: fmt.pct(t) }),
    ]),
    xs.map((v) => svg('text', { x: x(v), y: TOP + H + 18, 'text-anchor': 'middle', text: String(v) })),
    svg('text', { x: 0, y: 14, text: yLabel }),
    svg('text', { x: LEFT + W / 2, y: TOP + H + 40, 'text-anchor': 'middle', text: xLabel }),
    series.map((s) => [
      svg('path', { class: `line${s.strong ? ' strong' : ''}${s.dashed ? ' dashed' : ''}`,
        d: s.points.map(([px, py], i) => `${i ? 'L' : 'M'}${x(px).toFixed(1)} ${y(py).toFixed(1)}`).join('') }),
      s.points.map(([px, py]) => svg('circle', {
        class: `dot${s.strong ? ' strong' : ''}`, cx: x(px), cy: y(py), r: s.strong ? 4.5 : 3.5,
        'data-tip': tip ? tip(s, px, py) : `${s.label}: ${fmt.pct(py)}`,
      })),
    ]),
    ends.map(({ s, at }) => svg('text', {
      class: `end-label${s.strong ? ' strong' : ''}`, x: LEFT + W + 10, y: at + 4, text: s.label,
    }))));
}
