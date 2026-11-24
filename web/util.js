// Small DOM and formatting helpers. Text always goes in through textContent,
// so a URL typed into the page can never be interpreted as markup.

export const $ = (id) => document.getElementById(id);

function build(node, props, children) {
  for (const [key, value] of Object.entries(props)) {
    if (value == null || value === false) continue;
    if (key === 'class') node.setAttribute('class', value);
    else if (key === 'text') node.textContent = value;
    else if (key === 'style') for (const [p, v] of Object.entries(value)) node.style.setProperty(p, v);
    else if (key.startsWith('on')) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? '' : value);
  }
  node.append(...children.flat(Infinity).filter((c) => c != null && c !== false));
  return node;
}

export const el = (tag, props = {}, ...children) => build(document.createElement(tag), props, children);
export const svg = (tag, props = {}, ...children) =>
  build(document.createElementNS('http://www.w3.org/2000/svg', tag), props, children);

export function fill(container, ...children) {
  container.replaceChildren(...children.flat(Infinity).filter((c) => c != null && c !== false));
}

export const fmt = {
  int: (n) => Math.round(n).toLocaleString('en-GB'),
  pct: (x, digits = 0) => `${(x * 100).toFixed(digits)}%`,
  // one decimal only where a whole number would hide the difference
  pctAuto: (x) => fmt.pct(x, x > 0 && x < 0.1 ? 1 : 0),
  score: (x) => x.toFixed(2),
  signed: (x, digits = 3) => `${x < 0 ? '−' : '+'}${Math.abs(x).toFixed(digits)}`,
  time: (d = new Date()) => d.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' }),
  // A feature's value for one URL, printed according to the kind describe.py gives it.
  feature(kind, v) {
    switch (kind) {
      case 'flag': return v ? 'yes' : 'no';
      case 'count': return fmt.int(v);
      case 'chars': return `${fmt.int(v)} ${Math.round(v) === 1 ? 'character' : 'characters'}`;
      case 'share': return fmt.pct(v, v > 0 && v < 0.1 ? 1 : 0);
      case 'bits': return `${v.toFixed(2)} bits`;
      case 'score': return fmt.signed(v, 2);
      default: return Number.isInteger(v) ? fmt.int(v) : String(Number(v.toPrecision(3)));
    }
  },
};

export async function getJSON(path, options) {
  const response = await fetch(path, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw Object.assign(new Error(body.error || 'The server returned an error.'), { status: response.status });
  return body;
}

// One shared tooltip for anything carrying data-tip.
export function startTips(tip) {
  const show = (target, x, y) => {
    tip.textContent = target.dataset.tip;
    tip.hidden = false;
    const box = tip.getBoundingClientRect();
    tip.style.left = `${Math.max(8, Math.min(x + 12, innerWidth - box.width - 8))}px`;
    tip.style.top = `${y - box.height - 12 < 8 ? y + 18 : y - box.height - 12}px`;
  };
  document.addEventListener('pointermove', (event) => {
    const target = event.target.closest?.('[data-tip]');
    if (target) show(target, event.clientX, event.clientY);
    else tip.hidden = true;
  });
  document.addEventListener('pointerleave', () => { tip.hidden = true; });
  addEventListener('scroll', () => { tip.hidden = true; }, { passive: true });
}
