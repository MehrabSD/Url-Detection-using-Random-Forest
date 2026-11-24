"""Red team: search for the cheapest edits that sneak a flagged URL past the model.

    python src/adversary.py "http://paypal.com.secure-login-verify.xyz/webscr?cmd=_login"
    python src/adversary.py --mode strict "http://194.48.152.10/bins/x86.sh"

A detector is only as good as its weakest blind spot, and an attacker who sees a
link flagged will simply change it. This module plays that attacker. It knows a
fixed menu of edits that real phishing and malware campaigns make (move to a
.com, drop the word "login", pad the link with tracking parameters, ...), gives
each edit a price, and runs a beam search over sequences of edits, scoring every
candidate with the real model, until the score drops below the chosen cut-off.

Prices reflect what the edit costs the attacker, not how many characters change:

    1  free         rename a page or parameter on a server they already control
    2  new domain   register another address (a few dollars and some time)
    3  weaker bait  remove the brand or lure the victim was meant to trust

The answer is a *counterfactual explanation* ("this exact link would have passed")
and a robustness measurement in one: red_team.py runs it over hundreds of
phishing URLs the model catches and reports how many can be slipped through,
at what price. Only URL text is edited and scored; nothing is ever fetched.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import features as F

PRICES = {
    1: ("Free", "Rename something on a server the attacker already controls."),
    2: ("New domain", "Register another address: a few dollars and a little time."),
    3: ("Weaker bait", "Remove the brand or lure the victim was meant to trust."),
}

# id -> (short label, price, what the attacker does)
MOVES = {
    "pad_tracking": ("Add tracking parameters", 1,
                     "Append utm_ tags of the kind every newsletter link carries."),
    "pad_path": ("Add an ordinary-looking folder", 1,
                 "Put the page under a folder such as /en-us/help/."),
    "pad_path_suffix": ("Add ordinary folders after the page", 1,
                        "Route a longer address such as /login/en-us/help to the same page."),
    "drop_query": ("Drop the query string", 1,
                   "Serve the page without the ?parameters."),
    "drop_fragment": ("Drop the # fragment", 1, "Leave out the part after #."),
    "drop_extension": ("Hide the file extension", 1,
                       "Rewrite /claim.php as /claim, which any web server can do."),
    "drop_path_word": ("Rename a bait word in the path", 1,
                       "Remove words such as login, verify or account from the path or query."),
    "trim_path": ("Remove a folder from the path", 1, "Serve the page one folder higher."),
    "shorten_token": ("Shorten a long random token", 1,
                      "Replace a long random id with a short one."),
    "lowercase_path": ("Lower-case the path", 1, "Use only small letters in the path."),
    "swap_tld": ("Move to a mainstream ending", 2,
                 "Register the same name under .com, .net or .org instead."),
    "ip_to_name": ("Use a domain name instead of an IP", 2,
                   "Point a freshly registered, bland name at the same server."),
    "plain_name": ("Register a plainer name", 2,
                   "The same name without hyphens and digits."),
    "drop_host_word": ("Register a name without the bait word", 2,
                       "The same name with words such as secure or login removed."),
    "drop_subdomains": ("Remove the subdomain", 3,
                        "Drop the levels in front of the real domain, including any brand "
                        "name placed there to fool the reader."),
}

_BLAND_NAMES = ("northfield.com", "bluepeak.net", "riverstone.org")
_TRACKING = ("utm_source=newsletter&utm_medium=email&utm_campaign=autumn",
             "utm_source=partner&utm_medium=referral")
_BENIGN_FOLDERS = ("/en-us/help", "/blog/2024")
_LONG_TOKEN = re.compile(r"[A-Za-z0-9_\-]{16,}")
_MAINSTREAM = ("com", "net", "org")


def _join(userinfo, host, port, path, query, fragment) -> str:
    u = (userinfo + "@" if userinfo else "") + host + (":" + port if port else "") + path
    if query:
        u += "?" + query
    if fragment:
        u += "#" + fragment
    return u


def _strip_words(text: str) -> list[tuple[str, str]]:
    """(text with one bait word and a separator next to it removed, that word)."""
    out = []
    low = text.lower()
    for word in sorted(set(F._SUSP_RE.findall(low)), key=len, reverse=True):
        i = low.find(word)
        a, b = i, i + len(word)
        if b < len(text) and text[b] in "-_.":
            b += 1
        elif a > 0 and text[a - 1] in "-_.":
            a -= 1
        out.append((text[:a] + text[b:], word))
    return out


def candidates(u: str) -> list[tuple[str, str, str]]:
    """Every one-move edit of the normalised URL u: (new_url, move_id, detail)."""
    userinfo, host, port, path, query, fragment = F.split_url(u)
    parts = dict(userinfo=userinfo, host=host, port=port, path=path, query=query, fragment=fragment)
    out: list[tuple[str, str, str]] = []

    def emit(move, detail, **change):
        new = _join(**{**parts, **change})
        if new != u and len(F.normalize_url(new)) >= 4 and "." in new:
            out.append((new, move, detail))

    is_ip = F._is_ip(host)
    if is_ip or not host:
        sub, dom, suffix = "", host, ""
    else:
        ext = F._TLD.extract_str(host)
        sub, dom, suffix = ext.subdomain, ext.domain, ext.suffix

    # ---- free: path, query, fragment
    for tag in _TRACKING:
        emit("pad_tracking", f"?{tag}", query=f"{query}&{tag}" if query else tag)
    if path.count("/") <= 4:
        for folder in _BENIGN_FOLDERS:
            emit("pad_path", folder + "/", path=folder + (path if path.startswith("/") else "/" + path))
            emit("pad_path_suffix", folder, path=path.rstrip("/") + folder)
    if query:
        emit("drop_query", f"?{query[:40]}", query="")
    if fragment:
        emit("drop_fragment", f"#{fragment[:40]}", fragment="")
    segments = path.split("/")
    last = segments[-1]
    m = F._EXT_RE.search(last)
    if m and len(last) > len(m.group(0)):
        emit("drop_extension", m.group(0), path="/".join(segments[:-1] + [last[: -len(m.group(0))]]))
    for new_path, word in _strip_words(path):
        emit("drop_path_word", f'"{word}" from the path', path=new_path)
    for new_query, word in _strip_words(query):
        emit("drop_path_word", f'"{word}" from the query', query=new_query)
    real = [i for i, s in enumerate(segments) if s]
    if len(real) >= 2:
        for i in real[:-1][:3]:
            emit("trim_path", f"/{segments[i]}", path="/".join(segments[:i] + segments[i + 1:]))
    for match in list(_LONG_TOKEN.finditer(path))[:2]:
        short = match.group(0)[:6]
        emit("shorten_token", match.group(0)[:24], path=path[:match.start()] + short + path[match.end():])
    for match in list(_LONG_TOKEN.finditer(query))[:2]:
        short = match.group(0)[:6]
        emit("shorten_token", match.group(0)[:24], query=query[:match.start()] + short + query[match.end():])
    if path != path.lower():
        emit("lowercase_path", "", path=path.lower())

    # ---- new domain
    if is_ip:
        for name in _BLAND_NAMES:
            emit("ip_to_name", f"{host} -> {name}", host=name, port="")
    elif dom and suffix:
        prefix = sub + "." if sub else ""
        if suffix not in _MAINSTREAM:
            for tld in _MAINSTREAM:
                emit("swap_tld", f".{suffix} -> .{tld}", host=f"{prefix}{dom}.{tld}")
        plain = re.sub(r"[-\d]", "", dom)
        if plain != dom and len(plain) >= 3:
            emit("plain_name", f"{dom} -> {plain}", host=f"{prefix}{plain}.{suffix}")
        for new_dom, _word in _strip_words(dom):
            new_dom = new_dom.strip("-")
            if len(new_dom) >= 3:
                emit("drop_host_word", f"{dom} -> {new_dom}", host=f"{prefix}{new_dom}.{suffix}")
        if sub:
            emit("drop_subdomains", f"{sub}.", host=f"{dom}.{suffix}", userinfo="")
    if userinfo:
        emit("drop_subdomains", f"{userinfo}@", userinfo="")
    return out


@dataclass(order=True)
class _Path:
    score: float
    cost: int
    url: str = field(compare=False)
    steps: list = field(compare=False, default_factory=list)


def attack(clf, url: str, threshold: float, *, beam: int = 6, depth: int = 5,
           max_cost: int = 8) -> dict:
    """Beam search for the cheapest sequence of moves that takes the score below
    `threshold`. Every candidate is scored by the real model in batches."""
    started = time.perf_counter()
    start = F.normalize_url(url)
    start_score = float(clf.score([start])[0])
    result = {"url": start, "score": start_score, "threshold": threshold,
              "already_passes": start_score < threshold}
    if result["already_passes"]:
        return {**result, "found": False, "steps": [], "evaluated": 1,
                "single_moves": [], "milliseconds": _ms(started)}

    frontier = [_Path(start_score, 0, start)]
    seen = {start}
    best: _Path | None = None
    evaluated = 1
    singles: dict[str, tuple[float, str, str]] = {}

    for level in range(depth):
        batch = []
        for state in frontier:
            for new, move, detail in candidates(state.url):
                cost = state.cost + MOVES[move][1]
                if new in seen or cost > max_cost or (best and cost > best.cost):
                    continue
                seen.add(new)
                batch.append((state, new, move, detail, cost))
        if not batch:
            break
        scores = clf.score([b[1] for b in batch])
        evaluated += len(batch)
        children = []
        for (state, new, move, detail, cost), s in zip(batch, scores):
            step = {"move": move, "label": MOVES[move][0], "price": MOVES[move][1],
                    "detail": detail, "url": new, "score": float(s)}
            child = _Path(float(s), cost, new, state.steps + [step])
            if level == 0 and (move not in singles or s < singles[move][0]):
                singles[move] = (float(s), new, detail)
            if s < threshold:
                if best is None or (cost, s) < (best.cost, best.score):
                    best = child
            else:
                children.append(child)
        # Stop when no deeper path can be cheaper than the one already found.
        if best and best.cost <= min((c.cost for c in children), default=99) + 1:
            break
        # Keep the lowest scores, but give each move type a chance (diversity).
        children.sort()
        kept, per_move = [], {}
        for c in children:
            key = c.steps[-1]["move"]
            if per_move.get(key, 0) < 2:
                kept.append(c)
                per_move[key] = per_move.get(key, 0) + 1
            if len(kept) == beam:
                break
        frontier = kept

    single_moves = sorted(
        ({"move": m, "label": MOVES[m][0], "price": MOVES[m][1], "detail": d, "url": u,
          "score": s, "change": s - start_score} for m, (s, u, d) in singles.items()),
        key=lambda r: r["score"])
    return {**result, "found": best is not None,
            "steps": best.steps if best else [],
            "final_url": best.url if best else None,
            "final_score": best.score if best else None,
            "total_price": best.cost if best else None,
            "evaluated": evaluated, "single_moves": single_moves,
            "milliseconds": _ms(started)}


def _ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)


def catalogue() -> dict:
    return {"moves": [{"id": k, "label": v[0], "price": v[1], "description": v[2]} for k, v in MOVES.items()],
            "prices": {str(k): {"label": v[0], "description": v[1]} for k, v in PRICES.items()}}


def main() -> int:
    from predict import MODES, UrlClassifier
    ap = argparse.ArgumentParser(description="Find the cheapest edits that get a URL past the model.")
    ap.add_argument("urls", nargs="+")
    ap.add_argument("--mode", choices=list(MODES), default="balanced")
    args = ap.parse_args()
    clf = UrlClassifier()
    thr = clf.thresholds[MODES[args.mode]]
    for url in args.urls:
        r = attack(clf, url, thr)
        print(f"\n{r['url']}  score {r['score']:.3f}  (cut-off {thr:.3f})")
        if r["already_passes"]:
            print("  already passes - nothing to do")
            continue
        for i, s in enumerate(r["steps"], 1):
            print(f"  {i}. [{PRICES[s['price']][0]:<11}] {s['label']:<38} -> {s['score']:.3f}  {s['url']}")
        print(f"  {'evaded' if r['found'] else 'NOT evaded'}; price {r['total_price']}; "
              f"{r['evaluated']} candidates scored in {r['milliseconds']:.0f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
