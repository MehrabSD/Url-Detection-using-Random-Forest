"""Inbox triage: find every link in a pasted email or message and rank them.

    python src/triage.py message.txt
    python src/triage.py message.eml --mode strict

Phishing arrives as a message, not as a lone URL. This takes the whole text (plain
text, HTML source or a raw .eml), pulls out every web address in it, scores them
all in one batch, and lists the riskiest first. It also understands two things
analysts deal with every day:

    defanged links   hxxp://evil[.]com, evil(.)com, evil[dot]com - the way threat
                     reports write live links so nobody clicks them by accident
    hidden targets   in HTML, <a href="http://evil.xyz">paypal.com</a> shows one
                     site and opens another; such links are reported as mismatches

Nothing is fetched: like the model, this reads text only.
"""
from __future__ import annotations

import argparse
import html
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import features as F

MAX_TEXT = 200_000
MAX_URLS = 300

_REFANG = [
    (re.compile(r"\bhxxp(s?)(?=\[?:)", re.I), r"http\1"),
    (re.compile(r"\bh\*\*p(s?)(?=:)", re.I), r"http\1"),
    (re.compile(r"\[:\]|\(:\)"), ":"),
    (re.compile(r"\[\.\]|\(\.\)|\{\.\}|\[dot\]|\(dot\)", re.I), "."),
    (re.compile(r"\[/\]"), "/"),
]
_ANCHOR = re.compile(r"<a\b[^>]*?\bhref\s*=\s*([\"'])(.*?)\1[^>]*>(.*?)</a\s*>", re.I | re.S)
_TAG = re.compile(r"<[^>]+>")
_SCHEME_URL = re.compile(r"(?:https?|ftp)://[^\s<>\"'`{}|\\^\[\]]+", re.I)
# bare addresses: www.example.com/..., or name.tld/... with a known ending
_BARE_URL = re.compile(
    r"(?<![@\w.\-/])((?:[a-z0-9](?:[a-z0-9\-]{0,61}[a-z0-9])?\.)+[a-z]{2,24})(?::\d{2,5})?(/[^\s<>\"'`{}|\\^\[\]]*)?",
    re.I)
_TRAILING = ".,;:!?)]}'\">*"
_FILE_ENDINGS = {"txt", "pdf", "doc", "docx", "png", "jpg", "jpeg", "gif", "zip", "exe", "js", "py",
                 "csv", "html", "htm", "php", "json", "xml", "css", "md", "eml", "msg", "xls", "xlsx"}
_QP_SOFT_BREAK = re.compile(r"=\r?\n")


def refang(text: str) -> str:
    for pattern, repl in _REFANG:
        text = pattern.sub(repl, text)
    return text


def _tidy(url: str) -> str:
    url = html.unescape(url).strip()
    while url and url[-1] in _TRAILING:
        # keep a closing bracket that has its opening partner inside the URL
        if url[-1] == ")" and url.count("(") >= url.count(")"):
            break
        url = url[:-1]
    return url


def _plausible_bare(host: str) -> bool:
    ext = F._TLD.extract_str(host.lower())
    if not ext.suffix or not ext.domain:
        return False
    # "report.pdf" or "setup.exe" are file names, not sites
    return not (ext.suffix in _FILE_ENDINGS and not ext.subdomain)


def extract(text: str) -> dict:
    """URLs in order of first appearance, with how often each occurs, plus
    HTML links whose visible text names a different site from their target."""
    text = _QP_SOFT_BREAK.sub("", text[:MAX_TEXT])     # quoted-printable .eml line breaks
    text = text.replace("=3D", "=")
    text = refang(text)
    found: dict[str, dict] = {}
    mismatches = []

    def add(url: str, how: str):
        url = _tidy(url)
        norm = F.normalize_url(url)
        if len(norm) < 4 or "." not in norm or re.search(r"[\s\x00-\x1f\x7f]", norm):
            return
        if norm in found:
            found[norm]["count"] += 1
        elif len(found) < MAX_URLS:
            found[norm] = {"url": url, "normalized": norm, "count": 1, "found_as": how}

    for m in _ANCHOR.finditer(text):
        href, shown = m.group(2), _TAG.sub("", m.group(3)).strip()
        if not re.match(r"(?:https?|ftp)://|www\.", href, re.I):
            continue
        add(href, "link")
        shown_url = _SCHEME_URL.search(shown) or _BARE_URL.search(shown)
        if shown_url:
            a = F.registered_domain(_tidy(shown_url.group(0)))
            b = F.registered_domain(_tidy(href))
            if a and b and a != b:
                mismatches.append({"shows": _tidy(shown_url.group(0))[:120], "opens": _tidy(href)[:200],
                                   "shows_site": a, "opens_site": b})

    # A link's visible text is not a destination (the mismatch check covers it).
    plain = _ANCHOR.sub(" ", text)
    plain = _TAG.sub(" ", plain)
    for m in _SCHEME_URL.finditer(plain):
        add(m.group(0), "text")
    without_schemes = _SCHEME_URL.sub(" ", plain)
    for m in _BARE_URL.finditer(without_schemes):
        if _plausible_bare(m.group(1)):
            add(m.group(0), "text")
    return {"urls": list(found.values()), "mismatches": mismatches,
            "truncated": len(text) >= MAX_TEXT or len(found) >= MAX_URLS}


def triage(clf, text: str, thresholds: dict, lookalike_check=None) -> dict:
    """Extract, score in one batch, attach lookalike findings, sort riskiest first."""
    out = extract(text)
    rows = out["urls"]
    if rows:
        scores = clf.score([r["url"] for r in rows])
        for r, s in zip(rows, scores):
            r["score"] = float(s)
            r["site"] = F.registered_domain(r["normalized"])
            r["verdicts"] = {k: ("malicious" if s >= t else "benign") for k, t in thresholds.items()}
            if lookalike_check:
                findings = lookalike_check(r["url"])["findings"]
                r["lookalike"] = findings[0] if findings else None
    mismatch_sites = {m["opens_site"] for m in out["mismatches"]}
    for r in rows:
        r["hidden_target"] = r.get("site") in mismatch_sites

    def risk(r):
        boost = (r.get("lookalike") or {}).get("severity") == "high" or r["hidden_target"]
        return (-(r["score"] + (1.0 if boost else 0.0)))
    rows.sort(key=risk)
    return out


def main() -> int:
    from lookalike import check
    from predict import MODES, UrlClassifier
    ap = argparse.ArgumentParser(description="Rank every link in a message by risk.")
    ap.add_argument("file", help="text, HTML or .eml file")
    ap.add_argument("--mode", choices=list(MODES), default="balanced")
    args = ap.parse_args()
    text = Path(args.file).read_text(encoding="utf-8", errors="replace")
    clf = UrlClassifier()
    thr = {m: clf.thresholds[k] for m, k in MODES.items()}
    out = triage(clf, text, thr, check)
    for r in out["urls"]:
        flag = r["verdicts"][args.mode].upper() if r["verdicts"][args.mode] == "malicious" else "benign"
        extra = f"  [{r['lookalike']['label']}: {r['lookalike']['brand']}]" if r.get("lookalike") else ""
        print(f"{flag:<10} {r['score']:.3f}  {r['url']}{extra}")
    for m in out["mismatches"]:
        print(f"MISMATCH   link shows {m['shows']} but opens {m['opens']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
