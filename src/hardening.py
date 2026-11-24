"""Multi-view scoring: a defence against padding attacks, with no retraining.

The red team (adversary.py) found that most flagged phishing URLs slip under the
cut-off after one free edit: adding tracking parameters or ordinary folders. The
cause is the model's strongest input, url_ngram_score, which *averages* the
evidence of every 3-5 character sequence. Benign padding adds benign sequences
and dilutes the average; the malicious part is still there, just outnumbered.

An attacker can only add text around the payload, never make the payload itself
look innocent for free. So the hardened scorer looks at several *views* of the
same URL and keeps the most suspicious one:

    full         the URL as given
    no query     host + path, without ?query and #fragment
    one folder   host + one path segment, for each of the (up to) 4 longest segments

    hardened score = max(model score of each view)

Taking a maximum raises every score a little, benign ones included, so the
cut-offs are re-chosen on the validation sites exactly as train.py chose the
model's (best F1, and at most 1 % false alarms). red_team.py does that and saves
them to reports/red_team.json; this module only reads them. The saved model file
is never changed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import features as F

REPORT_PATH = Path(__file__).resolve().parent.parent / "reports" / "red_team.json"
MAX_SEGMENT_VIEWS = 4

# Named defence levels, from weakest to strongest (red_team.py measures each).
LEVELS = {
    "plain": "The model as trained: one score for the URL as given.",
    "no_query": "Also score the URL without its query string and fragment; keep the higher score.",
    "multi_view": "Also score the URL without its query, and the host with each of its longest "
                  "folders on its own; keep the highest score.",
}


def views(url: str, level: str = "multi_view") -> list[str]:
    """The texts the hardened scorer looks at for one URL (first = the URL itself)."""
    u = F.normalize_url(url)
    if level == "plain":
        return [u]
    userinfo, host, port, path, query, fragment = F.split_url(u)
    base = (userinfo + "@" if userinfo else "") + host + (":" + port if port else "")
    out = [u]
    if query or fragment:
        out.append(base + path)
    if level == "multi_view":
        segments = [s for s in path.split("/") if s]
        if len(segments) >= 2:
            for seg in sorted(segments, key=len, reverse=True)[:MAX_SEGMENT_VIEWS]:
                out.append(f"{base}/{seg}")
    seen, unique = set(), []
    for v in out:
        if v not in seen and len(v) >= 4:
            seen.add(v)
            unique.append(v)
    return unique


class HardenedScorer:
    """Same interface as UrlClassifier.score, with the max over views."""

    def __init__(self, clf, level: str = "multi_view", thresholds: dict | None = None):
        self.clf = clf
        self.level = level
        self.thresholds = thresholds if thresholds is not None else load_thresholds(level)

    def score(self, urls) -> np.ndarray:
        return self.score_views(urls)[0]

    def score_views(self, urls) -> tuple[np.ndarray, list[list[tuple[str, float]]]]:
        """Hardened scores, and every view with its own score (for the demo page)."""
        urls = list(urls)
        flat, owner = [], []
        for i, u in enumerate(urls):
            for v in views(u, self.level):
                flat.append(v)
                owner.append(i)
        scores = self.clf.score(flat) if flat else np.empty(0)
        out = np.zeros(len(urls))
        np.maximum.at(out, np.asarray(owner, dtype=int), scores)
        per_url: list[list[tuple[str, float]]] = [[] for _ in urls]
        for v, i, s in zip(flat, owner, scores):
            per_url[i].append((v, float(s)))
        return out, per_url


def load_thresholds(level: str = "multi_view") -> dict | None:
    """Cut-offs chosen on validation sites by red_team.py, or None if it has not run."""
    try:
        report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        return report["defences"][level]["thresholds"]
    except (OSError, ValueError, KeyError):
        return None
