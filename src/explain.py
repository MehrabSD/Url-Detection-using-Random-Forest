"""Explain one prediction of the saved model, exactly.

Used by serve.py for the demo page. Three things are computed for a URL:

    contributions   how far each feature moved the score. The score is the mean of
                    two tree models' probabilities (train.py):
                    forest   follow the URL down every tree; each split changes
                             the share of malicious training URLs in the current
                             node, and that change is credited to the feature the
                             split tested (Saabas' method).
                    booster  the same walk through every boosting round, in
                             log-odds: each split changes the node's value. The
                             log-odds changes are then rescaled by one factor so
                             they add up to the booster's change in probability.
                    Averaged, with no approximation (tests/test_explain.py checks):
                        score = baseline + sum(contributions)
    tree_votes      each forest tree's own answer, so the page can show the vote.
    halves          the forest's and the booster's probabilities, whose mean is the score.
    character heat  the URL n-gram score split back onto the characters that
                    produced it, to show which stretches look malicious.

Nothing here contacts the URL: like the model, it reads the text only.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

import describe
from compact_forest import CompactForest
import features as F

MAX_URL_LENGTH = 2000
_WHITE_SPACE = re.compile(r"\s\s+")  # same collapsing scikit-learn's char analyser does
# prepare_data.py (step 2) dropped training rows that, once normalised, contain
# whitespace or control characters, have no dot, or are under 4 characters. The
# model never saw such text, so its score for one would mean nothing.
_JUNK_CHARS = re.compile(r"[\s\x00-\x1f\x7f]")
_MIN_LENGTH = 4
_URL_NGRAMS = (3, 5)                 # must match learned_features.LearnedFeatures._vectorise
_HEAT_FULL_SCALE = 5.0               # a full-height bar, in standard deviations of the n-gram weights


def _sigmoid(z: float) -> float:
    return float(1.0 / (1.0 + np.exp(-z)))


def _ngram_count(length: int) -> int:
    lo, hi = _URL_NGRAMS
    return sum(max(0, length - n + 1) for n in range(lo, hi + 1))


def url_parts(u: str) -> list[dict]:
    """Where userinfo, host, port, path, query and fragment sit in a normalised URL.

    Mirrors features.split_url, but returns character positions.
    """
    hash_at = u.find("#")
    rest_end = hash_at if hash_at != -1 else len(u)
    question_at = u.find("?", 0, rest_end)
    path_end = question_at if question_at != -1 else rest_end
    slash_at = u.find("/", 0, path_end)
    authority_end = slash_at if slash_at != -1 else path_end

    authority = u[:authority_end]
    at = authority.rfind("@")
    host_start = at + 1 if at != -1 else 0
    host_end = authority_end
    port_span = None
    host_and_port = authority[host_start:]
    if not host_and_port.startswith("["):
        host, sep, port = host_and_port.rpartition(":")
        if sep and port.isdigit():
            host_end = host_start + len(host)
            port_span = (host_end + 1, authority_end)

    spans = [("userinfo", 0, at if at != -1 else 0), ("host", host_start, host_end)]
    if port_span:
        spans.append(("port", *port_span))
    spans.append(("path", authority_end, path_end))
    if question_at != -1:
        spans.append(("query", question_at + 1, rest_end))
    if hash_at != -1:
        spans.append(("fragment", hash_at + 1, len(u)))
    return [{"kind": k, "start": a, "end": b} for k, a, b in spans if b > a]


def character_heat(learned, u: str) -> list[float] | None:
    """Split the URL n-gram score onto the characters of the normalised URL `u`.

    Positive = that stretch looks like malicious training URLs, negative = benign.
    The values add up to the model's url_ngram_score for this URL.
    """
    weights = getattr(learned, "w_url", None)
    if weights is None:
        return None
    text = _WHITE_SPACE.sub(" ", u.lower())
    if len(text) != len(u):
        return None  # positions would no longer line up
    lo, hi = _URL_NGRAMS
    grams, spans = [], []
    for n in range(lo, min(hi, len(text)) + 1):
        for i in range(len(text) - n + 1):
            grams.append(text[i:i + n])
            spans.append((i, n))
    heat = np.zeros(len(text))
    if grams:
        hasher = HashingVectorizer(analyzer=lambda gram: [gram], n_features=len(weights),
                                   alternate_sign=False, norm=None)
        columns = hasher.transform(grams).indices
        per_gram = weights[columns] / (len(grams) + 1.0)
        for (start, n), w in zip(spans, per_gram):
            heat[start:start + n] += w / n
    return [round(float(h), 6) for h in heat]


class Explainer:
    def __init__(self, clf):
        self.clf = clf
        self.modes = describe.modes(clf.thresholds)
        # All trees as flat arrays: split feature, threshold, children and the share
        # of malicious training URLs in every node (compact_forest.py). A compact
        # model already is one; a scikit-learn forest is converted once.
        model = clf.model
        self._forest = model if isinstance(model, CompactForest) else CompactForest.from_sklearn(model)
        self.forest_baseline = float(np.mean(self._forest.share[self._forest.roots]))
        # the booster (UrlClassifier keeps it as a CompactBooster) starts every URL at
        # its baseline log-odds plus each round's root value
        self._booster = getattr(clf, "booster", None)
        if self._booster is not None:
            b = self._booster
            self._booster_start = b.baseline + float(b.value[b.roots].sum())
            self.booster_baseline = _sigmoid(self._booster_start)
            self.baseline = (self.forest_baseline + self.booster_baseline) / 2
        else:
            self.baseline = self.forest_baseline
        weights = getattr(clf.learned, "w_url", None)
        self._heat_scale = _HEAT_FULL_SCALE * float(np.std(weights)) if weights is not None else 0.0

    def _walk(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        f = self._forest
        feature, threshold, left, right, share = f.feature, f.threshold, f.left, f.right, f.share
        moved = np.zeros(len(x))
        votes = np.empty(f.n_estimators)
        for i, root in enumerate(f.roots):
            node = int(root)
            while left[node] != node:  # a leaf points to itself
                feat = feature[node]
                child = int(left[node] if x[feat] <= threshold[node] else right[node])
                moved[feat] += share[child] - share[node]
                node = child
            votes[i] = share[node]
        return moved / f.n_estimators, votes

    def _walk_booster(self, x: np.ndarray) -> tuple[np.ndarray, float]:
        """Log-odds moved by each feature over all boosting rounds, and the log-odds."""
        b = self._booster
        x = np.asarray(x, dtype=np.float32).astype(np.float64)   # as CompactBooster.raw
        moved = np.zeros(len(x))
        node = b.roots.astype(np.int64)
        for _ in range(b.max_depth):          # all rounds step down together
            feat = b.feature[node]
            child = np.where(b.go_left(x[feat], node), b.left[node], b.right[node])
            np.add.at(moved, feat, b.value[child] - b.value[node])   # 0 once at a leaf
            node = child
        return moved, self._booster_start + float(moved.sum())

    def explain(self, url: str) -> dict:
        url = str(url).strip()
        if not url:
            raise ValueError("Enter a URL to check.")
        if len(url) > MAX_URL_LENGTH:
            raise ValueError(f"That URL is longer than {MAX_URL_LENGTH} characters.")
        norm = F.normalize_url(url)
        if not norm:
            raise ValueError("Enter a URL to check.")
        if _JUNK_CHARS.search(norm):
            raise ValueError("A web address cannot contain spaces or control characters.")
        if len(norm) < _MIN_LENGTH or "." not in norm:
            raise ValueError("That does not look like a web address yet. Type a full one, such as example.com.")

        x = self.clf.featurise([url])[0]
        moved, votes = self._walk(x)
        forest_score = float(votes.mean())
        score, halves = forest_score, None
        if self._booster is not None:
            log_moved, log_odds = self._walk_booster(x)
            booster_score = _sigmoid(log_odds)
            change = log_odds - self._booster_start
            # one factor turns the log-odds changes into probability changes that add up
            # exactly to booster_score - booster_baseline
            factor = ((booster_score - self.booster_baseline) / change if abs(change) > 1e-12
                      else self.booster_baseline * (1 - self.booster_baseline))
            moved = (moved + factor * log_moved) / 2
            score = (forest_score + booster_score) / 2
            halves = {"forest": forest_score, "booster": booster_score,
                      "forest_baseline": self.forest_baseline,
                      "booster_baseline": self.booster_baseline,
                      "trees": int(self._forest.n_estimators), "rounds": int(self._booster.n_rounds)}

        contributions = [
            {"name": name, "value": round(float(v), 6), "contribution": round(float(c), 6)}
            for name, v, c in zip(self.clf.feature_names, x, moved)
        ]
        contributions.sort(key=lambda c: -abs(c["contribution"]))

        site = F.registered_domain(norm)
        host = next((p for p in url_parts(norm) if p["kind"] == "host"), None)
        site_span = None
        if host and site and norm[host["start"]:host["end"]].endswith(site):
            site_span = {"start": host["end"] - len(site), "end": host["end"]}

        heat = character_heat(self.clf.learned, norm)
        if heat is not None:
            # The score divides by the number of n-grams, which would make long URLs
            # look faint. Undo that for display and put every URL on one fixed scale.
            undo = (_ngram_count(len(norm)) + 1.0) / self._heat_scale if self._heat_scale else 0.0
            heat = {"total": round(sum(heat), 6),
                    "strength": [round(max(-1.0, min(1.0, h * undo)), 4) for h in heat]}

        return {
            "input": url,
            "normalized": norm,
            "parts": url_parts(norm),
            "site": site,
            "site_span": site_span,
            "score": score,
            "baseline": self.baseline,
            "verdicts": {m["id"]: "malicious" if score >= m["threshold"] else "benign"
                         for m in self.modes},
            "tree_votes": [round(float(v), 4) for v in votes],
            "halves": halves,
            "contributions": contributions,
            "heat": heat,
        }
