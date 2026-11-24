"""Score URLs with the trained model.

    python src/predict.py "http://paypal.com.secure-login.xyz/webscr?cmd=_login"
    python src/predict.py --file urls.txt            # one URL per line
    python src/predict.py --mode strict  <url> ...    # fewer false alarms
    python src/predict.py --explain <url>             # also print the feature values
    python src/predict.py --hardened <url>            # multi-view scoring (see hardening.py)

Modes (decision thresholds chosen on validation data during training):
    balanced  best F1 - the default
    strict    at most ~1 % of benign URLs flagged; misses more malicious ones
    default   plain 0.5 cut-off
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import numpy as np

import features as F
from compact_forest import CompactBooster

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "malicious_url_rf.joblib"
MODES = {"balanced": "max_f1", "strict": "low_fpr", "default": "default"}


class UrlClassifier:
    """The saved model: a random forest and a gradient-boosted model, whose
    probabilities are averaged. `path` is the bundle train.py wrote (.joblib,
    scikit-learn objects) or its compact export (.npz, see compact_forest.py);
    both predict alike."""

    def __init__(self, path: Path = MODEL_PATH):
        self.path = Path(path)
        if self.path.suffix == ".npz":
            from compact_forest import load
            bundle = load(self.path)
        else:
            bundle = joblib.load(self.path)
            bundle["model"].set_params(n_jobs=1, verbose=0)
        self.model = bundle["model"]
        booster = bundle.get("booster")
        if booster is not None and not isinstance(booster, CompactBooster):
            booster = CompactBooster.from_sklearn(booster)   # same predictions, flat arrays
        self.booster = booster
        self.learned = bundle["learned"]
        self.feature_names = bundle["feature_names"]
        self.thresholds = bundle["thresholds"]
        self.trained_at = bundle.get("trained_at")

    @property
    def n_trees(self) -> int:
        return int(self.model.n_estimators)

    def featurise(self, urls: list[str]) -> np.ndarray:
        norm = [F.normalize_url(u) for u in urls]
        return np.hstack([F.extract_many(norm, n_jobs=1), self.learned.transform(norm)])

    @property
    def n_rounds(self) -> int:
        return self.booster.n_rounds if self.booster is not None else 0

    def score_parts(self, X: np.ndarray) -> dict:
        """Each half's probability for already-featurised rows."""
        parts = {"forest": self.model.predict_proba(X)[:, 1]}
        if self.booster is not None:
            parts["booster"] = self.booster.predict_proba(X)[:, 1]
        return parts

    def score_features(self, X: np.ndarray) -> np.ndarray:
        """The model's score: the mean of the forest's and the booster's probabilities
        (just the forest's for a bundle saved before the booster was added)."""
        return np.mean(list(self.score_parts(X).values()), axis=0)

    def score(self, urls: list[str]) -> np.ndarray:
        """Probability-like score in [0, 1]; higher = more likely malicious."""
        return self.score_features(self.featurise(urls))

    def predict(self, urls: list[str], mode: str = "balanced") -> list[dict]:
        thr = self.thresholds[MODES[mode]]
        return [
            {"url": u, "score": float(s), "verdict": "MALICIOUS" if s >= thr else "benign"}
            for u, s in zip(urls, self.score(urls))
        ]


def main() -> int:
    ap = argparse.ArgumentParser(description="Classify URLs as benign or malicious.")
    ap.add_argument("urls", nargs="*", help="URLs to score")
    ap.add_argument("--file", help="text file with one URL per line")
    ap.add_argument("--mode", choices=list(MODES), default="balanced")
    ap.add_argument("--explain", action="store_true", help="print non-zero feature values")
    ap.add_argument("--hardened", action="store_true",
                    help="score with the padding-resistant multi-view scorer (needs reports/red_team.json)")
    args = ap.parse_args()

    urls = list(args.urls)
    if args.file:
        urls += [ln.strip() for ln in Path(args.file).read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not urls:
        ap.error("give at least one URL or --file")
    if not MODEL_PATH.exists():
        print("Model not found - run `python src/train.py` first.", file=sys.stderr)
        return 1

    clf = UrlClassifier()
    thr = clf.thresholds[MODES[args.mode]]
    if args.hardened:
        from hardening import HardenedScorer, load_thresholds
        hard_thr = load_thresholds()
        if hard_thr is None:
            print("Run `python src/red_team.py` first to calibrate the hardened scorer.", file=sys.stderr)
            return 1
        thr = hard_thr[MODES[args.mode]]
        print(f"mode = {args.mode}, hardened (flag as malicious when score >= {thr:.3f})\n")
        for u, s in zip(urls, HardenedScorer(clf, thresholds=hard_thr).score(urls)):
            print(f"{'MALICIOUS' if s >= thr else 'benign':<10} {s:.3f}  {u}")
        return 0
    print(f"mode = {args.mode} (flag as malicious when score >= {thr:.3f})\n")
    X = clf.featurise(urls) if args.explain else None
    for i, r in enumerate(clf.predict(urls, args.mode)):
        print(f"{r['verdict']:<10} {r['score']:.3f}  {r['url']}")
        if args.explain:
            shown = [f"{n}={v:.3g}" for n, v in zip(clf.feature_names, X[i]) if v]
            print("           " + ", ".join(shown) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
