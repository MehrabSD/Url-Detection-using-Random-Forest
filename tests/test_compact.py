"""The compact model (src/compact_forest.py) must behave exactly like the forest
and the gradient-boosted model train.py saved: same scores, same explanations. The
demo is served from it.

    .venv\\Scripts\\python.exe -m unittest discover -s tests -v

Needs models/malicious_url_rf.joblib and models/malicious_url_rf.npz.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

import adversary
import compact_forest
import explain
import features as F
from predict import MODEL_PATH, UrlClassifier

SEEDS = [
    "http://paypal.com.secure-login-verify.xyz/webscr?cmd=_login",
    "http://194.48.152.10/bins/x86.sh",
    "github.com/scikit-learn/scikit-learn",
    "https://accounts.google.com/signin/v2/identifier",
    "http://user:pw@example.co.uk:8080/a/b.php?x=1&y=22#top",
    "free-gift-card-winner.tk/claim.php",
    "en.wikipedia.org/wiki/Random_forest",
    "xn--pypal-4ve.com/signin",
]


def many_urls() -> list[str]:
    """The seeds plus every one-edit variant the red team would try: a few hundred
    varied URLs, so many different paths through the trees are exercised."""
    out = list(SEEDS)
    for url in SEEDS:
        first = [new for new, _move, _detail in adversary.candidates(F.normalize_url(url))]
        out += first
        for new in first:   # and one more edit deep
            out += [n for n, _m, _d in adversary.candidates(new)]
    return list(dict.fromkeys(out))


class CompactTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.full = UrlClassifier(MODEL_PATH)
        cls.small = UrlClassifier(compact_forest.COMPACT_PATH)
        cls.urls = many_urls()

    def test_probabilities_match_scikit_learn(self):
        X = self.full.featurise(self.urls)
        expected = self.full.model.predict_proba(X)
        got = self.small.model.predict_proba(X)
        self.assertGreater(len(self.urls), 200)
        np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)

    def test_booster_matches_scikit_learn(self):
        import joblib
        booster = joblib.load(MODEL_PATH)["booster"]     # the scikit-learn object
        X = self.full.featurise(self.urls)
        np.testing.assert_allclose(self.small.booster.predict_proba(X), booster.predict_proba(X),
                                   rtol=0, atol=1e-12)
        self.assertEqual(self.small.n_rounds, booster.n_iter_)

    def test_score_is_the_mean_of_the_two_halves(self):
        X = self.small.featurise(self.urls)
        parts = self.small.score_parts(X)
        self.assertEqual(set(parts), {"forest", "booster"})
        np.testing.assert_allclose(self.small.score(self.urls), (parts["forest"] + parts["booster"]) / 2,
                                   rtol=0, atol=1e-15)

    def test_whole_classifier_matches(self):
        np.testing.assert_allclose(self.small.score(self.urls), self.full.score(self.urls), rtol=0, atol=1e-12)
        self.assertEqual(self.small.thresholds, {k: float(v) for k, v in self.full.thresholds.items()})
        self.assertEqual(self.small.feature_names, list(self.full.feature_names))
        self.assertEqual(self.small.n_trees, self.full.n_trees)

    def test_explanations_match(self):
        a, b = explain.Explainer(self.full), explain.Explainer(self.small)
        self.assertAlmostEqual(a.baseline, b.baseline, places=12)
        for url in SEEDS:
            ra, rb = a.explain(url), b.explain(url)
            self.assertAlmostEqual(ra["score"], rb["score"], places=12)
            self.assertEqual(ra["contributions"], rb["contributions"])
            self.assertEqual(ra["tree_votes"], rb["tree_votes"])
            self.assertEqual(ra["halves"], rb["halves"])

    def test_compact_file_matches_the_trained_model(self):
        meta = compact_forest.load()
        self.assertEqual(meta["source_sha256"], compact_forest.file_sha256(MODEL_PATH),
                         "run `python src/compact_forest.py` after training")

    def test_server_prefers_the_compact_file_only_when_current(self):
        import serve
        self.assertEqual(serve.serving_path(), compact_forest.COMPACT_PATH)
        with mock.patch.object(serve, "file_sha256", return_value="0" * 64):
            self.assertEqual(serve.serving_path(), MODEL_PATH)   # stale export is not served


if __name__ == "__main__":
    unittest.main()
