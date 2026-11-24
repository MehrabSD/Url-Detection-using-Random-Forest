"""Checks that the explanations shown in the demo page are exact, not approximate.

    .venv\\Scripts\\python.exe -m unittest discover -s tests -v

Needs the trained model (models/malicious_url_rf.joblib).
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

import describe
import explain
import features as F
from predict import UrlClassifier

URLS = [
    "http://paypal.com.secure-login-verify.xyz/webscr?cmd=_login",
    "http://194.48.152.10/bins/x86.sh",
    "github.com/scikit-learn/scikit-learn",
    "https://accounts.google.com/signin/v2/identifier",
    "http://user:pw@example.co.uk:8080/a/b.php?x=1&y=22#top",
    "a.io",
]


class ExplainTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clf = UrlClassifier()
        cls.explainer = explain.Explainer(cls.clf)

    def test_contributions_add_up_to_the_score(self):
        scores = self.clf.score(URLS)
        for url, score in zip(URLS, scores):
            result = self.explainer.explain(url)
            total = result["baseline"] + sum(c["contribution"] for c in result["contributions"])
            self.assertAlmostEqual(result["score"], float(score), places=6, msg=url)
            self.assertAlmostEqual(total, float(score), places=5, msg=url)

    def test_one_contribution_per_model_feature(self):
        result = self.explainer.explain(URLS[0])
        names = [c["name"] for c in result["contributions"]]
        self.assertEqual(sorted(names), sorted(self.clf.feature_names))

    def test_tree_votes_average_to_the_score(self):
        result = self.explainer.explain(URLS[0])
        self.assertEqual(len(result["tree_votes"]), self.clf.n_trees)
        self.assertAlmostEqual(float(np.mean(result["tree_votes"])), result["score"], places=3)

    def test_character_heat_adds_up_to_the_pattern_score(self):
        for url in URLS:
            norm = F.normalize_url(url)
            heat = explain.character_heat(self.clf.learned, norm)
            expected = float(self.clf.learned.transform([norm])[0, 2])
            self.assertEqual(len(heat), len(norm), msg=url)
            self.assertAlmostEqual(sum(heat), expected, places=4, msg=url)

    def test_parts_match_the_feature_code(self):
        for url in URLS:
            norm = F.normalize_url(url)
            found = {p["kind"]: norm[p["start"]:p["end"]] for p in explain.url_parts(norm)}
            userinfo, host, port, path, query, fragment = F.split_url(norm)
            expected = {"userinfo": userinfo, "host": host, "port": port, "path": path,
                        "query": query, "fragment": fragment}
            self.assertEqual(found, {k: v for k, v in expected.items() if v}, msg=url)

    def test_verdict_follows_each_threshold(self):
        result = self.explainer.explain(URLS[0])
        for mode in describe.modes(self.clf.thresholds):
            flagged = result["score"] >= mode["threshold"]
            self.assertEqual(result["verdicts"][mode["id"]], "malicious" if flagged else "benign")

    def test_result_is_plain_json(self):
        json.dumps(self.explainer.explain(URLS[4]), allow_nan=False)

    def test_blank_url_is_rejected(self):
        with self.assertRaises(ValueError):
            self.explainer.explain("   ")

    def test_text_the_training_data_dropped_as_junk_is_rejected(self):
        # the same rule as prepare_data.py step 2: spaces, no dot, under 4 characters
        for junk in ("#", "...", "abc", "github", "a  b   c.com", "http://www.a.b", "bad\x07.com"):
            with self.assertRaises(ValueError, msg=junk):
                self.explainer.explain(junk)


class DescribeTest(unittest.TestCase):
    def test_every_current_feature_is_described(self):
        for name in F.FEATURE_NAMES + ["tld_malicious_rate", "host_ngram_score", "url_ngram_score"]:
            info = describe.feature(name)
            self.assertNotEqual(info["group"], "other", msg=name)
            self.assertTrue(info["description"], msg=name)

    def test_unknown_feature_still_gets_a_readable_label(self):
        info = describe.feature("n_emoji_in_host")
        self.assertEqual(info["label"], "N emoji in host")
        self.assertEqual(info["group"], "other")

    def test_unknown_threshold_becomes_a_mode(self):
        modes = describe.modes({"default": 0.5, "max_f1": 0.42, "high_recall": 0.2,
                                "low_fpr_target": 0.01})
        ids = [m["id"] for m in modes]
        self.assertIn("high_recall", ids)       # new threshold shows up on its own
        self.assertNotIn("low_fpr_target", ids)  # a setting, not a threshold


class ServerTest(unittest.TestCase):
    """The demo server, spoken to over real HTTP on a free port."""

    @classmethod
    def setUpClass(cls):
        import threading
        from http.server import ThreadingHTTPServer

        import serve

        class QuietHandler(serve.Handler):
            def log_message(self, *args):  # the 4xx answers below are expected
                pass

        QuietHandler.model = serve.Model()
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), QuietHandler)
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def request(self, path, body=None):
        from urllib.error import HTTPError
        from urllib.request import Request, urlopen
        data = body if body is None or isinstance(body, bytes) else json.dumps(body).encode()
        req = Request(self.base + path, data=data, headers={"Content-Type": "application/json"})
        try:
            with urlopen(req, timeout=30) as r:
                return r.status, r.headers, r.read()
        except HTTPError as e:
            return e.code, e.headers, e.read()

    def test_page_and_scripts_are_served_with_the_right_types(self):
        for path, kind in (("/", "text/html"), ("/app.js", "text/javascript"),
                           ("/style.css", "text/css"), ("/fonts/archivo.woff2", "font/woff2")):
            status, headers, _ = self.request(path)
            self.assertEqual(status, 200, msg=path)
            self.assertTrue(headers["Content-Type"].startswith(kind), msg=path)
            self.assertEqual(headers["X-Content-Type-Options"], "nosniff", msg=path)

    def test_folders_are_not_listed_and_files_outside_web_are_unreachable(self):
        for path in ("/fonts/", "/../src/serve.py", "/%2e%2e/src/serve.py"):
            self.assertEqual(self.request(path)[0], 404, msg=path)

    def test_state_describes_the_saved_model(self):
        status, _, body = self.request("/api/state")
        state = json.loads(body)
        self.assertEqual(status, 200)
        self.assertEqual(state["model"]["features"], len(state["features"]))
        self.assertTrue(state["modes"])

    def test_score_answers_a_url(self):
        status, _, body = self.request("/api/score", {"url": URLS[1]})
        self.assertEqual(status, 200)
        self.assertGreaterEqual(json.loads(body)["score"], 0.5)

    def test_score_rejects_anything_but_a_url_string(self):
        for body in ({"url": None}, {"url": 123}, {"url": ["a.com"]}, [1, 2], "a.com",
                     b"not json", b'{"url": "\xff"}', {"url": "#"}):
            status, _, answer = self.request("/api/score", body)
            self.assertEqual(status, 400, msg=repr(body))
            self.assertIn("error", json.loads(answer), msg=repr(body))


if __name__ == "__main__":
    unittest.main()
