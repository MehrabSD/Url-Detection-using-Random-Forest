"""Checks for the added layers: lookalike check, red-team search, multi-view
hardening and inbox triage, plus their HTTP endpoints.

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

import adversary
import describe
import features as F
import hardening
import lookalike
import triage
from predict import UrlClassifier


def techniques(url: str) -> list[str]:
    return [f["technique"] for f in lookalike.check(url)["findings"]]


class LookalikeTest(unittest.TestCase):
    def test_each_impersonation_trick_is_named(self):
        cases = {
            "paypa1.com": "swap",                        # digit for a letter
            "rnicrosoft.com/login": "swap",              # rn reads as m
            "xn--pypal-4ve.com": "homoglyph",            # punycode for a Cyrillic 'а'
            "pаypal.com": "homoglyph",              # the same, typed directly
            "goggle.com": "typo",
            "micorsoft-support.com": "typo",             # typo inside a hyphenated name
            "amazon-account-verify.net": "combo",
            "paypal.com.secure-login.xyz/x": "subdomain",
            "paypal.com@evil.xyz": "userinfo",
            "evil.xyz/paypal/login": "path",
            "paypal.co": "ending",
        }
        for url, expected in cases.items():
            self.assertIn(expected, techniques(url), msg=url)

    def test_real_sites_and_ordinary_words_are_left_alone(self):
        for url in ("paypal.com/signin", "accounts.google.com/signin", "github.com/paypal",
                    "purchase.com", "stream.com", "applebees.com", "wikipedia.org",
                    "amazon.co.uk/gp/cart", "194.48.152.10/x.sh", "bbc.co.uk/news"):
            self.assertEqual(techniques(url), [], msg=url)

    def test_alignment_spells_both_names(self):
        ops = lookalike.edit_ops("paypa1", "paypal")
        self.assertEqual("".join(o[1] for o in ops), "paypa1")
        self.assertEqual("".join(o[2] for o in ops), "paypal")
        self.assertEqual([o[0] for o in ops].count("change"), 1)
        self.assertEqual(lookalike.distance("googel", "google"), 1)   # one transposition

    def test_punycode_is_shown_as_the_browser_would(self):
        r = lookalike.check("xn--pypal-4ve.com")
        self.assertEqual(r["display_host"], "pаypal.com")
        self.assertEqual(r["scripts"], ["Latin", "Cyrillic"])


class ModelLayersTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clf = UrlClassifier()
        cls.cut = cls.clf.thresholds["max_f1"]

    def test_every_candidate_is_a_valid_one_move_edit(self):
        u = F.normalize_url("http://paypal.com.secure-login-verify.xyz/webscr/a.php?cmd=_login#x")
        seen_moves = set()
        for new, move, _detail in adversary.candidates(u):
            self.assertIn(move, adversary.MOVES)
            self.assertNotEqual(new, u)
            seen_moves.add(move)
        for move in ("pad_tracking", "drop_query", "drop_fragment", "drop_extension",
                     "swap_tld", "drop_subdomains", "drop_path_word"):
            self.assertIn(move, seen_moves)

    def test_attack_path_is_scored_by_the_real_model(self):
        r = adversary.attack(self.clf, "http://paypal.com.secure-login-verify.xyz/webscr?cmd=_login", self.cut)
        self.assertTrue(r["found"])
        self.assertLess(r["final_score"], self.cut)
        # every step's score is exactly what the model says about that step's URL
        scores = self.clf.score([s["url"] for s in r["steps"]])
        for step, s in zip(r["steps"], scores):
            self.assertAlmostEqual(step["score"], float(s), places=6)
        self.assertEqual(r["total_price"], sum(s["price"] for s in r["steps"]))

    def test_a_url_that_already_passes_is_not_attacked(self):
        r = adversary.attack(self.clf, "github.com/scikit-learn/scikit-learn", self.cut)
        self.assertTrue(r["already_passes"])
        self.assertEqual(r["steps"], [])

    def test_hardened_score_is_the_highest_view(self):
        url = "paypal.com.secure-login-verify.xyz/blog/2024/webscr?cmd=_login&utm_source=x"
        views = hardening.views(url)
        self.assertEqual(views[0], F.normalize_url(url))
        self.assertIn("paypal.com.secure-login-verify.xyz/webscr", views)
        scorer = hardening.HardenedScorer(self.clf, thresholds={})
        hard, per_url = scorer.score_views([url])
        self.assertAlmostEqual(hard[0], float(np.max(self.clf.score(views))), places=6)
        self.assertEqual([v for v, _ in per_url[0]], views)
        self.assertGreaterEqual(hard[0], float(self.clf.score([url])[0]))

    def test_plain_level_is_the_model_itself(self):
        self.assertEqual(hardening.views("a.com/x/y?q=1", "plain"), ["a.com/x/y?q=1"])


class TriageTest(unittest.TestCase):
    def test_links_are_found_refanged_and_counted(self):
        out = triage.extract(describe.EXAMPLE_MESSAGE + "\nagain: hxxp://194.48.152[.]10/bins/invoice.exe")
        urls = {r["normalized"]: r for r in out["urls"]}
        self.assertIn("194.48.152.10/bins/invoice.exe", urls)               # defanged form read
        self.assertEqual(urls["194.48.152.10/bins/invoice.exe"]["count"], 2)
        self.assertIn("paypa1-support.com/restore/account?id=88217", urls)  # bare address
        self.assertNotIn("paypa1-support.com", urls)                        # e-mail sender is not a link
        self.assertNotIn("paypal.com/myaccount", urls)                      # visible link text is not a target

    def test_link_text_naming_another_site_is_reported(self):
        out = triage.extract('<a href="http://evil.xyz/login">https://www.paypal.com</a> '
                             '<a href="https://paypal.com/x">paypal.com</a>')
        self.assertEqual(len(out["mismatches"]), 1)
        self.assertEqual(out["mismatches"][0]["opens_site"], "evil.xyz")

    def test_file_names_and_trailing_punctuation(self):
        out = triage.extract("See report.pdf, then (https://en.wikipedia.org/wiki/A_(b)). Done: example.org.")
        found = [r["normalized"] for r in out["urls"]]
        self.assertIn("en.wikipedia.org/wiki/A_(b)", found)
        self.assertIn("example.org", found)
        self.assertNotIn("report.pdf", found)


class ServerTest(unittest.TestCase):
    """The new endpoints over real HTTP."""

    @classmethod
    def setUpClass(cls):
        import threading
        from http.server import ThreadingHTTPServer

        import serve

        class QuietHandler(serve.Handler):
            def log_message(self, *args):
                pass

        QuietHandler.model = serve.Model()
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), QuietHandler)
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def post(self, path, body):
        from urllib.error import HTTPError
        from urllib.request import Request, urlopen
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        req = Request(self.base + path, data=data, headers={"Content-Type": "application/json"})
        try:
            with urlopen(req, timeout=60) as r:
                return r.status, json.loads(r.read())
        except HTTPError as e:
            return e.code, json.loads(e.read())

    def test_score_carries_the_lookalike_check(self):
        status, body = self.post("/api/score", {"url": "paypa1.com/signin"})
        self.assertEqual(status, 200)
        self.assertEqual(body["lookalike"]["findings"][0]["technique"], "swap")

    def test_attack_endpoint(self):
        status, body = self.post("/api/attack", {"url": "http://paypal.com.secure-login-verify.xyz/webscr?cmd=_login",
                                                 "mode": "balanced", "defence": "plain"})
        self.assertEqual(status, 200)
        self.assertTrue(body["found"])
        self.assertEqual(body["mode"], "balanced")

    def test_attack_rejects_bad_input(self):
        for body in ({"url": 5}, {"url": "#"}, {"url": "a.com/x", "defence": "magic"}):
            status, answer = self.post("/api/attack", body)
            self.assertEqual(status, 400, msg=repr(body))
            self.assertIn("error", answer)

    def test_triage_endpoint(self):
        status, body = self.post("/api/triage", {"text": describe.EXAMPLE_MESSAGE})
        self.assertEqual(status, 200)
        self.assertTrue(body["mismatches"])
        top = body["urls"][0]
        self.assertTrue(top["hidden_target"] or top["lookalike"])

    def test_triage_needs_links(self):
        for body in ({"text": ""}, {"text": "no links here at all"}, {"text": None}, [1]):
            status, answer = self.post("/api/triage", body)
            self.assertEqual(status, 400, msg=repr(body))
            self.assertIn("error", answer)


if __name__ == "__main__":
    unittest.main()
