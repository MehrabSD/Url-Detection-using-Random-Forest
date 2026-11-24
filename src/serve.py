"""Demo page for the trained model.

    python src/serve.py                 # opens http://127.0.0.1:8000
    python src/serve.py --port 9000 --no-open
    python src/serve.py --host 0.0.0.0 --no-open   # inside a container (see Dockerfile)

The page (web/) hard-codes nothing about the model. Every time it loads it asks
this server for the feature list, thresholds and reports, so after re-running
train.py the page shows the new model without any change to the frontend. The
server also notices a new model file while it is running and reloads it.

Only the Python standard library is used for serving. By default it listens on
this computer only; `--host 0.0.0.0` (or the HOST variable) opens it to other
machines, which is what a hosting platform needs. The port can also come from
the PORT variable, which most platforms set. URLs are scored from their text;
nothing is ever fetched.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import webbrowser
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import adversary
import describe
import explain
import hardening
import lookalike
import triage
from predict import MODEL_PATH, UrlClassifier

ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT / "web"
REPORTS = {
    "metrics": ROOT / "reports" / "metrics.json",
    "external": ROOT / "reports" / "external_eval.json",
    "prepare": ROOT / "data" / "processed" / "prepare_report.json",
    "red_team": hardening.REPORT_PATH,
}
MAX_BODY_BYTES = 16_384
MAX_TRIAGE_BYTES = 256_000   # a long email in HTML source
# The page loads nothing from other sites, so the browser may refuse anything that tries.
CONTENT_SECURITY_POLICY = ("default-src 'self'; img-src 'self' data:; object-src 'none'; "
                           "base-uri 'none'; form-action 'self'")
# Explicit types: on Windows, Python otherwise asks the registry, which on some
# machines maps .js to text/plain - and browsers refuse to run a module served so.
CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
    ".woff2": "font/woff2",
    ".txt": "text/plain; charset=utf-8",
}


def _stamp(path: Path) -> tuple[int, int] | None:
    try:
        st = path.stat()
        return st.st_mtime_ns, st.st_size
    except OSError:
        return None


class Model:
    """The loaded model, re-read when the file on disk changes."""

    def __init__(self):
        self._lock = threading.Lock()
        self._stamp = None
        self.clf = None
        self.explainer = None
        self.load_error = None
        self._load(_stamp(MODEL_PATH))

    def _load(self, stamp) -> None:
        clf = UrlClassifier()
        self.clf, self.explainer = clf, explain.Explainer(clf)
        self._stamp, self.load_error = stamp, None

    def current(self) -> tuple[UrlClassifier, explain.Explainer]:
        stamp = _stamp(MODEL_PATH)
        if stamp is not None and stamp != self._stamp:
            with self._lock:
                if stamp != self._stamp:
                    try:
                        self._load(stamp)
                        print(f"  reloaded model ({datetime.now():%H:%M:%S})", flush=True)
                    except Exception as exc:  # file still being written by train.py
                        self.load_error = str(exc)
        return self.clf, self.explainer

    def version(self) -> str:
        self.current()
        stamps = [self._stamp] + [_stamp(p) for p in REPORTS.values()]
        return "-".join(str(s[0]) if s else "none" for s in stamps)


def _read_report(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def state(model: Model) -> dict:
    clf, explainer = model.current()
    modes = explainer.modes
    bundle_time = getattr(clf, "trained_at", None)
    return {
        "version": model.version(),
        "model": {
            "trees": len(clf.model.estimators_),
            "features": len(clf.feature_names),
            "file": MODEL_PATH.name,
            "file_modified": datetime.fromtimestamp(MODEL_PATH.stat().st_mtime).isoformat(timespec="seconds"),
            "trained_at": bundle_time,
        },
        "modes": modes,
        "default_mode": "balanced" if any(m["id"] == "balanced" for m in modes) else modes[0]["id"],
        "baseline": explainer.baseline,
        "features": [describe.feature(name) for name in clf.feature_names],
        "groups": describe.GROUPS,
        "examples": describe.EXAMPLES,
        "example_message": describe.EXAMPLE_MESSAGE,
        "attack": adversary.catalogue(),
        "defences": [{"id": level, "description": text,
                      "ready": level == "plain" or hardening.load_thresholds(level) is not None}
                     for level, text in hardening.LEVELS.items()],
        "heat_feature": "url_ngram_score" if getattr(clf.learned, "w_url", None) is not None else None,
        "reports": {name: _read_report(path) for name, path in REPORTS.items()},
    }


class Handler(SimpleHTTPRequestHandler):
    model: Model  # set in main()
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map, **CONTENT_TYPES}
    timeout = 30                # drop connections that stall instead of holding a thread forever
    server_version = "url-demo"
    sys_version = ""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def log_message(self, fmt, *args):  # keep the terminal quiet during a demo
        if args and str(args[1])[:1] in "45":
            super().log_message(fmt, *args)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", CONTENT_SECURITY_POLICY)
        super().end_headers()

    def list_directory(self, path):  # never show folder listings
        self.send_error(404, "File not found")
        return None

    def _json(self, payload, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/api/state":
            self._json(state(self.model))
        elif path == "/api/version":
            self._json({"version": self.model.version()})
        elif path.startswith("/api/"):
            self._json({"error": "Unknown address."}, 404)
        else:
            super().do_GET()

    def _body(self, limit: int) -> dict | None:
        """The request's JSON object, or None after answering with an error."""
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if not 0 < length <= limit:
            self._json({"error": "Send a JSON object no larger than "
                                 f"{limit // 1000} kB."}, 400 if length <= limit else 413)
            return None
        try:
            body = json.loads(self.rfile.read(length).decode("utf-8"))
        except ValueError:  # includes bad JSON and bad UTF-8
            body = None
        if not isinstance(body, dict):
            self._json({"error": "Send a JSON object."}, 400)
            return None
        return body

    def _text(self, body: dict, key: str) -> str | None:
        value = body.get(key, "")
        if not isinstance(value, str):
            self._json({"error": f"Send a JSON object with a {key}."}, 400)
            return None
        return value

    def do_POST(self):
        route = {"/api/score": (self._score, MAX_BODY_BYTES),
                 "/api/attack": (self._attack, MAX_BODY_BYTES),
                 "/api/triage": (self._triage, MAX_TRIAGE_BYTES)}.get(self.path.split("?", 1)[0])
        if route is None:
            return self._json({"error": "Unknown address."}, 404)
        handler, limit = route
        body = self._body(limit)
        if body is None:
            return
        started = time.perf_counter()
        try:
            result = handler(body)
        except ValueError as exc:
            return self._json({"error": str(exc) or "That could not be read as a URL."}, 400)
        if result is not None:
            result["milliseconds"] = round((time.perf_counter() - started) * 1000, 1)
            self._json(result)

    def _score(self, body: dict) -> dict | None:
        url = self._text(body, "url")
        if url is None:
            return None
        result = self.model.current()[1].explain(url)
        result["lookalike"] = lookalike.check(result["normalized"])
        return result

    def _attack(self, body: dict) -> dict | None:
        """Red team: the cheapest edits that take this URL under the chosen cut-off."""
        url = self._text(body, "url")
        if url is None:
            return None
        clf, explainer = self.model.current()
        explainer.explain(url)            # same input checks as scoring
        mode = next((m for m in explainer.modes if m["id"] == body.get("mode")), explainer.modes[0])
        level = body.get("defence", "plain")
        if level not in hardening.LEVELS:
            raise ValueError("Unknown defence.")
        if level == "plain":
            scorer, threshold = clf, mode["threshold"]
        else:
            thresholds = hardening.load_thresholds(level)
            if thresholds is None or mode["key"] not in thresholds:
                raise ValueError("Run src/red_team.py first to calibrate the hardened scorer.")
            scorer = hardening.HardenedScorer(clf, level, thresholds)
            threshold = float(thresholds[mode["key"]])
        result = adversary.attack(scorer, url, threshold)
        result.update(mode=mode["id"], defence=level)
        if level != "plain":
            # which view of the URL gave the hardened score, before and after
            shown = [result["url"]] + ([result["final_url"]] if result.get("final_url") else [])
            _, per_url = scorer.score_views(shown)
            result["views"] = [[{"text": v, "score": s} for v, s in views] for views in per_url]
        return result

    def _triage(self, body: dict) -> dict | None:
        text = self._text(body, "text")
        if text is None:
            return None
        if not text.strip():
            raise ValueError("Paste a message to check.")
        clf, explainer = self.model.current()
        thresholds = {m["id"]: m["threshold"] for m in explainer.modes}
        result = triage.triage(clf, text, thresholds, lookalike.check)
        if not result["urls"]:
            raise ValueError("No web addresses found in that text.")
        return result


def main() -> int:
    ap = argparse.ArgumentParser(description="Serve the demo page for the trained model.")
    ap.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"),
                    help="address to listen on (default 127.0.0.1, this computer only; "
                         "0.0.0.0 accepts other machines, as a container needs)")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    ap.add_argument("--no-open", action="store_true", help="do not open the browser")
    args = ap.parse_args()

    if not MODEL_PATH.exists():
        print("Model not found - run `python src/train.py` first.", file=sys.stderr)
        return 1
    print("Loading model ...", flush=True)
    Handler.model = Model()
    try:
        server = ThreadingHTTPServer((args.host, args.port), Handler)
    except OSError:
        print(f"Port {args.port} is in use - try `--port {args.port + 1}`.", file=sys.stderr)
        return 1
    address = f"http://127.0.0.1:{args.port}"
    listening = "" if args.host == "127.0.0.1" else f"   listening on {args.host}:{args.port}"
    print(f"Demo page: {address}{listening}   (Ctrl+C to stop)", flush=True)
    if not args.no_open:
        threading.Timer(0.5, webbrowser.open, args=(address,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
