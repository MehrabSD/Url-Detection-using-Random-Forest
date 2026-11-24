# Malicious URL detection (random forest) - mini project

## Layout
- `src/download_data.py` -> `data/raw/<dataset>/` (never edit raw files)
- `src/prepare_data.py` -> `data/processed/dataset.parquet` + `prepare_report.json`
- `src/train.py` -> `models/malicious_url_rf.joblib` + `reports/metrics.json`
- `src/make_figures.py` -> `reports/figures/*.png` (reads the JSON reports only)
- `src/predict.py` - CLI / `UrlClassifier` class for scoring URLs
- `src/evaluate_external.py` -> `reports/external_eval.json` (saved model scored on
  `data/raw/hannousse_phishing/`, a test-only dataset that must never be used for training)
- `src/red_team.py` -> `reports/red_team.json` (adversarial evasion test, calibration of the
  hardened scorer on the validation bucket, lookalike-layer evaluation; external data only,
  never the test buckets). Re-run it after train.py, and after editing `adversary.py`,
  `hardening.py` or `lookalike.py` (about 15 min).
- `src/adversary.py` (attacker edits + beam search), `src/hardening.py` (multi-view scorer),
  `src/lookalike.py` (rule-based brand-impersonation check, shown beside the score, never
  mixed into it), `src/triage.py` (finds and ranks every link in a pasted message)
- `src/features.py` (53 fixed lexical features), `src/learned_features.py` (3 learned features)
- `src/serve.py` - local demo page (`web/`, plain HTML/CSS/JS, no build step); `src/explain.py`
  explains one prediction; `src/describe.py` holds feature/mode labels and example URLs
- `tests/test_explain.py`, `tests/test_red_team.py` - run with `.venv\Scripts\python.exe -m unittest discover -s tests`
  (includes HTTP tests that start `serve.py` on a free port)
- `Dockerfile` + `requirements-serve.txt` (runtime-only deps) package the demo;
  `deploy/push_to_hf_space.py` uploads it to a Hugging Face Space. If `serve.py` starts
  reading a new file, add it to the Dockerfile's COPY lines and to `FIXED_FILES` in that script.
  Keep `scikit-learn` in `requirements-serve.txt` equal to the version that saved the model.

## Running
Use the project venv: `.venv\Scripts\python.exe src\<script>.py`.
Demo page: `.venv\Scripts\python.exe src\serve.py` (restart it after editing a `src/` file).
Full rebuild order: download_data -> prepare_data -> train -> evaluate_external -> red_team ->
make_figures (about 30 min total).

## Rules for this project
- **Keep `TECHNICAL_DOCUMENTATION.md` in sync.** Any change to features, data cleaning, the
  split, hyper-parameters or results must be reflected there, with numbers copied from
  `reports/metrics.json` (never from memory), and a line added to its change log.
- Evaluation is always on a **site-disjoint** split (`bucket` column = hash of the registered
  domain; buckets 0-1 test, 2 validation, 3-9 train). Do not report random-split numbers as
  the headline result - they are inflated (see the leakage experiment).
- The test buckets are used once, at the end of `train.py`. Tune on the validation bucket.
- Features must be computable from the URL string alone (no network lookups), and must be
  computed on the normalised URL (scheme and `www.` stripped) - the scheme is a dataset
  artifact, not a real signal.
- If `features.py` changes, re-run prepare_data, train and make_figures, in that order.
- The demo page must not hard-code features, thresholds or numbers: it reads them from the
  model bundle and the JSON reports through `serve.py`. When adding a feature, add its label
  to `src/describe.py` (it shows up without one, named after its column).
