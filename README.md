# Malicious URL detection with a random forest and gradient boosting

**Live demo: https://malicious-url-detector-33wl.onrender.com** (free plan: the first visit after
a quiet spell takes about a minute while it wakes up)

Classifies a URL as benign or malicious from its text alone (no site visit).
The model averages a random forest with a gradient-boosted model (chosen after comparing
stacking, KNN and SVM ensembles on validation and external data). Trained on 1.65 M URLs from
three public datasets; on websites never seen in training it reaches 88.7 % accuracy,
F1 0.830, ROC-AUC 0.946.

**It attacks itself.** A built-in red team searches for the cheapest realistic edits that
sneak a flagged link past the model. It found that one free edit (adding newsletter tracking
tags) beat the model on 83.1 % of caught phishing URLs; a multi-view scoring defence, with no
retraining, cuts that to 12.7 % (external-dataset false alarms 13.9 % → 13.1 %).
Around the model: a lookalike-domain check (typosquats, homoglyphs, punycode), whole-email
triage and QR-code scanning.

**Full explanation of the data, features, model and results, the red team, a from-scratch
build guide, deployment, and interview preparation:
[TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md)**

## Quick start (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

.\.venv\Scripts\python.exe src\download_data.py    # fetch datasets (~185 MB)
.\.venv\Scripts\python.exe src\prepare_data.py     # clean + extract features
.\.venv\Scripts\python.exe src\train.py            # train + evaluate (~14 min)
.\.venv\Scripts\python.exe src\evaluate_external.py # check on a fourth dataset
.\.venv\Scripts\python.exe src\red_team.py          # attack + defence + lookalike (~15 min)
.\.venv\Scripts\python.exe src\make_figures.py     # charts in reports/figures
```

## Scoring URLs

```powershell
.\.venv\Scripts\python.exe src\predict.py "http://paypal.com.secure-login-verify.xyz/webscr?cmd=_login"
.\.venv\Scripts\python.exe src\predict.py --file urls.txt --mode strict
.\.venv\Scripts\python.exe src\predict.py --explain "http://194.48.152.10/bins/x86.sh"
.\.venv\Scripts\python.exe src\predict.py --hardened "paypal.com.secure-login.xyz/webscr?utm_source=x"
.\.venv\Scripts\python.exe src\adversary.py "http://free-gift-card-winner.tk/claim.php"   # attack it
.\.venv\Scripts\python.exe src\lookalike.py paypa1.com xn--pypal-4ve.com                  # brand imitation
.\.venv\Scripts\python.exe src\triage.py message.eml                                      # every link in an email
```

From Python:

```python
import sys; sys.path.insert(0, "src")
from predict import UrlClassifier
clf = UrlClassifier()
print(clf.predict(["github.com/scikit-learn", "http://free-gift-card-winner.tk/claim.php"]))
```

## Demo page

Double-click `Start demo.bat` in the project folder. It starts the server and opens the page;
close its window to stop. Or from a terminal:

```powershell
.\.venv\Scripts\python.exe src\serve.py    # opens http://127.0.0.1:8000
```

Type a URL and the page scores it as you type, then shows why: which stretches of the address
look suspicious, what the forest and the booster each said, how each of the forest's 200 trees
voted, and how far each feature moved the score.
It flags brand imitations letter by letter and reads QR codes from a photo, a pasted screenshot
or the camera. *Try to sneak it past the model* attacks the URL live, against the model or the
hardened scorer; *Check a whole message* ranks every link in a pasted email and exports CSV.
Below that it shows the test results, the comparison with simpler models, the robustness
report and the model's known mistakes. Everything on the page is read from the saved model and the JSON reports, so after
re-running `train.py` it shows the new model with no change to the page. It works offline, and
a URL is never opened. `--port 9000` changes the port; `--no-open` skips opening the browser.

## Putting the demo online (free)

The demo runs from the `Dockerfile` on Render's free plan. The server loads a compact copy of
the model (`models/malicious_url_rf.npz`, made by `src/compact_forest.py`): identical
predictions, 304 MB peak memory instead of ~940 MB, so it fits the free 512 MB.

1. Push this repository to GitHub (the compact model is committed; the `.joblib` is not).
2. On render.com: **New → Blueprint**, pick the repository, **Apply** (`render.yaml` sets it up).
3. Open the address Render shows. Every `git push` redeploys.

The free service sleeps after 15 idle minutes and wakes in about a minute. Hugging Face Docker
Spaces now need PRO; `deploy/push_to_hf_space.py` still works for PRO accounts. Details:
section 19 of the technical documentation.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

## Files

| Path | Contents |
|---|---|
| `src/serve.py`, `web/` | demo page and its local server |
| `src/explain.py` | per-URL explanation used by the demo page |
| `src/describe.py` | plain-language labels for features and modes, and the example URLs |
| `tests/` | 42 tests: explanations add up exactly to the model's score, the red team scores with the real model, lookalike rules, triage, the compact model equals scikit-learn, and the server over HTTP |
| `src/adversary.py`, `src/hardening.py`, `src/red_team.py` → `reports/red_team.json` | the attacker, the multi-view defence, and the experiment measuring both |
| `src/lookalike.py`, `src/triage.py` | brand-impersonation rules; link extraction from whole messages |
| `Dockerfile`, `requirements-serve.txt`, `deploy/` | container for the demo and the Hugging Face upload script |
| `src/features.py` | 53 lexical features |
| `src/learned_features.py` | 3 features learned from training labels |
| `src/download_data.py`, `prepare_data.py`, `train.py`, `make_figures.py`, `predict.py` | pipeline steps |
| `models/malicious_url_rf.joblib` | trained model: forest + booster (90 MB, not in git) |
| `models/malicious_url_rf.npz`, `src/compact_forest.py` | the same model as flat arrays (41 MB), served by the demo |
| `render.yaml` | free hosting on Render |
| `reports/metrics.json`, `reports/figures/` | all results and charts |
| `src/evaluate_external.py` → `reports/external_eval.json` | the saved model scored on a fourth dataset it was not built from (86.9 % accuracy on unseen sites) |
