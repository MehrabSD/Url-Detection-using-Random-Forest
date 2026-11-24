# Malicious URL detection with a random forest

Classifies a URL as benign or malicious from its text alone (no site visit).
Trained on 1.65 M URLs from three public datasets; on websites never seen in training it
reaches 88 % accuracy, F1 0.82, ROC-AUC 0.94.

**It attacks itself.** A built-in red team searches for the cheapest realistic edits that
sneak a flagged link past the model. It found that one free edit (adding newsletter tracking
tags) beat the model on 82.1 % of caught phishing URLs; a multi-view scoring defence, with no
retraining, cuts that to 11.1 % and lowers false alarms on an external dataset (14.5 % → 12.1 %).
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
look suspicious, how each of the 200 trees voted, and how far each feature moved the score.
It flags brand imitations letter by letter and reads QR codes from a photo, a pasted screenshot
or the camera. *Try to sneak it past the model* attacks the URL live, against the model or the
hardened scorer; *Check a whole message* ranks every link in a pasted email and exports CSV.
Below that it shows the test results, the comparison with simpler models, the robustness
report and the model's known mistakes. Everything on the page is read from the saved model and the JSON reports, so after
re-running `train.py` it shows the new model with no change to the page. It works offline, and
a URL is never opened. `--port 9000` changes the port; `--no-open` skips opening the browser.

## Putting the demo online

The demo runs in a container (`Dockerfile`) and can be published to a free Hugging Face Space:

```powershell
.\.venv\Scripts\python.exe -m pip install -r deploy\requirements-deploy.txt
.\.venv\Scripts\hf.exe auth login                        # once; needs a "write" token
.\.venv\Scripts\python.exe deploy\push_to_hf_space.py    # -> huggingface.co/spaces/<you>/malicious-url-detector
```

The server needs about 1 GB of memory, so 512 MB free tiers are too small. Details,
alternatives and updating after a retrain: section 19 of the technical documentation.

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
| `tests/` | 34 tests: explanations add up exactly to the model's score, the red team scores with the real model, lookalike rules, triage, and the server over HTTP |
| `src/adversary.py`, `src/hardening.py`, `src/red_team.py` → `reports/red_team.json` | the attacker, the multi-view defence, and the experiment measuring both |
| `src/lookalike.py`, `src/triage.py` | brand-impersonation rules; link extraction from whole messages |
| `Dockerfile`, `requirements-serve.txt`, `deploy/` | container for the demo and the Hugging Face upload script |
| `src/features.py` | 53 lexical features |
| `src/learned_features.py` | 3 features learned from training labels |
| `src/download_data.py`, `prepare_data.py`, `train.py`, `make_figures.py`, `predict.py` | pipeline steps |
| `models/malicious_url_rf.joblib` | trained model (88 MB) |
| `reports/metrics.json`, `reports/figures/` | all results and charts |
| `src/evaluate_external.py` → `reports/external_eval.json` | the saved model scored on a fourth dataset it was not built from (86.6 % accuracy on unseen sites) |
