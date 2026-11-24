"""Red-team the saved model, measure a defence, and check the lookalike layer.

    python src/red_team.py              # about 15 minutes
    python src/red_team.py --sample 100 # quicker, noisier

Reads   models/malicious_url_rf.joblib
        data/processed/dataset.parquet           (validation bucket only)
        data/raw/hannousse_phishing/*.parquet    (the external, test-only dataset)
Writes  reports/red_team.json

Three experiments, none of which fits or changes the model:

1. Calibrate the defences (hardening.py). Each hardened scorer gets its own
   cut-offs on the validation sites, by the same rules train.py used for the
   model: best F1 ("balanced") and at most 1 % false alarms ("strict").
2. Clean accuracy and attack. On external URLs from sites the model never
   trained on, measure each scorer's ordinary accuracy, then take a fixed random
   sample of the phishing URLs and let adversary.py try to sneak every one of them
   past each scorer, recording the cheapest successful price.
3. Lookalike layer (lookalike.py): how often it fires on benign and on malicious
   URLs, and how many of the phishing URLs the forest misses it would catch.

The test buckets of the main dataset are not touched: they are used once, by
train.py. Everything here is on validation sites (for choosing cut-offs) or on
the external dataset (for measuring).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

import adversary
import evaluate_external as E
import features as F
import hardening
import lookalike
from predict import MODES, UrlClassifier
from train import LOW_FPR_TARGET, SEED, TEST_BUCKETS, VAL_BUCKETS, metrics_at

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "processed" / "dataset.parquet"
PRICE_BUDGETS = (1, 2, 3, 4, 5, 6)
MAX_PRICE = 6
N_EXAMPLES = 4


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def choose_thresholds(y, p) -> dict:
    """Same rules as train.py step 5."""
    prec, rec, thr = precision_recall_curve(y, p)
    f1s = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12)
    fpr, _tpr, roc_thr = roc_curve(y, p)
    return {"default": 0.5, "max_f1": float(thr[int(np.argmax(f1s))]),
            "low_fpr": float(roc_thr[np.where(fpr <= LOW_FPR_TARGET)[0][-1]]),
            "low_fpr_target": LOW_FPR_TARGET}


def external_unseen(clf) -> pd.DataFrame:
    """External URLs whose site is absent from the train and validation buckets."""
    df = E.clean(E.load_external(), {})
    df["domain"] = F.registered_domains(df["url"].tolist(), n_jobs=1)
    corpus = pd.read_parquet(CORPUS, columns=["domain", "bucket"])
    used = set(corpus.loc[~corpus["bucket"].isin(TEST_BUCKETS), "domain"])
    return df[~df["domain"].isin(used)].reset_index(drop=True)


def lookalike_flags(urls) -> tuple[np.ndarray, Counter]:
    flagged = np.zeros(len(urls), dtype=bool)
    techniques: Counter = Counter()
    for i, u in enumerate(urls):
        found = [f for f in lookalike.check(u)["findings"] if f["severity"] in ("high", "medium")]
        if found:
            flagged[i] = True
            techniques[found[0]["technique"]] += 1
    return flagged, techniques


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sample", type=int, default=400, help="phishing URLs to attack (default 400)")
    args = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    clf = UrlClassifier()
    scorers = {"plain": clf}
    scorers.update({lvl: hardening.HardenedScorer(clf, lvl, thresholds={})
                    for lvl in hardening.LEVELS if lvl != "plain"})
    report: dict = {"config": {"seed": SEED, "attack_sample": args.sample, "max_price": MAX_PRICE,
                               "beam": 6, "depth": 5, "attack_mode": "balanced",
                               "moves": adversary.catalogue()["moves"],
                               "prices": adversary.catalogue()["prices"]},
                    "defences": {}}

    # ---------------------------------------------------------------- 1
    log("1/3  calibrating each scorer on the validation sites")
    val = pd.read_parquet(CORPUS, columns=["url", "label", "bucket"])
    val = val[val["bucket"].isin(VAL_BUCKETS)]
    yv = val["label"].to_numpy()
    for level, scorer in scorers.items():
        started = time.time()
        p = scorer.score(val["url"].tolist())
        thr = clf.thresholds if level == "plain" else choose_thresholds(yv, p)
        if level != "plain":
            scorer.thresholds = thr
        report["defences"][level] = {
            "description": hardening.LEVELS[level],
            "views_per_url": float(np.mean([len(hardening.views(u, level))
                                            for u in val["url"].iloc[:20000]])),
            "thresholds": {k: float(v) for k, v in thr.items()},
            "validation": {mode: metrics_at(yv, p, thr[key]) for mode, key in MODES.items()},
        }
        b = report["defences"][level]["validation"]["balanced"]
        log(f"     {level:<11} F1 {b['f1']:.4f}  FPR {b['false_positive_rate']:.4f}  "
            f"cut-off {thr['max_f1']:.3f}  ({time.time() - started:.0f}s)")
    del val

    # ---------------------------------------------------------------- 2
    log("2/3  external unseen sites: clean accuracy, then the attack")
    ext = external_unseen(clf)
    ye = ext["label"].to_numpy()
    raw = ext["raw_url"].tolist()
    malicious_rows = np.flatnonzero(ye == 1)
    sample_rows = rng.choice(malicious_rows, size=min(args.sample, len(malicious_rows)), replace=False)
    sample = [ext["url"].iloc[i] for i in sample_rows]
    report["external"] = {"dataset": "Hannousse & Yahiouche (2021), sites unseen in training",
                          "rows": int(len(ext)), "malicious": int(ye.sum()),
                          "attack_sample": len(sample)}
    plain_scores = None
    for level, scorer in scorers.items():
        thr = report["defences"][level]["thresholds"]
        p = scorer.score(raw)
        if level == "plain":
            plain_scores = p
        report["defences"][level]["external_unseen"] = {
            mode: metrics_at(ye, p, thr[key]) for mode, key in MODES.items()}

        started = time.time()
        cut = thr["max_f1"]
        start_scores = scorer.score(sample)
        prices, moves_used, single_change = [], Counter(), {}
        examples = []
        for url, s in zip(sample, start_scores):
            if s < cut:
                prices.append(0)          # missed before any attack
                continue
            r = adversary.attack(scorer, url, cut, max_cost=MAX_PRICE)
            prices.append(r["total_price"] if r["found"] else None)
            for m in r["single_moves"]:
                single_change.setdefault(m["move"], []).append(m["change"])
            if r["found"]:
                moves_used.update(step["move"] for step in r["steps"])
                if len(examples) < N_EXAMPLES and len(r["steps"]) >= 1 and len(url) < 90:
                    examples.append({"url": url, "score": r["score"], "steps": [
                        {k: step[k] for k in ("move", "label", "url", "score", "price")}
                        for step in r["steps"]]})
        caught = [x for x in prices if x != 0]
        n_caught = len(caught)

        def share(values, c):
            return float(np.mean([v is not None and v <= c for v in values])) if values else 0.0
        report["defences"][level]["attack"] = {
            "cut_off": cut,
            "sample": len(sample),
            "caught_before_attack": n_caught,
            "evaded_within_price": {str(c): share(caught, c) for c in PRICE_BUDGETS},
            "attacker_success_within_price": {str(c): share(prices, c) for c in (0,) + PRICE_BUDGETS},
            "never_evaded_within_max_price": float(np.mean([v is None for v in caught])) if caught else 0.0,
            "mean_price_when_evaded": float(np.mean([v for v in caught if v is not None]))
                                      if any(v is not None for v in caught) else None,
            "moves_in_cheapest_paths": dict(moves_used.most_common()),
            "mean_single_move_change": {m: float(np.mean(v)) for m, v in
                                        sorted(single_change.items(), key=lambda kv: np.mean(kv[1]))},
            "examples": examples,
            "seconds": round(time.time() - started, 1),
        }
        a = report["defences"][level]["attack"]
        log(f"     {level:<11} caught {n_caught}/{len(sample)}  evaded with one free edit "
            f"{a['evaded_within_price']['1']:.1%}  within price {MAX_PRICE} "
            f"{a['evaded_within_price'][str(MAX_PRICE)]:.1%}  ({a['seconds']:.0f}s)")

    # ---------------------------------------------------------------- 3
    log("3/3  lookalike layer")
    flags, techniques = lookalike_flags(ext["url"].tolist())
    missed = (plain_scores < clf.thresholds["max_f1"]) & (ye == 1)
    report["lookalike"] = {
        "external_unseen": {
            "benign_flagged": float(flags[ye == 0].mean()),
            "malicious_flagged": float(flags[ye == 1].mean()),
            "malicious_missed_by_model": int(missed.sum()),
            "of_those_flagged_by_lookalike": int((flags & missed).sum()),
            "techniques_on_malicious": dict(lookalike_flags(ext.loc[ye == 1, "url"].tolist())[1]),
        },
    }
    val = pd.read_parquet(CORPUS, columns=["url", "label", "bucket"])
    val = val[val["bucket"].isin(VAL_BUCKETS)]
    vflags, _ = lookalike_flags(val["url"].tolist())
    yv = val["label"].to_numpy()
    report["lookalike"]["validation"] = {
        "benign_flagged": float(vflags[yv == 0].mean()),
        "malicious_flagged": float(vflags[yv == 1].mean()),
        "rows": int(len(val)),
        # Read these: many "benign" rows it flags are phishing the source mislabelled.
        "benign_labelled_examples": [u[:140] for u in
                                     val.loc[vflags & (yv == 0), "url"].head(12).tolist()],
    }
    lk = report["lookalike"]
    log(f"     external: flags {lk['external_unseen']['malicious_flagged']:.1%} of phishing, "
        f"{lk['external_unseen']['benign_flagged']:.2%} of benign; catches "
        f"{lk['external_unseen']['of_those_flagged_by_lookalike']} of "
        f"{lk['external_unseen']['malicious_missed_by_model']} phishing URLs the model misses")
    log(f"     validation: flags {lk['validation']['benign_flagged']:.2%} of benign")

    report["evaluated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    report["minutes"] = round((time.time() - t0) / 60, 1)
    hardening.REPORT_PATH.write_text(json.dumps(report, indent=2))
    log(f"saved {hardening.REPORT_PATH.name} in {report['minutes']} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
