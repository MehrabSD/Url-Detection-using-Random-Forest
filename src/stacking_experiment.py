"""Experiment: does stacking a gradient-boosted model on the random forest help?

    python src/stacking_experiment.py        # about 10-15 minutes

Idea (stacking with passthrough). Give a gradient-boosting model the 56 usual
features *plus* the random forest's own prediction as a 57th column, so it can
spend its trees correcting the rows where the forest is wrong. For that column to
be honest on training rows it must be *out-of-fold*: each training row's value
comes from a forest that never saw that row's site. The training buckets 3-9
already are site-disjoint folds (the learned features use them the same way), so:

    for b in 3..9:  fit a forest on the other six buckets -> predict bucket b

Validation and external rows get the saved model's prediction (fitted on all
training buckets), exactly as a new URL would.

Compared, all on the same features and split:

    rf            the saved model (train.py)
    hgb           gradient boosting alone on the 56 features
    average       mean of rf and hgb (a plain blend, no learning)
    stack         gradient boosting on 56 features + out-of-fold forest score

Booster: scikit-learn's HistGradientBoostingClassifier (the LightGBM algorithm;
no extra dependency). Honesty rules: the test buckets 0-1 are never read. The
validation sites are split in two *by site*: half A chooses the number of
boosting rounds and the cut-offs, half B and the external dataset (sites unseen in
training) report the results. Writes reports/stacking.json. The saved model is
not changed.
"""
from __future__ import annotations

import json
import sys
import time
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import average_precision_score

import features as F
from learned_features import LearnedFeatures
from predict import UrlClassifier
from red_team import choose_thresholds, external_unseen
from train import N_JOBS, SEED, TEST_BUCKETS, VAL_BUCKETS, metrics_at

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "dataset.parquet"
REPORT = ROOT / "reports" / "stacking.json"

HGB = dict(learning_rate=0.1, max_iter=600, max_leaf_nodes=63, min_samples_leaf=50,
           l2_regularization=1.0, early_stopping=False, random_state=SEED)
CHECK_EVERY = 25


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def best_rounds(model, X, y) -> tuple[int, list]:
    """Number of boosting rounds with the best PR-AUC on (X, y)."""
    curve = []
    for i, p in enumerate(model.staged_predict_proba(X), start=1):
        if i % CHECK_EVERY == 0:
            curve.append({"rounds": i, "pr_auc": float(average_precision_score(y, p[:, 1]))})
    best = max(curve, key=lambda c: c["pr_auc"])
    return best["rounds"], curve


def staged_at(model, X, rounds: int) -> np.ndarray:
    for i, p in enumerate(model.staged_predict_proba(X), start=1):
        if i == rounds:
            return p[:, 1]
    raise ValueError(rounds)


def summary(y, p, thr: dict) -> dict:
    bal, strict = metrics_at(y, p, thr["max_f1"]), metrics_at(y, p, thr["low_fpr"])
    return {"pr_auc": bal["pr_auc"], "roc_auc": bal["roc_auc"],
            "balanced": {k: bal[k] for k in ("f1", "precision", "recall", "false_positive_rate", "accuracy")},
            "strict": {k: strict[k] for k in ("recall", "false_positive_rate", "precision")}}


def main() -> int:
    t0 = time.time()
    clf = UrlClassifier()
    rf_params = {k: v for k, v in clf.model.get_params().items()
                 if k in ("n_estimators", "min_samples_leaf", "max_samples", "max_depth")}

    log("loading training and validation rows (the test buckets are dropped unread)")
    df = pd.read_parquet(DATA)
    df = df[~df["bucket"].isin(TEST_BUCKETS)].reset_index(drop=True)
    X_base = df[F.FEATURE_NAMES].to_numpy(np.float32)
    y = df["label"].to_numpy()
    bucket = df["bucket"].to_numpy()
    urls = df["url"].to_numpy()
    domains = df["domain"].to_numpy()
    is_val = np.isin(bucket, VAL_BUCKETS)
    is_train = ~is_val
    del df

    log("learned features: out-of-fold on train (as train.py), saved model's on validation")
    L = np.zeros((len(y), 3), dtype=np.float32)
    L[is_train] = LearnedFeatures().fit_transform_oof(urls[is_train], y[is_train],
                                                      domains[is_train], bucket[is_train])
    L[is_val] = clf.learned.transform(urls[is_val])
    X = np.hstack([X_base, L])
    del X_base, L

    # sanity: the rebuilt validation rows give exactly the saved model's scores
    p_check = clf.model.predict_proba(X[is_val][:2000])[:, 1]
    p_saved = clf.score(list(urls[is_val][:2000]))
    assert np.allclose(p_check, p_saved), "features differ from the saved model's"

    # validation split in two halves by site: A tunes, B reports
    half_b = np.array([zlib.crc32(d.encode()) & 1 for d in domains[is_val]], dtype=bool)
    Xva, yva = X[is_val], y[is_val]
    Xtr, ytr, btr = X[is_train], y[is_train], bucket[is_train]
    A, B = ~half_b, half_b

    log("level 1: out-of-fold random-forest scores for the training rows (7 site folds)")
    rf_oof = np.zeros(len(ytr))
    for b in np.unique(btr):
        started = time.time()
        fold = btr == b
        m = RandomForestClassifier(n_jobs=N_JOBS, random_state=SEED, **rf_params)
        m.fit(Xtr[~fold], ytr[~fold])
        rf_oof[fold] = m.predict_proba(Xtr[fold])[:, 1]
        log(f"    bucket {b}: {fold.sum():,} rows ({time.time() - started:.0f}s)")
        del m
    rf_val = clf.model.predict_proba(Xva)[:, 1]

    log("external dataset (sites unseen in training)")
    ext = external_unseen(clf)
    Xext = clf.featurise(ext["raw_url"].tolist())
    yext = ext["label"].to_numpy()
    rf_ext = clf.model.predict_proba(Xext)[:, 1]

    report: dict = {"config": {"seed": SEED, "rf_params": rf_params, "hgb": HGB,
                               "validation_rows": {"A_tune": int(A.sum()), "B_report": int(B.sum())},
                               "external_rows": int(len(yext)),
                               "rf_oof_pr_auc_on_train": float(average_precision_score(ytr, rf_oof))},
                    "models": {}}
    scores_val, scores_ext = {"rf": rf_val}, {"rf": rf_ext}

    for name, extra_tr, extra_va, extra_ext in (
            ("hgb", None, None, None),
            ("stack", rf_oof, rf_val, rf_ext)):
        log(f"level 2: {name}")
        started = time.time()
        add = (lambda M, e: M if e is None else np.hstack([M, e[:, None].astype(np.float32)]))
        model = HistGradientBoostingClassifier(**HGB).fit(add(Xtr, extra_tr), ytr)
        va_in, ext_in = add(Xva, extra_va), add(Xext, extra_ext)
        rounds, curve = best_rounds(model, va_in[A], yva[A])
        scores_val[name] = staged_at(model, va_in, rounds)
        scores_ext[name] = staged_at(model, ext_in, rounds)
        report["models"][name] = {"rounds": rounds, "curve_on_A": curve,
                                  "fit_seconds": round(time.time() - started, 1)}
        log(f"    best rounds {rounds} ({time.time() - started:.0f}s)")
    scores_val["average"] = (scores_val["rf"] + scores_val["hgb"]) / 2
    scores_ext["average"] = (scores_ext["rf"] + scores_ext["hgb"]) / 2

    for name in ("rf", "hgb", "average", "stack"):
        thr = choose_thresholds(yva[A], scores_val[name][A])
        entry = report["models"].setdefault(name, {})
        entry["thresholds_from_A"] = {k: float(v) for k, v in thr.items()}
        entry["validation_B"] = summary(yva[B], scores_val[name][B], thr)
        entry["external_unseen"] = summary(yext, scores_ext[name], thr)
        v, e = entry["validation_B"], entry["external_unseen"]
        log(f"    {name:<8} val-B PR-AUC {v['pr_auc']:.4f} F1 {v['balanced']['f1']:.4f} "
            f"strict recall {v['strict']['recall']:.4f} | external PR-AUC {e['pr_auc']:.4f} "
            f"F1 {e['balanced']['f1']:.4f} FPR {e['balanced']['false_positive_rate']:.4f}")

    report["minutes"] = round((time.time() - t0) / 60, 1)
    report["evaluated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    REPORT.write_text(json.dumps(report, indent=2))
    log(f"saved {REPORT.name} in {report['minutes']} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
