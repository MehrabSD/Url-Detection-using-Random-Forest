"""Train and evaluate the malicious-URL random forest.

    python src/train.py

Reads   data/processed/dataset.parquet        (made by prepare_data.py)
Writes  models/malicious_url_rf.joblib        the model bundle used by predict.py
        reports/metrics.json                  every number quoted in the docs

The script runs six steps; each one answers a question that is discussed in
TECHNICAL_DOCUMENTATION.md:

    1. leakage      How much does a random split flatter the model?
    2. cross        Does a model trained on one dataset work on another?
    3. learned      Build the three learned features (out-of-fold).
    4. search       Pick random-forest hyper-parameters on the validation sites.
    5. final        Fit the final forest, choose the decision threshold.
    6. test         Score once on the held-out test sites + baselines + importances.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, precision_recall_curve,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

import features as F
from learned_features import LearnedFeatures

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "dataset.parquet"
MODEL_PATH = ROOT / "models" / "malicious_url_rf.joblib"
REPORT_PATH = ROOT / "reports" / "metrics.json"

SEED = 42
N_JOBS = max(1, (os.cpu_count() or 2) - 2)

# Site-disjoint split: every registered domain hashes to one bucket 0-9.
TEST_BUCKETS = (0, 1)   # ~20 % of sites - touched exactly once, at the end
VAL_BUCKETS = (2,)      # ~10 % of sites - hyper-parameters and threshold
#                         buckets 3-9   - training

# Settings for the quick diagnostic experiments (steps 1-2).
QUICK_RF = dict(n_estimators=100, min_samples_leaf=2, max_samples=0.5)

# Candidates for step 4. First entry = scikit-learn defaults.
SEARCH_SPACE = [
    dict(),
    dict(min_samples_leaf=2, max_samples=0.3),
    dict(min_samples_leaf=5, max_samples=0.3),
    dict(min_samples_leaf=5, max_samples=0.5),
    dict(min_samples_leaf=10, max_samples=0.5),
    dict(min_samples_leaf=3, max_samples=0.2, max_depth=25),
    dict(min_samples_leaf=5, max_samples=0.3, max_features=0.3),
    dict(min_samples_leaf=5, max_samples=0.3, criterion="entropy"),
    dict(min_samples_leaf=5, max_samples=0.3, class_weight="balanced_subsample"),
]
SEARCH_TREES = 60
SEARCH_TOLERANCE = 0.002   # prefer the smallest forest within this PR-AUC of the best
FINAL_TREES = 200
LOW_FPR_TARGET = 0.01


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def rf(**params) -> RandomForestClassifier:
    return RandomForestClassifier(n_jobs=N_JOBS, random_state=SEED, **params)


def metrics_at(y, p, thr: float = 0.5) -> dict:
    y = np.asarray(y)
    tn, fp, fn, tp = confusion_matrix(y, p >= thr, labels=[0, 1]).ravel()
    both = 0 < y.sum() < len(y)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {
        "n": int(len(y)),
        "threshold": float(thr),
        "accuracy": float((tp + tn) / len(y)),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(2 * prec * rec / (prec + rec)) if prec + rec else 0.0,
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "roc_auc": float(roc_auc_score(y, p)) if both else None,
        "pr_auc": float(average_precision_score(y, p)) if both else None,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def curve_points(y, p, n: int = 150) -> dict:
    """Down-sampled ROC and precision-recall curves, small enough to keep in JSON."""
    fpr, tpr, _ = roc_curve(y, p)
    prec, rec, _ = precision_recall_curve(y, p)
    i = np.unique(np.linspace(0, len(fpr) - 1, n).astype(int))
    j = np.unique(np.linspace(0, len(rec) - 1, n).astype(int))
    return {"fpr": fpr[i].round(5).tolist(), "tpr": tpr[i].round(5).tolist(),
            "recall": rec[j].round(5).tolist(), "precision": prec[j].round(5).tolist()}


def forest_size(model: RandomForestClassifier) -> dict:
    return {"nodes_per_tree": int(np.mean([e.tree_.node_count for e in model.estimators_])),
            "mean_depth": float(np.mean([e.tree_.max_depth for e in model.estimators_]))}


# ------------------------------------------------------------------ step 1
def step_leakage(df: pd.DataFrame, X: np.ndarray) -> dict:
    """Kaggle dataset only: random 80/20 split vs. site-disjoint split."""
    k = (df["source"] == "kaggle").to_numpy()
    Xk, yk = X[k], df["label"].to_numpy()[k]
    bucket, domain = df["bucket"].to_numpy()[k], df["domain"].to_numpy()[k]
    idx = np.arange(len(yk))
    out = {}

    tr, te = train_test_split(idx, test_size=0.2, stratify=yk, random_state=SEED)
    m = rf(**QUICK_RF).fit(Xk[tr], yk[tr])
    out["random_split"] = metrics_at(yk[te], m.predict_proba(Xk[te])[:, 1])
    out["random_split"]["test_urls_whose_site_is_in_train"] = float(
        np.isin(domain[te], np.unique(domain[tr])).mean())

    te = np.isin(bucket, TEST_BUCKETS)
    m = rf(**QUICK_RF).fit(Xk[~te], yk[~te])
    out["site_disjoint_split"] = metrics_at(yk[te], m.predict_proba(Xk[te])[:, 1])
    out["site_disjoint_split"]["test_urls_whose_site_is_in_train"] = 0.0
    return out


# ------------------------------------------------------------------ step 2
def step_cross_dataset(df: pd.DataFrame, X: np.ndarray) -> dict:
    """Train on one source, test on every source (always on unseen sites)."""
    y, src = df["label"].to_numpy(), df["source"].to_numpy()
    te = df["bucket"].isin(TEST_BUCKETS).to_numpy()
    sources = ["kaggle", "phiusiil", "benchmark"]
    out = {}
    for train_on in sources + ["all three"]:
        rows = ~te if train_on == "all three" else (~te & (src == train_on))
        m = rf(**QUICK_RF).fit(X[rows], y[rows])
        out[train_on] = {}
        for test_on in sources:
            r = te & (src == test_on)
            res = metrics_at(y[r], m.predict_proba(X[r])[:, 1])
            out[train_on][test_on] = {k: res[k] for k in ("f1", "roc_auc", "accuracy", "recall",
                                                          "false_positive_rate")}
    return out


def main() -> int:
    t_start = time.time()
    report: dict = {"config": {
        "seed": SEED, "sklearn": sklearn.__version__, "test_buckets": TEST_BUCKETS,
        "val_buckets": VAL_BUCKETS, "quick_rf": QUICK_RF, "final_trees": FINAL_TREES,
    }}

    log("loading dataset")
    df = pd.read_parquet(DATA)
    X_base = df[F.FEATURE_NAMES].to_numpy(np.float32)
    meta = df[["url", "label", "source", "subtype", "domain", "bucket"]].copy()
    del df
    y = meta["label"].to_numpy()
    bucket = meta["bucket"].to_numpy()
    is_test = np.isin(bucket, TEST_BUCKETS)
    is_val = np.isin(bucket, VAL_BUCKETS)
    is_train = ~is_test & ~is_val
    report["split"] = {
        name: {"rows": int(m.sum()), "malicious_share": float(y[m].mean()),
               "sites": int(meta.loc[m, "domain"].nunique())}
        for name, m in (("train", is_train), ("validation", is_val), ("test", is_test))
    }
    log(f"rows: train {is_train.sum():,}  val {is_val.sum():,}  test {is_test.sum():,}")

    log("step 1/6  leakage check (random vs site-disjoint split, Kaggle only)")
    report["leakage"] = step_leakage(meta, X_base)
    log("    random F1 {:.4f}   site-disjoint F1 {:.4f}".format(
        report["leakage"]["random_split"]["f1"], report["leakage"]["site_disjoint_split"]["f1"]))

    log("step 2/6  cross-dataset generalisation")
    report["cross_dataset"] = step_cross_dataset(meta, X_base)

    # -------------------------------------------------------------- step 3
    log("step 3/6  learned features (out-of-fold on train)")
    learned = LearnedFeatures()
    urls = meta["url"].to_numpy()
    L = np.zeros((len(meta), 3), dtype=np.float32)
    L[is_train] = learned.fit_transform_oof(urls[is_train], y[is_train],
                                            meta["domain"].to_numpy()[is_train], bucket[is_train])
    L[~is_train] = learned.transform(urls[~is_train])
    X = np.hstack([X_base, L])
    feature_names = F.FEATURE_NAMES + LearnedFeatures.NAMES
    Xtr, ytr, Xva, yva, Xte, yte = X[is_train], y[is_train], X[is_val], y[is_val], X[is_test], y[is_test]

    # -------------------------------------------------------------- step 4
    log("step 4/6  hyper-parameter search (train -> validation)")
    search = []
    for params in SEARCH_SPACE:
        t0 = time.time()
        m = rf(n_estimators=SEARCH_TREES, **params).fit(Xtr, ytr)
        p = m.predict_proba(Xva)[:, 1]
        res = metrics_at(yva, p)
        row = {"params": params, "fit_seconds": round(time.time() - t0, 1),
               "val_pr_auc": res["pr_auc"], "val_roc_auc": res["roc_auc"], "val_f1": res["f1"],
               **forest_size(m)}
        search.append(row)
        log(f"    {params or 'sklearn defaults'}: PR-AUC {row['val_pr_auc']:.4f}  "
            f"F1 {row['val_f1']:.4f}  nodes/tree {row['nodes_per_tree']:,}")
    best_score = max(r["val_pr_auc"] for r in search)
    eligible = [r for r in search if r["val_pr_auc"] >= best_score - SEARCH_TOLERANCE]
    chosen = min(eligible, key=lambda r: r["nodes_per_tree"])
    best_params = chosen["params"]
    report["search"] = {"candidates": search, "chosen": best_params, "tolerance": SEARCH_TOLERANCE,
                        "trees_during_search": SEARCH_TREES}
    log(f"    chosen: {best_params}")

    # -------------------------------------------------------------- step 5
    log(f"step 5/6  final forest ({FINAL_TREES} trees)")
    t0 = time.time()
    model = rf(n_estimators=FINAL_TREES, **best_params).fit(Xtr, ytr)
    fit_seconds = time.time() - t0
    p_val = model.predict_proba(Xva)[:, 1]

    # how many trees are actually needed?
    running, curve = np.zeros(len(yva)), []
    checkpoints = {1, 2, 5, 10, 20, 35, 50, 75, 100, 150, FINAL_TREES}
    for i, est in enumerate(model.estimators_, start=1):
        running += est.predict_proba(Xva)[:, 1]
        if i in checkpoints:
            res = metrics_at(yva, running / i)
            curve.append({"trees": i, "val_pr_auc": res["pr_auc"], "val_f1": res["f1"]})
    report["trees_curve"] = curve

    # decision thresholds, chosen on validation sites only
    prec, rec, thr = precision_recall_curve(yva, p_val)
    f1s = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12)
    thr_f1 = float(thr[int(np.argmax(f1s))])
    fpr, tpr, roc_thr = roc_curve(yva, p_val)
    thr_low_fpr = float(roc_thr[np.where(fpr <= LOW_FPR_TARGET)[0][-1]])
    report["thresholds"] = {"default": 0.5, "max_f1": thr_f1, "low_fpr": thr_low_fpr,
                            "low_fpr_target": LOW_FPR_TARGET}
    report["validation"] = {name: metrics_at(yva, p_val, t) for name, t in
                            (("default", 0.5), ("max_f1", thr_f1), ("low_fpr", thr_low_fpr))}
    log(f"    fit {fit_seconds:.0f}s   thresholds: max-F1 {thr_f1:.3f}, <=1% FPR {thr_low_fpr:.3f}")

    # -------------------------------------------------------------- step 6
    log("step 6/6  held-out test sites")
    p_test = model.predict_proba(Xte)[:, 1]
    test_meta = meta[is_test].reset_index(drop=True)
    test = {name: metrics_at(yte, p_test, t) for name, t in
            (("default", 0.5), ("max_f1", thr_f1), ("low_fpr", thr_low_fpr))}
    test["curves"] = curve_points(yte, p_test)

    src = test_meta["source"].to_numpy()
    test["by_source"] = {s: metrics_at(yte[src == s], p_test[src == s], thr_f1)
                         for s in ("kaggle", "phiusiil", "benchmark")}
    flagged = p_test >= thr_f1
    test["detection_rate_by_kind"] = {
        f"{s} / {k}": {"n": int(len(g)), "detected": float(flagged[g.index].mean())}
        for (s, k), g in test_meta[yte == 1].groupby(["source", "subtype"])
    }

    # examples of confident mistakes, for the error analysis in the docs
    order = np.argsort(p_test)
    fn_idx = [i for i in order if yte[i] == 1][:15]
    fp_idx = [i for i in order[::-1] if yte[i] == 0][:15]
    test["confident_errors"] = {
        "false_negatives": [{"url": test_meta.url[i][:160], "source": src[i],
                             "kind": test_meta.subtype[i], "score": float(p_test[i])} for i in fn_idx],
        "false_positives": [{"url": test_meta.url[i][:160], "source": src[i],
                             "score": float(p_test[i])} for i in fp_idx],
    }
    report["test"] = test
    log("    test @max-F1 threshold: acc {accuracy:.4f}  prec {precision:.4f}  rec {recall:.4f}  "
        "F1 {f1:.4f}  ROC-AUC {roc_auc:.4f}".format(**test["max_f1"]))

    # baselines + ablation, all trained on the same rows, scored on the same test sites
    log("    baselines and ablation")
    comparison = {}

    def add(name, p):
        comparison[name] = {**metrics_at(yte, p), "curves": curve_points(yte, p, 80)}
        log(f"      {name}: F1 {comparison[name]['f1']:.4f}  ROC-AUC {comparison[name]['roc_auc']:.4f}")

    dummy = DummyClassifier(strategy="most_frequent").fit(Xtr, ytr)
    comparison["Always predict benign"] = metrics_at(yte, dummy.predict_proba(Xte)[:, 1])
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300)).fit(Xtr, ytr)
    add("Logistic regression", lr.predict_proba(Xte)[:, 1])
    tree = DecisionTreeClassifier(min_samples_leaf=best_params.get("min_samples_leaf", 1),
                                  random_state=SEED).fit(Xtr, ytr)
    add("Single decision tree", tree.predict_proba(Xte)[:, 1])
    nb = len(F.FEATURE_NAMES)
    rf_base = rf(n_estimators=FINAL_TREES, **best_params).fit(Xtr[:, :nb], ytr)
    add("Random forest, 53 hand-made features", rf_base.predict_proba(Xte[:, :nb])[:, 1])
    add("Random forest, all 56 features (final)", p_test)
    report["comparison"] = comparison
    del rf_base, tree, lr

    # feature importance: impurity-based (from training) and permutation (on test sites)
    log("    feature importances")
    rng = np.random.default_rng(SEED)
    sample = rng.choice(len(yte), size=min(20_000, len(yte)), replace=False)
    perm = permutation_importance(model, Xte[sample], yte[sample], scoring="average_precision",
                                  n_repeats=3, random_state=SEED, n_jobs=1)
    report["feature_importance"] = sorted(
        ({"feature": f, "impurity": float(i), "permutation": float(pm), "permutation_std": float(ps)}
         for f, i, pm, ps in zip(feature_names, model.feature_importances_,
                                 perm.importances_mean, perm.importances_std)),
        key=lambda r: -r["permutation"])

    # ---------------------------------------------------------------- save
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "learned": learned,
        "feature_names": feature_names,
        "thresholds": report["thresholds"],
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sklearn_version": sklearn.__version__,
    }
    joblib.dump(bundle, MODEL_PATH, compress=3)
    # the same forest as flat arrays, for serving in little memory (compact_forest.py)
    import compact_forest
    compact_forest.export(MODEL_PATH)

    # end-to-end speed: raw URL string -> verdict
    t0 = time.time()
    some = test_meta["url"].head(10_000).tolist()
    Xs = np.hstack([F.extract_many(some, n_jobs=1), learned.transform(some)])
    model.predict_proba(Xs)
    per_url_ms = (time.time() - t0) / len(some) * 1000

    report["model"] = {
        "params": {"n_estimators": FINAL_TREES, **best_params},
        "n_features": len(feature_names),
        "fit_seconds": round(fit_seconds, 1),
        "file_mb": round(MODEL_PATH.stat().st_size / 1e6, 1),
        "batch_ms_per_url": round(per_url_ms, 4),
        **forest_size(model),
    }
    report["total_minutes"] = round((time.time() - t_start) / 60, 1)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    log(f"saved {MODEL_PATH.name} ({report['model']['file_mb']} MB) and {REPORT_PATH.name} "
        f"in {report['total_minutes']} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
