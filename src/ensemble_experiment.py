"""Experiment: combine the random forest with k-nearest neighbours and SVMs.

    python src/ensemble_experiment.py        # about 10-15 minutes

Different kinds of model make different mistakes: a forest splits on one feature
at a time, k-nearest neighbours looks at the most similar training URLs, and a
support vector machine draws one smooth boundary through all features at once.
Combining them can cancel some mistakes out. This script measures whether it does.

Members (all trained on the training buckets 3-9, scaled features for KNN/SVM):

    rf     the saved random forest (train.py)
    knn    k-nearest neighbours on a 100,000-row sample, distance-weighted votes
    lsvm   linear SVM on every training row
    rsvm   SVM with an RBF kernel on a 20,000-row sample

KNN and the kernel SVM compare a URL with stored training URLs, so their cost grows
with the training set; samples keep them practical, the linear SVM scales to all rows.

Combinations (learned on validation half A only):

    soft vote     each member's score is turned into a probability by Platt scaling
                  (a one-feature logistic regression on half A), then averaged
    learned blend a logistic regression on the members' scores decides how much to
                  trust each one (stacking with a held-out blending set)
    + ablations   forest+KNN and forest+RBF-SVM soft votes, to see who helps

Honesty rules as in stacking_experiment.py: the test buckets 0-1 are never read;
the validation sites are split in two by site - half A chooses every setting,
weight and cut-off, half B and the external dataset report. Writes
reports/ensemble.json. The saved model is not changed.
"""
from __future__ import annotations

import json
import sys
import time
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

import features as F
from learned_features import LearnedFeatures
from predict import UrlClassifier
from red_team import choose_thresholds, external_unseen
from stacking_experiment import summary
from train import N_JOBS, SEED, TEST_BUCKETS, VAL_BUCKETS

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "dataset.parquet"
REPORT = ROOT / "reports" / "ensemble.json"

KNN_TRAIN_ROWS, KNN_K = 100_000, (5, 15, 45)
LINEAR_SVM_C = (0.1, 1.0)
RBF_SVM_TRAIN_ROWS, RBF_SVM_C = 20_000, (1.0, 10.0)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def knn_scores(dist: np.ndarray, ind: np.ndarray, y_train: np.ndarray, k: int) -> np.ndarray:
    """Distance-weighted share of malicious neighbours among the k nearest, as
    scikit-learn's KNeighborsClassifier(weights="distance") computes it (an exact
    match outvotes everything else). One neighbour search serves every k."""
    d, lab = dist[:, :k], y_train[ind[:, :k]]
    exact = d == 0
    with np.errstate(divide="ignore"):
        w = np.where(exact.any(axis=1, keepdims=True), exact.astype(float), 1.0 / d)
    return (w * lab).sum(axis=1) / w.sum(axis=1)


def in_parallel(fn, X: np.ndarray) -> np.ndarray:
    chunks = np.array_split(np.arange(len(X)), N_JOBS)
    return np.concatenate(joblib.Parallel(n_jobs=N_JOBS)(joblib.delayed(fn)(X[c]) for c in chunks))


def pr_auc(y, p) -> float:
    return float(average_precision_score(y, p))


def main() -> int:
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    clf = UrlClassifier()

    log("loading training and validation rows (the test buckets are dropped unread)")
    df = pd.read_parquet(DATA)
    df = df[~df["bucket"].isin(TEST_BUCKETS)].reset_index(drop=True)
    X_base = df[F.FEATURE_NAMES].to_numpy(np.float32)
    y, bucket = df["label"].to_numpy(), df["bucket"].to_numpy()
    urls, domains = df["url"].to_numpy(), df["domain"].to_numpy()
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
    Xtr, ytr, Xva, yva = X[is_train], y[is_train], X[is_val], y[is_val]
    assert np.allclose(clf.model.predict_proba(Xva[:2000])[:, 1], clf.score(list(urls[is_val][:2000])))
    B = np.array([zlib.crc32(d.encode()) & 1 for d in domains[is_val]], dtype=bool)
    A = ~B

    log("external dataset (sites unseen in training)")
    ext = external_unseen(clf)
    Xext, yext = clf.featurise(ext["raw_url"].tolist()), ext["label"].to_numpy()

    scaler = StandardScaler().fit(Xtr)
    Str, Sva, Sext = scaler.transform(Xtr), scaler.transform(Xva), scaler.transform(Xext)
    val_s, ext_s, settings = {}, {}, {}

    log("member: random forest (saved model)")
    val_s["rf"], ext_s["rf"] = clf.model.predict_proba(Xva)[:, 1], clf.model.predict_proba(Xext)[:, 1]

    log(f"member: k-nearest neighbours ({KNN_TRAIN_ROWS:,}-row sample)")
    started = time.time()
    rows = rng.choice(len(ytr), size=KNN_TRAIN_ROWS, replace=False)
    nn = NearestNeighbors(n_neighbors=max(KNN_K), n_jobs=N_JOBS).fit(Str[rows])
    dva, iva = nn.kneighbors(Sva)
    dext, iext = nn.kneighbors(Sext)
    tried = {k: pr_auc(yva[A], knn_scores(dva[A], iva[A], ytr[rows], k)) for k in KNN_K}
    k = max(tried, key=tried.get)
    val_s["knn"], ext_s["knn"] = knn_scores(dva, iva, ytr[rows], k), knn_scores(dext, iext, ytr[rows], k)
    settings["knn"] = {"training_rows": KNN_TRAIN_ROWS, "k": k, "weights": "distance",
                       "pr_auc_on_A_by_k": {str(kk): v for kk, v in tried.items()},
                       "seconds": round(time.time() - started, 1)}
    log(f"    k = {k} ({time.time() - started:.0f}s)")
    del nn, dva, iva

    log("member: linear SVM (all training rows)")
    started, best = time.time(), None
    tried = {}
    for c in LINEAR_SVM_C:
        m = LinearSVC(C=c, dual=False).fit(Str, ytr)
        tried[c] = pr_auc(yva[A], m.decision_function(Sva[A]))
        if best is None or tried[c] > tried[best[0]]:
            best = (c, m)
    val_s["lsvm"], ext_s["lsvm"] = best[1].decision_function(Sva), best[1].decision_function(Sext)
    settings["lsvm"] = {"training_rows": int(len(ytr)), "C": best[0],
                        "pr_auc_on_A_by_C": {str(c): v for c, v in tried.items()},
                        "seconds": round(time.time() - started, 1)}
    log(f"    C = {best[0]} ({time.time() - started:.0f}s)")

    log(f"member: SVM with RBF kernel ({RBF_SVM_TRAIN_ROWS:,}-row sample)")
    started, best = time.time(), None
    tried = {}
    rows = rng.choice(len(ytr), size=RBF_SVM_TRAIN_ROWS, replace=False)
    for c in RBF_SVM_C:
        m = SVC(kernel="rbf", C=c, gamma="scale", cache_size=2000).fit(Str[rows], ytr[rows])
        tried[c] = pr_auc(yva[A], in_parallel(m.decision_function, Sva[A]))
        if best is None or tried[c] > tried[best[0]]:
            best = (c, m)
    val_s["rsvm"] = in_parallel(best[1].decision_function, Sva)
    ext_s["rsvm"] = in_parallel(best[1].decision_function, Sext)
    settings["rsvm"] = {"training_rows": RBF_SVM_TRAIN_ROWS, "C": best[0], "gamma": "scale",
                        "support_vectors": int(best[1].n_support_.sum()),
                        "pr_auc_on_A_by_C": {str(c): v for c, v in tried.items()},
                        "seconds": round(time.time() - started, 1)}
    log(f"    C = {best[0]} ({time.time() - started:.0f}s)")

    log("combinations (all weights and calibrations learned on validation half A)")
    members = ["rf", "knn", "lsvm", "rsvm"]
    platt = {m: LogisticRegression().fit(val_s[m][A, None], yva[A]) for m in members}
    pv = {m: platt[m].predict_proba(val_s[m][:, None])[:, 1] for m in members}
    pe = {m: platt[m].predict_proba(ext_s[m][:, None])[:, 1] for m in members}

    def vote(names):
        return np.mean([pv[n] for n in names], axis=0), np.mean([pe[n] for n in names], axis=0)

    combos = {"vote: forest + KNN": ["rf", "knn"], "vote: forest + RBF SVM": ["rf", "rsvm"],
              "vote: forest + KNN + both SVMs": members}
    for name, names in combos.items():
        val_s[name], ext_s[name] = vote(names)

    Mv = np.column_stack([val_s[m] for m in members])
    Me = np.column_stack([ext_s[m] for m in members])
    meta_scale = StandardScaler().fit(Mv[A])
    meta = LogisticRegression().fit(meta_scale.transform(Mv[A]), yva[A])
    val_s["learned blend of all four"] = meta.predict_proba(meta_scale.transform(Mv))[:, 1]
    ext_s["learned blend of all four"] = meta.predict_proba(meta_scale.transform(Me))[:, 1]
    settings["learned_blend_weights"] = dict(zip(members, map(float, meta.coef_[0])))

    labels = {"rf": "Random forest alone (saved model)", "knn": "k-nearest neighbours alone",
              "lsvm": "Linear SVM alone", "rsvm": "RBF-kernel SVM alone"}
    report: dict = {"config": {"seed": SEED, "validation_rows": {"A_tune": int(A.sum()), "B_report": int(B.sum())},
                               "external_rows": int(len(yext)), "members": settings},
                    "models": {}}
    for key, scores in val_s.items():
        thr = choose_thresholds(yva[A], scores[A])
        name = labels.get(key, key)
        report["models"][name] = {"thresholds_from_A": {k: float(v) for k, v in thr.items()},
                                  "validation_B": summary(yva[B], scores[B], thr),
                                  "external_unseen": summary(yext, ext_s[key], thr)}
        v, e = report["models"][name]["validation_B"], report["models"][name]["external_unseen"]
        log(f"    {name:<34} val-B PR-AUC {v['pr_auc']:.4f} F1 {v['balanced']['f1']:.4f} "
            f"strict rec {v['strict']['recall']:.3f} | ext PR-AUC {e['pr_auc']:.4f} "
            f"ROC {e['roc_auc']:.4f} F1 {e['balanced']['f1']:.4f}")

    report["minutes"] = round((time.time() - t0) / 60, 1)
    report["evaluated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    REPORT.write_text(json.dumps(report, indent=2))
    log(f"saved {REPORT.name} in {report['minutes']} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
