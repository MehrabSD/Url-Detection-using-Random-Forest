"""Score the finished model on an external dataset it was not built from.

    python src/evaluate_external.py

Reads   data/raw/hannousse_phishing/*.parquet   (fetched by download_data.py)
        data/processed/dataset.parquet           (url / domain / bucket only, to measure overlap)
        models/malicious_url_rf.joblib
Writes  reports/external_eval.json

Nothing is fitted or tuned here. The model, its learned features and its three
decision thresholds are used exactly as train.py saved them, through the same
`UrlClassifier` that predict.py uses.

The external dataset (Hannousse & Yahiouche 2021) takes its legitimate URLs from
popular websites, many of which are also in our training corpus. So results are
reported for three groups:

    unseen_sites  registered domain absent from our train + validation buckets
                  -> the headline, consistent with the site-disjoint rule
    seen_sites    registered domain present there (the model has met the site)
    all           both together
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

import features as F
from predict import MODEL_PATH, UrlClassifier
from train import TEST_BUCKETS, curve_points, metrics_at

ROOT = Path(__file__).resolve().parent.parent
EXTERNAL = ROOT / "data" / "raw" / "hannousse_phishing"
CORPUS = ROOT / "data" / "processed" / "dataset.parquet"
REPORT_PATH = ROOT / "reports" / "external_eval.json"

MODES = (("balanced", "max_f1"), ("default", "default"), ("strict", "low_fpr"))
N_ERROR_EXAMPLES = 12


def load_external() -> pd.DataFrame:
    df = pd.concat([pd.read_parquet(EXTERNAL / f"{s}.parquet", columns=["url", "status"])
                    for s in ("train", "test")], ignore_index=True)
    return pd.DataFrame({
        "raw_url": df["url"].astype(str),
        "label": (df["status"] == "phishing").astype("int8"),
    })


def clean(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    """Same normalisation and cleaning rules as prepare_data.py."""
    df = df.assign(url=df["raw_url"].map(F.normalize_url))
    bad = (
        (df["url"].str.len() < 4)
        | ~df["url"].str.contains(".", regex=False)
        | df["url"].str.contains(r"[\s\x00-\x1f\x7f]", regex=True)
    )
    df = df[~bad]
    nlab = df.groupby("url")["label"].transform("nunique")
    dup = df.duplicated("url", keep="first") & (nlab == 1)
    report["cleaning"] = {"dropped_junk": int(bad.sum()),
                          "dropped_conflicting_labels": int((nlab > 1).sum()),
                          "dropped_duplicates": int(dup.sum())}
    return df[(nlab == 1) & ~dup].reset_index(drop=True)


def main() -> int:
    t0 = time.time()
    if not MODEL_PATH.exists():
        print("Model not found - run `python src/train.py` first.", file=sys.stderr)
        return 1
    report: dict = {"dataset": {
        "name": "Hannousse & Yahiouche (2021) web page phishing detection benchmark",
        "source": "https://huggingface.co/datasets/pirocheto/phishing-url",
        "licence": "CC BY 4.0",
    }}

    df = load_external()
    report["dataset"]["raw_rows"] = len(df)
    df = clean(df, report)
    df["domain"] = F.registered_domains(df["url"].tolist(), n_jobs=1)

    # What has the model already met? Sites in the train + validation buckets were
    # used for fitting / threshold selection; test-bucket sites never were.
    corpus = pd.read_parquet(CORPUS, columns=["url", "domain", "bucket"])
    used = ~corpus["bucket"].isin(TEST_BUCKETS)
    seen_site = df["domain"].isin(set(corpus.loc[used, "domain"])).to_numpy()
    seen_url = df["url"].isin(set(corpus.loc[used, "url"])).to_numpy()
    del corpus

    y = df["label"].to_numpy()
    report["overlap_with_training_corpus"] = {
        "rows_after_cleaning": int(len(df)),
        "exact_url_in_train_or_validation": int(seen_url.sum()),
        "site_in_train_or_validation": int(seen_site.sum()),
        "share_of_benign_on_seen_sites": float(seen_site[y == 0].mean()),
        "share_of_malicious_on_seen_sites": float(seen_site[y == 1].mean()),
    }

    clf = UrlClassifier()
    t1 = time.time()
    X = clf.featurise(df["raw_url"].tolist())   # raw strings in, as a user would pass them
    parts = clf.score_parts(X)
    p = clf.score_features(X)
    report["ms_per_url"] = round((time.time() - t1) / len(df) * 1000, 4)
    report["thresholds"] = clf.thresholds

    groups = {"unseen_sites": ~seen_site, "seen_sites": seen_site, "all": np.ones(len(df), bool)}
    report["results"] = {}
    for name, rows in groups.items():
        res = {mode: metrics_at(y[rows], p[rows], clf.thresholds[key]) for mode, key in MODES}
        res["rows"] = int(rows.sum())
        res["sites"] = int(df.loc[rows, "domain"].nunique())
        res["malicious_share"] = float(y[rows].mean())
        report["results"][name] = res
    report["results"]["unseen_sites"]["curves"] = curve_points(y[~seen_site], p[~seen_site], 80)
    # each half of the model on its own (threshold-free measures; the cut-offs belong to the average)
    report["halves_unseen_sites"] = {
        name: {"pr_auc": float(average_precision_score(y[~seen_site], q[~seen_site])),
               "roc_auc": float(roc_auc_score(y[~seen_site], q[~seen_site]))}
        for name, q in {**parts, "average": p}.items()}

    # Most confident mistakes on unseen sites, for the write-up.
    sub = df[~seen_site].assign(score=p[~seen_site])
    worst_fn = sub[sub["label"] == 1].nsmallest(N_ERROR_EXAMPLES, "score")
    worst_fp = sub[sub["label"] == 0].nlargest(N_ERROR_EXAMPLES, "score")
    report["confident_errors_unseen_sites"] = {
        kind: [{"url": u[:160], "score": float(s)} for u, s in zip(part["url"], part["score"])]
        for kind, part in (("false_negatives", worst_fn), ("false_positives", worst_fp))
    }

    report["evaluated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    report["seconds"] = round(time.time() - t0, 1)
    REPORT_PATH.write_text(json.dumps(report, indent=2))

    print(f"{len(df):,} URLs after cleaning; {int(seen_site.sum()):,} are on sites the model trained on")
    for name in groups:
        r = report["results"][name]
        b = r["balanced"]
        print(f"  {name:<13} n={r['rows']:>6,}  malicious {r['malicious_share']:.1%}  "
              f"acc {b['accuracy']:.4f}  prec {b['precision']:.4f}  rec {b['recall']:.4f}  "
              f"F1 {b['f1']:.4f}  FPR {b['false_positive_rate']:.4f}  ROC-AUC {b['roc_auc']:.4f}")
    print(f"saved {REPORT_PATH.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
