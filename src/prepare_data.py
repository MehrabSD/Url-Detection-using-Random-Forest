"""Merge the raw datasets into one clean, feature-extracted table.

    python src/prepare_data.py

Output: data/processed/dataset.parquet with columns
    url      normalised URL string
    label    0 = benign, 1 = malicious
    source   kaggle | phiusiil | benchmark
    subtype  benign | phishing | malware | defacement | malicious (unknown kind)
    domain   registered domain / IP (the "site" the URL belongs to)
    bucket   0-9, a stable hash of `domain`; used to build site-disjoint splits
    <53 feature columns>
and data/processed/prepare_report.json describing every cleaning step.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import features as F

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

BENCH_BENIGN_SAMPLE = 600_000   # benign rows sampled from the 2.7M in the benchmark
SEED = 42


def load_kaggle() -> pd.DataFrame:
    df = pd.concat(
        [pd.read_parquet(RAW / "kaggle_malicious_urls" / f"{s}.parquet") for s in ("train", "val", "test")],
        ignore_index=True,
    )
    return pd.DataFrame({
        "url": df["url"].astype(str),
        "label": (df["type"] != "benign").astype("int8"),
        "source": "kaggle",
        "subtype": df["type"].astype(str),
    })


def load_phiusiil() -> pd.DataFrame:
    df = pd.read_csv(RAW / "phiusiil" / "PhiUSIIL_Phishing_URL_Dataset.csv", usecols=["URL", "label"])
    # In this dataset label 1 = legitimate, 0 = phishing -> flip to our convention.
    malicious = (1 - df["label"]).astype("int8")
    return pd.DataFrame({
        "url": df["URL"].astype(str),
        "label": malicious,
        "source": "phiusiil",
        "subtype": np.where(malicious == 1, "phishing", "benign"),
    })


def load_benchmark() -> pd.DataFrame:
    df = pd.read_parquet(RAW / "url_benchmark" / "url_benchmark.parquet", columns=["url", "label"])
    mal = df[df["label"] == "malicious"]
    ben = df[df["label"] == "benign"].sample(n=BENCH_BENIGN_SAMPLE, random_state=SEED)
    df = pd.concat([mal, ben], ignore_index=True)
    is_mal = (df["label"] == "malicious").astype("int8")
    return pd.DataFrame({
        "url": df["url"].astype(str),
        "label": is_mal,
        "source": "benchmark",
        "subtype": np.where(is_mal == 1, "malicious", "benign"),
    })


def bucket_of(domain: str) -> int:
    return int(hashlib.md5(domain.encode("utf-8", "replace")).hexdigest(), 16) % 10


def main() -> int:
    t0 = time.time()
    report: dict = {"steps": {}}

    df = pd.concat([load_kaggle(), load_phiusiil(), load_benchmark()], ignore_index=True)
    report["raw_rows_by_source"] = df["source"].value_counts().to_dict()
    print(f"loaded {len(df):,} raw rows")

    # Artifact check on the untouched Kaggle data: how well does the one-line
    # rule "malicious if the string starts with http" do? (see docs, Normalisation)
    k = df[df["source"] == "kaggle"]
    rule = k["url"].str.lower().str.startswith("http").astype("int8")
    report["kaggle_raw_scheme_rule_accuracy"] = round(float((rule == k["label"]).mean()), 4)
    report["kaggle_raw_scheme_share"] = {
        t: round(float(g.str.lower().str.startswith("http").mean()), 4)
        for t, g in k.groupby("subtype")["url"]
    }

    # 1. normalise
    df["url"] = df["url"].map(F.normalize_url)

    # 2. drop junk: too short, no dot in it, whitespace / control characters
    bad = (
        (df["url"].str.len() < 4)
        | ~df["url"].str.contains(".", regex=False)
        | df["url"].str.contains(r"[\s\x00-\x1f\x7f]", regex=True)
    )
    report["steps"]["dropped_junk"] = int(bad.sum())
    df = df[~bad]

    # 3. URLs that appear with BOTH labels are ambiguous -> remove them entirely
    nlab = df.groupby("url")["label"].transform("nunique")
    report["steps"]["dropped_conflicting_labels"] = int((nlab > 1).sum())
    df = df[nlab == 1]

    # 4. exact duplicates (after normalisation) -> keep the first occurrence
    dup = df.duplicated("url", keep="first")
    report["steps"]["dropped_duplicates"] = int(dup.sum())
    df = df[~dup].reset_index(drop=True)
    print(f"{len(df):,} rows after cleaning")

    # 5. site identifier + split bucket
    urls = df["url"].tolist()
    df["domain"] = F.registered_domains(urls)
    df["bucket"] = df["domain"].map(bucket_of).astype("int8")

    # 6. features
    X = F.extract_many(urls)
    feats = pd.DataFrame(X, columns=F.FEATURE_NAMES)
    out = pd.concat([df, feats], axis=1)

    OUT.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT / "dataset.parquet", index=False)

    report["final_rows"] = len(out)
    report["final_by_source_label"] = {
        f"{s}/{'malicious' if l else 'benign'}": int(n)
        for (s, l), n in out.groupby(["source", "label"]).size().items()
    }
    report["final_by_subtype"] = out["subtype"].value_counts().to_dict()
    report["unique_domains"] = int(out["domain"].nunique())
    report["n_features"] = F.N_FEATURES
    report["seconds"] = round(time.time() - t0, 1)
    (OUT / "prepare_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
