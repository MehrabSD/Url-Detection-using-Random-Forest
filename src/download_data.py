"""Download the raw datasets used by the project.

Every dataset is saved into its own folder under data/raw/ and is never
modified afterwards. Run:

    python src/download_data.py

Files that already exist are skipped, so the script is safe to re-run.
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

HF = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"

# name -> list of (url, filename)
DATASETS: dict[str, list[tuple[str, str]]] = {
    # Kaggle "Malicious URLs dataset" (sid321axn), 651,191 URLs, 4 classes.
    # Pulled from a Hugging Face mirror because Kaggle needs an API key.
    "kaggle_malicious_urls": [
        (HF.format(repo="surajshelke/malicious_url", path=f"data/{split}-00000-of-00001.parquet"),
         f"{split}.parquet")
        for split in ("train", "val", "test")
    ],
    # PhiUSIIL Phishing URL dataset (UCI id 967), 235,795 URLs, CC BY 4.0.
    "phiusiil": [
        ("https://archive.ics.uci.edu/static/public/967/phiusiil+phishing+url+dataset.zip",
         "phiusiil.zip"),
    ],
    # Site-aware URL benchmark, 3,000,000 URLs (9:1 benign:malicious), Apache-2.0.
    "url_benchmark": [
        (HF.format(repo="JPxxx/url-benchmark-dataset", path="data/train-00000-of-00001.parquet"),
         "url_benchmark.parquet"),
    ],
    # EXTERNAL TEST ONLY - never read by prepare_data.py / train.py.
    # Hannousse & Yahiouche (2021) web-page phishing benchmark, 11,430 URLs, balanced, CC BY 4.0.
    # Used by evaluate_external.py to check the finished model on a dataset it was not built from.
    "hannousse_phishing": [
        (HF.format(repo="pirocheto/phishing-url", path=f"data/{split}.parquet"), f"{split}.parquet")
        for split in ("train", "test")
    ],
}


def download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  [skip] {dest.name} already present ({dest.stat().st_size / 1e6:.1f} MB)")
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        done = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                done += len(chunk)
    tmp.replace(dest)
    print(f"  [ok]   {dest.name} ({done / 1e6:.1f} MB)")


def main() -> int:
    for name, files in DATASETS.items():
        folder = RAW / name
        folder.mkdir(parents=True, exist_ok=True)
        print(f"{name}:")
        for url, filename in files:
            download(url, folder / filename)
            if filename.endswith(".zip"):
                with zipfile.ZipFile(folder / filename) as z:
                    for member in z.infolist():
                        # Only extract plain data files, flattened into the folder.
                        target = folder / Path(member.filename).name
                        if member.is_dir() or not member.filename.lower().endswith(".csv"):
                            continue
                        if not target.exists():
                            with z.open(member) as src, open(target, "wb") as dst:
                                dst.write(src.read())
                            print(f"  [ok]   extracted {target.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
