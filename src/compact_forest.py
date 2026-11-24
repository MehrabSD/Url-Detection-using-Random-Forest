"""The trained forest as a few flat arrays: same predictions, a fraction of the memory.

    python src/compact_forest.py        # models/malicious_url_rf.joblib -> models/malicious_url_rf.npz

Why. scikit-learn keeps each of the 200 trees as its own object, with a 64-byte
record per node plus a separate array of class counts. Loading the 4.98 million
nodes from the pickle peaks near 940 MB, which is more than free hosting plans
offer (Render's free plan has 512 MB). The demo server only needs to *use* the
forest, so this module stores just what prediction and explanation read:

    feature    int16    which feature a node tests
    threshold  float64  the split value (kept at full precision: see below)
    left/right int32    children, as indices into the concatenated arrays
    share      float64  share of malicious training URLs reaching the node
    roots      int32    where each tree starts

All trees are concatenated, and a leaf points to itself on both sides. Then a
whole batch of URLs can go down all 200 trees at once: repeat "go left or right"
max_depth times with numpy fancy indexing, and every path ends parked on its leaf.

Exactness. scikit-learn compares the float32 feature value with a float64
threshold; so does this code (numpy promotes the float32 value), so every URL
reaches the same leaf. tests/test_compact.py checks the probabilities match
predict_proba to 1e-12 on thousands of URLs.

The file is a plain .npz plus a JSON string - no pickle - so loading it cannot
run code, and it does not depend on the scikit-learn version.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FULL_PATH = ROOT / "models" / "malicious_url_rf.joblib"
COMPACT_PATH = ROOT / "models" / "malicious_url_rf.npz"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class CompactForest:
    """predict_proba-compatible forest built from flat arrays."""

    def __init__(self, feature, threshold, left, right, share, roots, max_depth: int):
        self.feature, self.threshold = feature, threshold
        self.left, self.right, self.share = left, right, share
        self.roots, self.max_depth = roots, int(max_depth)
        self.n_estimators = len(roots)

    @classmethod
    def from_sklearn(cls, model) -> "CompactForest":
        parts = {k: [] for k in ("feature", "threshold", "left", "right", "share")}
        roots, offset, depth = [], 0, 0
        for est in model.estimators_:
            t = est.tree_
            n = t.node_count
            idx = np.arange(n, dtype=np.int64)
            leaf = t.children_left == -1
            value = t.value[:, 0, :]
            parts["feature"].append(np.where(leaf, 0, t.feature).astype(np.int16))
            parts["threshold"].append(t.threshold.astype(np.float64))
            parts["left"].append((np.where(leaf, idx, t.children_left) + offset).astype(np.int32))
            parts["right"].append((np.where(leaf, idx, t.children_right) + offset).astype(np.int32))
            parts["share"].append((value[:, 1] / value.sum(axis=1)).astype(np.float64))
            roots.append(offset)
            offset += n
            depth = max(depth, t.max_depth)
        arrays = {k: np.concatenate(v) for k, v in parts.items()}
        return cls(roots=np.asarray(roots, dtype=np.int32), max_depth=depth, **arrays)

    def leaves(self, X: np.ndarray) -> np.ndarray:
        """(n_urls, n_trees) index of the leaf each URL reaches in each tree."""
        X = np.asarray(X, dtype=np.float32)
        rows = np.arange(len(X))[:, None]
        node = np.broadcast_to(self.roots, (len(X), self.n_estimators)).copy()
        for _ in range(self.max_depth):
            go_left = X[rows, self.feature[node]] <= self.threshold[node]
            node = np.where(go_left, self.left[node], self.right[node])
        return node

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = self.share[self.leaves(X)].mean(axis=1) if len(X) else np.empty(0)
        return np.column_stack([1.0 - p, p])

    def is_leaf(self, node: int) -> bool:
        return self.left[node] == node


def export(full_path: Path = FULL_PATH, out_path: Path = COMPACT_PATH) -> Path:
    """Write the compact file for the model bundle saved by train.py."""
    import joblib  # only needed here, not when serving

    bundle = joblib.load(full_path)
    forest = CompactForest.from_sklearn(bundle["model"])
    learned = bundle["learned"]
    meta = {
        "feature_names": list(bundle["feature_names"]),
        "thresholds": {k: float(v) for k, v in bundle["thresholds"].items()},
        "trained_at": bundle.get("trained_at"),
        "sklearn_version": bundle.get("sklearn_version"),
        "source_sha256": file_sha256(full_path),
        "max_depth": forest.max_depth,
        "learned": {"alpha": learned.alpha, "hash_bits": learned.hash_bits,
                    "tld_smoothing": learned.tld_smoothing, "prior": float(learned.prior),
                    "tld_rate": {str(k): float(v) for k, v in learned.tld_rate.items()}},
    }
    np.savez_compressed(out_path, feature=forest.feature, threshold=forest.threshold,
                        left=forest.left, right=forest.right, share=forest.share,
                        roots=forest.roots, w_host=learned.w_host, w_url=learned.w_url,
                        meta=np.array(json.dumps(meta)))
    return out_path


def load(path: Path = COMPACT_PATH) -> dict:
    """Everything UrlClassifier needs, without scikit-learn's forest objects."""
    from learned_features import LearnedFeatures

    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        forest = CompactForest(z["feature"], z["threshold"], z["left"], z["right"], z["share"],
                               z["roots"], meta["max_depth"])
        cfg = meta["learned"]
        learned = LearnedFeatures(alpha=cfg["alpha"], hash_bits=cfg["hash_bits"],
                                  tld_smoothing=cfg["tld_smoothing"])
        learned.w_host, learned.w_url = z["w_host"], z["w_url"]
    learned.tld_rate, learned.prior = cfg["tld_rate"], cfg["prior"]
    return {"model": forest, "learned": learned, "feature_names": meta["feature_names"],
            "thresholds": meta["thresholds"], "trained_at": meta["trained_at"],
            "sklearn_version": meta["sklearn_version"], "source_sha256": meta["source_sha256"]}


def main() -> int:
    if not FULL_PATH.exists():
        print("Model not found - run `python src/train.py` first.", file=sys.stderr)
        return 1
    out = export()
    print(f"wrote {out.name}: {out.stat().st_size / 1e6:.1f} MB "
          f"(from {FULL_PATH.name}, {FULL_PATH.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
