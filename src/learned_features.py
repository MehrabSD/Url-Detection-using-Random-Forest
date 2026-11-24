"""Three features that are *learned from labelled training URLs*.

The 53 features in features.py are fixed formulas. These three are small
statistical models whose output is fed to the random forest as extra columns:

    tld_malicious_rate  smoothed share of malicious sites seen under this TLD
    host_ngram_score    character 2-4-gram Naive-Bayes log-odds of the host name
    url_ngram_score     character 3-5-gram Naive-Bayes log-odds of the whole URL

Because they are fitted on labels, training rows must receive *out-of-fold*
values (computed by a model that never saw that row's site), otherwise the
forest would learn to trust them far more than it should. `fit_transform_oof`
does exactly that; `transform` is used for validation, test and new URLs.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer

import features as F


def _nb_weights(c0: np.ndarray, c1: np.ndarray, alpha: float) -> np.ndarray:
    """Per-n-gram log-likelihood ratio log P(g|malicious) - log P(g|benign)."""
    v = c0.shape[0]
    lp1 = np.log(c1 + alpha) - np.log(c1.sum() + alpha * v)
    lp0 = np.log(c0 + alpha) - np.log(c0.sum() + alpha * v)
    return (lp1 - lp0).astype(np.float32)


def _score(m, w: np.ndarray) -> np.ndarray:
    """Average log-odds per n-gram, so long and short strings are comparable."""
    n = np.asarray(m.sum(axis=1)).ravel()
    return (m @ w) / (n + 1.0)


class LearnedFeatures:
    NAMES = ["tld_malicious_rate", "host_ngram_score", "url_ngram_score"]

    def __init__(self, alpha: float = 0.1, hash_bits: int = 20, tld_smoothing: float = 20.0):
        self.alpha = alpha
        self.hash_bits = hash_bits
        self.tld_smoothing = tld_smoothing
        self.w_host: np.ndarray | None = None
        self.w_url: np.ndarray | None = None
        self.tld_rate: dict[str, float] = {}
        self.prior: float = 0.5

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _hosts_tlds(urls) -> tuple[list[str], list[str]]:
        hosts, tlds = [], []
        for u in urls:
            h = F.split_url(u)[1]
            hosts.append(h)
            tlds.append("<ip>" if F._is_ip(h) else h.rsplit(".", 1)[-1])
        return hosts, tlds

    def _vectorise(self, urls, hosts):
        kw = dict(analyzer="char", n_features=2 ** self.hash_bits, alternate_sign=False,
                  norm=None, lowercase=True)
        m_host = HashingVectorizer(ngram_range=(2, 4), **kw).transform(hosts)
        m_url = HashingVectorizer(ngram_range=(3, 5), **kw).transform(urls)
        return m_host, m_url

    def _tld_table(self, total: pd.DataFrame, prior: float) -> dict[str, float]:
        rate = (total["s"] + prior * self.tld_smoothing) / (total["n"] + self.tld_smoothing)
        return rate.to_dict()

    # ------------------------------------------------------------------- API
    def fit_transform_oof(self, urls, y, domains, folds) -> np.ndarray:
        """Fit on all rows and return out-of-fold features for those same rows.

        `urls` must already be normalised. `folds` must keep every domain in a
        single fold (we pass the domain-hash bucket).
        """
        urls = list(urls)
        y = np.asarray(y).astype(np.int8)
        folds = np.asarray(folds)
        hosts, tlds = self._hosts_tlds(urls)
        m_host, m_url = self._vectorise(urls, hosts)
        out = np.zeros((len(urls), 3), dtype=np.float32)
        fold_ids = np.unique(folds)

        # --- n-gram Naive Bayes: class counts are additive, so
        #     "everything except fold b" = total - fold b.
        for col, m in ((1, m_host), (2, m_url)):
            counts = {}
            for b in fold_ids:
                rows = folds == b
                counts[b] = tuple(
                    np.asarray(m[rows & (y == k)].sum(axis=0)).ravel() for k in (0, 1)
                )
            tot0 = sum(c[0] for c in counts.values())
            tot1 = sum(c[1] for c in counts.values())
            for b in fold_ids:
                rows = np.where(folds == b)[0]
                w = _nb_weights(tot0 - counts[b][0], tot1 - counts[b][1], self.alpha)
                out[rows, col] = _score(m[rows], w)
            w_all = _nb_weights(tot0, tot1, self.alpha)
            if col == 1:
                self.w_host = w_all
            else:
                self.w_url = w_all

        # --- TLD malicious rate, counted once per *site* so that a single huge
        #     domain cannot dominate its TLD.
        site = (
            pd.DataFrame({"domain": list(domains), "tld": tlds, "y": y, "fold": folds})
            .groupby("domain", sort=False)
            .agg(tld=("tld", "first"), y=("y", "mean"), fold=("fold", "first"))
        )
        self.prior = float(site["y"].mean())
        per_fold = site.groupby(["fold", "tld"])["y"].agg(s="sum", n="count")
        total = site.groupby("tld")["y"].agg(s="sum", n="count")
        tld_arr = np.array(tlds, dtype=object)
        for b in fold_ids:
            rest = total.sub(per_fold.xs(b, level="fold"), fill_value=0)
            fold_sites = site[site["fold"] != b]
            table = self._tld_table(rest, float(fold_sites["y"].mean()))
            rows = folds == b
            out[rows, 0] = pd.Series(tld_arr[rows]).map(table).fillna(self.prior).to_numpy()
        self.tld_rate = self._tld_table(total, self.prior)
        return out

    def transform(self, urls) -> np.ndarray:
        urls = list(urls)
        hosts, tlds = self._hosts_tlds(urls)
        m_host, m_url = self._vectorise(urls, hosts)
        out = np.zeros((len(urls), 3), dtype=np.float32)
        out[:, 0] = [self.tld_rate.get(t, self.prior) for t in tlds]
        out[:, 1] = _score(m_host, self.w_host)
        out[:, 2] = _score(m_url, self.w_url)
        return out
