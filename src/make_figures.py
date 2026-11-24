"""Draw every chart used in the documentation from the saved JSON reports.

    python src/make_figures.py

Reads reports/metrics.json, data/processed/prepare_report.json and (when
red_team.py has run) reports/red_team.json, writes PNGs
to reports/figures/. No model or dataset is needed, so the charts can be
restyled without retraining.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "reports" / "figures"

# --- palette (light surface) -------------------------------------------------
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"   # categorical slots 1-3
GRAY = "#c3c2b7"                                        # de-emphasised marks
SEQ = LinearSegmentedColormap.from_list("blue_seq", ["#cde2fb", "#6da7ec", "#2a78d6", "#0d366b"])

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": INK2,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.titlesize": 12, "axes.titleweight": "semibold", "axes.titlelocation": "left",
    "axes.titlepad": 12, "legend.frameon": False, "figure.dpi": 110, "savefig.dpi": 160,
    "savefig.bbox": "tight", "xtick.major.size": 0, "ytick.major.size": 0,
})


def save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name)
    plt.close(fig)
    print("  wrote", name)


def hbars(ax, labels, values, colors, fmt="{:.3f}", xlim=None):
    """Thin ranked horizontal bars with the value written at the tip."""
    ypos = np.arange(len(labels))[::-1]
    ax.barh(ypos, values, height=0.52, color=colors, linewidth=0)
    ax.set_yticks(ypos, labels)
    ax.spines["left"].set_visible(False)
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    if xlim:
        ax.set_xlim(*xlim)
    span = ax.get_xlim()[1] - ax.get_xlim()[0]
    for yy, v in zip(ypos, values):
        ax.text(v + span * 0.012, yy, fmt.format(v), va="center", ha="left", color=INK2, fontsize=9)


def fig_dataset(prep: dict) -> None:
    counts = prep["final_by_source_label"]
    names = {"kaggle": "Kaggle Malicious URLs", "phiusiil": "PhiUSIIL (UCI)", "benchmark": "Site-aware benchmark (sample)"}
    src = ["benchmark", "kaggle", "phiusiil"]
    ben = np.array([counts[f"{s}/benign"] for s in src]) / 1e3
    mal = np.array([counts[f"{s}/malicious"] for s in src]) / 1e3
    fig, ax = plt.subplots(figsize=(8, 2.9))
    ypos = np.arange(len(src))[::-1]
    ax.barh(ypos, ben, height=0.5, color=BLUE, linewidth=0, label="Benign")
    ax.barh(ypos, mal, left=ben + ben.max() * 0.006, height=0.5, color=ORANGE, linewidth=0, label="Malicious")
    for yy, b, m in zip(ypos, ben, mal):
        ax.text(b / 2, yy, f"{b:,.0f}K", ha="center", va="center", color="white", fontsize=9)
        ax.text(b + m / 2, yy, f"{m:,.0f}K", ha="center", va="center", color="white", fontsize=9)
    ax.set_yticks(ypos, [names[s] for s in src])
    ax.spines["left"].set_visible(False)
    ax.xaxis.grid(True); ax.set_axisbelow(True)
    ax.set_xlabel("URLs after cleaning (thousands)")
    ax.set_title(f"Training corpus: {prep['final_rows']:,} unique URLs from three sources")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2)
    save(fig, "dataset_composition.png")


def fig_leakage(m: dict) -> None:
    lk = m["leakage"]
    metrics = [("accuracy", "Accuracy"), ("f1", "F1"), ("recall", "Recall"), ("roc_auc", "ROC-AUC")]
    a = [lk["random_split"][k] for k, _ in metrics]
    b = [lk["site_disjoint_split"][k] for k, _ in metrics]
    x = np.arange(len(metrics))
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    w = 0.3
    ax.bar(x - w / 2 - 0.01, a, w, color=GRAY, linewidth=0, label="Random 80/20 split (same sites in train and test)")
    ax.bar(x + w / 2 + 0.01, b, w, color=BLUE, linewidth=0, label="Site-disjoint split (test sites never seen)")
    for xi, va, vb in zip(x, a, b):
        ax.text(xi - w / 2 - 0.01, va + 0.012, f"{va:.3f}", ha="center", fontsize=9, color=INK2)
        ax.text(xi + w / 2 + 0.01, vb + 0.012, f"{vb:.3f}", ha="center", fontsize=9, color=INK2)
    ax.set_xticks(x, [n for _, n in metrics])
    ax.tick_params(axis="x", colors=INK2)
    ax.set_ylim(0, 1.08); ax.yaxis.grid(True); ax.set_axisbelow(True)
    ax.set_title("The same model looks far better when test sites leak into training\n"
                 "(Kaggle dataset, identical random forest settings)", fontsize=11)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=1)
    save(fig, "leakage_random_vs_site_split.png")


def fig_cross(m: dict) -> None:
    cd = m["cross_dataset"]
    rows = list(cd.keys())
    cols = ["kaggle", "phiusiil", "benchmark"]
    pretty = {"kaggle": "Kaggle", "phiusiil": "PhiUSIIL", "benchmark": "Benchmark", "all three": "All three"}
    z = np.array([[cd[r][c]["f1"] for c in cols] for r in rows])
    fig, ax = plt.subplots(figsize=(5.8, 3.9))
    im = ax.imshow(z, cmap=SEQ, vmin=0, vmax=1, aspect="auto")
    for i in range(len(rows)):
        for j in range(len(cols)):
            ax.text(j, i, f"{z[i, j]:.2f}", ha="center", va="center",
                    color="white" if z[i, j] > 0.55 else INK, fontsize=11)
    ax.set_xticks(range(len(cols)), [pretty[c] for c in cols])
    ax.set_yticks(range(len(rows)), [pretty[r] for r in rows])
    ax.tick_params(axis="x", colors=INK2, top=True, labeltop=True, bottom=False, labelbottom=False)
    ax.set_xlabel("Tested on (unseen sites)"); ax.xaxis.set_label_position("top")
    ax.set_ylabel("Trained on")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-.5, len(cols)), minor=True); ax.set_yticks(np.arange(-.5, len(rows)), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2); ax.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03); cb.outline.set_visible(False)
    cb.set_label("F1 score", color=INK2)
    fig.suptitle("A model trained on one dataset transfers poorly to the others", x=0.02, ha="left",
                 fontsize=12, fontweight="semibold", y=1.06)
    save(fig, "cross_dataset_f1.png")


def fig_comparison(m: dict) -> None:
    comp = {k: v for k, v in m["comparison"].items()}
    order = sorted(comp, key=lambda k: comp[k]["f1"], reverse=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.2), sharey=True)
    for ax, key, title in ((axes[0], "f1", "F1 at the 0.5 cut-off"), (axes[1], "roc_auc", "ROC-AUC")):
        vals = [comp[k][key] if comp[k][key] is not None else 0.5 for k in order]
        hbars(ax, order, vals, [BLUE if "final" in k else GRAY for k in order], xlim=(0, 1.1))
        ax.set_title(title, fontsize=10.5)
    fig.suptitle("Model comparison on the held-out test sites", x=0.02, ha="left", fontsize=12,
                 fontweight="semibold", y=1.05)
    save(fig, "model_comparison.png")


def fig_curves(m: dict) -> None:
    comp = m["comparison"]
    series = [("Random forest, all 56 features (final)", "Random forest (final)", BLUE),
              ("Logistic regression", "Logistic regression", ORANGE),
              ("Single decision tree", "Single decision tree", AQUA)]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for key, label, color in series:
        c = comp[key]["curves"]
        axes[0].plot(c["fpr"], c["tpr"], color=color, lw=2, label=f"{label} (AUC {comp[key]['roc_auc']:.3f})")
        axes[1].plot(c["recall"], c["precision"], color=color, lw=2, label=f"{label} (AP {comp[key]['pr_auc']:.3f})")
    axes[0].plot([0, 1], [0, 1], color=AXIS, lw=1)
    axes[0].text(0.62, 0.56, "random guessing", color=MUTED, fontsize=8.5, rotation=38)
    axes[0].set(xlabel="False positive rate (benign URLs flagged)", ylabel="True positive rate (malicious caught)",
                title="ROC curve", xlim=(0, 1), ylim=(0, 1.02))
    axes[1].set(xlabel="Recall (malicious caught)", ylabel="Precision (flags that are right)",
                title="Precision-recall curve", xlim=(0, 1), ylim=(0, 1.02))
    for ax in axes:
        ax.grid(True); ax.set_axisbelow(True)
        ax.legend(loc="lower right" if ax is axes[0] else "lower left", fontsize=9)
    save(fig, "roc_pr_curves.png")


def fig_confusion(m: dict) -> None:
    t = m["test"]["max_f1"]
    z = np.array([[t["tn"], t["fp"]], [t["fn"], t["tp"]]])
    share = z / z.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.9, 3.9))
    ax.imshow(share, cmap=SEQ, vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{z[i, j]:,}\n{share[i, j]:.1%}", ha="center", va="center",
                    color="white" if share[i, j] > 0.55 else INK, fontsize=11)
    ax.set_xticks([0, 1], ["Predicted benign", "Predicted malicious"])
    ax.set_yticks([0, 1], ["Actually\nbenign", "Actually\nmalicious"])
    ax.tick_params(axis="x", colors=INK2)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([0.5], minor=True); ax.set_yticks([0.5], minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2); ax.tick_params(which="minor", length=0)
    ax.set_title(f"Confusion matrix on test sites (threshold {t['threshold']:.2f})", fontsize=11)
    save(fig, "confusion_matrix.png")


def fig_importance(m: dict) -> None:
    top = m["feature_importance"][:20]
    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    hbars(ax, [r["feature"] for r in top], [r["permutation"] for r in top], BLUE, fmt="{:.4f}")
    ax.set_xlim(0, max(r["permutation"] for r in top) * 1.16)
    ax.set_xlabel("Drop in PR-AUC when the feature is shuffled")
    ax.set_title("Top 20 features by permutation importance (test sites)")
    save(fig, "feature_importance.png")


def fig_trees(m: dict) -> None:
    c = m["trees_curve"]
    x, yv = [r["trees"] for r in c], [r["val_pr_auc"] for r in c]
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    ax.plot(x, yv, color=BLUE, lw=2, marker="o", ms=6, mec=SURFACE, mew=1.5)
    ax.set_xscale("log"); ax.set_xticks(x, [str(v) for v in x]); ax.minorticks_off()
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    ax.set(xlabel="Number of trees in the forest (log scale)", ylabel="Validation PR-AUC",
           title="More trees help quickly, then the gain flattens")
    ax.annotate(f"{yv[-1]:.4f}", (x[-1], yv[-1]), textcoords="offset points", xytext=(0, 9), ha="center",
                color=INK2, fontsize=9)
    ax.annotate(f"{yv[0]:.4f}", (x[0], yv[0]), textcoords="offset points", xytext=(12, -3), ha="left",
                color=INK2, fontsize=9)
    save(fig, "trees_vs_performance.png")


def fig_detection(m: dict) -> None:
    d = m["test"]["detection_rate_by_kind"]
    pretty = {"benchmark / malicious": "Benchmark - mixed phishing/malware", "kaggle / defacement": "Kaggle - defacement",
              "kaggle / malware": "Kaggle - malware", "kaggle / phishing": "Kaggle - phishing",
              "phiusiil / phishing": "PhiUSIIL - phishing"}
    order = sorted(d, key=lambda k: d[k]["detected"], reverse=True)
    fig, ax = plt.subplots(figsize=(7.6, 2.9))
    hbars(ax, [f"{pretty.get(k, k)}  (n={d[k]['n']:,})" for k in order], [d[k]["detected"] for k in order],
          BLUE, fmt="{:.1%}", xlim=(0, 1.12))
    ax.set_xticks([0, .25, .5, .75, 1], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_title("Share of malicious test URLs caught, by dataset and attack type")
    save(fig, "detection_rate_by_kind.png")


DEFENCE_NAMES = {"plain": "Model as trained", "no_query": "+ query-stripped view",
                 "multi_view": "+ per-folder views (hardened)"}


def fig_red_team(r: dict) -> None:
    """Evasion curves of each scorer, and the edits that hurt the plain model most."""
    d = r["defences"]
    styles = {"plain": (GRAY, "-", 2.2), "no_query": (ORANGE, "--", 2.0), "multi_view": (BLUE, "-", 2.8)}
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    for level, info in d.items():
        curve = info["attack"]["evaded_within_price"]
        xs, ys = [int(k) for k in curve], list(curve.values())
        color, dash, width = styles.get(level, (INK2, "-", 2))
        ax.plot(xs, ys, dash, color=color, linewidth=width, marker="o", markersize=4.5)
        ax.text(xs[-1] + 0.12, ys[-1], DEFENCE_NAMES.get(level, level), va="center", fontsize=9, color=INK2)
        ax.text(xs[0] - 0.12, ys[0], f"{ys[0]:.0%}", va="center", ha="right", fontsize=9, color=INK2)
    ax.set_xlim(0.4, 9.2); ax.set_ylim(0, 1.04)
    ax.set_xticks(range(1, 7))
    ax.set_yticks([0, .25, .5, .75, 1], ["0%", "25%", "50%", "75%", "100%"])
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    ax.set_xlabel("Attacker's budget: total price of the edits (1 = one free rename)")
    ax.set_ylabel("Caught phishing that gets through")
    ax.set_title(f"Red team: {r['config']['attack_sample']} phishing URLs from unseen sites "
                 "(external dataset), balanced cut-offs", fontsize=11)
    save(fig, "red_team_evasion.png")

    moves = {m["id"]: m for m in r["config"]["moves"]}
    change = d["plain"]["attack"]["mean_single_move_change"]
    order = sorted(change, key=lambda k: change[k])[:10]
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    hbars(ax, [f"{moves[k]['label']} (price {moves[k]['price']})" for k in order],
          [-change[k] for k in order], BLUE, fmt="-{:.2f}")
    ax.set_title("Average fall in score from one edit, model as trained", fontsize=11)
    save(fig, "red_team_single_edits.png")


def main() -> int:
    red_team = ROOT / "reports" / "red_team.json"
    if red_team.exists():
        fig_red_team(json.loads(red_team.read_text()))
    metrics = json.loads((ROOT / "reports" / "metrics.json").read_text())
    prep = json.loads((ROOT / "data" / "processed" / "prepare_report.json").read_text())
    fig_dataset(prep)
    fig_leakage(metrics)
    fig_cross(metrics)
    fig_comparison(metrics)
    fig_curves(metrics)
    fig_confusion(metrics)
    fig_importance(metrics)
    fig_trees(metrics)
    fig_detection(metrics)
    return 0


if __name__ == "__main__":
    sys.exit(main())
