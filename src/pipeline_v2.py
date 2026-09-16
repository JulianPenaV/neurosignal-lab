"""
v2 pipeline: CSP + linear classifiers, evaluated on both imagined and executed
motor-movement conditions, across multiple subjects, with per-decode
significance testing and a paired imagined-vs-executed comparison.

This directly answers the two open questions v1 (results/REPORT.md) left
unresolved:
    1. Is naive band-power decoding weak because of the feature set, or
       because imagined movement is inherently hard to decode? -> CSP tests
       the feature-set hypothesis; running both conditions tests the other.
    2. How much of a gap is there between imagined and executed decoding,
       specifically, on the same subjects? -> paired comparison below.

Usage:
    ./.venv/bin/python -m src.pipeline_v2 1 2 3 4 5 6 7 8 9 10
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

from .classify_csp import evaluate_csp, fit_csp_for_visualization
from .download_data import EXECUTED_RUNS, IMAGERY_RUNS, download_subject
from .layout import build_layout
from .preprocessing import epoch_dataset

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "v2")
CONDITIONS = {"imagined": IMAGERY_RUNS, "executed": EXECUTED_RUNS}
N_COMPONENTS = 6


def run_subject_condition(subject_id, runs):
    paths = download_subject(subject_id, runs=runs)
    epochs, labels, channels = epoch_dataset(paths, channels=None)  # full 64-ch montage
    results = evaluate_csp(epochs, labels, n_components=N_COMPONENTS)
    return results, epochs, labels, channels


def run(subjects):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # {condition: {subject: {classifier: metrics}}}
    all_results = {cond: {} for cond in CONDITIONS}
    # keep the raw epochs/labels/channels for the best-accuracy case, for the pattern plot
    best = {"acc": -1, "epochs": None, "labels": None, "channels": None, "tag": None}

    for sid in subjects:
        for cond, runs in CONDITIONS.items():
            print(f"--- subject {sid}, {cond} ---")
            results, epochs, labels, channels = run_subject_condition(sid, runs)
            all_results[cond][sid] = results
            lda_acc = results["LDA"]["accuracy"]
            print(f"  LDA acc={lda_acc:.3f} (p={results['LDA']['p_value']:.4f}, "
                  f"n={results['LDA']['n_epochs']}), "
                  f"SVM acc={results['Linear SVM']['accuracy']:.3f}")
            if lda_acc > best["acc"]:
                best.update(acc=lda_acc, epochs=epochs, labels=labels,
                            channels=channels, tag=f"S{sid:03d}_{cond}")

    # ---- Paired imagined-vs-executed comparison (LDA accuracy, per subject) ----
    subj_ids = sorted(all_results["imagined"].keys())
    imagined_acc = np.array([all_results["imagined"][s]["LDA"]["accuracy"] for s in subj_ids])
    executed_acc = np.array([all_results["executed"][s]["LDA"]["accuracy"] for s in subj_ids])

    try:
        stat, wilcoxon_p = wilcoxon(executed_acc, imagined_acc)
    except ValueError:
        # wilcoxon fails if all differences are zero
        stat, wilcoxon_p = float("nan"), float("nan")

    # ---- Plot: paired per-subject accuracy, imagined vs executed ----
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(subj_ids))
    width = 0.35
    ax.bar(x - width / 2, imagined_acc, width, label="imagined", color="tab:blue")
    ax.bar(x + width / 2, executed_acc, width, label="executed", color="tab:green")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="chance")
    ax.set_xticks(x)
    ax.set_xticklabels([f"S{s:03d}" for s in subj_ids])
    ax.set_ylabel("LDA accuracy (5-fold CV)")
    ax.set_title("CSP decoding accuracy: imagined vs. executed movement")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "imagined_vs_executed.png"), dpi=150)
    plt.close(fig)

    # ---- Plot: CSP spatial patterns for the best-accuracy subject/condition ----
    csp = fit_csp_for_visualization(best["epochs"], best["labels"], n_components=N_COMPONENTS)
    pos = build_layout(best["channels"])
    xs = np.array([pos[c][0] for c in best["channels"] if c in pos])
    ys = np.array([pos[c][1] for c in best["channels"] if c in pos])
    keep_idx = [i for i, c in enumerate(best["channels"]) if c in pos]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    for ax, comp_idx, title in [(axes[0], 0, "Pattern 1 (class A)"),
                                  (axes[1], -1, "Pattern 6 (class B)")]:
        weights = csp.patterns_[comp_idx, keep_idx]
        vmax = np.abs(weights).max()
        sc = ax.scatter(xs, ys, c=weights, cmap="RdBu_r", vmin=-vmax, vmax=vmax, s=160,
                         edgecolors="k", linewidths=0.5)
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.axis("off")
        fig.colorbar(sc, ax=ax, shrink=0.7)
    fig.suptitle(f"CSP spatial patterns -- {best['tag']} (schematic layout, see src/layout.py)")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "csp_patterns.png"), dpi=150)
    plt.close(fig)

    # ---- Summary JSON ----
    def summarize(cond):
        return {
            str(s): {
                clf: {
                    "accuracy": round(all_results[cond][s][clf]["accuracy"], 4),
                    "p_value": round(all_results[cond][s][clf]["p_value"], 5),
                    "n_epochs": all_results[cond][s][clf]["n_epochs"],
                }
                for clf in ("LDA", "Linear SVM")
            }
            for s in subj_ids
        }

    summary = {
        "subjects": subj_ids,
        "n_components": N_COMPONENTS,
        "results": {cond: summarize(cond) for cond in CONDITIONS},
        "paired_comparison": {
            "mean_imagined_lda_acc": round(float(imagined_acc.mean()), 4),
            "mean_executed_lda_acc": round(float(executed_acc.mean()), 4),
            "wilcoxon_statistic": float(stat),
            "wilcoxon_p_value": float(wilcoxon_p),
        },
        "best_case": best["tag"],
    }
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    subjects = [int(s) for s in sys.argv[1:]] or list(range(1, 11))
    run(subjects)
