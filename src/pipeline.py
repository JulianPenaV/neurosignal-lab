"""
End-to-end pipeline: download -> preprocess -> epoch -> feature-extract ->
classify -> report, for one or more subjects from the PhysioNet EEGMMIDB
open motor-imagery dataset.

Usage:
    ./.venv/bin/python -m src.pipeline 1 2 3
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

from .classify import evaluate
from .download_data import download_subject
from .features import BANDS, extract_features
from .preprocessing import CLASS_NAMES, FS, epoch_dataset

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def run(subjects):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_paths = []
    for sid in subjects:
        all_paths.extend(download_subject(sid))

    epochs, labels, channels = epoch_dataset(all_paths)
    print(f"Epoched dataset: {epochs.shape[0]} epochs, {epochs.shape[1]} channels, "
          f"{epochs.shape[2]} samples/epoch")
    print(f"Class balance: {CLASS_NAMES[0]}={ (labels==0).sum() }, "
          f"{CLASS_NAMES[1]}={ (labels==1).sum() }")

    X, feature_names = extract_features(epochs)
    results = evaluate(X, labels)

    # --- PSD plot: mu/beta power on C3 vs C4, split by class ---
    c3_idx, c4_idx = channels.index("C3"), channels.index("C4")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for cls, name, color in [(0, "left fist", "tab:blue"), (1, "right fist", "tab:orange")]:
        mask = labels == cls
        for ax, ch_idx, title in [(axes[0], c3_idx, "C3 (left motor cortex)"),
                                    (axes[1], c4_idx, "C4 (right motor cortex)")]:
            psds = []
            for e in np.where(mask)[0]:
                freqs, psd = welch(epochs[e, ch_idx, :], fs=FS, nperseg=256)
                psds.append(psd)
            mean_psd = np.mean(psds, axis=0)
            ax.plot(freqs, mean_psd, label=name, color=color)
            ax.set_title(title)
            ax.set_xlim(4, 35)
            ax.set_xlabel("Hz")
    axes[0].set_ylabel("Power")
    axes[0].legend()
    fig.suptitle("Mean PSD by imagined movement side (ERD/ERS check)")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "psd_by_class.png"), dpi=150)
    plt.close(fig)

    # --- Confusion matrices ---
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4))
    if len(results) == 1:
        axes = [axes]
    for ax, (name, r) in zip(axes, results.items()):
        cm = r["confusion_matrix"]
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_xticklabels(CLASS_NAMES)
        ax.set_yticks([0, 1]); ax.set_yticklabels(CLASS_NAMES)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"{name} (acc={r['accuracy']:.2f})")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "confusion_matrices.png"), dpi=150)
    plt.close(fig)

    # --- Summary JSON ---
    summary = {
        "subjects": subjects,
        "n_epochs": int(epochs.shape[0]),
        "n_channels": epochs.shape[1],
        "channels": channels,
        "class_balance": {CLASS_NAMES[0]: int((labels == 0).sum()),
                           CLASS_NAMES[1]: int((labels == 1).sum())},
        "accuracy": {name: float(r["accuracy"]) for name, r in results.items()},
    }
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    subjects = [int(s) for s in sys.argv[1:]] or [1, 2, 3]
    run(subjects)
