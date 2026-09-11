"""
Feature extraction: per-epoch, per-channel band power in the mu and beta bands,
via Welch's method (averaged periodogram -- more stable than a single FFT on a
short, noisy epoch).

This gives each epoch a compact feature vector (n_channels x n_bands) instead of
raw time series, which is both what classical BCI pipelines use and much easier
to reason about / plot than a black-box deep model for a first pass.
"""

import numpy as np
from scipy.signal import welch

from .preprocessing import FS

BANDS = {
    "mu": (8.0, 12.0),
    "beta": (13.0, 30.0),
}


def band_power(epoch_1d: np.ndarray, fs: float = FS, band=(8.0, 12.0)):
    """Average power spectral density within [band[0], band[1]] Hz for one channel."""
    freqs, psd = welch(epoch_1d, fs=fs, nperseg=min(256, len(epoch_1d)))
    mask = (freqs >= band[0]) & (freqs <= band[1])
    return np.trapz(psd[mask], freqs[mask])


def extract_features(epochs: np.ndarray, fs: float = FS, bands=BANDS):
    """
    epochs: (n_epochs, n_channels, n_times)
    returns: (n_epochs, n_channels * n_bands) feature matrix, and feature names
    """
    n_epochs, n_channels, _ = epochs.shape
    band_names = list(bands.keys())
    feat = np.zeros((n_epochs, n_channels * len(band_names)))

    for e in range(n_epochs):
        col = 0
        for ch in range(n_channels):
            for band_name in band_names:
                feat[e, col] = band_power(epochs[e, ch, :], fs, bands[band_name])
                col += 1

    feature_names = [
        f"ch{ch}_{band_name}" for ch in range(n_channels) for band_name in band_names
    ]
    return feat, feature_names
