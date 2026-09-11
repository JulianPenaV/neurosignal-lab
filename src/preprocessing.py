"""
Preprocessing: load raw EDF recordings, band-pass filter to the sensorimotor
mu/beta range, and slice into labeled epochs around each imagery cue.

Design choices (and why):
- 8-30 Hz band-pass: motor imagery event-related desynchronization/synchronization
  (ERD/ERS) lives in the mu (8-12 Hz) and beta (13-30 Hz) rhythms over sensorimotor
  cortex. Filtering to this band removes slow drift and high-frequency EMG/noise
  without discarding the signal of interest.
- 4th-order Butterworth, zero-phase (filtfilt): standard choice for EEG -- avoids
  the phase distortion a causal filter would introduce.
- Epoch window 0.5s-3.5s post-cue: skips the initial visual-cue reaction transient
  and captures the sustained imagery period (cues are ~4.1-4.2s long here).
- T0 (rest) epochs are dropped: this is a binary left-vs-right decode, not a
  rest-vs-task decode.
"""

import numpy as np
import pyedflib
from scipy.signal import butter, filtfilt

FS = 160.0  # Hz, fixed by the EEGMMIDB recording hardware
LOWCUT, HIGHCUT = 8.0, 30.0
EPOCH_START, EPOCH_END = 0.5, 3.5  # seconds relative to cue onset

LABEL_MAP = {"T1": 0, "T2": 1}  # T1 = left fist imagery, T2 = right fist imagery
CLASS_NAMES = ["left_fist", "right_fist"]


def bandpass_filter(data: np.ndarray, fs: float = FS, low=LOWCUT, high=HIGHCUT, order=4):
    """Zero-phase Butterworth band-pass, applied along the last axis."""
    nyq = fs / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, data, axis=-1)


def load_raw(edf_path: str):
    """Return (signals [n_channels, n_samples], channel_labels, annotations)."""
    f = pyedflib.EdfReader(edf_path)
    n_ch = f.signals_in_file
    n_samp = f.getNSamples()[0]
    signals = np.zeros((n_ch, n_samp))
    for i in range(n_ch):
        signals[i, :] = f.readSignal(i)
    labels = [l.strip(".").strip() for l in f.getSignalLabels()]
    onsets, durations, descriptions = f.readAnnotations()
    f.close()
    return signals, labels, list(zip(onsets, durations, descriptions))


def epoch_recording(edf_path: str, channels=("C3", "C1", "Cz", "C2", "C4")):
    """
    Load one EDF run, band-pass filter, and slice into (n_epochs, n_channels, n_times)
    labeled epochs for the T1/T2 imagery cues.
    """
    signals, ch_labels, annotations = load_raw(edf_path)
    filtered = bandpass_filter(signals, FS)

    ch_idx = [ch_labels.index(c) for c in channels]

    epochs, labels = [], []
    for onset, _duration, desc in annotations:
        if desc not in LABEL_MAP:
            continue
        start = int((onset + EPOCH_START) * FS)
        end = int((onset + EPOCH_END) * FS)
        if end > filtered.shape[1]:
            continue
        epochs.append(filtered[np.ix_(ch_idx, range(start, end))])
        labels.append(LABEL_MAP[desc])

    return np.array(epochs), np.array(labels), list(channels)


def epoch_dataset(edf_paths, channels=("C3", "C1", "Cz", "C2", "C4")):
    """Epoch and concatenate multiple runs (e.g. all imagery runs for one subject)."""
    all_epochs, all_labels = [], []
    for path in edf_paths:
        epochs, labels, ch_names = epoch_recording(path, channels)
        all_epochs.append(epochs)
        all_labels.append(labels)
    return np.concatenate(all_epochs, axis=0), np.concatenate(all_labels, axis=0), list(channels)
