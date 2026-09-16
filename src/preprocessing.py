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

LABEL_MAP = {"T1": 0, "T2": 1}  # T1 = left fist (imagined or executed), T2 = right fist
CLASS_NAMES = ["left_fist", "right_fist"]

# Full 64-channel BCI2000 montage, in the order EEGMMIDB stores them.
ALL_CHANNELS = [
    "Fc5", "Fc3", "Fc1", "Fcz", "Fc2", "Fc4", "Fc6", "C5", "C3", "C1", "Cz", "C2",
    "C4", "C6", "Cp5", "Cp3", "Cp1", "Cpz", "Cp2", "Cp4", "Cp6", "Fp1", "Fpz", "Fp2",
    "Af7", "Af3", "Afz", "Af4", "Af8", "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6",
    "F8", "Ft7", "Ft8", "T7", "T8", "T9", "T10", "Tp7", "Tp8", "P7", "P5", "P3", "P1",
    "Pz", "P2", "P4", "P6", "P8", "Po7", "Po3", "Poz", "Po4", "Po8", "O1", "Oz", "O2", "Iz",
]


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
    labeled epochs for the T1/T2 cues (left vs. right fist, imagined or executed
    depending on which run was loaded).

    channels=None uses the full 64-channel montage (needed for CSP, which relies
    on having enough channels to find a good spatial filter).
    """
    signals, ch_labels, annotations = load_raw(edf_path)
    filtered = bandpass_filter(signals, FS)

    use_channels = list(ch_labels) if channels is None else list(channels)
    ch_idx = [ch_labels.index(c) for c in use_channels]

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

    return np.array(epochs), np.array(labels), use_channels


def epoch_dataset(edf_paths, channels=("C3", "C1", "Cz", "C2", "C4")):
    """Epoch and concatenate multiple runs (e.g. all runs of one condition for one subject)."""
    all_epochs, all_labels, ch_names = [], [], None
    for path in edf_paths:
        epochs, labels, ch_names = epoch_recording(path, channels)
        all_epochs.append(epochs)
        all_labels.append(labels)
    return np.concatenate(all_epochs, axis=0), np.concatenate(all_labels, axis=0), ch_names
