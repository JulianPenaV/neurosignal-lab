"""
Download open-access EEG recordings from PhysioNet's EEG Motor Movement/Imagery
Database (EEGMMIDB). No account or license agreement required -- the dataset is
public domain (ODC-PDDL).

Dataset details: 64-channel EEG at 160 Hz, 10-10 system, subjects performed or
imagined opening/closing their left or right fist while cued.

Runs used here (per-subject):
    R04, R08, R12  -> imagined motor movement, left fist (T1) vs right fist (T2)

Reference:
Schalk, G., McFarland, D.J., Hinterberger, T., Birbaumer, N., Wolpaw, J.R.
BCI2000: A General-Purpose Brain-Computer Interface (BCI) System.
IEEE TBME 51(6):1034-1043, 2004.
Goldberger AL, et al. PhysioBank, PhysioToolkit, and PhysioNet. Circulation 101(23), 2000.
"""

import os
import urllib.request

BASE_URL = "https://physionet.org/files/eegmmidb/1.0.0"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# Imagined-movement runs: left fist vs right fist
IMAGERY_RUNS = ["04", "08", "12"]


def download_subject(subject_id: int, runs=IMAGERY_RUNS, data_dir=DATA_DIR):
    """Download the imagined-movement runs for one subject (e.g. subject_id=1 -> S001)."""
    os.makedirs(data_dir, exist_ok=True)
    subj = f"S{subject_id:03d}"
    paths = []
    for run in runs:
        fname = f"{subj}R{run}.edf"
        dest = os.path.join(data_dir, fname)
        if not os.path.exists(dest):
            url = f"{BASE_URL}/{subj}/{fname}"
            print(f"Downloading {url} -> {dest}")
            urllib.request.urlretrieve(url, dest)
        else:
            print(f"Already have {dest}")
        paths.append(dest)
    return paths


if __name__ == "__main__":
    import sys

    subjects = [int(s) for s in sys.argv[1:]] or [1]
    for sid in subjects:
        download_subject(sid)
