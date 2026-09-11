# NeuroSignal Lab

A from-scratch EEG signal processing and motor-imagery classification pipeline,
built on open-access data from PhysioNet's EEG Motor Movement/Imagery Database
(EEGMMIDB).

## Why this project exists

This is the first hands-on technical building block toward **SynapseArt** (EEG
signal processing → generative art, one of the [Synaptica](../Synaptica_Project_Description.md.pdf)
sub-projects) and toward the **Frequency Generator**'s eventual need to *validate*
that a given frequency protocol actually changes measurable brain activity. Both
of those currently exist as concepts; this repo is a real, working, versioned
piece of the underlying skill — reading raw biosignal files, filtering them
correctly, turning them into features, and evaluating a classifier honestly
(cross-validated, not cherry-picked).

It doubles as a portfolio piece for two audiences at once:

- **PhD admissions committees** want evidence of independent research capacity —
  not a claim of a skill on a resume, but a repo they can open, a pipeline that
  runs, and a write-up that reasons about *why* a result looks the way it does
  (including when it's not impressive). See [`results/REPORT.md`](results/REPORT.md).
- **Biotech / neurotech hiring managers** want to see the specific technical
  skills listed on a resume (EEG signal processing, biosensing, ML) actually
  exercised on real data, not just named.

## What it does

1. **Download** — pulls open, license-free EEG recordings directly from
   PhysioNet (no account needed) for one or more subjects performing *imagined*
   left-fist / right-fist motor movement. ([`src/download_data.py`](src/download_data.py))
2. **Preprocess** — band-pass filters each recording to the mu/beta band
   (8–30 Hz, where motor-imagery ERD/ERS lives) and slices it into labeled
   epochs around each cue. ([`src/preprocessing.py`](src/preprocessing.py))
3. **Feature extraction** — computes mu and beta band power (via Welch PSD) at
   each sensorimotor electrode (C3, C1, Cz, C2, C4) per epoch.
   ([`src/features.py`](src/features.py))
4. **Classification** — evaluates LDA and linear SVM classifiers with
   stratified 5-fold cross-validation. ([`src/classify.py`](src/classify.py))
5. **Report** — saves PSD plots, confusion matrices, and an accuracy summary to
   `results/`. ([`src/pipeline.py`](src/pipeline.py))

## Running it

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
./.venv/bin/python -m src.pipeline 1 2 3 4 5   # subject IDs to download + analyze
```

Data is cached in `data/` after the first download (not committed to git — see
`.gitignore` — it's a few MB per subject and trivially re-downloadable from the
public dataset).

## Results (current)

See [`results/REPORT.md`](results/REPORT.md) for the full write-up. Short
version: pooled cross-subject decoding sits at chance (~50%), and so does naive
per-subject decoding with this simple feature set — an honest, expected result
given the known difficulty of *imagined* (vs. executed) movement and the
limitations of raw band-power features without spatial filtering. The report
lays out exactly what the standard next step (Common Spatial Patterns) is and
why it should help.

## Data source & citation

Schalk, G., McFarland, D.J., Hinterberger, T., Birbaumer, N., Wolpaw, J.R.
*BCI2000: A General-Purpose Brain-Computer Interface (BCI) System.*
IEEE Transactions on Biomedical Engineering 51(6):1034-1043, 2004.

Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh, Mark RG, Mietus JE,
Moody GB, Peng CK, Stanley HE. *PhysioBank, PhysioToolkit, and PhysioNet:
Components of a New Research Resource for Complex Physiologic Signals.*
Circulation 101(23):e215-e220, 2000.

## Next steps

- [ ] Add Common Spatial Patterns (CSP) spatial filtering before feature
      extraction — the standard fix for the weak band-power baseline seen here.
- [ ] Compare *executed* movement runs (R03/R07/R11) against *imagined* runs to
      quantify the imagery-vs-execution gap directly, on this same dataset.
- [ ] Try a per-subject-normalized pooled model (z-score features within
      subject before pooling) as a cheaper alternative to full per-subject models.
- [ ] Once the classification baseline is solid, start the actual SynapseArt
      fork: map band-power feature vectors to generative visual/audio parameters
      instead of a class label.
- [ ] Swap in a mental-wellness-relevant public dataset (e.g. a stress/affect
      EEG or peripheral-physiology dataset) once this pipeline is validated on a
      clean, well-studied benchmark — ties the same pipeline directly back to
      Synaptica's mental-wellness mission and the Frequency Generator's
      validation needs.
