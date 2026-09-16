# NeuroSignal Lab

A from-scratch EEG signal processing and motor-imagery classification pipeline,
built on open-access data from PhysioNet's EEG Motor Movement/Imagery Database
(EEGMMIDB).

## Why this project exists

This is the first hands-on technical building block toward **SynapseArt** (EEG
signal processing → generative art, one of the Synaptica ecosystem's
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

There are two pipelines, v1 and v2, kept side by side deliberately — the
repo's history is itself part of the portfolio value: a real result, an
honest diagnosis of its weakness, and a follow-up that specifically targets
that weakness and reports what happened.

**v1 — band-power baseline** (chance-level result; establishes the pipeline
and the honest-reporting standard for everything after it)
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

**v2 — Common Spatial Patterns (CSP), from scratch** (fixes the v1 weakness;
adds a second condition and real statistics)
1. **CSP feature extraction**, implemented from the generalized-eigenvalue
   formulation (no MNE dependency) — finds spatial filters that maximize the
   variance ratio between left- and right-fist trials across all 64 channels,
   with the fit correctly nested inside each cross-validation fold to avoid
   label leakage. ([`src/csp.py`](src/csp.py), [`src/classify_csp.py`](src/classify_csp.py))
2. **Two conditions, same subjects** — runs both *imagined* and *executed*
   movement through the identical pipeline and statistically compares them
   (Wilcoxon signed-rank, paired by subject). ([`src/pipeline_v2.py`](src/pipeline_v2.py))
3. **Per-decode significance** — exact binomial test against chance for every
   subject/condition result, not just an accuracy number.
4. **Spatial pattern visualization** — plots which electrodes each CSP filter
   actually weights, on a schematic scalp layout derived from the standard
   10-10 electrode naming convention. ([`src/layout.py`](src/layout.py))

## Running it

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt

./.venv/bin/python -m src.pipeline 1 2 3 4 5              # v1: band power
./.venv/bin/python -m src.pipeline_v2 1 2 3 4 5 6 7 8 9 10 # v2: CSP, both conditions
```

Data is cached in `data/` after the first download (not committed to git — see
`.gitignore` — it's a few MB per subject and trivially re-downloadable from the
public dataset).

## Results (current)

**v1** ([`results/REPORT.md`](results/REPORT.md)): pooled and per-subject
decoding both sat at chance (~50%) with simple band-power features — an
honest null result that pointed straight at CSP as the fix.

**v2** ([`results/v2/REPORT_v2.md`](results/v2/REPORT_v2.md)): CSP lifts mean
per-subject accuracy to **64%** (imagined and executed, 10 subjects), with
individual subjects reaching **95.6%** (p < 0.0001) — confirming the v1
feature-set diagnosis was correct. Accuracy varies widely by subject (40–96%),
consistent with the well-documented "BCI illiteracy" phenomenon in the BCI
literature. Imagined vs. executed movement showed no significant difference
on this sample (Wilcoxon p=1.0, n=10 subjects) — a genuinely open question the
report discusses rather than glosses over.

## Data source & citation

Schalk, G., McFarland, D.J., Hinterberger, T., Birbaumer, N., Wolpaw, J.R.
*BCI2000: A General-Purpose Brain-Computer Interface (BCI) System.*
IEEE Transactions on Biomedical Engineering 51(6):1034-1043, 2004.

Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh, Mark RG, Mietus JE,
Moody GB, Peng CK, Stanley HE. *PhysioBank, PhysioToolkit, and PhysioNet:
Components of a New Research Resource for Complex Physiologic Signals.*
Circulation 101(23):e215-e220, 2000.

## Next steps

v1's next steps (CSP, imagined-vs-executed comparison) are done in v2. What's
still open, in priority order — see [`results/v2/REPORT_v2.md`](results/v2/REPORT_v2.md#next-steps)
for the full reasoning:

- [ ] Scale to more of the 109 available EEGMMIDB subjects to get a real
      estimate of the "BCI illiteracy" rate instead of an n=10 sketch of it.
- [ ] Per-subject CSP component-count tuning, nested inside cross-validation.
- [ ] Add an artifact-rejection step (EOG regression or ICA).
- [ ] Fork into the actual SynapseArt prototype: replace the classifier step
      with a mapping from CSP features to generative visual/audio parameters.
- [ ] Swap in a mental-wellness-relevant public dataset (e.g. a stress/affect
      EEG or peripheral-physiology dataset) once the pipeline is validated on a
      clean, well-studied benchmark — ties this directly back to Synaptica's
      mental-wellness mission and the Frequency Generator's validation needs.
