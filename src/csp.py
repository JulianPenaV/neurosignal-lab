"""
Common Spatial Patterns (CSP) -- from scratch, no MNE dependency.

CSP finds linear combinations of channels (spatial filters) that maximize the
variance of one class while minimizing the variance of the other. It's the
standard feature-extraction step for two-class motor-imagery BCI decoding
(Ramoser, Muller-Gerking & Pfurtscheller, 1998; Blankertz et al., 2008), and
the direct answer to the weak baseline documented in v1 (results/REPORT.md):
raw per-electrode band power ignores the *spatial* pattern of activity across
electrodes, which is exactly what distinguishes left- from right-hand imagery
(contralateral sensorimotor ERD).

Algorithm (binary case):
1. For each trial (epoch), compute the spatial covariance matrix, normalized
   by its trace so trial-to-trial amplitude differences don't dominate.
2. Average the normalized covariances within each class -> C1, C2.
3. Solve the generalized eigenvalue problem  C1 w = lambda (C1 + C2) w.
   Eigenvectors with lambda near 1 maximize variance for class 1 (and
   minimize it for class 2); eigenvectors with lambda near 0 do the reverse.
4. Keep the top-m and bottom-m eigenvectors as spatial filters.
5. Feature per trial per filter: log-normalized variance of the filtered
   signal -- the standard CSP feature (Ramoser et al., 1998, Eq. 5).

IMPORTANT -- data leakage: CSP filters are *learned from labeled data*, so
(like any other data-dependent feature extractor) they must be fit only on
the training fold of each cross-validation split, never on the full dataset.
This module's `fit`/`transform` split enforces that when used correctly from
the evaluation loop in pipeline_v2.py.
"""

import numpy as np
from scipy.linalg import eigh


class CSP:
    def __init__(self, n_components: int = 6):
        """n_components: total spatial filters kept = n_components (split evenly
        between the two variance extremes, so n_components should be even)."""
        if n_components % 2 != 0:
            raise ValueError("n_components must be even (split between class extremes)")
        self.n_components = n_components
        self.filters_ = None  # (n_components, n_channels)
        self.patterns_ = None  # (n_channels, n_components) -- for visualization

    @staticmethod
    def _trial_covariance(trial: np.ndarray) -> np.ndarray:
        """trial: (n_channels, n_times) -> normalized spatial covariance (n_channels, n_channels)."""
        cov = trial @ trial.T
        trace = np.trace(cov)
        return cov / trace if trace > 0 else cov

    def fit(self, X: np.ndarray, y: np.ndarray):
        """X: (n_epochs, n_channels, n_times), y: (n_epochs,) in {0, 1}."""
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError("CSP as implemented here is binary-only")

        covs = {}
        for c in classes:
            trial_covs = [self._trial_covariance(X[i]) for i in np.where(y == c)[0]]
            covs[c] = np.mean(trial_covs, axis=0)

        C1, C2 = covs[classes[0]], covs[classes[1]]
        # Generalized eigenvalue problem: C1 w = lambda (C1 + C2) w
        eigvals, eigvecs = eigh(C1, C1 + C2)  # ascending eigval order

        m = self.n_components // 2
        # Smallest eigenvalues -> best for class[1]; largest -> best for class[0]
        low = eigvecs[:, :m]
        high = eigvecs[:, -m:]
        W = np.concatenate([high, low], axis=1)  # (n_channels, n_components)

        self.filters_ = W.T  # (n_components, n_channels), rows are spatial filters
        self.patterns_ = np.linalg.pinv(W)  # (n_components, n_channels): the associated
        # spatial *patterns* (how each filter's source projects back onto the scalp) --
        # this is what should be visualized, not the filter itself (Haufe et al., 2014).
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """X: (n_epochs, n_channels, n_times) -> (n_epochs, n_components) log-variance features."""
        if self.filters_ is None:
            raise RuntimeError("CSP must be fit before transform")
        n_epochs = X.shape[0]
        feats = np.zeros((n_epochs, self.n_components))
        for i in range(n_epochs):
            projected = self.filters_ @ X[i]  # (n_components, n_times)
            var = np.var(projected, axis=1)
            feats[i] = np.log(var / var.sum())
        return feats

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)
