"""
Nested cross-validation for CSP + linear classifier.

"Nested" here means: for every fold, CSP is *fit only on that fold's training
epochs*, then used to transform both train and test epochs. This is not
optional -- CSP is a supervised, data-dependent feature extractor (it looks at
the labels), so fitting it on the full dataset before cross-validating would
leak test-set label information into the features and inflate accuracy. This
is the single most common correctness bug in from-scratch CSP implementations
and the reason v1's simpler band-power features (which don't look at labels)
didn't need this extra care.
"""

import numpy as np
from scipy.stats import binomtest
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .csp import CSP


def _make_classifier(name: str):
    if name == "LDA":
        return LinearDiscriminantAnalysis()
    if name == "Linear SVM":
        return SVC(kernel="linear", C=1.0)
    raise ValueError(name)


def evaluate_csp(
    epochs: np.ndarray,
    labels: np.ndarray,
    n_components: int = 6,
    n_splits: int = 5,
    classifiers=("LDA", "Linear SVM"),
    random_state: int = 42,
):
    """
    epochs: (n_epochs, n_channels, n_times), labels: (n_epochs,) in {0, 1}
    Returns {classifier_name: {"accuracy", "confusion_matrix", "predictions",
                                "fold_accuracies", "p_value", "n_epochs"}}
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    n = len(labels)

    results = {name: {"preds": np.zeros(n, dtype=int), "fold_acc": []} for name in classifiers}

    for train_idx, test_idx in cv.split(epochs, labels):
        csp = CSP(n_components=n_components)
        csp.fit(epochs[train_idx], labels[train_idx])
        X_train = csp.transform(epochs[train_idx])
        X_test = csp.transform(epochs[test_idx])

        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        for name in classifiers:
            clf = _make_classifier(name)
            clf.fit(X_train, labels[train_idx])
            fold_preds = clf.predict(X_test)
            results[name]["preds"][test_idx] = fold_preds
            results[name]["fold_acc"].append(float((fold_preds == labels[test_idx]).mean()))

    out = {}
    for name in classifiers:
        preds = results[name]["preds"]
        acc = float((preds == labels).mean())
        n_correct = int((preds == labels).sum())
        # Two-sided exact binomial test against chance (p=0.5) -- standard way to report
        # whether a single-subject BCI decode is significantly above chance (Muller-Putz
        # et al., 2008 recommend this over a normal approximation at these sample sizes).
        p_value = binomtest(n_correct, n, 0.5).pvalue
        out[name] = {
            "accuracy": acc,
            "confusion_matrix": confusion_matrix(labels, preds),
            "predictions": preds,
            "fold_accuracies": results[name]["fold_acc"],
            "p_value": float(p_value),
            "n_epochs": n,
            "n_correct": n_correct,
        }
    return out


def fit_csp_for_visualization(epochs: np.ndarray, labels: np.ndarray, n_components: int = 6):
    """
    Fit CSP on the FULL dataset (train+test) purely for plotting spatial patterns.
    Never use this fit for accuracy numbers -- see the leakage note in evaluate_csp.
    """
    csp = CSP(n_components=n_components)
    csp.fit(epochs, labels)
    return csp
