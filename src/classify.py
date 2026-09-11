"""
Classification: left-fist vs right-fist motor imagery from mu/beta band power
features, evaluated with stratified k-fold cross-validation (not a single
train/test split -- with a few hundred epochs, one split is noisy).

Two classifiers are compared:
- LDA: the classical BCI baseline (Blankertz et al., 2011); a good sanity check
  because it has almost no hyperparameters to accidentally overfit.
- Linear SVM: usually a mild improvement, included for comparison.
"""

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def evaluate(X: np.ndarray, y: np.ndarray, n_splits: int = 5, random_state: int = 42):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    models = {
        "LDA": make_pipeline(StandardScaler(), LinearDiscriminantAnalysis()),
        "Linear SVM": make_pipeline(StandardScaler(), SVC(kernel="linear", C=1.0)),
    }

    results = {}
    for name, model in models.items():
        preds = cross_val_predict(model, X, y, cv=cv)
        acc = (preds == y).mean()
        cm = confusion_matrix(y, preds)
        results[name] = {"accuracy": acc, "confusion_matrix": cm, "predictions": preds}
    return results
