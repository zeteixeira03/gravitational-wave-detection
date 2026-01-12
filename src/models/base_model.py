"""
Baseline feature engineering and logistic regression model for G2Net.

- compute_features(sample): (3, 4096) -> (n_features,) - imported from data.features
- LogRegBaseline: simple sklearn logistic regression on those features.
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression

# Import feature computation from the features module
from src.data.features import compute_features

# Re-export for backwards compatibility
__all__ = ['LogRegBaseline', 'compute_features']


# ---------------------------------------------------------------------
# Logistic regression baseline
# ---------------------------------------------------------------------

@dataclass
class LogRegBaseline:
    """
    Thin wrapper around sklearn's LogisticRegression for convenience.

    Example
    -------
    model = LogRegBaseline()
    model.fit(X_train, y_train)
    y_val_proba = model.predict_proba(X_val)
    y_val_pred  = model.predict(X_val, threshold=0.5)
    """
    # from @dataclass
    max_iter: int = 1000
    penalty: str | None = "l2"      # l2 regularization
    C: float = 1.0
    solver: str = "lbfgs"

    def __post_init__(self) -> None:
        self.model = LogisticRegression(
            max_iter=self.max_iter,
            penalty=self.penalty,
            C=self.C,
            solver=self.solver,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the logistic regression model on (X, y)."""
        self.model.fit(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability of class 1 for each row of X."""
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict binary labels given a decision threshold on p(y=1|x).

        Parameters
        ----------
        threshold : float
            Values >= threshold are mapped to class 1, else 0.
        """
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

