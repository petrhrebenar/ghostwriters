"""Per-author binary classifiers for authorship attribution.

Each author gets an independent binary classifier (one-vs-rest).
Supports LogisticRegression and XGBoost backends.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def _build_binary_classifier(
    classifier_type: str = "logistic",
    n_positive: int = 1,
    n_negative: int = 1,
) -> LogisticRegression | XGBClassifier:
    """Create an unfitted binary classifier.

    Parameters
    ----------
    classifier_type : str
        "logistic" or "xgboost".
    n_positive : int
        Number of positive samples (used for XGBoost scale_pos_weight).
    n_negative : int
        Number of negative samples.

    Returns
    -------
    Unfitted sklearn-compatible estimator.
    """
    if classifier_type == "logistic":
        return LogisticRegression(
            solver="lbfgs",
            class_weight="balanced",
            max_iter=1000,
            C=1.0,
            random_state=42,
        )
    elif classifier_type == "xgboost":
        scale = n_negative / max(n_positive, 1)
        return XGBClassifier(
            max_depth=3,
            n_estimators=100,
            learning_rate=0.1,
            scale_pos_weight=scale,
            reg_alpha=1.0,
            reg_lambda=1.0,
            random_state=42,
            eval_metric="logloss",
            verbosity=0,
        )
    else:
        raise ValueError(f"Unknown classifier_type: {classifier_type}. Use 'logistic' or 'xgboost'.")


class PerAuthorClassifier:
    """Wrapper that trains one binary classifier per author.

    Parameters
    ----------
    classifier_type : str
        "logistic" or "xgboost".
    authors : list[str], optional
        Ordered list of author names. If None, derived from y during fit().
    """

    def __init__(
        self,
        classifier_type: str = "logistic",
        authors: Optional[list[str]] = None,
    ):
        self.classifier_type = classifier_type
        self.authors: list[str] = authors or []
        self.classifiers: dict[str, LogisticRegression | XGBClassifier] = {}
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PerAuthorClassifier":
        """Train one binary classifier per author.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Feature matrix.
        y : np.ndarray, shape (n_samples,)
            Author labels.

        Returns
        -------
        self
        """
        if not self.authors:
            self.authors = sorted(np.unique(y).tolist())

        self.classifiers = {}
        for author in self.authors:
            y_binary = (y == author).astype(int)
            n_pos = int(y_binary.sum())
            n_neg = len(y_binary) - n_pos

            clf = _build_binary_classifier(
                classifier_type=self.classifier_type,
                n_positive=n_pos,
                n_negative=n_neg,
            )
            clf.fit(X, y_binary)
            self.classifiers[author] = clf

        self._fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability of authorship for each author.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        np.ndarray, shape (n_samples, n_authors)
            Probability of the positive class (= this author) per classifier.
        """
        if not self._fitted:
            raise RuntimeError("PerAuthorClassifier has not been fitted yet.")

        probas = np.zeros((X.shape[0], len(self.authors)))
        for i, author in enumerate(self.authors):
            clf = self.classifiers[author]
            # predict_proba returns (n_samples, 2) — column 1 is P(positive)
            probas[:, i] = clf.predict_proba(X)[:, 1]
        return probas

    def predict_rank(self, X: np.ndarray) -> list[list[str]]:
        """Rank authors by probability for each sample.

        Returns
        -------
        list[list[str]]
            For each sample, authors sorted by descending probability.
        """
        probas = self.predict_proba(X)
        rankings = []
        for row in probas:
            ranked_idx = np.argsort(-row)
            rankings.append([self.authors[i] for i in ranked_idx])
        return rankings

    def get_author_names(self) -> list[str]:
        """Return the ordered list of author names."""
        return list(self.authors)

    def get_coefficients(self) -> dict[str, np.ndarray]:
        """Extract feature coefficients (logistic regression only).

        Returns
        -------
        dict[str, np.ndarray]
            Mapping from author to coefficient vector (n_features,).

        Raises
        ------
        ValueError
            If classifier_type is not 'logistic'.
        """
        if self.classifier_type != "logistic":
            raise ValueError("Coefficients only available for logistic regression.")
        return {
            author: clf.coef_[0]
            for author, clf in self.classifiers.items()
        }

    def get_feature_importances(self) -> dict[str, np.ndarray]:
        """Extract feature importances (XGBoost only).

        Returns
        -------
        dict[str, np.ndarray]
            Mapping from author to importance vector (n_features,).

        Raises
        ------
        ValueError
            If classifier_type is not 'xgboost'.
        """
        if self.classifier_type != "xgboost":
            raise ValueError("Feature importances only available for XGBoost.")
        return {
            author: clf.feature_importances_
            for author, clf in self.classifiers.items()
        }
