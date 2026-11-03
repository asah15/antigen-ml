"""
models_classical.py — Classical ML track for protein antigenicity.

Engineered descriptors (features.py) -> standardisation -> classifier.
Four models that span the usual strong baselines:

  logreg : L2 logistic regression (linear, calibrated-ish, interpretable)
  svm    : RBF-kernel SVM (classic VaxiJen-style nonlinear baseline)
  rf     : Random Forest (robust, handles the 475-dim sparse-ish space)
  xgb    : Gradient-boosted trees (usually the top classical performer)

All return an sklearn-compatible estimator so train/evaluate code is uniform.
"""

from __future__ import annotations
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


def _scaled(est):
    # tree models do not need scaling, but a uniform pipeline keeps the API clean
    return Pipeline([("scale", StandardScaler()), ("clf", est)])


def build_model(name: str, seed: int = 42):
    name = name.lower()
    if name == "logreg":
        return _scaled(LogisticRegression(max_iter=2000, C=1.0,
                                          class_weight="balanced"))
    if name == "svm":
        return _scaled(SVC(C=2.0, gamma="scale", probability=True,
                           class_weight="balanced", random_state=seed))
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=600, max_depth=None, min_samples_leaf=2,
            max_features="sqrt", class_weight="balanced_subsample",
            n_jobs=-1, random_state=seed)
    if name == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=700, max_depth=6, learning_rate=0.03,
            subsample=0.85, colsample_bytree=0.7, reg_lambda=2.0,
            min_child_weight=2, eval_metric="logloss",
            tree_method="hist", n_jobs=-1, random_state=seed)
    raise ValueError(f"unknown model '{name}'")


CLASSICAL_MODELS = ["logreg", "svm", "rf", "xgb"]
