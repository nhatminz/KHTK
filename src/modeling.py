"""Fold-local preprocessing and modest CPU model searches."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from .feature_engineering import CATEGORICAL_FEATURES, FeatureBuilder, feature_list


class FoldBalancedClassifier(ClassifierMixin, BaseEstimator):
    """Set XGBoost positive weight from the current fit fold, never validation/test."""

    def __init__(self, estimator, balance=False):
        self.estimator = estimator
        self.balance = balance

    def fit(self, X, y):
        self.estimator_ = clone(self.estimator)
        positives = int(np.sum(np.asarray(y) == 1))
        negatives = len(y) - positives
        self.scale_pos_weight_ = negatives / max(positives, 1) if self.balance else 1.0
        self.estimator_.set_params(scale_pos_weight=self.scale_pos_weight_)
        self.estimator_.fit(X, y)
        self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X):
        check_is_fitted(self)
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        check_is_fitted(self)
        return self.estimator_.predict_proba(X)


def build_pipeline(estimator, scale=False, variant="safe", engineered=False):
    features = feature_list(variant, engineered)
    categorical = [c for c in features if c in CATEGORICAL_FEATURES]
    numeric = [c for c in features if c not in categorical]
    numeric_steps = [("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="<MISSING>", keep_empty_features=True)),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return Pipeline([
        ("features", FeatureBuilder(variant, engineered)),
        ("preprocess", ColumnTransformer([
            ("numeric", Pipeline(numeric_steps), numeric),
            ("categorical", categorical_pipeline, categorical),
        ], remainder="drop", sparse_threshold=0)),
        ("model", estimator),
    ])


def model_specs(seed=2026, jobs=2):
    models = {
        "Dummy": (DummyClassifier(strategy="prior"), False, {}),
        "LogisticRegression": (LogisticRegression(max_iter=2500, random_state=seed), True,
            {"model__C": [0.1, 1.0, 10.0], "model__class_weight": [None, "balanced"]}),
        "RandomForest": (RandomForestClassifier(random_state=seed, n_jobs=jobs), False,
            {"model__n_estimators": [120, 200], "model__max_depth": [6, 12, None],
             "model__min_samples_leaf": [2, 5, 10], "model__class_weight": [None, "balanced_subsample"]}),
        # Disable internal row-random early stopping so there is no hidden vehicle overlap.
        "HistGradientBoosting": (HistGradientBoostingClassifier(random_state=seed, early_stopping=False), False,
            {"model__max_iter": [100, 180], "model__learning_rate": [0.05, 0.1],
             "model__max_leaf_nodes": [7, 15], "model__l2_regularization": [0.0, 1.0],
             "model__class_weight": [None, "balanced"]}),
    }
    unavailable = {}
    try:
        from xgboost import XGBClassifier
        estimator = XGBClassifier(objective="binary:logistic", eval_metric="logloss", tree_method="hist",
                                  n_jobs=jobs, random_state=seed)
        models["XGBoost"] = (FoldBalancedClassifier(estimator), False,
            {"model__estimator__n_estimators": [100, 180], "model__estimator__max_depth": [2, 4],
             "model__estimator__learning_rate": [0.05, 0.1], "model__estimator__subsample": [0.8, 1.0],
             "model__estimator__colsample_bytree": [0.8, 1.0], "model__balance": [False, True]})
    except ImportError:
        unavailable["XGBoost"] = "Not installed; optional. sklearn baselines are still trained."
    return models, unavailable


def rebuild_variant(fitted_pipeline, variant="safe", engineered=False):
    return build_pipeline(clone(fitted_pipeline.named_steps["model"]),
                          scale="scaler" in fitted_pipeline.named_steps["preprocess"].named_transformers_["numeric"].named_steps,
                          variant=variant, engineered=engineered)
