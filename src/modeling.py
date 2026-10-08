"""Fold-local preprocessing and modest CPU model searches."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from .feature_engineering import CATEGORICAL_FEATURES, FeatureBuilder, feature_list


class LightweightTabPFN(ClassifierMixin, BaseEstimator):
    """Native TabPFN inputs; full train context by default and lazy reload."""

    def __init__(self, random_state=2026, max_train_rows=0, device="cpu",
                 n_estimators=1, model_version="v2", n_jobs=2):
        self.random_state = random_state
        self.max_train_rows = max_train_rows
        self.device = device
        self.n_estimators = n_estimators
        self.model_version = model_version
        self.n_jobs = n_jobs

    def _restore(self):
        import os
        # Never open a login browser implicitly; report missing access instead.
        os.environ.setdefault("TABPFN_NO_BROWSER", "1")
        from tabpfn import TabPFNClassifier
        self.estimator_ = TabPFNClassifier.create_default_for_version(
            self.model_version, device=self.device, n_estimators=self.n_estimators,
            random_state=self.random_state, n_preprocessing_jobs=self.n_jobs,
            ignore_pretraining_limits=(self.device == "cpu" and len(self.context_y_) > 1000),
            fit_mode="fit_preprocessors")
        self.estimator_.fit(self.context_X_, self.context_y_)

    def fit(self, X, y):
        if self.max_train_rows < 0 or self.max_train_rows == 1 or self.n_estimators < 1:
            raise ValueError("TabPFN needs max_train_rows=0 (all) or >= 2, and n_estimators >= 1.")
        y = np.asarray(y)
        if len(np.unique(y)) != 2:
            raise ValueError("TabPFN binary training needs both classes.")
        indices = np.arange(len(y))
        if self.max_train_rows and len(indices) > self.max_train_rows:
            _, indices = next(StratifiedShuffleSplit(n_splits=1, test_size=self.max_train_rows,
                random_state=self.random_state).split(X, y))
        self.context_X_ = X.iloc[indices].copy() if hasattr(X, "iloc") else X[indices].copy()
        self.context_y_ = y[indices].copy()
        self.classes_ = np.unique(y)
        self.context_rows_ = len(indices)
        self._restore()
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "context_y_")
        if not hasattr(self, "estimator_"):
            self._restore()
        return self.estimator_.predict_proba(X)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def __getstate__(self):
        # Store fitted context, not a live torch engine. Rebuild lazily on first
        # prediction from the same cached checkpoint/version, without reading test labels.
        state = self.__dict__.copy()
        state.pop("estimator_", None)
        return state


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
    if isinstance(estimator, LightweightTabPFN):
        return Pipeline([("features", FeatureBuilder(variant, engineered)),
                         ("preprocess", "passthrough"), ("model", estimator)])
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


def model_specs(seed=2026, jobs=2, include_tabpfn=False, tabpfn_rows=0, tabpfn_version="v2"):
    models = {
        "Dummy": (DummyClassifier(strategy="prior"), False, {}),
        "LogisticRegression": (LogisticRegression(max_iter=2500, random_state=seed), True,
            {"model__C": [0.1, 1.0, 10.0], "model__class_weight": [None, "balanced"]}),
        "MLP": (MLPClassifier(hidden_layer_sizes=(32, 16), activation="relu", solver="adam",
            batch_size=64, max_iter=200, early_stopping=False, n_iter_no_change=15,
            random_state=seed), True, {"model__alpha": [0.001, 0.01],
                "model__learning_rate_init": [0.001, 0.003]}),
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
    if include_tabpfn:
        try:
            from tabpfn import TabPFNClassifier
            models["TabPFN"] = (LightweightTabPFN(random_state=seed, n_jobs=jobs,
                max_train_rows=tabpfn_rows, model_version=tabpfn_version), False, {})
        except ImportError as exc:
            unavailable["TabPFN"] = f"Not installed or dependency missing: {exc}"
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
    if isinstance(fitted_pipeline.named_steps["model"], LightweightTabPFN):
        return build_pipeline(clone(fitted_pipeline.named_steps["model"]), variant=variant, engineered=engineered)
    return build_pipeline(clone(fitted_pipeline.named_steps["model"]),
                          scale="scaler" in fitted_pipeline.named_steps["preprocess"].named_transformers_["numeric"].named_steps,
                          variant=variant, engineered=engineered)
