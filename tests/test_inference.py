import json
import os
from pathlib import Path
import subprocess
import sys
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from predict import predict_frame
from src.data_processing import ROOT, sha256_file, write_json
from src.evaluation import metrics, tune_threshold
from src.feature_engineering import SAFE_FEATURES, TARGETS
from src.modeling import build_pipeline, model_specs


@pytest.fixture
def saved_models(dataset, tmp_path):
    schema = {"targets": {}}
    for i, target in enumerate(TARGETS, 1):
        pipeline = build_pipeline(LogisticRegression(max_iter=2000), scale=True)
        pipeline.fit(dataset.iloc[:160], dataset.iloc[:160][target])
        path = tmp_path / f"y{i}_pipeline.joblib"
        joblib.dump(pipeline, path)
        reloaded = joblib.load(path)
        np.testing.assert_array_equal(pipeline.predict_proba(dataset.iloc[160:165]), reloaded.predict_proba(dataset.iloc[160:165]))
        schema["targets"][target] = {"required_raw_features": SAFE_FEATURES, "threshold": 0.37,
            "model_file": path.name, "model_sha256": sha256_file(path), "training_ranges": {}}
    write_json(tmp_path / "inference_schema.json", schema)
    return tmp_path


def test_saved_pipelines_inference_thresholds_and_probabilities(dataset, saved_models):
    with pytest.warns(UserWarning, match="giả lập"):
        output = predict_frame(dataset.iloc[160:165], saved_models)
    for target in TARGETS:
        probabilities = output[target + "_probability"]
        assert probabilities.between(0, 1).all()
        assert output[target + "_prediction"].equals((probabilities >= 0.37).astype(int))


def test_inference_rejects_missing_schema_invalid_and_all_missing(dataset, saved_models):
    for kind in ["missing", "invalid", "all_missing"]:
        query = dataset.iloc[:5].copy()
        if kind == "missing":
            query = query.drop(columns="am_luong_pct")
        elif kind == "invalid":
            query["am_luong_pct"] = 120
        else:
            query["am_luong_pct"] = np.nan
        with pytest.warns(UserWarning):
            with pytest.raises(ValueError):
                predict_frame(query, saved_models)


def test_threshold_tuning_honors_fpr_and_single_class_metrics():
    y = np.array([0, 0, 0, 0, 1, 1]); prob = np.array([0.1, 0.2, 0.3, 0.55, 0.6, 0.8])
    chosen, table = tune_threshold(y, prob, max_fpr=0)
    assert chosen["f1"] == 1 and chosen["false_positive_rate"] == 0
    assert chosen["f1"] > metrics(y, prob)["f1"]
    assert metrics(np.zeros(3), np.array([0.1, 0.2, 0.3]))["roc_auc"] is None


def test_optional_xgboost_weight_uses_fit_fold(dataset):
    pytest.importorskip("xgboost")
    specs, _ = model_specs()
    estimator = specs["XGBoost"][0].set_params(balance=True, estimator__n_estimators=3)
    train = dataset.iloc[:160]
    pipeline = build_pipeline(estimator).fit(train, train[TARGETS[1]])
    assert pipeline.named_steps["model"].scale_pos_weight_ == pytest.approx((train[TARGETS[1]] == 0).sum() / (train[TARGETS[1]] == 1).sum())


def test_training_saving_loading_and_csv_cli_end_to_end(tmp_path):
    environment = {**os.environ, "PYTHONUTF8": "1", "MPLBACKEND": "Agg"}
    command = [sys.executable, str(ROOT / "huan_luyen_yaris_ml.py"), "--output-dir", str(tmp_path),
               "--models", "Dummy", "LogisticRegression", "--search-iterations", "1",
               "--skip-ablations", "--skip-profile-experiment"]
    completed = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    evaluation = json.loads((tmp_path / "reports/evaluation.json").read_text(encoding="utf-8"))
    locked = json.loads((tmp_path / "reports/selection_lock.json").read_text(encoding="utf-8"))
    assert evaluation["test_evaluation_passes"] == 1
    for target in TARGETS:
        assert evaluation["targets"][target]["threshold"] == locked[target]["threshold"]
    output = tmp_path / "predictions.csv"
    result = subprocess.run([sys.executable, str(ROOT / "predict.py"), "--input", str(tmp_path / "reports/example_input.csv"),
        "--models-dir", str(tmp_path / "models"), "--output", str(output)], cwd=ROOT, env=environment,
        capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    predicted = pd.read_csv(output)
    assert len(predicted) == 5
    assert predicted[[c for c in predicted if c.endswith("_probability")]].map(lambda v: 0 <= v <= 1).all().all()
