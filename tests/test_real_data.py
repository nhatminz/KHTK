"""Regression tests for train-only real-data import; no invented persisted observations."""
import numpy as np
import pytest
from src.feature_engineering import IDENTIFIERS, SAFE_FEATURES, TARGETS
from src.real_data import load_real_training


def real_fixture(dataset):
    # Test fixture only, never an actual measured dataset or training artifact.
    frame = dataset.iloc[:8][[*IDENTIFIERS, *SAFE_FEATURES, *TARGETS]].copy()
    frame["scenario_id"] = [f"TEST_REAL_SESSION_{i}" for i in range(len(frame))]
    frame["vehicle_id"] = "TEST_REAL_CAR_1"
    frame["measurement_timestamp"] = "2026-10-08T10:00:00+07:00"
    frame["labeler"] = "test fixture"
    frame["label_evidence"] = "unit test only; not measured observations"
    frame["data_origin"] = "REAL_MEASURED"
    return frame


def test_real_import_requires_distinct_ids_and_evidence(dataset, tmp_path):
    frame = real_fixture(dataset)
    path = tmp_path / "fixture.csv"
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    imported, report = load_real_training(path, dataset)
    assert len(imported) == 8 and report["rows"] == 8
    frame["vehicle_id"] = dataset.iloc[0]["vehicle_id"]
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    with pytest.raises(ValueError, match="overlaps"):
        load_real_training(path, dataset)


def test_real_import_rejects_unknown_labels_and_entire_missing_feature(dataset, tmp_path):
    for column in [TARGETS[0], "label_evidence", SAFE_FEATURES[0]]:
        frame = real_fixture(dataset)
        frame[column] = np.nan
        path = tmp_path / "fixture.csv"
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        with pytest.raises(ValueError):
            load_real_training(path, dataset)
