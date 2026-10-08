import json
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.data_processing import DEFAULT_DATA, audit_dataset, load_dataset, validate_training_data
from src.feature_engineering import ALL_COLUMNS, SAFE_FEATURES, TARGETS, invalid_numeric_mask
from src.modeling import build_pipeline


def test_original_bom_schema_and_targets(dataset):
    assert DEFAULT_DATA.read_bytes().startswith(b"\xef\xbb\xbf")
    assert dataset.shape == (2880, 43)
    assert set(ALL_COLUMNS) == set(dataset.columns)
    assert list(dataset.columns[-2:]) == TARGETS
    validate_training_data(dataset)


def test_loading_blanks_and_duplicate_headers(tmp_path):
    path = tmp_path / "input.csv"
    path.write_text("name,value\n ,1\n", encoding="utf-8-sig")
    assert pd.isna(load_dataset(path, training=False).iloc[0]["name"])
    path.write_text("name,name\na,b\n", encoding="utf-8-sig")
    with pytest.raises(ValueError, match="Duplicate CSV column"):
        load_dataset(path, training=False)


@pytest.mark.parametrize("column,value", [("am_luong_pct", 101), ("ty_le_dung_cho_no_may_do_thi", -0.1),
    ("do_nang_luong_am_thanh_RMS_0_1", 1.1), ("vong_tua_may_rpm", 0),
    ("amp_ngoai_co_lap", 0.5), ("den_pha_A", -1), ("den_pha_A", np.inf), ("den_pha_A", "bad")])
def test_invalid_ranges(column, value):
    assert invalid_numeric_mask(pd.Series([value]), column).iloc[0]


def test_signed_headroom_is_valid():
    assert not invalid_numeric_mask(pd.Series([-119.08, 0, 2]), "du_dia_dien_moi_A_GIA_LAP").any()


def test_identifier_duplicates_and_missing_labels_fail(dataset):
    bad = dataset.copy()
    bad.loc[1, "scenario_id"] = bad.loc[0, "scenario_id"]
    with pytest.raises(ValueError, match="Duplicate"):
        validate_training_data(bad)
    bad = dataset.copy(); bad.loc[0, "vehicle_id"] = np.nan
    with pytest.raises(ValueError, match="Missing"):
        validate_training_data(bad)
    bad = dataset.copy(); bad.loc[0, TARGETS[0]] = np.nan
    with pytest.raises(ValueError, match="Target"):
        validate_training_data(bad)


def test_imputation_unseen_categories_and_train_only_statistics(dataset):
    train = dataset.iloc[:80].copy()
    train.loc[train.index[0], "am_luong_pct"] = np.nan
    train.loc[train.index[1], "hang_xe"] = np.nan
    train["xang_do_thi_L_100km"] = np.nan  # Keep all-empty feature, even on a train fold.
    pipeline = build_pipeline(LogisticRegression(max_iter=2000), scale=True, variant="context")
    pipeline.fit(train, train[TARGETS[0]])
    before = pipeline.named_steps["preprocess"].named_transformers_["numeric"].named_steps["imputer"].statistics_.copy()
    validation = dataset.iloc[80:85].copy()
    validation["hang_xe"] = "brand_never_seen"
    validation["mau_xe"] = "model_never_seen"
    validation["am_luong_pct"] = np.nan
    probabilities = pipeline.predict_proba(validation)[:, 1]
    assert np.isfinite(probabilities).all() and ((probabilities >= 0) & (probabilities <= 1)).all()
    np.testing.assert_array_equal(before, pipeline.named_steps["preprocess"].named_transformers_["numeric"].named_steps["imputer"].statistics_)
    features = pipeline.named_steps["features"].transform(validation)
    transformed = pipeline.named_steps["preprocess"].transform(features)
    assert np.isfinite(transformed).all()


def test_audit_preserves_missing_fuel_rows(dataset, tmp_path):
    before = dataset.copy(deep=True)
    report = audit_dataset(dataset, tmp_path)
    pd.testing.assert_frame_equal(dataset, before)
    assert report["shape"] == [2880, 43]
    missing = pd.read_csv(tmp_path / "missing_values.csv").set_index("column")
    assert missing.loc["xang_do_thi_L_100km", "missing_count"] == 208
    assert sum(report["invalid_numeric_counts"].values()) == 0
    assert json.loads((tmp_path / "data_quality_report.json").read_text(encoding="utf-8"))["duplicate_rows"] == 0


def test_binary_categories_stable_when_missing_promotes_dtype(dataset):
    pipeline = build_pipeline(LogisticRegression(max_iter=2000), scale=True)
    pipeline.fit(dataset.iloc[:160], dataset.iloc[:160][TARGETS[0]])
    query = dataset.iloc[160:165].copy()
    expected = pipeline.predict_proba(query)
    query["amp_ngoai_co_lap"] = query["amp_ngoai_co_lap"].astype(float)
    np.testing.assert_array_equal(expected, pipeline.predict_proba(query))
    query.loc[query.index[0], "amp_ngoai_co_lap"] = np.nan
    features = pipeline.named_steps["features"].transform(query)
    assert set(features["amp_ngoai_co_lap"].dropna()) <= {"0", "1"}
    assert np.isfinite(pipeline.predict_proba(query)).all()
