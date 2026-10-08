import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from src.feature_engineering import (IDENTIFIERS, LEAKAGE_FEATURES, MEASURED_FEATURES, MUSIC_FEATURES,
    SAFE_FEATURES, TARGETS, FeatureBuilder, feature_list, feature_policy)
from src.modeling import build_pipeline


def test_whitelist_forbids_targets_ids_and_leakage(dataset):
    forbidden = set(TARGETS + IDENTIFIERS + LEAKAGE_FEATURES + MEASURED_FEATURES + MUSIC_FEATURES + ["loi_bom_xang_GIA_LAP"])
    assert not forbidden & set(SAFE_FEATURES)
    assert not forbidden & set(feature_list("safe", True))
    builder = FeatureBuilder(engineered=True).fit(dataset)
    assert not forbidden & set(builder.transform(dataset).columns)
    assert len(feature_policy(dataset.columns)) == 43


def test_predictions_do_not_depend_on_targets_or_forbidden_columns(dataset):
    pipeline = build_pipeline(RandomForestClassifier(n_estimators=12, max_depth=4, random_state=1), engineered=True)
    train = dataset.iloc[:160]
    pipeline.fit(train, train[TARGETS[0]])
    query = dataset.iloc[160:170].copy()
    before = pipeline.predict_proba(query)
    for column in [*TARGETS, *IDENTIFIERS, *LEAKAGE_FEATURES, *MEASURED_FEATURES, *MUSIC_FEATURES]:
        query[column] = "forbidden_values_should_be_ignored"
    np.testing.assert_array_equal(before, pipeline.predict_proba(query))


def test_engineering_formulas_and_missing_propagation(dataset):
    row = dataset.iloc[:1].copy()
    builder = FeatureBuilder(engineered=True).fit(row)
    result = builder.transform(row)
    assert result.iloc[0]["phu_tai_co_ban_tong_A"] == pytest.approx(sum(row.iloc[0][c] for c in ["tai_dien_co_ban_A", "den_pha_A", "quat_gio_dieu_hoa_A", "suoi_kinh_A"]))
    assert result.iloc[0]["volume_RMS_proxy"] == pytest.approx(row.iloc[0]["am_luong_pct"] / 100 * row.iloc[0]["do_nang_luong_am_thanh_RMS_0_1"])
    row["den_pha_A"] = np.nan
    assert pd.isna(builder.transform(row).iloc[0]["phu_tai_co_ban_tong_A"])
