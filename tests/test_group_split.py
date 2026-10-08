import numpy as np
from sklearn.model_selection import GroupKFold
from src.data_processing import assert_group_disjoint, group_split
from src.feature_engineering import TARGETS


def test_three_way_group_split_and_class_presence(dataset):
    parts = group_split(dataset)
    assert_group_disjoint(dataset, parts)
    assert [len(parts[name]) for name in ["train", "validation", "test"]] == [2016, 432, 432]
    for name, indices in parts.items():
        assert dataset.iloc[indices]["vehicle_id"].nunique() == (252 if name == "train" else 54)
        assert all(dataset.iloc[indices][target].nunique() == 2 for target in TARGETS)
    repeat = group_split(dataset)
    for name in parts:
        np.testing.assert_array_equal(parts[name], repeat[name])


def test_inner_cv_is_group_disjoint(dataset):
    train = dataset.iloc[group_split(dataset)["train"]]
    for fit, validation in GroupKFold(3).split(train, groups=train["vehicle_id"]):
        assert set(train.iloc[fit]["vehicle_id"]).isdisjoint(train.iloc[validation]["vehicle_id"])
