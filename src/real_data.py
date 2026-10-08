"""Validate independently labeled measured observations before train-only import."""
import pandas as pd

from .data_processing import load_dataset, sha256_file
from .feature_engineering import IDENTIFIERS, SAFE_FEATURES, TARGETS, invalid_numeric_mask

REQUIRED_REAL_COLUMNS = [*IDENTIFIERS, *SAFE_FEATURES, *TARGETS,
                         "measurement_timestamp", "labeler", "label_evidence", "data_origin"]


def load_real_training(path, original):
    extra = load_dataset(path, training=False)
    missing = sorted(set(REQUIRED_REAL_COLUMNS) - set(extra))
    if missing:
        raise ValueError(f"Real training CSV missing columns: {missing}")
    for c in [*IDENTIFIERS, *TARGETS, "measurement_timestamp", "labeler", "label_evidence", "data_origin"]:
        if extra[c].isna().any():
            raise ValueError(f"Real training requires non-missing {c}.")
    if not extra["data_origin"].eq("REAL_MEASURED").all():
        raise ValueError("Real CSV data_origin must be REAL_MEASURED; do not label synthetic rows as real.")
    if pd.to_datetime(extra["measurement_timestamp"], errors="coerce", utc=True).isna().any():
        raise ValueError("measurement_timestamp must be valid ISO datetime.")
    if extra["scenario_id"].duplicated().any() or extra.duplicated().any():
        raise ValueError("Duplicate real scenarios/rows.")
    for c in ["scenario_id", "vehicle_id"]:
        if set(extra[c]) & set(original[c]):
            raise ValueError(f"Real {c} overlaps original dataset; use stable distinct real IDs.")
    if (extra.groupby("vehicle_id")["profile_id"].nunique() > 1).any():
        raise ValueError("Real vehicle_id maps to multiple profiles.")
    for c in [*SAFE_FEATURES, *TARGETS]:
        if invalid_numeric_mask(extra[c], c).any():
            raise ValueError(f"Invalid numeric/range in real data: {c}")
        extra[c] = pd.to_numeric(extra[c], errors="raise")
        if c in SAFE_FEATURES and extra[c].isna().all():
            raise ValueError(f"No measured/independently estimated values supplied for {c}.")
    report = {"path": str(path), "sha256": sha256_file(path), "rows": len(extra),
              "vehicles": int(extra["vehicle_id"].nunique()), "origin": "REAL_MEASURED",
              "usage": "Appended only after original train/validation/test split; never inserted into holdouts.",
              "target_definition": "Independent specialist labels with recorded evidence; do not copy simulator rules or model predictions.",
              "evaluation_scope": "Original validation/test remain synthetic; metrics do not measure real-car validity."}
    return extra, report
