"""Schema-checked inference with persisted validation thresholds."""
import argparse
import json
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd

from src.data_processing import ROOT, SIMULATION_WARNING, load_dataset, sha256_file
from src.feature_engineering import NUMERIC_COLUMNS, TARGETS, categorical_value, invalid_numeric_mask


def predict_frame(df, models_dir):
    models_dir = Path(models_dir)
    schema = json.loads((models_dir / "inference_schema.json").read_text(encoding="utf-8"))
    if df.empty:
        raise ValueError("No observations to predict.")
    warnings.warn(SIMULATION_WARNING, UserWarning, stacklevel=2)
    result = df[[c for c in ["scenario_id", "vehicle_id", "profile_id"] if c in df]].copy()
    for target in TARGETS:
        entry = schema["targets"][target]
        required = entry["required_raw_features"]
        missing = sorted(set(required) - set(df.columns))
        if missing:
            raise ValueError(f"Missing required features for {target}: {missing}")
        features = df[required].copy()
        for c in required:
            if c in NUMERIC_COLUMNS:
                if invalid_numeric_mask(features[c], c).any():
                    raise ValueError(f"Invalid numeric data or range: {c}")
                features[c] = pd.to_numeric(features[c], errors="raise")
            if features[c].isna().all():
                raise ValueError(f"Important feature entirely missing: {c}. Supply valid measurements/estimates.")
            if features[c].isna().any():
                warnings.warn(f"Missing values in {c}: using training-fitted imputer.", UserWarning, stacklevel=2)
            if c in entry["training_ranges"]:
                lower, upper = entry["training_ranges"][c]
                if ((features[c] < lower) | (features[c] > upper)).any():
                    warnings.warn(f"{c} outside observed training range [{lower}, {upper}]; extrapolation.", UserWarning, stacklevel=2)
        for c, known in entry.get("training_categories", {}).items():
            if (~features[c].dropna().map(lambda v: categorical_value(v, c)).isin(known)).any():
                warnings.warn(f"Unseen categories in {c}; encoder ignores unseen levels.", UserWarning, stacklevel=2)
        pipeline_path = models_dir / entry["model_file"]
        if sha256_file(pipeline_path) != entry["model_sha256"]:
            raise ValueError(f"Model and inference metadata do not match: {pipeline_path.name}")
        pipeline = joblib.load(pipeline_path)
        probabilities = pipeline.predict_proba(features)[:, 1]
        if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
            raise ValueError("Model emitted invalid probabilities.")
        result[target + "_probability"] = probabilities
        result[target + "_prediction"] = (probabilities >= entry["threshold"]).astype(int)
        result[target + "_threshold"] = entry["threshold"]
    result["interpretation"] = "SIMULATION_RESEARCH_ONLY"
    return result


def main():
    parser = argparse.ArgumentParser(description="Predict Y1/Y2 probabilities on a new UTF-8 CSV.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, default=ROOT / "outputs/models")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/predictions.csv")
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("Output must not overwrite input CSV.")
    if args.output.resolve().is_relative_to((ROOT / "data/raw").resolve()):
        parser.error("Cannot write predictions into data/raw; preserve original datasets.")
    df = load_dataset(args.input, training=False)
    result = predict_frame(df, args.models_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Saved {len(result)} predictions to {args.output}")


if __name__ == "__main__":
    main()
