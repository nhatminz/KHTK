"""Create empty measured-data entry files without inventing observations."""
import argparse
from pathlib import Path

import pandas as pd

from src.data_processing import ROOT, write_json
from src.real_data import REQUIRED_REAL_COLUMNS, load_real_training
from src.feature_engineering import SAFE_FEATURES, TARGETS, RANGES
from src.data_processing import DEFAULT_DATA, load_dataset


def main():
    parser = argparse.ArgumentParser(description="Create templates or validate labeled real-car measurements.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/real")
    parser.add_argument("--validate", type=Path, help="Check a filled real CSV; does not train.")
    args = parser.parse_args()
    if args.validate:
        _, report = load_real_training(args.validate, load_dataset(DEFAULT_DATA))
        print(report)
        return
    directory = args.output_dir.resolve()
    if directory.is_relative_to((ROOT / "data/raw").resolve()):
        raise ValueError("Templates must be outside data/raw.")
    directory.mkdir(parents=True, exist_ok=True)
    # Never overwrite measurements that the user may already have entered.
    paths = [directory / name for name in ["training_template.csv", "measurements_template.csv", "entry_schema.json"]]
    if any(path.exists() for path in paths):
        raise FileExistsError("Template already exists; choose another --output-dir to preserve entered data.")
    pd.DataFrame(columns=REQUIRED_REAL_COLUMNS).to_csv(paths[0], index=False, encoding="utf-8-sig")
    measurements = ["scenario_id", "vehicle_id", "measurement_timestamp", "VIN_or_vehicle_code",
        "engine_code", "alternator_part_number", "head_unit_model", "external_amplifier_model",
        "engine_rpm", "battery_voltage_engine_off_V", "battery_voltage_idle_audio_off_V",
        "battery_voltage_idle_audio_on_V", "battery_voltage_2000rpm_load_V",
        "alternator_current_A", "head_unit_current_idle_A", "head_unit_current_playing_A",
        "external_amplifier_current_A", "battery_SOH_pct", "battery_SOC_pct",
        "positive_cable_voltage_drop_V", "ground_voltage_drop_V",
        "installation_inspection", "fault_codes", "observed_stall", "technician", "notes"]
    pd.DataFrame(columns=measurements).to_csv(paths[1], index=False, encoding="utf-8-sig")
    write_json(paths[2], {
        "observations_created": 0, "training_file": paths[0].name,
        "measurement_file": paths[1].name,
        "required_training_columns": REQUIRED_REAL_COLUMNS,
        "origin": "Set data_origin=REAL_MEASURED only for actual measured observations.",
        "ids": "Use stable REAL_CAR_* vehicle IDs; one REAL_SESSION_* per measured session; repeat vehicle ID across sessions of the same car.",
        "timestamp": "ISO 8601 with timezone, e.g. 2026-10-08T10:00:00+07:00",
        "features": {c: {"range": RANGES.get(c), "instruction": "Measured or independently estimated BEFORE label decision; suffix _GIA_LAP retained for compatibility only."} for c in SAFE_FEATURES},
        "alternator_health": "Percent of measured available capacity versus an independently specified reference at matching RPM/conditions; record method in evidence. Do not infer from target or assume 100%.",
        "labels": {TARGETS[0]: "0/1: specialist assessment of audio system contribution to electrical deficit using controlled A/B evidence, not simply stall occurrence.",
                   TARGETS[1]: "0/1: independent specialist assessment of capacity upgrade need after ruling out charging/installation faults; do not derive from model predictions."},
        "labeler": "Specialist who independently assigned labels.",
        "label_evidence": "Measurement record reference and justification; missing/unknown labels must stay out of training.",
        "measurements_mapping": "Measurements worksheet records evidence; no automatic conversion of voltage to alternator health or guessed currents.",
        "usage": "Pass a filled copy via --additional-train-data; original validation/test stay synthetic. These empty templates add no training observations."})
    print(f"Created empty templates: {directory}; no real observations fabricated.")


if __name__ == "__main__":
    main()
