"""Train two independent classifiers on synthetic scenarios: python huan_luyen_yaris_ml.py."""
import argparse
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter
from zoneinfo import ZoneInfo

import joblib
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from threadpoolctl import threadpool_limits

from src.data_processing import (DEFAULT_DATA, ROOT, SIMULATION_WARNING, audit_dataset, group_split,
    load_dataset, sha256_file, split_summary, validate_training_data, write_json)
from src.evaluation import metrics, save_evaluation_plots, tune_threshold
from src.feature_engineering import (ENGINEERED, SAFE_FEATURES, TARGETS, feature_list, feature_policy)
from src.modeling import build_pipeline, model_specs, rebuild_variant
from src.real_data import load_real_training
from src.feature_engineering import categorical_value


def cli():
    parser = argparse.ArgumentParser(description="EDA + group-aware training for simulated Y1/Y2 labels.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--models", nargs="+", default=None,
                        help="Dummy LogisticRegression MLP RandomForest HistGradientBoosting XGBoost TabPFN; TabPFN requires explicit selection")
    parser.add_argument("--additional-train-data", type=Path, help="Independently labeled real CSV; appended to train only")
    parser.add_argument("--tabpfn-rows", type=int, default=0,
                        help="0 (default): use every training row; positive value: optional context limit")
    parser.add_argument("--tabpfn-version", default="v2", choices=["v2", "v2.5", "v3", "v3.5-fast"],
                        help="v2 avoids newer gated checkpoints; other versions may require PriorLabs access")
    parser.add_argument("--skip-permutation-importance", action="store_true", help="Reduce repeated CPU inference")
    parser.add_argument("--search-iterations", type=int, default=4)
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--jobs", type=int, default=2, help="CPU threads; search runs sequentially to avoid oversubscription")
    parser.add_argument("--max-fpr", type=float, default=0.10, help="Validation threshold false-positive-rate limit; fix before training")
    parser.add_argument("--eda-only", action="store_true")
    parser.add_argument("--skip-ablations", action="store_true", help="Skip music/context/measured validation experiments")
    parser.add_argument("--skip-profile-experiment", action="store_true")
    args = parser.parse_args()
    if args.search_iterations < 1 or args.cv_folds < 2 or args.jobs < 1 or (args.tabpfn_rows != 0 and args.tabpfn_rows < 32) or not 0 <= args.max_fpr <= 1:
        parser.error("iterations >= 1, folds >= 2, jobs >= 1, tabpfn-rows = 0 or >= 32, max-fpr in [0, 1] required")
    return args


def validate_cv(df, target, folds, group="vehicle_id"):
    if df[group].nunique() < folds:
        raise ValueError(f"Not enough {group} groups for {folds} CV folds.")
    for train, val in GroupKFold(folds).split(df, groups=df[group]):
        if set(df.iloc[train][group]) & set(df.iloc[val][group]):
            raise ValueError("Cross-validation group overlap.")
        if any(df.iloc[i][target].nunique() != 2 for i in (train, val)):
            raise ValueError(f"CV fold for {target} lacks one class; reduce folds or redesign group CV.")


def compare_record(target, name, variant, engineered, stage, met, cv_ap=None, selected=False):
    return {"target": target, "model": name, "variant": variant, "engineered": engineered,
            "stage": stage, "cv_average_precision": cv_ap, "selected": selected,
            **{k: v for k, v in met.items() if k != "confusion_matrix"},
            "TN": met["confusion_matrix"][0][0], "FP": met["confusion_matrix"][0][1],
            "FN": met["confusion_matrix"][1][0], "TP": met["confusion_matrix"][1][1]}


def profile_experiment(development, output, seed, folds):
    """Separate fixed HGB experiment; folds hold out whole profiles, no main test access."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    report = {"scope": "Only train+validation vehicles; profile-disjoint group CV, fixed model/threshold, never used to select main pipelines.",
              "model": "Fixed HistGradientBoosting", "parameters": {"max_iter": 100, "max_leaf_nodes": 15,
              "learning_rate": 0.1, "early_stopping": False}, "targets": {}}
    for target in TARGETS:
        validate_cv(development, target, folds, "profile_id")
        probabilities = np.empty(len(development)); fold_reports = []
        for fold, (train, val) in enumerate(GroupKFold(folds).split(development, groups=development["profile_id"])):
            estimator = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15, learning_rate=0.1,
                                                       early_stopping=False, random_state=seed)
            pipeline = build_pipeline(estimator)
            pipeline.fit(development.iloc[train], development.iloc[train][target])
            probabilities[val] = pipeline.predict_proba(development.iloc[val])[:, 1]
            fold_reports.append({"fold": fold, "held_out_profiles": sorted(development.iloc[val]["profile_id"].unique()),
                                  "metrics_at_0_5": metrics(development.iloc[val][target], probabilities[val])})
        report["targets"][target] = {"folds": fold_reports, "pooled_out_of_fold": metrics(development[target], probabilities)}
    write_json(output / "reports/profile_generalization.json", report)


def train(args):
    started = perf_counter()
    output = args.output_dir.resolve()
    if output.is_relative_to((ROOT / "data/raw").resolve()):
        raise ValueError("Output directory must be outside data/raw to preserve source files.")
    for folder in ["models", "reports", "plots"]:
        (output / folder).mkdir(parents=True, exist_ok=True)
    df = load_dataset(args.data)
    audit_dataset(df, output / "eda", args.data.parent / "thong_so_xe_nguon_thuc.csv",
                  args.data.parent / "yaris_2008_dataset_classification.xlsx", args.data)
    print(f"EDA saved: {output / 'eda'}", flush=True)
    validate_training_data(df)
    if args.eda_only:
        return
    parts = group_split(df, args.seed)
    train_df, val_df, test_df = [df.iloc[parts[name]] for name in ["train", "validation", "test"]]
    summary = split_summary(df, parts)
    real_report = None
    if args.additional_train_data:
        extra, real_report = load_real_training(args.additional_train_data, df)
        train_df = pd.concat([train_df, extra], ignore_index=True)
        write_json(output / "reports/real_training_import.json", real_report)
    write_json(output / "reports/split_summary.json", summary)
    assignments = df[["scenario_id", "vehicle_id", "profile_id"]].copy()
    assignments["split"] = ""
    for name, indices in parts.items():
        assignments.iloc[indices, assignments.columns.get_loc("split")] = name
    if real_report:
        extra_assignments = extra[["scenario_id", "vehicle_id", "profile_id"]].copy()
        extra_assignments["split"] = "train_real"
        assignments = pd.concat([assignments, extra_assignments], ignore_index=True)
    assignments.to_csv(output / "reports/split_assignments.csv", index=False, encoding="utf-8-sig")
    print("Split:", {k: (v["rows"], v["vehicles"]) for k, v in summary.items()}, flush=True)
    specs, unavailable = model_specs(args.seed, args.jobs,
        include_tabpfn=bool(args.models and "TabPFN" in args.models),
        tabpfn_rows=args.tabpfn_rows, tabpfn_version=args.tabpfn_version)
    if args.models:
        unknown = set(args.models) - set(specs)
        if unknown:
            raise ValueError(f"Unavailable/unknown models: {sorted(unknown)}; {unavailable}")
        specs = {k: v for k, v in specs.items() if k in args.models or k == "Dummy"}
    versions = {}
    for package in ["numpy", "pandas", "scikit-learn", "joblib", "matplotlib", "openpyxl", "xgboost", "torch", "tabpfn"]:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not installed"
    config = {"warning": SIMULATION_WARNING, "created_at": datetime.now(ZoneInfo("Asia/Bangkok")).isoformat(),
              "data_path": str(args.data.resolve()), "data_sha256": sha256_file(args.data),
              "seed": args.seed, "models": list(specs), "unavailable_models": unavailable,
              "versions": versions, "cv_folds": args.cv_folds, "search_iterations": args.search_iterations,
              "cpu_threads": args.jobs, "split": summary, "max_validation_fpr": args.max_fpr,
              "selection": "Safe/engineered candidates only. Max validation AP, then threshold F1, recall, lower FPR. Hyperparameters use train-only vehicle GroupKFold AP.",
              "threshold_selection": "Max validation F1 subject to predeclared FPR cap; ties: recall, lower FPR, closer to 0.5.",
              "test_policy": "Lock all models/thresholds before one test reporting pass; no choice uses test metrics. No refit on validation after threshold tuning.",
              "search_spaces": {k: v[2] for k, v in specs.items()}, "safe_features": SAFE_FEATURES,
              "engineered_formulas": ENGINEERED, "oversampling": False,
              "skip_ablations": args.skip_ablations, "skip_profile_experiment": args.skip_profile_experiment}
    config.update({"real_training_import": real_report, "actual_training_rows": len(train_df),
        "tabpfn": {"enabled": "TabPFN" in specs, "version": args.tabpfn_version,
                   "max_train_context_rows": args.tabpfn_rows or None,
                   "actual_train_context_rows": min(args.tabpfn_rows, len(train_df)) if args.tabpfn_rows else len(train_df),
                   "context_policy": "All train rows" if args.tabpfn_rows == 0 else "Optional stratified train-only subset",
                   "device": "cpu", "ensemble_size": 1,
                   "input": "Native named DataFrame; no OHE/scaling/imputation outside TabPFN",
                   "persistence": "Save fitted contextual data; restore same pretrained checkpoint on first prediction"},
        "MLP": {"hidden_layers": [32, 16], "max_iter": 200, "batch_size": 64,
                "early_stopping": False, "device": "cpu"},
        "skip_permutation_importance": args.skip_permutation_importance})
    write_json(output / "reports/training_config.json", config)
    rows, all_candidates, selections, feature_reports = [], {}, {}, {}
    for target_index, target in enumerate(TARGETS, 1):
        print(f"Training Y{target_index}...", flush=True)
        validate_cv(train_df, target, args.cv_folds)
        candidates = []
        tuning_records = []
        for name, (estimator, scale, space) in specs.items():
            pipeline = build_pipeline(estimator, scale=scale)
            cv_ap = None
            if space:
                search = RandomizedSearchCV(pipeline, space, n_iter=args.search_iterations,
                    scoring="average_precision", cv=GroupKFold(args.cv_folds), random_state=args.seed,
                    n_jobs=1, refit=True, error_score="raise")
                search.fit(train_df, train_df[target], groups=train_df["vehicle_id"])
                fitted, cv_ap = search.best_estimator_, float(search.best_score_)
                pd.DataFrame(search.cv_results_).to_csv(output / f"reports/y{target_index}_{name}_cv.csv", index=False)
                tuning_records.append({"model": name, "best_params": search.best_params_, "cv_ap": cv_ap})
            else:
                fitted = pipeline.fit(train_df, train_df[target])
            for engineered in ([False] if name in ["Dummy", "TabPFN"] else [False, True]):
                candidate_pipeline = fitted if not engineered else rebuild_variant(fitted, engineered=True).fit(train_df, train_df[target])
                probabilities = candidate_pipeline.predict_proba(val_df)[:, 1]
                tuned, curve = tune_threshold(val_df[target], probabilities, args.max_fpr)
                suffix = "engineered" if engineered else "raw"
                curve.to_csv(output / f"reports/y{target_index}_{name}_{suffix}_thresholds.csv", index=False)
                default = metrics(val_df[target], probabilities)
                candidates.append({"model": name, "engineered": engineered, "pipeline": candidate_pipeline,
                                   "validation": tuned, "validation_default": default, "cv_ap": cv_ap if not engineered else None})
                rows.append(compare_record(target, name, "safe", engineered, "validation_tuned", tuned, cv_ap if not engineered else None))
                rows.append(compare_record(target, name, "safe", engineered, "validation_0.5", default, cv_ap if not engineered else None))
                print(f"  {name}/{suffix}: val AP={tuned['average_precision']:.4f}, F1={tuned['f1']:.4f}, recall={tuned['recall']:.4f}, FPR={tuned['false_positive_rate']:.4f}", flush=True)
        winner = max(candidates, key=lambda c: (c["validation"]["average_precision"], c["validation"]["f1"],
            c["validation"]["recall"], -c["validation"]["false_positive_rate"]))
        selections[target] = winner
        all_candidates[target] = candidates
        for row in rows:
            if row["target"] == target and row["model"] == winner["model"] and row["engineered"] == winner["engineered"]:
                row["selected"] = True
        write_json(output / f"reports/y{target_index}_tuning.json", tuning_records)
        if not args.skip_ablations and winner["model"] != "TabPFN":
            raw = next(c for c in candidates if c["model"] == winner["model"] and not c["engineered"])
            for variant in ["music", "context", "measured"]:
                ablation = rebuild_variant(raw["pipeline"], variant=variant).fit(train_df, train_df[target])
                probabilities = ablation.predict_proba(val_df)[:, 1]
                tuned, curve = tune_threshold(val_df[target], probabilities, args.max_fpr)
                curve.to_csv(output / f"reports/y{target_index}_{variant}_thresholds.csv", index=False)
                rows.append(compare_record(target, raw["model"], variant, False, "ablation_validation_tuned", tuned))
                rows.append(compare_record(target, raw["model"], variant, False, "ablation_validation_0.5", metrics(val_df[target], probabilities)))
                if not args.skip_permutation_importance:
                    importance = permutation_importance(ablation, val_df[feature_list(variant)], val_df[target],
                        scoring="average_precision", n_repeats=3, random_state=args.seed, n_jobs=1)
                    pd.DataFrame({"feature": feature_list(variant), "mean_AP_drop": importance.importances_mean,
                        "std_AP_drop": importance.importances_std}).sort_values("mean_AP_drop", ascending=False).to_csv(
                            output / f"reports/y{target_index}_{variant}_permutation_importance.csv", index=False)
        required = feature_list("safe")
        if not args.skip_permutation_importance and winner["model"] != "TabPFN":
            importance = permutation_importance(winner["pipeline"], val_df[required], val_df[target],
                scoring="average_precision", n_repeats=5, random_state=args.seed, n_jobs=1)
            pd.DataFrame({"feature": required, "mean_AP_drop": importance.importances_mean,
                "std_AP_drop": importance.importances_std}).sort_values("mean_AP_drop", ascending=False).to_csv(
                    output / f"reports/y{target_index}_permutation_importance.csv", index=False)
        feature_reports[target] = {"variant": "safe", "engineered": winner["engineered"], "required_raw_features": required,
            "model_features": feature_list("safe", winner["engineered"]),
            "excluded_columns": [c for c in df if c not in required],
            "transformed_features": (feature_list("safe", winner["engineered"]) if winner["model"] == "TabPFN"
                else winner["pipeline"].named_steps["preprocess"].get_feature_names_out().tolist())}
    if not args.skip_profile_experiment:
        print("Running fixed model profile-disjoint experiment on development vehicles...", flush=True)
        profile_experiment(df.iloc[np.r_[parts["train"], parts["validation"]]], output, args.seed, args.cv_folds)
    locked = {t: {"model": c["model"], "engineered": c["engineered"],
                  "threshold": c["validation"]["threshold"], "validation": c["validation"],
                  "validation_at_0_5": c["validation_default"]} for t, c in selections.items()}
    write_json(output / "reports/selection_lock.json", locked)
    write_json(output / "reports/selected_features.json", {"targets": feature_reports, "column_audit": feature_policy(df.columns),
        "engineered": ENGINEERED, "policy": "Ablations are validation only; measured/music/context never eligible for main pipelines."})
    evaluation = {"warning": SIMULATION_WARNING, "split": summary,
                  "test_evaluation_passes": 1, "targets": {}}
    test_predictions = test_df[["scenario_id", "vehicle_id"]].copy()
    for target_index, target in enumerate(TARGETS, 1):
        winner = selections[target]
        for candidate in all_candidates[target]:
            probabilities = candidate["pipeline"].predict_proba(test_df)[:, 1]
            tuned = metrics(test_df[target], probabilities, candidate["validation"]["threshold"])
            default = metrics(test_df[target], probabilities)
            selected = candidate is winner
            rows.append(compare_record(target, candidate["model"], "safe", candidate["engineered"], "test_tuned", tuned, candidate["cv_ap"], selected))
            rows.append(compare_record(target, candidate["model"], "safe", candidate["engineered"], "test_0.5", default, candidate["cv_ap"], selected))
            if selected:
                evaluation["targets"][target] = {**locked[target], "test": tuned, "test_at_0_5": default}
                test_predictions[target + "_actual"] = test_df[target]
                test_predictions[target + "_probability"] = probabilities
                test_predictions[target + "_prediction"] = (probabilities >= tuned["threshold"]).astype(int)
                save_evaluation_plots(test_df[target], probabilities, tuned["threshold"], output / "plots", f"y{target_index}")
                print(f"Selected Y{target_index}: {winner['model']}, engineered={winner['engineered']}, test AP={tuned['average_precision']:.4f}, F1={tuned['f1']:.4f}", flush=True)
        pipeline_path = output / f"models/y{target_index}_pipeline.joblib"
        joblib.dump(winner["pipeline"], pipeline_path)
        reloaded = joblib.load(pipeline_path)
        smoke = val_df.iloc[:5]
        np.testing.assert_allclose(reloaded.predict_proba(smoke), winner["pipeline"].predict_proba(smoke))
    test_predictions.to_csv(output / "reports/test_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(rows).to_csv(output / "reports/model_comparison.csv", index=False, encoding="utf-8-sig")
    write_json(output / "reports/evaluation.json", evaluation)
    write_json(output / "models/thresholds.json", {t: c["validation"]["threshold"] for t, c in selections.items()})
    schema = {"warning": SIMULATION_WARNING, "targets": {}, "versions": versions,
              "training_data_sha256": sha256_file(args.data)}
    for target_index, target in enumerate(TARGETS, 1):
        schema["targets"][target] = {"model_file": f"y{target_index}_pipeline.joblib",
            "model_sha256": sha256_file(output / f"models/y{target_index}_pipeline.joblib"),
            "threshold": selections[target]["validation"]["threshold"],
            "required_raw_features": feature_reports[target]["required_raw_features"],
            "training_ranges": {c: [float(train_df[c].min()), float(train_df[c].max())] for c in SAFE_FEATURES},
            "training_categories": {"amp_ngoai_co_lap": [categorical_value(v, "amp_ngoai_co_lap")
                for v in train_df["amp_ngoai_co_lap"].dropna().unique()]}}
    write_json(output / "models/inference_schema.json", schema)
    val_df.iloc[:5][["scenario_id", *SAFE_FEATURES]].to_csv(output / "reports/example_input.csv", index=False, encoding="utf-8-sig")
    config["elapsed_seconds"] = round(perf_counter() - started, 2)
    write_json(output / "reports/training_config.json", config)
    print(f"Saved pipelines, reports and plots to {output}; elapsed={config['elapsed_seconds']}s", flush=True)


if __name__ == "__main__":
    arguments = cli()
    print(SIMULATION_WARNING, flush=True)
    with threadpool_limits(limits=arguments.jobs):
        train(arguments)
