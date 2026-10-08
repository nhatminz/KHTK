"""Metrics and validation-only threshold selection."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
    ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay, precision_recall_curve)


def metrics(y, probabilities, threshold=0.5):
    y = np.asarray(y)
    probabilities = np.asarray(probabilities)
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Predicted probabilities must be finite and in [0, 1].")
    predicted = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    both_classes = len(np.unique(y)) == 2
    return {"threshold": float(threshold), "accuracy": float(accuracy_score(y, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)) if both_classes else float(accuracy_score(y, predicted)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "average_precision": float(average_precision_score(y, probabilities)) if np.any(y == 1) else None,
        "roc_auc": float(roc_auc_score(y, probabilities)) if both_classes else None,
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else None,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "rows": len(y), "positive_rate": float(y.mean())}


def tune_threshold(y, probabilities, max_fpr=0.10):
    """Maximize validation F1 subject to a predeclared FPR limit; ties prefer recall then low FPR."""
    y = np.asarray(y); probabilities = np.asarray(probabilities)
    # Actual breakpoints avoid a coarse grid missing the best decision boundary.
    _, _, thresholds = precision_recall_curve(y, probabilities)
    candidates = np.unique(np.r_[0.5, thresholds, np.nextafter(1.0, 2.0)])
    records = []
    for threshold in candidates:
        predicted = probabilities >= threshold
        tp = int(np.sum(predicted & (y == 1))); fp = int(np.sum(predicted & (y == 0)))
        fn = int(np.sum(~predicted & (y == 1))); tn = int(np.sum(~predicted & (y == 0)))
        records.append({"threshold": float(threshold), "precision": tp / max(tp + fp, 1),
                        "recall": tp / max(tp + fn, 1), "f1": 2 * tp / max(2 * tp + fp + fn, 1),
                        "false_positive_rate": fp / max(fp + tn, 1)})
    feasible = [m for m in records if m["false_positive_rate"] <= max_fpr]
    best = max(feasible, key=lambda m: (m["f1"], m["recall"], -m["false_positive_rate"], -abs(m["threshold"] - 0.5)))
    return metrics(y, probabilities, best["threshold"]), pd.DataFrame(records)


def save_evaluation_plots(y, probabilities, threshold, output, prefix):
    import matplotlib.pyplot as plt
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    ConfusionMatrixDisplay.from_predictions(y, (probabilities >= threshold).astype(int), labels=[0, 1], ax=axes[0], colorbar=False)
    PrecisionRecallDisplay.from_predictions(y, probabilities, ax=axes[1])
    if len(np.unique(y)) == 2:
        RocCurveDisplay.from_predictions(y, probabilities, ax=axes[2])
    axes[0].set_title(f"{prefix} | threshold={threshold:.3f}")
    fig.tight_layout(); fig.savefig(output / f"{prefix}_evaluation.png", dpi=130); plt.close(fig)
