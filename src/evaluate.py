"""Shared metrics, plots, and markdown comparison tables."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from src.utils import apply_plot_style, models_dir, reports_dir


def classification_metrics(y_true, y_pred, y_proba=None) -> dict:
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_proba is not None:
        out["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    return out


def regression_metrics(y_true, y_pred) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def plot_confusion_matrix(y_true, y_pred, title: str, path: Path) -> Path:
    apply_plot_style()
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Greys")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1], ["0", "1"])
    ax.set_yticks([0, 1], ["0", "1"])
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_feature_importance(names, values, title: str, path: Path, top_n: int = 20) -> Path:
    apply_plot_style()
    order = np.argsort(np.abs(values))[-top_n:]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(np.array(names)[order], np.array(values)[order], color="#2c5282")
    ax.set_title(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def extract_feature_importance(pipeline) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    pre = pipeline.named_steps["pre"]
    model = pipeline.named_steps["model"]
    try:
        names = np.array(pre.get_feature_names_out())
    except Exception:
        return None, None
    if hasattr(model, "feature_importances_"):
        return names, model.feature_importances_
    if hasattr(model, "coef_"):
        coef = model.coef_
        if getattr(coef, "ndim", 1) > 1:
            coef = np.ravel(np.abs(coef).mean(axis=0) if coef.shape[0] > 1 else coef[0])
        return names, np.ravel(coef)
    return None, None


def write_metrics_json(name: str, metrics: dict) -> Path:
    path = reports_dir() / f"{name}_metrics.json"
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return path


def write_comparison_tables() -> Path:
    """Combine classifier + regressor JSON sidecar files into one markdown table."""
    rows_clf, rows_reg = [], []
    for path in sorted(reports_dir().glob("*_metrics.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        task = payload.get("task")
        if task == "classification":
            rows_clf.append(payload)
        elif task == "regression":
            rows_reg.append(payload)

    lines = ["# Model comparison", ""]
    if rows_clf:
        lines.append("## Classification — Good Investment")
        lines.append("")
        lines.append("| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for r in sorted(rows_clf, key=lambda x: x["metrics"].get("f1", 0), reverse=True):
            m = r["metrics"]
            lines.append(
                f"| {r['model']} | {m.get('accuracy', float('nan')):.4f} | "
                f"{m.get('precision', float('nan')):.4f} | {m.get('recall', float('nan')):.4f} | "
                f"{m.get('f1', float('nan')):.4f} | {m.get('roc_auc', float('nan')):.4f} |"
            )
        lines.append("")
    if rows_reg:
        lines.append("## Regression — Future price (5 years, lakhs)")
        lines.append("")
        lines.append("| Model | RMSE | MAE | R2 |")
        lines.append("|---|---:|---:|---:|")
        for r in sorted(rows_reg, key=lambda x: x["metrics"].get("rmse", 1e18)):
            m = r["metrics"]
            lines.append(
                f"| {r['model']} | {m.get('rmse', float('nan')):.4f} | "
                f"{m.get('mae', float('nan')):.4f} | {m.get('r2', float('nan')):.4f} |"
            )
        lines.append("")
    lines.extend(
        [
            "Notes:",
            "- The classification label is a 6-point rule. Even after dropping the exact rule features, related raw columns remain, so strong metrics are expected.",
            "- The regression target is `Price_in_Lakhs * (1+r)^5` with r from a known rate table, so R2 can be very high. That is a property of the synthetic target, not proof of a real price-forecasting engine.",
            "",
        ]
    )
    out = reports_dir() / "model_comparison.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def default_model_paths() -> dict:
    d = models_dir()
    return {
        "classifier": d / "good_investment_pipeline.joblib",
        "regressor": d / "future_price_pipeline.joblib",
    }
