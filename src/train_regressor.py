"""Train 5-year price regressors, log to MLflow, register the best model."""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")

import os

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from xgboost import XGBRegressor

from src.evaluate import (
    extract_feature_importance,
    plot_feature_importance,
    regression_metrics,
    write_comparison_tables,
    write_metrics_json,
)
from src.features import build_model_pipeline, raw_frame
from src.utils import RANDOM_STATE, TEST_SIZE, data_processed_path, models_dir, reports_dir

EXPERIMENT = "future_price_regression"
REGISTERED_NAME = "FuturePriceRegressor"
TUNE_SAMPLE = 30_000
TUNE_ITERS = 6


def load_xy() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(data_processed_path())
    X = raw_frame(df)
    y = df["Future_Price_5Y_Lakhs"].astype(float)
    return X, y


def log_run(name: str, pipe, params: dict, X_train, y_train, X_test, y_test, extra: dict | None = None):
    y_pred = pipe.predict(X_test)
    metrics = regression_metrics(y_test, y_pred)

    names, values = extract_feature_importance(pipe)
    fi_path = None
    if names is not None:
        fi_path = reports_dir() / "figures" / f"fi_reg_{name}.png"
        plot_feature_importance(names, values, f"Feature importance — {name}", fi_path)

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=name):
        mlflow.log_params({k: str(v) for k, v in params.items()})
        if extra:
            mlflow.log_params({f"extra_{k}": str(v) for k, v in extra.items()})
        mlflow.log_metrics(metrics)
        if fi_path:
            mlflow.log_artifact(str(fi_path))
        mlflow.sklearn.log_model(pipe, "model")

    write_metrics_json(
        f"reg_{name}",
        {"task": "regression", "model": name, "metrics": metrics, "params": params},
    )
    print(f"{name}: {metrics}")
    return metrics, pipe


def run():
    mlflow.set_tracking_uri("file:" + str(ROOT / "mlruns"))
    X, y = load_xy()
    print("future price describe:\n", y.describe())
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    candidates = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=60,
            max_depth=12,
            min_samples_leaf=20,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "xgboost": XGBRegressor(
            n_estimators=80,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            tree_method="hist",
        ),
    }

    results = {}
    fitted = {}
    for name, est in candidates.items():
        print(f"\n=== Training {name} ===")
        pipe = build_model_pipeline(est, task="regressor")
        pipe.fit(X_train, y_train)
        metrics, pipe = log_run(name, pipe, est.get_params(), X_train, y_train, X_test, y_test)
        results[name] = metrics
        fitted[name] = pipe

    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.choice(len(X_train), size=min(TUNE_SAMPLE, len(X_train)), replace=False)
    X_s, y_s = X_train.iloc[idx], y_train.iloc[idx]

    search_spaces = {
        "random_forest": (
            RandomForestRegressor(n_jobs=-1, random_state=RANDOM_STATE),
            {
                "model__n_estimators": [40, 80, 120],
                "model__max_depth": [8, 12, 16],
                "model__min_samples_leaf": [10, 20, 40],
            },
        ),
        "xgboost": (
            XGBRegressor(n_jobs=-1, random_state=RANDOM_STATE, tree_method="hist"),
            {
                "model__n_estimators": [60, 100, 140],
                "model__max_depth": [4, 6, 8],
                "model__learning_rate": [0.05, 0.1, 0.2],
                "model__subsample": [0.7, 0.9],
            },
        ),
    }

    for name, (est, grid) in search_spaces.items():
        print(f"\n=== RandomizedSearch {name} (n={len(X_s)}) ===")
        pipe = build_model_pipeline(est, task="regressor")
        search = RandomizedSearchCV(
            pipe,
            grid,
            n_iter=TUNE_ITERS,
            scoring="r2",
            cv=3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=1,
        )
        search.fit(X_s, y_s)
        best_pipe = build_model_pipeline(est, task="regressor")
        best_pipe.set_params(**search.best_params_)
        best_pipe.fit(X_train, y_train)
        tuned_name = f"{name}_tuned"
        metrics, best_pipe = log_run(
            tuned_name,
            best_pipe,
            search.best_params_,
            X_train,
            y_train,
            X_test,
            y_test,
            extra={"search_sample": len(X_s)},
        )
        results[tuned_name] = metrics
        fitted[tuned_name] = best_pipe

    best_name = min(results, key=lambda k: results[k]["rmse"])
    best_pipe = fitted[best_name]
    out_path = models_dir() / "future_price_pipeline.joblib"
    joblib.dump(best_pipe, out_path)
    print(f"\nBest regressor: {best_name} -> {out_path}")

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=f"register_{best_name}"):
        mlflow.log_params({"registered_from": best_name})
        mlflow.log_metrics(results[best_name])
        mlflow.sklearn.log_model(
            best_pipe,
            "model",
            registered_model_name=REGISTERED_NAME,
        )

    meta = {"best_model": best_name, "path": str(out_path), "metrics": results[best_name]}
    (models_dir() / "regressor_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    write_comparison_tables()
    return best_name, results


if __name__ == "__main__":
    run()
