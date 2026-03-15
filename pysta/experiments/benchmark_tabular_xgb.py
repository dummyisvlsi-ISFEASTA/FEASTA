"""
Benchmark FEASTA tabular ML on a CSV dump using XGBoost.

This script measures the tabular-only pipeline stages needed for the
comparison table:
  - PySTA Load CSV
  - PySTA Pre-Processing
  - PySTA Convert to Tensors
  - Training
  - Accuracy (R^2, MAE in ns)

Feature generation runtime is supplied separately from the OpenSTA run
because it is measured during CSV export.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pysta import Design
from pysta.experiments.pysta_tabular_ml import preprocess_tabular


def _parse_feature_gen_runtime(args):
    if args.feature_gen_rt is not None:
        return float(args.feature_gen_rt)

    if args.feature_gen_log:
        with open(args.feature_gen_log, "r", encoding="utf-8", errors="ignore") as f:
            for line in reversed(f.readlines()):
                line = line.strip()
                if line.startswith("ELAPSED="):
                    return float(line.split("=", 1)[1])
    return None


def _make_tensors(X_train, X_test, y_train, y_test, device):
    t0 = time.time()

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        X_train_t = X_train_t.to("cuda")
        X_test_t = X_test_t.to("cuda")
        y_train_t = y_train_t.to("cuda")
        y_test_t = y_test_t.to("cuda")

    tensor_time = time.time() - t0
    return tensor_time, X_train_t, X_test_t, y_train_t, y_test_t


def _make_model(device, seed, n_estimators, max_depth, learning_rate):
    kwargs = dict(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=seed,
        n_jobs=-1,
        tree_method="hist",
    )

    if device == "cuda":
        kwargs["device"] = "cuda"
    else:
        kwargs["device"] = "cpu"

    return xgb.XGBRegressor(**kwargs)


def main():
    parser = argparse.ArgumentParser(description="Benchmark FEASTA tabular ML with XGBoost")
    parser.add_argument("--csv_dir", required=True, help="Directory containing FEASTA CSVs")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Tensor/XGBoost device")
    parser.add_argument("--feature-gen-rt", type=float, default=None, help="Feature generation runtime in seconds")
    parser.add_argument("--feature-gen-log", default=None, help="Log file containing a trailing ELAPSED=<sec> line")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test fraction")
    parser.add_argument("--seed", type=int, default=8026728, help="Random seed")
    parser.add_argument("--target-mode", choices=["raw", "log"], default="log", help="Train on raw delay or log1p(delay)")
    parser.add_argument("--n-estimators", type=int, default=500, help="XGBoost boosting rounds")
    parser.add_argument("--max-depth", type=int, default=8, help="XGBoost max depth")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="XGBoost learning rate")
    args = parser.parse_args()

    feature_gen_rt = _parse_feature_gen_runtime(args)

    t0 = time.time()
    design = Design(args.csv_dir, lazy_topology=True, verbose=False)
    load_time = time.time() - t0

    t0 = time.time()
    result = preprocess_tabular(design, verbose=False)
    preprocess_time = time.time() - t0

    X = result["X"]
    y_raw = result["y"]
    y_train_target = result["y_log"] if args.target_mode == "log" else y_raw

    X_train, X_test, y_train, y_test, _y_train_raw, y_test_raw = train_test_split(
        X,
        y_train_target,
        y_raw,
        test_size=args.test_size,
        random_state=args.seed,
    )

    tensor_time, _, _, _, _ = _make_tensors(X_train, X_test, y_train, y_test, args.device)

    model = _make_model(
        args.device,
        args.seed,
        args.n_estimators,
        args.max_depth,
        args.learning_rate,
    )

    train_t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - train_t0

    pred = model.predict(X_test)
    if args.target_mode == "log":
        pred_ns = np.expm1(pred)
    else:
        pred_ns = pred

    pred_ns = np.maximum(pred_ns, 0.0)

    r2 = r2_score(y_test_raw, pred_ns)
    mae = mean_absolute_error(y_test_raw, pred_ns)

    print("=" * 70)
    print("FEASTA Tabular Benchmark")
    print("=" * 70)
    print(f"CSV dir               : {args.csv_dir}")
    print(f"Device                : {args.device}")
    print(f"Samples               : {len(X):,}")
    print(f"Features              : {X.shape[1]}")
    print(f"Target mode           : {args.target_mode}")
    print("-" * 70)
    print(f"Feature Generation RT : {feature_gen_rt:.2f}s" if feature_gen_rt is not None else "Feature Generation RT : N/A")
    print(f"PySTA Load CSV        : {load_time:.2f}s")
    print(f"PySTA Pre-Processing  : {preprocess_time:.2f}s")
    print(f"PySTA Convert Tensors : {tensor_time:.2f}s")
    print(f"Training              : {train_time:.2f}s (XGBoost)")
    print(f"Accuracy (R^2)        : {r2:.4f}")
    print(f"Accuracy (MAE)        : {mae:.5f} ns")
    print("=" * 70)

    print("TABLE_ROW")
    print(
        "\t".join(
            [
                f"{feature_gen_rt:.2f}s" if feature_gen_rt is not None else "N/A",
                f"{load_time:.2f}s",
                f"{preprocess_time:.2f}s",
                f"{tensor_time:.2f}s",
                f"{train_time:.2f}s (XGBoost)",
                f"{r2:.4f}",
                f"{mae:.5f} ns",
            ]
        )
    )


if __name__ == "__main__":
    main()
