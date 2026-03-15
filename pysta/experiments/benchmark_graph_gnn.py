"""
Benchmark FEASTA graph ML on a CSV dump using the DGL TimingGCN model.

This script measures:
  - PySTA Load CSV
  - PySTA Pre-Processing
  - PySTA Convert to Tensors
  - Training
  - Accuracy (R^2, MAE in ns)

It also supports trimming tiny-delay net arcs via --min-delay-ns.
"""

import argparse
import os
import random
import sys
import time

import dgl
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pysta import Design
from pysta.experiments.data_graph import pysta_to_dgl_graph, LOG_EPS, LOG_SHIFT
from pysta.experiments.model import TimingGCN


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


def _inverse_log_delay(values):
    values = np.asarray(values, dtype=np.float32)
    return np.maximum(np.exp(values - LOG_SHIFT) - LOG_EPS, 0.0)


def _make_edge_split(num_edges, test_size, seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(num_edges)
    test_count = max(1, int(num_edges * test_size))
    test_idx = perm[:test_count]
    train_idx = perm[test_count:]
    if len(train_idx) == 0:
        train_idx = perm[:-1]
        test_idx = perm[-1:]
    return train_idx, test_idx


def main():
    parser = argparse.ArgumentParser(description="Benchmark FEASTA graph ML with TimingGCN")
    parser.add_argument("--csv_dir", required=True, help="Directory containing FEASTA CSVs")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="DGL/Torch device")
    parser.add_argument("--feature-gen-rt", type=float, default=None, help="Feature generation runtime in seconds")
    parser.add_argument("--feature-gen-log", default=None, help="Log file containing ELAPSED=<sec>")
    parser.add_argument("--min-delay-ns", type=float, default=0.0, help="Drop net arcs with delay below this threshold")
    parser.add_argument("--epochs", type=int, default=300, help="Training epochs")
    parser.add_argument("--lr", type=float, default=5e-4, help="Adam learning rate")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test fraction over edges")
    parser.add_argument("--seed", type=int, default=8026728, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    feature_gen_rt = _parse_feature_gen_runtime(args)

    load_t0 = time.time()
    design = Design(args.csv_dir, lazy_topology=True, verbose=False)
    load_time = time.time() - load_t0

    preprocess_t0 = time.time()
    # Pre-processing is measured separately from graph conversion:
    # count valid net arcs and define the edge split after trimming.
    arcs = design.arcs
    if 'ArcType' in arcs.columns:
        arcs = arcs[arcs['ArcType'] == 'net_arc'].copy()
    delay_cols = [c for c in ['Delay_Max_RR', 'Delay_Max_RF', 'Delay_Max_FR', 'Delay_Max_FF'] if c in arcs.columns]
    import pandas as pd
    for col in delay_cols:
        arcs[col] = pd.to_numeric(arcs[col], errors='coerce')
    arcs['_delay_ns'] = arcs[delay_cols].max(axis=1).fillna(0.0).astype(np.float32)
    if args.min_delay_ns > 0.0:
        arcs = arcs[arcs['_delay_ns'] >= args.min_delay_ns].copy()
    preprocess_time = time.time() - preprocess_t0

    convert_t0 = time.time()
    g = pysta_to_dgl_graph(
        design,
        device=args.device,
        min_delay_ns=args.min_delay_ns,
        verbose=False,
    )
    convert_time = time.time() - convert_t0

    num_edges = g.num_edges('net_out')
    if num_edges < 2:
        raise ValueError("Not enough edges after trimming; lower --min-delay-ns")

    train_idx_np, test_idx_np = _make_edge_split(num_edges, args.test_size, args.seed)
    train_idx = torch.tensor(train_idx_np, dtype=torch.long, device=g.device)
    test_idx = torch.tensor(test_idx_np, dtype=torch.long, device=g.device)

    model = TimingGCN()
    if args.device == "cuda":
        try:
            model = model.to("cuda")
        except Exception as e:
            raise RuntimeError(f"CUDA requested for model but unavailable: {e}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_t0 = time.time()
    for _ in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(g).squeeze(-1)
        truth = g.edges['net_out'].data['net_delays_log'].squeeze(-1)
        loss = F.mse_loss(pred[train_idx], truth[train_idx])
        loss.backward()
        optimizer.step()
    if args.device == "cuda":
        torch.cuda.synchronize()
    train_time = time.time() - train_t0

    model.eval()
    with torch.no_grad():
        pred = model(g).squeeze(-1)
        truth = g.edges['net_out'].data['net_delays_log'].squeeze(-1)

    pred_test = pred[test_idx].detach().cpu().numpy()
    truth_test = truth[test_idx].detach().cpu().numpy()

    pred_ns = _inverse_log_delay(pred_test)
    truth_ns = _inverse_log_delay(truth_test)

    r2 = r2_score(truth_ns, pred_ns)
    mae = mean_absolute_error(truth_ns, pred_ns)

    print("=" * 70)
    print("FEASTA Graph Benchmark")
    print("=" * 70)
    print(f"CSV dir               : {args.csv_dir}")
    print(f"Device                : {args.device}")
    print(f"Min delay threshold   : {args.min_delay_ns:.6f} ns")
    print(f"Nodes                 : {g.number_of_nodes():,}")
    print(f"Edges                 : {num_edges:,}")
    print("-" * 70)
    print(f"Feature Generation RT : {feature_gen_rt:.2f}s" if feature_gen_rt is not None else "Feature Generation RT : N/A")
    print(f"PySTA Load CSV        : {load_time:.2f}s")
    print(f"PySTA Pre-Processing  : {preprocess_time:.2f}s")
    print(f"PySTA Convert Tensors : {convert_time:.2f}s")
    print(f"Training              : {train_time:.2f}s (TimingGCN) : {args.epochs} epochs")
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
                f"{convert_time:.2f}s",
                f"{train_time:.2f}s (TimingGCN) : {args.epochs} epochs",
                f"{r2:.4f}",
                f"{mae:.5f} ns",
            ]
        )
    )


if __name__ == "__main__":
    main()
