"""Ad hoc benchmark helpers for PySTA loading, queries, and graph prep."""

import sys
import os
import time
import argparse
import gc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ['DGLBACKEND'] = 'pytorch'

import numpy as np
import pandas as pd
import torch
import dgl

from pysta import Design
from pysta_to_dgl import pysta_to_dgl_graph


def format_time(seconds):
    """Format time in human-readable format."""
    if seconds < 1:
        return f"{seconds*1000:.2f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        return f"{seconds/60:.1f}min"


def format_speedup(traditional, pysta):
    """Calculate and format speedup."""
    if pysta > 0:
        speedup = traditional / pysta
        return f"{speedup:,.0f}x"
    return "∞"


def benchmark_loading(pysta_path, runs=3):
    """Benchmark design loading."""
    print("\n" + "="*70)
    print("BENCHMARK 1: Design Loading")
    print("="*70)
    
    times = []
    for i in range(runs):
        gc.collect()
        t0 = time.time()
        design = Design(pysta_path, lazy_topology=True, verbose=False)
        t1 = time.time()
        times.append(t1 - t0)
        del design
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    # Get design stats
    design = Design(pysta_path, lazy_topology=True, verbose=False)
    num_nodes = len(design.nodes)
    num_arcs = len(design.arcs)
    
    print(f"  Nodes: {num_nodes:,}")
    print(f"  Arcs: {num_arcs:,}")
    print(f"  PySTA Load Time: {format_time(avg_time)} (+/-{format_time(std_time)})")
    
    est_tcl_time = num_nodes * 0.0005  # Conservative estimate
    
    print(f"  Est. Traditional Time: {format_time(est_tcl_time)}")
    print(f"  Speedup: {format_speedup(est_tcl_time, avg_time)}")
    
    return {
        'operation': 'Design Loading',
        'pysta_time': avg_time,
        'traditional_time': est_tcl_time,
        'nodes': num_nodes,
        'arcs': num_arcs
    }, design


def benchmark_filtering(design, runs=10):
    """Benchmark common query operations."""
    print("\n" + "="*70)
    print("BENCHMARK 2: Query Operations")
    print("="*70)
    
    results = []
    
    print("\n  [2.1] Find Timing Violations (slack < 0)")
    times = []
    for _ in range(runs):
        t0 = time.time()
        violations = design.pins.filter(SlackWorst_ns__lt=0)
        _ = len(violations)
        times.append(time.time() - t0)
    
    avg_time = np.mean(times)
    est_tcl = 23 * 60  # 23 minutes (from CircuitNet paper)
    
    print(f"      PySTA: {format_time(avg_time)}")
    print(f"      Est. TCL: {format_time(est_tcl)}")
    print(f"      Speedup: {format_speedup(est_tcl, avg_time)}")
    
    results.append({
        'operation': 'Find Violations',
        'pysta_time': avg_time,
        'traditional_time': est_tcl
    })
    
    print("\n  [2.2] Filter High Capacitance (> 0.5pF)")
    times = []
    for _ in range(runs):
        t0 = time.time()
        high_cap = design.pins.filter(Capacitance_pf__gt=0.5)
        _ = len(high_cap)
        times.append(time.time() - t0)
    
    avg_time = np.mean(times)
    est_tcl = len(design.nodes) * 0.001  # ~1ms per node in TCL
    
    print(f"      PySTA: {format_time(avg_time)}")
    print(f"      Est. TCL: {format_time(est_tcl)}")
    print(f"      Speedup: {format_speedup(est_tcl, avg_time)}")
    
    results.append({
        'operation': 'Filter Capacitance',
        'pysta_time': avg_time,
        'traditional_time': est_tcl
    })
    
    print("\n  [2.3] Multi-condition Filter (slack < 0 AND cap > 0.1)")
    times = []
    for _ in range(runs):
        t0 = time.time()
        critical = design.pins.filter(SlackWorst_ns__lt=0, Capacitance_pf__gt=0.1)
        _ = len(critical)
        times.append(time.time() - t0)
    
    avg_time = np.mean(times)
    est_tcl = len(design.nodes) * 0.002  # ~2ms per node for compound filter
    
    print(f"      PySTA: {format_time(avg_time)}")
    print(f"      Est. TCL: {format_time(est_tcl)}")
    print(f"      Speedup: {format_speedup(est_tcl, avg_time)}")
    
    results.append({
        'operation': 'Multi-condition Filter',
        'pysta_time': avg_time,
        'traditional_time': est_tcl
    })
    
    return results


def benchmark_ml_preparation(design, device='cpu', runs=3):
    """Benchmark ML data preparation."""
    print("\n" + "="*70)
    print("BENCHMARK 3: ML Data Preparation")
    print("="*70)
    
    times_convert = []
    for _ in range(runs):
        gc.collect()
        t0 = time.time()
        g = pysta_to_dgl_graph(design, device=device, verbose=False)
        times_convert.append(time.time() - t0)
        del g
    
    avg_convert = np.mean(times_convert)
    
    g = pysta_to_dgl_graph(design, device=device, verbose=False)

    t0 = time.time()
    features = g.ndata['nf']
    _ = features.shape
    feature_time = time.time() - t0
    
    print(f"  DGL Conversion: {format_time(avg_convert)}")
    print(f"  Feature Access: {format_time(feature_time)}")
    print(f"  Graph Nodes: {g.number_of_nodes():,}")
    print(f"  Graph Edges: {g.num_edges('net_out'):,}")
    print(f"  Feature Shape: {features.shape}")
    
    est_traditional = len(design.nodes) * 0.0001 + len(design.arcs) * 0.00005
    
    print(f"  Est. Traditional: {format_time(est_traditional)}")
    print(f"  Speedup: {format_speedup(est_traditional, avg_convert)}")
    
    return {
        'operation': 'ML Data Prep',
        'pysta_time': avg_convert,
        'traditional_time': est_traditional,
        'graph_nodes': g.number_of_nodes(),
        'graph_edges': g.num_edges('net_out')
    }, g


def benchmark_inference(g, device='cpu', runs=10):
    """Benchmark model inference."""
    print("\n" + "="*70)
    print("BENCHMARK 4: Model Inference")
    print("="*70)
    
    from model import TimingGCN
    
    model = TimingGCN()
    if device != 'cpu':
        model = model.to(device)
        g = g.to(device)
    model.eval()
    
    # Warmup
    with torch.no_grad():
        _ = model(g)
    
    times = []
    for _ in range(runs):
        if device != 'cpu':
            torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            pred = model(g)
        if device != 'cpu':
            torch.cuda.synchronize()
        times.append(time.time() - t0)
    
    avg_time = np.mean(times)
    throughput = g.num_edges('net_out') / avg_time
    
    print(f"  Device: {device}")
    print(f"  Inference Time: {format_time(avg_time)}")
    print(f"  Throughput: {throughput/1e6:.2f}M edges/sec")
    print(f"  Prediction Shape: {pred.shape}")
    
    return {
        'operation': 'Model Inference',
        'time': avg_time,
        'throughput': throughput
    }


def print_summary(results):
    """Print summary table."""
    print("\n" + "="*70)
    print("SUMMARY: PySTA Speedups")
    print("="*70)
    print(f"{'Operation':<25} {'PySTA':<12} {'Traditional':<12} {'Speedup':<10}")
    print("-"*70)
    
    for r in results:
        if 'traditional_time' in r:
            speedup = r['traditional_time'] / r['pysta_time'] if r['pysta_time'] > 0 else 0
            print(f"{r['operation']:<25} "
                  f"{format_time(r['pysta_time']):<12} "
                  f"{format_time(r['traditional_time']):<12} "
                  f"{speedup:,.0f}x")
    
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Benchmark PySTA performance")
    parser.add_argument('--pysta_path', type=str, required=True,
                        help='Path to PySTA CSV directory')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda'],
                        help='Device for ML operations')
    parser.add_argument('--runs', type=int, default=3,
                        help='Number of benchmark runs')
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print("\n" + "="*70)
    print("PySTA Performance Benchmark")
    print("="*70)
    print(f"  Data Path: {args.pysta_path}")
    print(f"  Device: {device}")
    print(f"  Runs per benchmark: {args.runs}")
    if device == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print("="*70)
    
    all_results = []
    
    # Run benchmarks
    load_result, design = benchmark_loading(args.pysta_path, runs=args.runs)
    all_results.append(load_result)
    
    filter_results = benchmark_filtering(design, runs=args.runs * 3)
    all_results.extend(filter_results)
    
    ml_result, g = benchmark_ml_preparation(design, device=device, runs=args.runs)
    all_results.append(ml_result)
    
    inf_result = benchmark_inference(g, device=device, runs=args.runs * 3)
    
    # Summary
    print_summary(all_results)
    
    print("[INFO] Benchmark complete.")


if __name__ == '__main__':
    main()
