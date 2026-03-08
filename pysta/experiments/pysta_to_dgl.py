"""
PySTA to DGL Graph Converter

Converts PySTA Design objects to DGL heterographs compatible with
CircuitNet's TimingGCN for net delay prediction.

Usage:
    python pysta_to_dgl.py --pysta_path /path/to/csv/dir --output_path ./graph.bin
"""

import sys
import os
import time
import argparse
import numpy as np

# Add PySTA to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set DGL backend before importing
os.environ['DGLBACKEND'] = 'pytorch'

import torch
import dgl

from pysta import Design


def pysta_to_dgl_graph(design, device='cpu', verbose=True):
    """
    Convert PySTA Design to DGL heterograph.
    
    Parameters
    ----------
    design : Design
        PySTA Design object with loaded CSVs
    device : str
        Device to place graph on ('cpu' or 'cuda')
    verbose : bool
        Print progress information
        
    Returns
    -------
    dgl.DGLHeteroGraph
        Heterograph with:
        - Node features: [x, y, capacitance, slew]
        - Edge types: 'net_out' (driver->sink), 'net_in' (sink->driver)
        - Edge data: net_delay (target for prediction)
    """
    nodes_df = design.nodes
    arcs_df = design.arcs
    
    if verbose:
        print(f"  Converting design to DGL graph...")
        print(f"    Nodes: {len(nodes_df):,}")
        print(f"    Arcs: {len(arcs_df):,}")
    
    # ====================
    # NODE FEATURES
    # ====================
    
    def get_col(df, col, default=0.0):
        """Safely extract column with default value."""
        if col in df.columns:
            return df[col].fillna(default).values.astype(np.float32)
        return np.full(len(df), default, dtype=np.float32)
    
    # Extract features
    x_coords = get_col(nodes_df, 'CoordX_um', 0.0)
    y_coords = get_col(nodes_df, 'CoordY_um', 0.0)
    capacitance = get_col(nodes_df, 'Capacitance_pf', 0.0)
    slew = get_col(nodes_df, 'SlewRise_ns', 0.0)
    
    # Normalize coordinates to [0, 1]
    x_range = x_coords.max() - x_coords.min()
    y_range = y_coords.max() - y_coords.min()
    
    if x_range > 0:
        x_coords = (x_coords - x_coords.min()) / x_range
    if y_range > 0:
        y_coords = (y_coords - y_coords.min()) / y_range
    
    # Log-normalize capacitance and slew (handle zeros/negatives)
    capacitance = np.log1p(np.maximum(capacitance, 0))
    slew = np.log1p(np.maximum(slew, 0))
    
    # Stack features: [x, y, cap, slew]
    node_features = np.stack([x_coords, y_coords, capacitance, slew], axis=1)
    
    if verbose:
        print(f"    Node features shape: {node_features.shape}")
        print(f"    Coord range: X=[{x_coords.min():.3f}, {x_coords.max():.3f}], "
              f"Y=[{y_coords.min():.3f}, {y_coords.max():.3f}]")
    
    # ====================
    # EDGES
    # ====================
    
    if len(arcs_df) == 0:
        raise ValueError("No arcs found - cannot build graph")
    
    # Get source and sink IDs
    if '_source_id' in arcs_df.columns and '_sink_id' in arcs_df.columns:
        src_ids = arcs_df['_source_id'].values
        sink_ids = arcs_df['_sink_id'].values
    else:
        # Build name-to-index mapping
        name_to_idx = {name: i for i, name in enumerate(nodes_df['Name'])}
        src_ids = arcs_df['Source'].map(name_to_idx).values
        sink_ids = arcs_df['Sink'].map(name_to_idx).values
    
    # Filter out invalid edges (NaN from missing mappings)
    valid_mask = ~(np.isnan(src_ids.astype(float)) | np.isnan(sink_ids.astype(float)))
    src_ids = src_ids[valid_mask].astype(np.int64)
    sink_ids = sink_ids[valid_mask].astype(np.int64)
    
    if verbose:
        print(f"    Valid edges: {len(src_ids):,} / {len(arcs_df):,}")
    
    # Get delay values (target for prediction)
    if 'Delay' in arcs_df.columns:
        delays = arcs_df['Delay'].fillna(0.0).values[valid_mask].astype(np.float32)
    else:
        delays = np.zeros(len(src_ids), dtype=np.float32)
        if verbose:
            print("    WARNING: No 'Delay' column found, using zeros")
    
    # ====================
    # BUILD HETEROGRAPH
    # ====================
    
    num_nodes = len(nodes_df)
    
    # Create heterograph with bidirectional edges
    graph_data = {
        ('node', 'net_out', 'node'): (src_ids, sink_ids),  # driver -> sink
        ('node', 'net_in', 'node'): (sink_ids, src_ids),   # sink -> driver
    }
    
    g = dgl.heterograph(graph_data, num_nodes_dict={'node': num_nodes})
    
    # Add node features
    g.ndata['nf'] = torch.tensor(node_features, dtype=torch.float32)
    
    # Add edge delay (raw)
    g.edges['net_out'].data['net_delay'] = torch.tensor(delays, dtype=torch.float32).unsqueeze(1)
    
    # Add log-transformed delay (target for training)
    # Same transformation as CircuitNet: log(delay + 1e-4) + 9.211
    log_delays = torch.log(torch.tensor(delays, dtype=torch.float32) + 1e-4) + 9.211
    g.edges['net_out'].data['net_delays_log'] = log_delays.unsqueeze(1)
    
    if verbose:
        print(f"    Graph created: {g.number_of_nodes():,} nodes, "
              f"{g.num_edges('net_out'):,} edges")
        print(f"    Delay range: [{delays.min():.6f}, {delays.max():.6f}]")
        print(f"    Log delay range: [{log_delays.min():.3f}, {log_delays.max():.3f}]")
    
    # Move to device
    if device != 'cpu':
        g = g.to(device)
    
    return g


def benchmark_pysta_to_dgl(pysta_path, output_path=None, design_name=None, device='cpu'):
    """
    Benchmark PySTA loading and DGL conversion.
    
    Parameters
    ----------
    pysta_path : str
        Path to directory containing PySTA CSVs
    output_path : str, optional
        Path to save DGL graph (.bin file)
    design_name : str, optional
        Name for the design
    device : str
        Device for the graph
        
    Returns
    -------
    tuple
        (results_dict, graph)
    """
    results = {}
    
    print("\n" + "="*60)
    print("PySTA -> DGL Conversion Benchmark")
    print("="*60)
    
    # Step 1: Load with PySTA
    print("\n[1] Loading design with PySTA...")
    t0 = time.time()
    design = Design(pysta_path, name=design_name, lazy_topology=True, verbose=True)
    t_load = time.time() - t0
    results['pysta_load_time'] = t_load
    print(f"  [INFO] PySTA load time: {t_load:.2f}s")
    
    # Step 2: Convert to DGL
    print("\n[2] Converting to DGL graph...")
    t0 = time.time()
    graph = pysta_to_dgl_graph(design, device=device, verbose=True)
    t_convert = time.time() - t0
    results['dgl_convert_time'] = t_convert
    print(f"  [INFO] DGL conversion time: {t_convert:.2f}s")
    
    # Step 3: Save graph (optional)
    if output_path:
        print(f"\n[3] Saving graph to {output_path}...")
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        t0 = time.time()
        dgl.save_graphs(output_path, graph)
        t_save = time.time() - t0
        results['save_time'] = t_save
        print(f"  [INFO] Save time: {t_save:.2f}s")
        print(f"  [INFO] File size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")
    
    # Summary
    results['total_time'] = results['pysta_load_time'] + results['dgl_convert_time']
    results['nodes'] = graph.number_of_nodes()
    results['edges'] = graph.num_edges('net_out')
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"  Nodes:        {results['nodes']:,}")
    print(f"  Edges:        {results['edges']:,}")
    print(f"  PySTA Load:   {results['pysta_load_time']:.2f}s")
    print(f"  DGL Convert:  {results['dgl_convert_time']:.2f}s")
    print(f"  TOTAL:        {results['total_time']:.2f}s")
    print("="*60 + "\n")
    
    return results, graph


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PySTA output to CircuitNet DGL format")
    parser.add_argument("--pysta_path", type=str, required=True,
                        help="Path to directory containing PySTA CSVs")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Path to save DGL graph (e.g., ./graph/design.bin)")
    parser.add_argument("--name", type=str, default=None,
                        help="Design name")
    parser.add_argument("--device", type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help="Device to place graph on")
    
    args = parser.parse_args()
    
    # Auto-detect CUDA
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available, falling back to CPU")
        args.device = 'cpu'
    
    results, graph = benchmark_pysta_to_dgl(
        args.pysta_path, 
        args.output_path,
        args.name,
        args.device
    )
