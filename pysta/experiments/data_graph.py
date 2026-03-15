"""
data_graph.py - PySTA-based data loader for CircuitNet GNN training

Instead, it loads design data directly from OpenSTA CSVs using PySTA.

Same interface: load_data(args) returns (data_train, data_test)
"""
import torch
import dgl
import random
import os
import sys
import time
import numpy as np

# Add PySTA to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pysta import Design

random.seed(8026728)

# Default path to ZipCPU CSVs
DEFAULT_PYSTA_PATH = './data/zipcpu'
DELAY_COLS = ['Delay_Max_RR', 'Delay_Max_RF', 'Delay_Max_FR', 'Delay_Max_FF']
EDGE_FEATURE_COLS = ['InputTransition_ns', 'OutputLoad_pf']
LOG_EPS = 1e-4
LOG_SHIFT = 9.211


def pysta_to_dgl_graph(design, device='cuda', min_delay_ns=0.0, verbose=True):
    """
    Convert PySTA Design to DGL heterograph (same format as CircuitNet).
    
    Returns a graph with:
    - Node features 'nf': [x, y, capacitance, slew] (4D)
    - Edge data 'net_delay': raw delay values
    - Edge data 'ef': [input_transition, output_load]
    """
    nodes_df = design.nodes
    arcs_df = design.arcs

    # Predict only net arcs for the graph task.
    if 'ArcType' in arcs_df.columns:
        arcs_df = arcs_df[arcs_df['ArcType'] == 'net_arc'].copy()

    if verbose:
        print(f"  Nodes: {len(nodes_df)}, Arcs: {len(arcs_df)}")
    
    # Build node index
    name_to_idx = {name: i for i, name in enumerate(nodes_df['Name'])}
    num_nodes = len(nodes_df)
    
    # Extract node features [x, y, capacitance, slew]
    def get_col(df, col, default=0.0):
        if col in df.columns:
            vals = df[col].fillna(default).values
            # Handle pyarrow arrays
            if hasattr(vals, 'to_numpy'):
                vals = vals.to_numpy()
            return vals.astype(np.float32)
        return np.full(len(df), default, dtype=np.float32)
    
    x_coords = get_col(nodes_df, 'CoordX_um', 0.0)
    y_coords = get_col(nodes_df, 'CoordY_um', 0.0)
    capacitance = get_col(nodes_df, 'Capacitance_pf', 0.0)
    slew = get_col(nodes_df, 'SlewRise_ns', 0.0)
    
    # Normalize coordinates to [0, 1]
    if x_coords.max() > x_coords.min():
        x_coords = (x_coords - x_coords.min()) / (x_coords.max() - x_coords.min())
    if y_coords.max() > y_coords.min():
        y_coords = (y_coords - y_coords.min()) / (y_coords.max() - y_coords.min())
    
    # Log-normalize capacitance and slew
    capacitance = np.log1p(np.abs(capacitance))
    slew = np.log1p(np.abs(slew))
    
    node_features = np.stack([x_coords, y_coords, capacitance, slew], axis=1)
    if verbose:
        print(f"  Node features shape: {node_features.shape}")
    
    arcs = arcs_df.copy()

    if '_source_id' in arcs.columns and '_sink_id' in arcs.columns:
        arcs['_src'] = arcs['_source_id']
        arcs['_snk'] = arcs['_sink_id']
    else:
        arcs['_src'] = arcs['Source'].map(name_to_idx)
        arcs['_snk'] = arcs['Sink'].map(name_to_idx)

    arcs = arcs.dropna(subset=['_src', '_snk'])

    import pandas as pd
    available_delay_cols = [c for c in DELAY_COLS if c in arcs.columns]
    if not available_delay_cols:
        raise ValueError("No delay columns found in network_arcs.csv")

    for col in available_delay_cols:
        arcs[col] = pd.to_numeric(arcs[col], errors='coerce')
    arcs['_delay_ns'] = arcs[available_delay_cols].max(axis=1).fillna(0.0).astype(np.float32)

    if min_delay_ns > 0.0:
        arcs = arcs[arcs['_delay_ns'] >= float(min_delay_ns)].copy()

    src_ids = arcs['_src'].astype(np.int64).values
    sink_ids = arcs['_snk'].astype(np.int64).values
    delays = arcs['_delay_ns'].values.astype(np.float32)

    edge_features = []
    for col in EDGE_FEATURE_COLS:
        if col in arcs.columns:
            values = pd.to_numeric(arcs[col], errors='coerce').fillna(0.0).values.astype(np.float32)
        else:
            values = np.zeros(len(arcs), dtype=np.float32)
        edge_features.append(values)
    edge_attr = np.stack(edge_features, axis=1) if len(arcs) else np.zeros((0, len(EDGE_FEATURE_COLS)), dtype=np.float32)

    if verbose:
        print(f"  Valid edges: {len(src_ids)} / {len(arcs_df)}")
        if len(delays):
            print(f"  Delay range: [{delays.min():.6f}, {delays.max():.6f}] ns")
    
    # Create DGL heterograph
    graph_data = {
        ('node', 'net_out', 'node'): (src_ids, sink_ids),
        ('node', 'net_in', 'node'): (sink_ids, src_ids),
    }
    
    g = dgl.heterograph(graph_data, num_nodes_dict={'node': num_nodes})
    
    # Add node features
    g.ndata['nf'] = torch.tensor(node_features, dtype=torch.float32)
    
    # Add edge data (delay) - must be 2D for CircuitNet compatibility
    delay_tensor = torch.tensor(delays, dtype=torch.float32).unsqueeze(1)
    g.edges['net_out'].data['net_delay'] = delay_tensor
    g.edges['net_out'].data['net_delays_log'] = torch.log(delay_tensor + LOG_EPS) + LOG_SHIFT
    edge_tensor = torch.tensor(edge_attr, dtype=torch.float32)
    g.edges['net_out'].data['ef'] = edge_tensor
    g.edges['net_in'].data['ef'] = edge_tensor

    # Move to device - try CUDA, fall back to CPU
    if device == 'cuda':
        try:
            g = g.to('cuda')
        except Exception as e:
            print(f"  Warning: Could not move to CUDA ({e}), using CPU")
            device = 'cpu'
    
    if verbose:
        print(f"  Graph: {g.number_of_nodes()} nodes, {g.num_edges('net_out')} edges")
    
    return g


def load_data(args):
    """
    Load data using PySTA - same interface as original data_graph.py
    
    Returns (data_train, data_test) where each is a dict of {name: graph}
    """
    # Get PySTA path from args or use default
    pysta_path = getattr(args, 'pysta_path', None) or getattr(args, 'data_path', DEFAULT_PYSTA_PATH)
    
    # Check if it's a path to CSVs or to .bin files
    if os.path.exists(os.path.join(pysta_path, 'network_nodes.csv')):
        # It's a PySTA CSV path - load using PySTA
        print(f"[PySTA] Loading design from {pysta_path}...")
        t0 = time.time()
        design = Design(pysta_path, lazy_topology=True, verbose=True)
        print(f"[PySTA] Load time: {time.time()-t0:.2f}s")
        
        print("[PySTA] Converting to DGL...")
        t0 = time.time()
        g = pysta_to_dgl_graph(
            design,
            device='cuda' if getattr(args, 'device', 'cpu') == 'cuda' else 'cpu',
            min_delay_ns=getattr(args, 'min_delay_ns', 0.0),
            verbose=True,
        )
        print(f"[PySTA] Convert time: {time.time()-t0:.2f}s")
        
        # Create data dict with single design
        data = {'zipcpu': g}
        
        # For single design, use same for train and test
        data_train = data
        data_test = data
        
        print(f"[PySTA] Loaded 1 design (ZipCPU)")
        
    else:
        # Fall back to original .bin file loading
        print(f"[Original] Loading from {pysta_path}...")
        available_data = os.listdir(pysta_path)
        
        available_data_temp = [i for i in available_data if i.endswith('.bin')]
        print(f"DEBUG: {len(available_data_temp)}, {args.train_data_number}")
        
        random.seed(8026728)
        train_data_keys = random.sample(available_data_temp, args.train_data_number)
        test_data_keys = [i for i in available_data_temp if i not in train_data_keys]
        random.seed(8026728)
        test_data_keys = random.sample(test_data_keys, args.test_data_number)
        
        data = {}
        for k in available_data_temp:
            g = dgl.load_graphs(os.path.join(pysta_path, k))[0][0].to('cuda')
            g.edges['net_out'].data['net_delays_log'] = (
                torch.log(0.0001 + g.edges['net_out'].data['net_delay']) + 9.211
            )
            data[k] = g
        
        data_train = {k: t for k, t in data.items() if k in train_data_keys}
        data_test = {k: t for k, t in data.items() if k in test_data_keys}
    
    return data_train, data_test
