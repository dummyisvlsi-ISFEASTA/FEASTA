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

# Add PySTA to path (relative to project root)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from pysta import Design

random.seed(8026728)

# Default path to ZipCPU CSVs
DEFAULT_PYSTA_PATH = './data/zipcpu'


def pysta_to_dgl_graph(design, device='cuda'):
    """
    Convert PySTA Design to DGL heterograph (same format as CircuitNet).
    
    Returns a graph with:
    - Node features 'nf': [x, y, capacitance, slew] (4D)
    - Edge data 'net_delay': delay values
    """
    import numpy as np
    
    nodes_df = design.nodes
    arcs_df = design.arcs
    
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
    print(f"  Node features shape: {node_features.shape}")
    
    # Build edges from arcs
    src_names = arcs_df['Source'].values
    sink_names = arcs_df['Sink'].values
    
    # Handle pyarrow arrays
    if hasattr(src_names, 'to_numpy'):
        src_names = src_names.to_numpy()
    if hasattr(sink_names, 'to_numpy'):
        sink_names = sink_names.to_numpy()
    
    src_ids = []
    sink_ids = []
    valid_indices = []
    
    for i, (src, sink) in enumerate(zip(src_names, sink_names)):
        if src in name_to_idx and sink in name_to_idx:
            src_ids.append(name_to_idx[src])
            sink_ids.append(name_to_idx[sink])
            valid_indices.append(i)
    
    src_ids = np.array(src_ids, dtype=np.int64)
    sink_ids = np.array(sink_ids, dtype=np.int64)
    
    print(f"  Valid edges: {len(src_ids)} / {len(arcs_df)}")
    
    # Get delay values
    if 'Delay' in arcs_df.columns:
        delays = arcs_df['Delay'].fillna(0.0).values
        if hasattr(delays, 'to_numpy'):
            delays = delays.to_numpy()
        delays = delays[valid_indices].astype(np.float32)
    else:
        delays = np.zeros(len(src_ids), dtype=np.float32)
    
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
    
    # Move to device - try CUDA, fall back to CPU
    if device == 'cuda':
        try:
            g = g.to('cuda')
        except Exception as e:
            print(f"  Warning: Could not move to CUDA ({e}), using CPU")
            device = 'cpu'
    
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
        g = pysta_to_dgl_graph(design)
        print(f"[PySTA] Convert time: {time.time()-t0:.2f}s")
        
        # Add log-delay (same transform as original)
        g.edges['net_out'].data['net_delays_log'] = (
            torch.log(0.0001 + g.edges['net_out'].data['net_delay']) + 9.211
        )
        
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
