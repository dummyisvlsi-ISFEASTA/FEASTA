"""Build adjacency and traversal structures from FEASTA arc data."""

from typing import Dict, List, Optional, Set, Tuple
from collections import deque

import pandas as pd
import numpy as np
from scipy import sparse

class TopologyBuilder:
    """Graph topology derived from the arc table."""
    
    def __init__(self, num_nodes: int):
        self.num_nodes = num_nodes
        self.forward_adj: Optional[sparse.csr_matrix] = None
        self.backward_adj: Optional[sparse.csr_matrix] = None
        self.edge_delays: Optional[np.ndarray] = None
        self.edge_types: Optional[np.ndarray] = None
        self.has_cycles: bool = False
        self.cycle_breaking_points: Set[Tuple[int, int]] = set()
        self.depth_from_input: Optional[np.ndarray] = None
        self.depth_to_output: Optional[np.ndarray] = None
    
    def build_adjacency(
        self,
        arcs_df: pd.DataFrame,
        source_col: str = "_source_id",
        sink_col: str = "_sink_id",
        delay_col: str = "Delay",
        type_col: str = "ArcType"
    ) -> Tuple[sparse.csr_matrix, sparse.csr_matrix]:
        """Build forward and backward CSR adjacency matrices."""
        if arcs_df is None or arcs_df.empty:
            # Empty graph
            self.forward_adj = sparse.csr_matrix((self.num_nodes, self.num_nodes))
            self.backward_adj = sparse.csr_matrix((self.num_nodes, self.num_nodes))
            return self.forward_adj, self.backward_adj
        
        # Filter valid arcs (non-null source/sink)
        valid_mask = arcs_df[source_col].notna() & arcs_df[sink_col].notna()
        valid_arcs = arcs_df[valid_mask]
        
        if len(valid_arcs) == 0:
            self.forward_adj = sparse.csr_matrix((self.num_nodes, self.num_nodes))
            self.backward_adj = sparse.csr_matrix((self.num_nodes, self.num_nodes))
            return self.forward_adj, self.backward_adj
        
        # Extract source and sink arrays
        sources = valid_arcs[source_col].astype(int).values
        sinks = valid_arcs[sink_col].astype(int).values
        
        # Filter out-of-range indices
        valid_idx = (sources >= 0) & (sources < self.num_nodes) & \
                    (sinks >= 0) & (sinks < self.num_nodes)
        sources = sources[valid_idx]
        sinks = sinks[valid_idx]
        
        num_edges = len(sources)
        
        # Edge weights (use 1.0 if delay not available)
        if delay_col in valid_arcs.columns:
            delays = valid_arcs[delay_col].astype(float).fillna(0.0).values[valid_idx].astype(np.float32)
        else:
            delays = np.ones(num_edges, dtype=np.float32)
        
        self.edge_delays = delays
        
        # Edge types
        if type_col in valid_arcs.columns:
            self.edge_types = valid_arcs[type_col].values[valid_idx]
        
        # Build forward adjacency (source -> sink)
        # Using COO format first, then convert to CSR
        data = np.ones(num_edges, dtype=np.float32)
        self.forward_adj = sparse.csr_matrix(
            (data, (sources, sinks)),
            shape=(self.num_nodes, self.num_nodes)
        )
        
        # Build backward adjacency (transpose)
        self.backward_adj = self.forward_adj.T.tocsr()
        
        return self.forward_adj, self.backward_adj
    
    def detect_cycles(self) -> Tuple[bool, Set[Tuple[int, int]]]:
        """Detect cycles and record edges that participate in them."""
        if self.forward_adj is None:
            return False, set()
        
        in_degree = np.array(self.forward_adj.sum(axis=0)).flatten().astype(int)
        queue = deque()
        for node in range(self.num_nodes):
            if in_degree[node] == 0:
                queue.append(node)
        
        processed = 0
        visited = set()
        
        while queue:
            node = queue.popleft()
            visited.add(node)
            processed += 1
            
            row = self.forward_adj.getrow(node)
            successors = row.indices
            
            for succ in successors:
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    queue.append(succ)
        
        self.has_cycles = processed < self.num_nodes
        if self.has_cycles:
            unvisited = set(range(self.num_nodes)) - visited
            
            for node in unvisited:
                row = self.forward_adj.getrow(node)
                for succ in row.indices:
                    self.cycle_breaking_points.add((node, succ))
        
        return self.has_cycles, self.cycle_breaking_points
    
    def compute_logic_depth(
        self,
        nodes_df: pd.DataFrame,
        port_col: str = "IsPort",
        direction_col: str = "Direction"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute BFS depth from inputs and to outputs."""
        self.depth_from_input = np.full(self.num_nodes, -1, dtype=np.int32)
        self.depth_to_output = np.full(self.num_nodes, -1, dtype=np.int32)
        
        if self.forward_adj is None or nodes_df is None:
            return self.depth_from_input, self.depth_to_output
        
        input_ports = set()
        output_ports = set()
        
        if port_col in nodes_df.columns and direction_col in nodes_df.columns:
            for idx, row in nodes_df.iterrows():
                is_port = row.get(port_col, False)
                direction = str(row.get(direction_col, "")).lower()
                node_id = row.get("_node_id", idx)
                
                if is_port:
                    if "input" in direction or "in" == direction:
                        input_ports.add(node_id)
                    elif "output" in direction or "out" == direction:
                        output_ports.add(node_id)
        
        if not input_ports:
            in_degree = np.array(self.backward_adj.sum(axis=1)).flatten()
            input_ports = set(np.where(in_degree == 0)[0])
        
        if not output_ports:
            out_degree = np.array(self.forward_adj.sum(axis=1)).flatten()
            output_ports = set(np.where(out_degree == 0)[0])
        
        self._bfs_depth(
            start_nodes=input_ports,
            adj_matrix=self.forward_adj,
            depth_array=self.depth_from_input
        )
        
        self._bfs_depth(
            start_nodes=output_ports,
            adj_matrix=self.backward_adj,
            depth_array=self.depth_to_output
        )
        
        return self.depth_from_input, self.depth_to_output
    
    def _bfs_depth(
        self,
        start_nodes: Set[int],
        adj_matrix: sparse.csr_matrix,
        depth_array: np.ndarray
    ):
        """Run BFS from a seed set and write depths into `depth_array`."""
        queue = deque()
        for node in start_nodes:
            if 0 <= node < self.num_nodes:
                depth_array[node] = 0
                queue.append((node, 0))
        
        visited = set(start_nodes)
        
        while queue:
            node, depth = queue.popleft()
            
            row = adj_matrix.getrow(node)
            successors = row.indices
            
            for succ in successors:
                if succ not in visited:
                    visited.add(succ)
                    depth_array[succ] = depth + 1
                    queue.append((succ, depth + 1))
    
    def get_fanout(self, node_id: int) -> np.ndarray:
        """Get all direct fanout nodes (successors)."""
        if self.forward_adj is None:
            return np.array([], dtype=int)
        
        if node_id < 0 or node_id >= self.num_nodes:
            return np.array([], dtype=int)
        
        row = self.forward_adj.getrow(node_id)
        return row.indices
    
    def get_fanin(self, node_id: int) -> np.ndarray:
        """Get all direct fanin nodes (predecessors)."""
        if self.backward_adj is None:
            return np.array([], dtype=int)
        
        if node_id < 0 or node_id >= self.num_nodes:
            return np.array([], dtype=int)
        
        row = self.backward_adj.getrow(node_id)
        return row.indices
    
    def get_fanout_depth(self, node_id: int, max_depth: int = 5) -> Dict[int, int]:
        """
        Get fanout cone up to max_depth hops.
        
        Returns dict of {node_id: depth}.
        """
        if self.forward_adj is None:
            return {}
        
        result = {node_id: 0}
        queue = deque([(node_id, 0)])
        
        while queue:
            current, depth = queue.popleft()
            
            if depth >= max_depth:
                continue
            
            successors = self.get_fanout(current)
            for succ in successors:
                if (current, succ) in self.cycle_breaking_points:
                    continue
                
                if succ not in result:
                    result[succ] = depth + 1
                    queue.append((succ, depth + 1))
        
        return result
    
    def get_fanin_depth(self, node_id: int, max_depth: int = 5) -> Dict[int, int]:
        """
        Get fanin cone up to max_depth hops.
        
        Returns dict of {node_id: depth}.
        """
        if self.backward_adj is None:
            return {}
        
        result = {node_id: 0}
        queue = deque([(node_id, 0)])
        
        while queue:
            current, depth = queue.popleft()
            
            if depth >= max_depth:
                continue
            
            predecessors = self.get_fanin(current)
            for pred in predecessors:
                if (pred, current) in self.cycle_breaking_points:
                    continue
                
                if pred not in result:
                    result[pred] = depth + 1
                    queue.append((pred, depth + 1))
        
        return result
    
    def get_stats(self) -> Dict:
        """Get topology statistics."""
        stats = {
            "num_nodes": self.num_nodes,
            "num_edges": self.forward_adj.nnz if self.forward_adj is not None else 0,
            "has_cycles": self.has_cycles,
            "num_cycle_edges": len(self.cycle_breaking_points),
            "max_depth_from_input": int(self.depth_from_input.max()) if self.depth_from_input is not None else -1,
            "max_depth_to_output": int(self.depth_to_output.max()) if self.depth_to_output is not None else -1,
        }
        return stats

def build_topology(
    nodes_df: pd.DataFrame,
    arcs_df: pd.DataFrame
) -> TopologyBuilder:
    """Build adjacency, cycle information, and depth annotations."""
    num_nodes = len(nodes_df) if nodes_df is not None else 0
    
    builder = TopologyBuilder(num_nodes)
    
    if arcs_df is not None and not arcs_df.empty:
        builder.build_adjacency(arcs_df)
    builder.detect_cycles()
    if nodes_df is not None:
        builder.compute_logic_depth(nodes_df)
    
    return builder
