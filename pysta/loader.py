"""Load FEASTA CSV exports into indexed pandas DataFrames."""

import os
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

from .utils import (
    SCHEMA,
    SchemaError,
    LoadError,
    validate_schema,
    cast_types,
    build_name_index,
    fast_merge_by_index,
    get_memory_usage,
    optimize_dtypes
)
from .topology import TopologyBuilder, build_topology
from .query import QueryEngine, FilterableDataFrame
from .export import TensorBridge

EXPECTED_FILES = {
    "network_nodes": "network_nodes.csv",
    "network_arcs": "network_arcs.csv",
    "pin_properties": "pin_properties.csv",
    "cell_properties": "cell_properties.csv"
}

class Design:
    """Loaded design data plus optional topology and query helpers."""
    
    def __init__(
        self,
        path: str,
        name: str = None,
        validate: bool = True,
        optimize_memory: bool = True,
        verbose: bool = True,
        lazy_topology: bool = True
    ):
        self._start_time = time.time()
        self._path = Path(path)
        self._name = name or self._path.name
        self._validate = validate
        self._optimize_memory = optimize_memory
        self._verbose = verbose
        self._lazy_topology = lazy_topology
        self._topology_built = False
        
        self._nodes_df: Optional[pd.DataFrame] = None
        self._arcs_df: Optional[pd.DataFrame] = None
        self._cells_df: Optional[pd.DataFrame] = None
        self._pin_properties_df: Optional[pd.DataFrame] = None
        
        self._name_to_id: Dict[str, int] = {}
        self._id_to_name: Dict[int, str] = {}
        
        self._topology: Optional[TopologyBuilder] = None
        
        self._query_engine: Optional[QueryEngine] = None
        self._tensor_bridge: Optional[TensorBridge] = None
        
        self._metadata: Dict[str, Any] = {}
        self._warnings: List[str] = []
        self._load()
    
    def _load(self):
        """Load all available CSV inputs and optional topology."""
        self._log("Loading design from", str(self._path))
        self._discover_files()
        self._load_nodes()
        self._load_arcs()
        self._load_cells()
        self._load_pin_properties()
        if not self._lazy_topology:
            self._build_topology()
        self._build_metadata()
        self._log(f"Load complete in {self._metadata['load_time_sec']:.2f}s")
    
    def _discover_files(self):
        """Check that required CSV files exist."""
        if not self._path.exists():
            raise LoadError(f"Path does not exist: {self._path}")
        
        if not self._path.is_dir():
            raise LoadError(f"Path is not a directory: {self._path}")
        
        self._files = {}
        missing = []
        
        for schema_name, filename in EXPECTED_FILES.items():
            filepath = self._path / filename
            if filepath.exists():
                self._files[schema_name] = filepath
            else:
                missing.append(filename)
        
        if "network_nodes" not in self._files:
            raise LoadError(
                f"Required file 'network_nodes.csv' not found in {self._path}"
            )
        
        if missing:
            self._warnings.append(f"Missing optional files: {missing}")
    
    def _load_nodes(self):
        """Load network_nodes.csv and build primary index."""
        self._log("Loading nodes...")
        
        filepath = self._files["network_nodes"]
        
        try:
            df = pd.read_csv(filepath, engine="pyarrow", dtype_backend="pyarrow")
        except Exception as e:
            self._log(f"  PyArrow failed ({e}), using default engine...")
            df = pd.read_csv(filepath, on_bad_lines="skip")
        
        if self._validate:
            warnings = validate_schema(df, "network_nodes", str(filepath))
            self._warnings.extend(warnings)
        
        df = cast_types(df, "network_nodes")
        if "Name" not in df.columns:
            raise SchemaError("Column 'Name' not found in network_nodes.csv")
        
        self._name_to_id, self._id_to_name = build_name_index(df["Name"])
        
        df["_node_id"] = df["Name"].map(self._name_to_id)
        if self._optimize_memory:
            df = optimize_dtypes(df)
        
        self._nodes_df = df
        self._log(f"  Loaded {len(df)} nodes")
    
    def _load_arcs(self):
        """Load network_arcs.csv."""
        if "network_arcs" not in self._files:
            self._arcs_df = pd.DataFrame()
            self._warnings.append("No arcs file found - graph traversal disabled")
            return
        
        self._log("Loading arcs...")
        
        filepath = self._files["network_arcs"]
        
        try:
            df = pd.read_csv(filepath, engine="pyarrow", dtype_backend="pyarrow")
        except Exception as e:
            self._log(f"  PyArrow failed ({e}), using default engine...")
            df = pd.read_csv(filepath, on_bad_lines="skip")
        
        if self._validate:
            warnings = validate_schema(df, "network_arcs", str(filepath))
            self._warnings.extend(warnings)
        
        df = cast_types(df, "network_arcs")
        if "Source" in df.columns:
            df["_source_id"] = df["Source"].map(self._name_to_id)
        if "Sink" in df.columns:
            df["_sink_id"] = df["Sink"].map(self._name_to_id)
        
        if self._optimize_memory:
            df = optimize_dtypes(df)
        
        self._arcs_df = df
        self._log(f"  Loaded {len(df)} arcs")
    
    def _load_cells(self):
        """Load cell_properties.csv."""
        if "cell_properties" not in self._files:
            self._cells_df = pd.DataFrame()
            return
        
        self._log("Loading cells...")
        
        filepath = self._files["cell_properties"]
        
        try:
            df = pd.read_csv(filepath, engine="pyarrow", dtype_backend="pyarrow")
        except Exception as e:
            self._log(f"  PyArrow failed ({e}), using default engine...")
            df = pd.read_csv(filepath, on_bad_lines="skip")
        
        if self._validate:
            warnings = validate_schema(df, "cell_properties", str(filepath))
            self._warnings.extend(warnings)
        
        df = cast_types(df, "cell_properties")
        if self._optimize_memory:
            df = optimize_dtypes(df)
        
        self._cells_df = df
        self._log(f"  Loaded {len(df)} cells")
    
    def _load_pin_properties(self):
        """Load pin_properties.csv and merge it into the node table."""
        if "pin_properties" not in self._files:
            return
        
        self._log("Loading pin properties...")
        
        filepath = self._files["pin_properties"]
        
        try:
            df = pd.read_csv(filepath, engine="pyarrow", dtype_backend="pyarrow")
        except Exception as e:
            self._log(f"  PyArrow failed ({e}), using default engine...")
            df = pd.read_csv(filepath, on_bad_lines="skip")
        
        if self._validate:
            warnings = validate_schema(df, "pin_properties", str(filepath))
            self._warnings.extend(warnings)
        
        df = cast_types(df, "pin_properties")
        
        self._pin_properties_df = df
        self._log(f"  Loaded {len(df)} pin properties")
        
        self._log("  Merging with nodes...")
        if "FullName" in df.columns and "Name" not in df.columns:
            df = df.rename(columns={"FullName": "Name"})
        self._nodes_df = fast_merge_by_index(
            self._nodes_df,
            df,
            key_column="Name"
        )
    
    def _build_topology(self):
        """Build graph topology and derived depth information."""
        if self._topology_built:
            return
            
        self._log("Building topology...")
        
        self._topology = build_topology(self._nodes_df, self._arcs_df)
        if self._topology.depth_from_input is not None:
            self._nodes_df["LogicDepthFromInput"] = self._topology.depth_from_input
        if self._topology.depth_to_output is not None:
            self._nodes_df["LogicDepthToOutput"] = self._topology.depth_to_output
        
        topo_stats = self._topology.get_stats()
        self._log(f"  Edges: {topo_stats['num_edges']}")
        
        if topo_stats["has_cycles"]:
            self._log(f"  Cycles detected: {topo_stats['num_cycle_edges']} back-edges")
            self._warnings.append(f"Design has {topo_stats['num_cycle_edges']} feedback loops")
        
        if topo_stats["max_depth_from_input"] >= 0:
            self._log(f"  Logic depth range: 0 to {topo_stats['max_depth_from_input']} stages")
        
        self._topology_built = True
    
    def _ensure_topology(self):
        """Build topology the first time it is requested."""
        if not self._topology_built:
            self._build_topology()
    
    def _build_metadata(self):
        """Build metadata dictionary."""
        load_time = time.time() - self._start_time
        
        topo_stats = self._topology.get_stats() if self._topology else {}
        
        self._metadata = {
            "design_name": self._name,
            "path": str(self._path),
            "node_count": len(self._nodes_df) if self._nodes_df is not None else 0,
            "arc_count": len(self._arcs_df) if self._arcs_df is not None else 0,
            "cell_count": len(self._cells_df) if self._cells_df is not None else 0,
            "load_time_sec": load_time,
            "memory_mb": self._get_total_memory(),
            "has_coordinates": self._has_column("CoordX_um"),
            "has_timing": self._has_column("SlackWorst_ns"),
            "has_power": self._cells_df is not None and "LeakagePower_pW" in (self._cells_df.columns if self._cells_df is not None else []),
            "has_cycles": topo_stats.get("has_cycles", False),
            "num_cycle_edges": topo_stats.get("num_cycle_edges", 0),
            "max_logic_depth": topo_stats.get("max_depth_from_input", -1),
            "warnings": self._warnings
        }
    
    def _get_total_memory(self) -> float:
        """Get total memory usage in MB."""
        total = 0.0
        if self._nodes_df is not None:
            total += get_memory_usage(self._nodes_df)
        if self._arcs_df is not None:
            total += get_memory_usage(self._arcs_df)
        if self._cells_df is not None:
            total += get_memory_usage(self._cells_df)
        return total
    
    def _has_column(self, col: str) -> bool:
        """Check if a column exists in nodes."""
        if self._nodes_df is None:
            return False
        return col in self._nodes_df.columns
    
    def _log(self, *args):
        """Print a log line when verbose output is enabled."""
        if self._verbose:
            print(" ".join(str(a) for a in args))
    
    @property
    def name(self) -> str:
        """Design name."""
        return self._name
    
    @property
    def nodes(self) -> pd.DataFrame:
        """Node table."""
        return self._nodes_df

    @property
    def pins(self) -> "FilterableDataFrame":
        """Node table with query helpers attached."""
        return FilterableDataFrame(
            self._nodes_df,
            self._topology,
            self._name_to_id,
            self._id_to_name
        )
    
    @property
    def arcs(self) -> pd.DataFrame:
        """Arc table."""
        return self._arcs_df
    
    @property
    def cells(self) -> pd.DataFrame:
        """Cell table."""
        return self._cells_df
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Design metadata."""
        return self._metadata
    
    @property
    def topology(self) -> TopologyBuilder:
        """Topology view."""
        return self._topology
    
    def filter(self, **conditions) -> pd.DataFrame:
        """Filter nodes."""
        qe = QueryEngine(
            self._nodes_df, self._topology,
            self._name_to_id, self._id_to_name
        )
        return qe.filter(**conditions)
    
    def get_fanin(self, node_name: str, depth: int = 1) -> pd.DataFrame:
        """Return a node's fanin cone."""
        qe = QueryEngine(
            self._nodes_df, self._topology,
            self._name_to_id, self._id_to_name
        )
        return qe.get_fanin(node_name, depth)
    
    def get_fanout(self, node_name: str, depth: int = 1) -> pd.DataFrame:
        """Return a node's fanout cone."""
        qe = QueryEngine(
            self._nodes_df, self._topology,
            self._name_to_id, self._id_to_name
        )
        return qe.get_fanout(node_name, depth)
    
    def get_critical_paths(self, top_k: int = 10) -> List[Dict]:
        """Return critical paths."""
        qe = QueryEngine(
            self._nodes_df, self._topology,
            self._name_to_id, self._id_to_name
        )
        return qe.get_critical_paths(top_k)
    
    def get_node(self, name: str) -> pd.Series:
        """Return one node by name."""
        if name not in self._name_to_id:
            raise KeyError(f"Node '{name}' not found in design")
        
        node_id = self._name_to_id[name]
        return self._nodes_df[self._nodes_df["_node_id"] == node_id].iloc[0]
    
    def get_node_id(self, name: str) -> int:
        """Return the integer ID for a node name."""
        if name not in self._name_to_id:
            raise KeyError(f"Node '{name}' not found in design")
        return self._name_to_id[name]
    
    def get_node_name(self, node_id: int) -> str:
        """Return the node name for an integer ID."""
        if node_id not in self._id_to_name:
            raise KeyError(f"Node ID {node_id} not found in design")
        return self._id_to_name[node_id]
    
    def to_pytorch_geometric(
        self,
        node_features: List[str],
        edge_weight: str = "Delay",
        edge_features: Optional[List[str]] = None,
        target: Optional[str] = None,
        normalize: bool = True
    ):
        """Export to a PyTorch Geometric data object."""
        bridge = TensorBridge(
            self._nodes_df, self._arcs_df, self._name_to_id
        )
        return bridge.to_pytorch_geometric(
            node_features, edge_weight, edge_features, target, normalize
        )
    
    def to_numpy(
        self,
        features: List[str],
        target: Optional[str] = None,
        normalize: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Export node features and optional target to NumPy arrays."""
        bridge = TensorBridge(
            self._nodes_df, self._arcs_df, self._name_to_id
        )
        return bridge.to_numpy(features, target, normalize)
    
    def summary(self) -> str:
        """Return a text summary of the loaded design."""
        lines = [
            "╔" + "═" * 50 + "╗",
            f"║ Design: {self._name:<41} ║",
            "╠" + "═" * 50 + "╣",
            f"║ Nodes:      {self._metadata['node_count']:>10,}                    ║",
            f"║ Arcs:       {self._metadata['arc_count']:>10,}                    ║",
            f"║ Cells:      {self._metadata['cell_count']:>10,}                    ║",
            "╠" + "═" * 50 + "╣",
            f"║ Memory:     {self._metadata['memory_mb']:>10.1f} MB                ║",
            f"║ Load Time:  {self._metadata['load_time_sec']:>10.2f} sec               ║",
            "╠" + "═" * 50 + "╣",
            f"║ Has Timing:      {'Yes' if self._metadata['has_timing'] else 'No':<27} ║",
            f"║ Has Coordinates: {'Yes' if self._metadata['has_coordinates'] else 'No':<27} ║",
            f"║ Has Power:       {'Yes' if self._metadata['has_power'] else 'No':<27} ║",
            "╠" + "═" * 50 + "╣",
            f"║ Has Cycles:      {'Yes' if self._metadata['has_cycles'] else 'No':<27} ║",
            f"║ Max Logic Depth: {self._metadata['max_logic_depth']:<27} ║",
            "╚" + "═" * 50 + "╝",
        ]
        
        if self._warnings:
            lines.append("\nWarnings:")
            for w in self._warnings[:5]:
                lines.append(f"  ⚠ {w}")
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        return f"Design('{self._name}', nodes={self._metadata['node_count']}, arcs={self._metadata['arc_count']})"
    
    def __str__(self) -> str:
        return self.summary()
