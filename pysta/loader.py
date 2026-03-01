"""
PySTA HyperLoader

The core loader that ingests OpenSTA CSV exports into a validated,
indexed, zero-copy in-memory representation with topology analysis.
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import numpy as np

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


# =============================================================================
# EXPECTED FILES
# =============================================================================

EXPECTED_FILES = {
    "network_nodes": "network_nodes.csv",
    "network_arcs": "network_arcs.csv",
    "pin_properties": "pin_properties.csv",
    "cell_properties": "cell_properties.csv"
}


# =============================================================================
# DESIGN CLASS
# =============================================================================

class Design:
    """
    The core design object that holds all timing analysis data.
    
    This class performs:
    1. Automatic CSV detection and loading
    2. Hash-based indexing for O(1) node lookups
    3. Schema validation and type optimization
    4. Zero-copy data merging
    5. Topology construction with cycle detection
    6. Logic depth computation via BFS
    
    Attributes:
        name (str): Design identifier
        nodes (DataFrame): All pins/nodes with properties
        arcs (DataFrame): All connectivity edges
        cells (DataFrame): Cell/instance-level properties
        pins (FilterableDataFrame): Nodes with filter/traversal methods
        metadata (dict): Design statistics and schema info
    
    Example:
        >>> design = Design("./dumps/zipcpu/")
        >>> print(design.summary())
        >>> violations = design.pins.filter(SlackWorst_ns__lt=0)
        >>> data = design.to_pytorch_geometric(node_features=["SlewRise_ns", "Capacitance_pf"])
    """
    
    def __init__(
        self,
        path: str,
        name: str = None,
        validate: bool = True,
        optimize_memory: bool = True,
        verbose: bool = True,
        lazy_topology: bool = True
    ):
        """
        Initialize a Design object from OpenSTA CSV exports.
        
        Parameters
        ----------
        path : str
            Path to directory containing CSV files from OpenSTA dump.
            
        name : str, optional
            Human-readable design name. If None, inferred from directory.
            
        validate : bool, default=True
            Perform schema validation on load.
            
        optimize_memory : bool, default=True
            Optimize dtypes to reduce memory usage.
            
        verbose : bool, default=True
            Print loading progress.
            
        lazy_topology : bool, default=True
            If True, defer topology building until first graph traversal.
            This reduces initial load time significantly for large designs.
        """
        self._start_time = time.time()
        self._path = Path(path)
        self._name = name or self._path.name
        self._validate = validate
        self._optimize_memory = optimize_memory
        self._verbose = verbose
        self._lazy_topology = lazy_topology
        self._topology_built = False
        
        # Internal storage
        self._nodes_df: Optional[pd.DataFrame] = None
        self._arcs_df: Optional[pd.DataFrame] = None
        self._cells_df: Optional[pd.DataFrame] = None
        self._pin_properties_df: Optional[pd.DataFrame] = None
        
        # Index structures
        self._name_to_id: Dict[str, int] = {}
        self._id_to_name: Dict[int, str] = {}
        
        # Topology
        self._topology: Optional[TopologyBuilder] = None
        
        # Query engine (created lazily)
        self._query_engine: Optional[QueryEngine] = None
        self._tensor_bridge: Optional[TensorBridge] = None
        
        # Metadata
        self._metadata: Dict[str, Any] = {}
        self._warnings: List[str] = []
        
        # Load the design
        self._load()
    
    # =========================================================================
    # LOADING
    # =========================================================================
    
    def _load(self):
        """Main loading sequence."""
        self._log("Loading design from", str(self._path))
        
        # Step 1: File discovery
        self._discover_files()
        
        # Step 2: Load nodes (creates master index)
        self._load_nodes()
        
        # Step 3: Load arcs
        self._load_arcs()
        
        # Step 4: Load cells
        self._load_cells()
        
        # Step 5: Load and merge pin properties
        self._load_pin_properties()
        
        # Step 6: Build topology (CSR + cycle detection + logic depth)
        # Deferred if lazy_topology=True (default) for faster loading
        if not self._lazy_topology:
            self._build_topology()
        
        # Step 7: Build metadata
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
        
        # network_nodes is required, others are optional
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
        
        # Try PyArrow C++ engine first (10x faster), fall back if CSV has issues
        try:
            df = pd.read_csv(filepath, engine="pyarrow", dtype_backend="pyarrow")
        except Exception as e:
            self._log(f"  PyArrow failed ({e}), using default engine...")
            df = pd.read_csv(filepath, on_bad_lines="skip")
        
        # Validate schema
        if self._validate:
            warnings = validate_schema(df, "network_nodes", str(filepath))
            self._warnings.extend(warnings)
        
        # Cast types
        df = cast_types(df, "network_nodes")
        
        # Build primary index from Name column
        if "Name" not in df.columns:
            raise SchemaError("Column 'Name' not found in network_nodes.csv")
        
        self._name_to_id, self._id_to_name = build_name_index(df["Name"])
        
        # Add integer ID column
        df["_node_id"] = df["Name"].map(self._name_to_id)
        
        # Optimize memory
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
        
        # Try PyArrow C++ engine first (10x faster), fall back if CSV has issues
        try:
            df = pd.read_csv(filepath, engine="pyarrow", dtype_backend="pyarrow")
        except Exception as e:
            self._log(f"  PyArrow failed ({e}), using default engine...")
            df = pd.read_csv(filepath, on_bad_lines="skip")
        
        # Validate schema
        if self._validate:
            warnings = validate_schema(df, "network_arcs", str(filepath))
            self._warnings.extend(warnings)
        
        # Cast types
        df = cast_types(df, "network_arcs")
        
        # Map Source/Sink to integer IDs
        if "Source" in df.columns:
            df["_source_id"] = df["Source"].map(self._name_to_id)
        if "Sink" in df.columns:
            df["_sink_id"] = df["Sink"].map(self._name_to_id)
        
        # Optimize memory
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
        
        # Try PyArrow C++ engine first (10x faster), fall back if CSV has issues
        try:
            df = pd.read_csv(filepath, engine="pyarrow", dtype_backend="pyarrow")
        except Exception as e:
            self._log(f"  PyArrow failed ({e}), using default engine...")
            df = pd.read_csv(filepath, on_bad_lines="skip")
        
        # Validate schema
        if self._validate:
            warnings = validate_schema(df, "cell_properties", str(filepath))
            self._warnings.extend(warnings)
        
        # Cast types
        df = cast_types(df, "cell_properties")
        
        # Optimize memory
        if self._optimize_memory:
            df = optimize_dtypes(df)
        
        self._cells_df = df
        self._log(f"  Loaded {len(df)} cells")
    
    def _load_pin_properties(self):
        """Load pin_properties.csv and merge with nodes (zero-copy)."""
        if "pin_properties" not in self._files:
            return
        
        self._log("Loading pin properties...")
        
        filepath = self._files["pin_properties"]
        
        # Try PyArrow C++ engine first (10x faster), fall back if CSV has issues
        try:
            df = pd.read_csv(filepath, engine="pyarrow", dtype_backend="pyarrow")
        except Exception as e:
            self._log(f"  PyArrow failed ({e}), using default engine...")
            df = pd.read_csv(filepath, on_bad_lines="skip")
        
        # Validate schema
        if self._validate:
            warnings = validate_schema(df, "pin_properties", str(filepath))
            self._warnings.extend(warnings)
        
        # Cast types
        df = cast_types(df, "pin_properties")
        
        self._pin_properties_df = df
        self._log(f"  Loaded {len(df)} pin properties")
        
        # Zero-copy merge with nodes
        self._log("  Merging with nodes (zero-copy)...")
        if "FullName" in df.columns and "Name" not in df.columns:
            df = df.rename(columns={"FullName": "Name"})
        self._nodes_df = fast_merge_by_index(
            self._nodes_df,
            df,
            key_column="Name"
        )
    
    def _build_topology(self):
        """Build graph topology, detect cycles, compute logic depth."""
        if self._topology_built:
            return  # Already built
            
        self._log("Building topology...")
        
        # Build topology using helper function
        self._topology = build_topology(self._nodes_df, self._arcs_df)
        
        # Add logic depth columns to nodes
        if self._topology.depth_from_input is not None:
            self._nodes_df["LogicDepthFromInput"] = self._topology.depth_from_input
        if self._topology.depth_to_output is not None:
            self._nodes_df["LogicDepthToOutput"] = self._topology.depth_to_output
        
        # Get stats
        topo_stats = self._topology.get_stats()
        self._log(f"  Edges: {topo_stats['num_edges']}")
        
        if topo_stats["has_cycles"]:
            self._log(f"  ⚠ Cycles detected: {topo_stats['num_cycle_edges']} back-edges")
            self._warnings.append(f"Design has {topo_stats['num_cycle_edges']} feedback loops")
        
        if topo_stats["max_depth_from_input"] >= 0:
            self._log(f"  Logic depth range: 0 to {topo_stats['max_depth_from_input']} stages")
        
        self._topology_built = True
    
    def _ensure_topology(self):
        """Lazily build topology on first access."""
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
            "has_power": self._has_column("Activity"),
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
        """Print if verbose mode is on."""
        if self._verbose:
            print(*args)
    
    # =========================================================================
    # PROPERTIES
    # =========================================================================
    
    @property
    def name(self) -> str:
        """Design name."""
        return self._name
    
    @property
    def nodes(self) -> pd.DataFrame:
        """
        All network nodes (pins) with their properties.
        
        Includes structural, timing, power, and computed features (LogicDepth).
        """
        return self._nodes_df
    
    @property
    def pins(self) -> "FilterableDataFrame":
        """
        Nodes with filter, traversal, and critical path methods.
        
        Example:
            >>> design.pins.filter(SlackWorst_ns__lt=0)
            >>> design.pins.get_fanout("reg/Q", depth=3)
            >>> design.pins.get_critical_paths(top_k=10)
        """
        return FilterableDataFrame(
            self._nodes_df,
            self._topology,
            self._name_to_id,
            self._id_to_name
        )
    
    @property
    def arcs(self) -> pd.DataFrame:
        """All connectivity arcs between nodes."""
        return self._arcs_df
    
    @property
    def cells(self) -> pd.DataFrame:
        """All cell/instance properties."""
        return self._cells_df
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Design statistics and schema information."""
        return self._metadata
    
    @property
    def topology(self) -> TopologyBuilder:
        """Topology builder with adjacency matrices and graph algorithms."""
        return self._topology
    
    # =========================================================================
    # QUERY METHODS
    # =========================================================================
    
    def filter(self, **conditions) -> pd.DataFrame:
        """
        Filter nodes using Django-style conditions.
        
        Example:
            >>> design.filter(SlackWorst_ns__lt=0, IsClock=True)
        """
        qe = QueryEngine(
            self._nodes_df, self._topology,
            self._name_to_id, self._id_to_name
        )
        return qe.filter(**conditions)
    
    def get_fanin(self, node_name: str, depth: int = 1) -> pd.DataFrame:
        """Get fanin cone of a node."""
        qe = QueryEngine(
            self._nodes_df, self._topology,
            self._name_to_id, self._id_to_name
        )
        return qe.get_fanin(node_name, depth)
    
    def get_fanout(self, node_name: str, depth: int = 1) -> pd.DataFrame:
        """Get fanout cone of a node."""
        qe = QueryEngine(
            self._nodes_df, self._topology,
            self._name_to_id, self._id_to_name
        )
        return qe.get_fanout(node_name, depth)
    
    def get_critical_paths(self, top_k: int = 10) -> List[Dict]:
        """Get top-K critical timing paths."""
        qe = QueryEngine(
            self._nodes_df, self._topology,
            self._name_to_id, self._id_to_name
        )
        return qe.get_critical_paths(top_k)
    
    # =========================================================================
    # INDEX ACCESS
    # =========================================================================
    
    def get_node(self, name: str) -> pd.Series:
        """O(1) lookup of a single node by name."""
        if name not in self._name_to_id:
            raise KeyError(f"Node '{name}' not found in design")
        
        node_id = self._name_to_id[name]
        return self._nodes_df[self._nodes_df["_node_id"] == node_id].iloc[0]
    
    def get_node_id(self, name: str) -> int:
        """Get integer ID for a node name."""
        if name not in self._name_to_id:
            raise KeyError(f"Node '{name}' not found in design")
        return self._name_to_id[name]
    
    def get_node_name(self, node_id: int) -> str:
        """Get name for an integer node ID."""
        if node_id not in self._id_to_name:
            raise KeyError(f"Node ID {node_id} not found in design")
        return self._id_to_name[node_id]
    
    # =========================================================================
    # EXPORT METHODS
    # =========================================================================
    
    def to_pytorch_geometric(
        self,
        node_features: List[str],
        edge_weight: str = "Delay",
        edge_features: Optional[List[str]] = None,
        target: Optional[str] = None,
        normalize: bool = True
    ):
        """
        Export to PyTorch Geometric Data object.
        
        Parameters
        ----------
        node_features : list
            Node feature columns (e.g., ["SlewRise_ns", "Capacitance_pf"])
        edge_weight : str, default="Delay"
            Edge weight column (for GNN convolution).
        edge_features : list, optional
            Additional edge features.
        target : str, optional
            Target column for prediction (y).
        normalize : bool, default=True
            Apply feature-specific normalization.
            
        Returns
        -------
        torch_geometric.data.Data
            Graph data with x, edge_index, edge_weight, y.
        """
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
        """
        Export to NumPy arrays (for scikit-learn, XGBoost).
        
        Returns (X, y) tuple.
        """
        bridge = TensorBridge(
            self._nodes_df, self._arcs_df, self._name_to_id
        )
        return bridge.to_numpy(features, target, normalize)
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    
    def summary(self) -> str:
        """Generate human-readable design summary."""
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
