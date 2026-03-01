"""
PySTA Query Engine

High-performance query engine for filtering, traversal, and aggregation.
Uses Django-style operators for intuitive filtering.
"""

import re
import operator
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import pandas as pd
import numpy as np

from .topology import TopologyBuilder


# =============================================================================
# OPERATOR MAPPINGS
# =============================================================================

OPERATORS = {
    "lt": operator.lt,
    "lte": operator.le,
    "gt": operator.gt,
    "gte": operator.ge,
    "eq": operator.eq,
    "ne": operator.ne,
}

STRING_OPERATORS = {
    "contains": lambda s, v: s.str.contains(v, na=False, regex=False),
    "startswith": lambda s, v: s.str.startswith(v, na=False),
    "endswith": lambda s, v: s.str.endswith(v, na=False),
    "regex": lambda s, v: s.str.contains(v, na=False, regex=True),
}


# =============================================================================
# QUERY ENGINE CLASS
# =============================================================================

class QueryEngine:
    """
    High-performance query engine for circuit data.
    
    Supports:
    - Django-style filtering (e.g., SlackWorst_ns__lt=0)
    - Graph traversal (fanin, fanout)
    - Critical path extraction
    - Aggregation
    
    Example:
        >>> qe = QueryEngine(design.nodes, design.topology)
        >>> violations = qe.filter(SlackWorst_ns__lt=0)
        >>> fanout = qe.get_fanout("reg/Q", depth=3)
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        topology: Optional[TopologyBuilder] = None,
        name_to_id: Optional[Dict[str, int]] = None,
        id_to_name: Optional[Dict[int, str]] = None
    ):
        """
        Initialize query engine.
        
        Parameters
        ----------
        df : pd.DataFrame
            Data to query (nodes, cells, or arcs).
        topology : TopologyBuilder, optional
            Topology for graph traversal.
        name_to_id : dict, optional
            Name to integer ID mapping.
        id_to_name : dict, optional
            Integer ID to name mapping.
        """
        self._df = df
        self._topology = topology
        self._name_to_id = name_to_id or {}
        self._id_to_name = id_to_name or {}
    
    # =========================================================================
    # FILTERING
    # =========================================================================
    
    def filter(self, **conditions) -> pd.DataFrame:
        """
        Filter data using Django-style conditions.
        
        Parameters
        ----------
        **conditions
            Keyword arguments in format: column__operator=value
            
            Operators:
                __lt    : Less than
                __lte   : Less than or equal
                __gt    : Greater than
                __gte   : Greater than or equal
                __eq    : Equal (default if no operator)
                __ne    : Not equal
                __in    : In list
                __nin   : Not in list
                __contains : String contains
                __startswith : String starts with
                __endswith : String ends with
                __regex : Regular expression match
                __isnull : Is null/NaN
                
        Returns
        -------
        pd.DataFrame
            Filtered results.
            
        Examples
        --------
        >>> qe.filter(SlackWorst_ns__lt=0)                # Violations
        >>> qe.filter(IsClock=True, Capacitance_pf__gt=0.5)  # High-cap clocks
        >>> qe.filter(Direction__in=["input", "output"])  # Ports only
        """
        if self._df is None or self._df.empty:
            return pd.DataFrame()
        
        mask = pd.Series([True] * len(self._df), index=self._df.index)
        
        for key, value in conditions.items():
            col, op = self._parse_condition(key)
            
            if col not in self._df.columns:
                raise KeyError(f"Column '{col}' not found in data")
            
            col_mask = self._apply_operator(self._df[col], op, value)
            mask = mask & col_mask
        
        return self._df[mask].copy()
    
    def _parse_condition(self, key: str) -> Tuple[str, str]:
        """Parse condition key into column and operator."""
        # Check for double-underscore operator
        for op in list(OPERATORS.keys()) + list(STRING_OPERATORS.keys()) + ["in", "nin", "isnull"]:
            suffix = f"__{op}"
            if key.endswith(suffix):
                col = key[:-len(suffix)]
                return col, op
        
        # No operator = equality
        return key, "eq"
    
    def _apply_operator(self, series: pd.Series, op: str, value: Any) -> pd.Series:
        """Apply operator to series."""
        if op in OPERATORS:
            return OPERATORS[op](series, value)
        
        if op in STRING_OPERATORS:
            return STRING_OPERATORS[op](series, value)
        
        if op == "in":
            return series.isin(value)
        
        if op == "nin":
            return ~series.isin(value)
        
        if op == "isnull":
            if value:
                return series.isna()
            else:
                return series.notna()
        
        # Default: equality
        return series == value
    
    # =========================================================================
    # POINT QUERY
    # =========================================================================
    
    def get(self, name: str) -> Optional[pd.Series]:
        """
        Get single row by name (O(1) lookup).
        
        Parameters
        ----------
        name : str
            Node/cell name.
            
        Returns
        -------
        pd.Series or None
            Row data if found, None otherwise.
        """
        if name not in self._name_to_id:
            return None
        
        node_id = self._name_to_id[name]
        
        if "_node_id" in self._df.columns:
            matches = self._df[self._df["_node_id"] == node_id]
        elif "Name" in self._df.columns:
            matches = self._df[self._df["Name"] == name]
        else:
            return None
        
        if len(matches) == 0:
            return None
        
        return matches.iloc[0]
    
    # =========================================================================
    # GRAPH TRAVERSAL
    # =========================================================================
    
    def get_fanout(
        self,
        node_name: str,
        depth: int = 1,
        include_properties: bool = True
    ) -> pd.DataFrame:
        """
        Get fanout cone from a node.
        
        Parameters
        ----------
        node_name : str
            Starting node name.
        depth : int, default=1
            Number of hops to traverse.
        include_properties : bool, default=True
            Include full node properties in result.
            
        Returns
        -------
        pd.DataFrame
            Fanout nodes with depth column.
        """
        if self._topology is None:
            return pd.DataFrame()
        
        if node_name not in self._name_to_id:
            raise KeyError(f"Node '{node_name}' not found")
        
        node_id = self._name_to_id[node_name]
        
        # Get fanout with depth
        fanout_dict = self._topology.get_fanout_depth(node_id, max_depth=depth)
        
        if not fanout_dict:
            return pd.DataFrame()
        
        # Build result DataFrame
        result_data = []
        for nid, d in fanout_dict.items():
            name = self._id_to_name.get(nid, f"node_{nid}")
            result_data.append({"Name": name, "_node_id": nid, "Depth": d})
        
        result = pd.DataFrame(result_data)
        
        # Merge with properties if requested
        if include_properties and self._df is not None:
            result = result.merge(
                self._df.drop(columns=["Depth"], errors="ignore"),
                on="Name",
                how="left",
                suffixes=("", "_prop")
            )
        
        return result.sort_values("Depth")
    
    def get_fanin(
        self,
        node_name: str,
        depth: int = 1,
        include_properties: bool = True
    ) -> pd.DataFrame:
        """
        Get fanin cone to a node.
        
        Parameters
        ----------
        node_name : str
            Target node name.
        depth : int, default=1
            Number of hops to traverse backward.
        include_properties : bool, default=True
            Include full node properties in result.
            
        Returns
        -------
        pd.DataFrame
            Fanin nodes with depth column.
        """
        if self._topology is None:
            return pd.DataFrame()
        
        if node_name not in self._name_to_id:
            raise KeyError(f"Node '{node_name}' not found")
        
        node_id = self._name_to_id[node_name]
        
        # Get fanin with depth
        fanin_dict = self._topology.get_fanin_depth(node_id, max_depth=depth)
        
        if not fanin_dict:
            return pd.DataFrame()
        
        # Build result DataFrame
        result_data = []
        for nid, d in fanin_dict.items():
            name = self._id_to_name.get(nid, f"node_{nid}")
            result_data.append({"Name": name, "_node_id": nid, "Depth": d})
        
        result = pd.DataFrame(result_data)
        
        # Merge with properties if requested
        if include_properties and self._df is not None:
            result = result.merge(
                self._df.drop(columns=["Depth"], errors="ignore"),
                on="Name",
                how="left",
                suffixes=("", "_prop")
            )
        
        return result.sort_values("Depth")
    
    # =========================================================================
    # CRITICAL PATHS
    # =========================================================================
    
    def get_critical_paths(
        self,
        top_k: int = 10,
        slack_column: str = "SlackWorst_ns",
        max_stages: int = 100
    ) -> List[Dict]:
        """
        Find top-K critical timing paths.
        
        Parameters
        ----------
        top_k : int, default=10
            Number of paths to return.
        slack_column : str, default="SlackWorst_ns"
            Column to use for slack.
        max_stages : int, default=100
            Maximum path length.
            
        Returns
        -------
        List[Dict]
            Each dict contains:
            - endpoint: str
            - startpoint: str
            - slack: float
            - stages: int
            - path: List[str]
        """
        if self._df is None or self._topology is None:
            return []
        
        if slack_column not in self._df.columns:
            return []
        
        # Get nodes sorted by worst slack
        slack_df = self._df[self._df[slack_column].notna()].copy()
        slack_df = slack_df.sort_values(slack_column)
        
        paths = []
        seen_endpoints = set()
        
        for _, row in slack_df.iterrows():
            if len(paths) >= top_k:
                break
            
            endpoint_name = row.get("Name", "")
            endpoint_id = row.get("_node_id", -1)
            slack = row.get(slack_column, 0)
            
            if endpoint_name in seen_endpoints:
                continue
            
            seen_endpoints.add(endpoint_name)
            
            # Trace back to find startpoint
            path_ids = self._trace_back(endpoint_id, max_stages)
            
            if path_ids:
                path_names = [self._id_to_name.get(nid, f"node_{nid}") for nid in path_ids]
                startpoint = path_names[0]
                
                paths.append({
                    "endpoint": endpoint_name,
                    "startpoint": startpoint,
                    "slack": float(slack),
                    "stages": len(path_ids) - 1,
                    "path": path_names
                })
        
        return paths
    
    def _trace_back(self, endpoint_id: int, max_stages: int) -> List[int]:
        """Trace back from endpoint to find critical path."""
        if self._topology is None:
            return [endpoint_id]
        
        path = [endpoint_id]
        current = endpoint_id
        visited = {endpoint_id}
        
        for _ in range(max_stages):
            predecessors = self._topology.get_fanin(current)
            
            if len(predecessors) == 0:
                break
            
            # Filter out cycle breaking points
            valid_preds = [
                p for p in predecessors 
                if (p, current) not in self._topology.cycle_breaking_points
                and p not in visited
            ]
            
            if not valid_preds:
                break
            
            # Take first predecessor (could be improved with delay-based selection)
            next_node = valid_preds[0]
            visited.add(next_node)
            path.insert(0, next_node)
            current = next_node
        
        return path
    
    # =========================================================================
    # AGGREGATION
    # =========================================================================
    
    def agg(self, aggregations: Dict[str, str]) -> pd.Series:
        """
        Aggregate data.
        
        Parameters
        ----------
        aggregations : dict
            Column -> aggregation function mapping.
            Functions: "sum", "mean", "min", "max", "count", "std"
            
        Returns
        -------
        pd.Series
            Aggregation results.
            
        Example
        -------
        >>> qe.filter(IsClock=True).agg({"Capacitance_pf": "sum"})
        """
        if self._df is None or self._df.empty:
            return pd.Series()
        
        results = {}
        for col, func in aggregations.items():
            if col in self._df.columns:
                results[f"{col}_{func}"] = self._df[col].agg(func)
        
        return pd.Series(results)
    
    # =========================================================================
    # SUBGRAPH EXTRACTION
    # =========================================================================
    
    def extract_subgraph(
        self,
        center_node: str,
        hops: int = 2
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Extract subgraph around a center node.
        
        Parameters
        ----------
        center_node : str
            Center node name.
        hops : int, default=2
            Radius in hops.
            
        Returns
        -------
        tuple
            (nodes_df, arcs_df) for the subgraph.
        """
        if self._topology is None:
            return pd.DataFrame(), pd.DataFrame()
        
        if center_node not in self._name_to_id:
            raise KeyError(f"Node '{center_node}' not found")
        
        center_id = self._name_to_id[center_node]
        
        # Get all nodes within hops
        fanin = self._topology.get_fanin_depth(center_id, max_depth=hops)
        fanout = self._topology.get_fanout_depth(center_id, max_depth=hops)
        
        all_nodes = set(fanin.keys()) | set(fanout.keys())
        
        # Filter nodes DataFrame
        if "_node_id" in self._df.columns:
            subgraph_nodes = self._df[self._df["_node_id"].isin(all_nodes)].copy()
        else:
            node_names = [self._id_to_name.get(nid, "") for nid in all_nodes]
            subgraph_nodes = self._df[self._df["Name"].isin(node_names)].copy()
        
        # Arcs would need to be filtered separately (not implemented here)
        subgraph_arcs = pd.DataFrame()
        
        return subgraph_nodes, subgraph_arcs


# =============================================================================
# FILTERABLE DATAFRAME WRAPPER
# =============================================================================

class FilterableDataFrame:
    """
    Wrapper that adds filter method to DataFrame.
    
    Allows: design.pins.filter(SlackWorst_ns__lt=0)
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        topology: Optional[TopologyBuilder] = None,
        name_to_id: Optional[Dict[str, int]] = None,
        id_to_name: Optional[Dict[int, str]] = None
    ):
        self._df = df
        self._qe = QueryEngine(df, topology, name_to_id, id_to_name)
    
    def filter(self, **conditions) -> pd.DataFrame:
        """Filter using Django-style conditions."""
        return self._qe.filter(**conditions)
    
    def get(self, name: str) -> Optional[pd.Series]:
        """Get single row by name."""
        return self._qe.get(name)
    
    def get_fanout(self, node_name: str, depth: int = 1) -> pd.DataFrame:
        """Get fanout cone."""
        return self._qe.get_fanout(node_name, depth)
    
    def get_fanin(self, node_name: str, depth: int = 1) -> pd.DataFrame:
        """Get fanin cone."""
        return self._qe.get_fanin(node_name, depth)
    
    def get_critical_paths(self, top_k: int = 10) -> List[Dict]:
        """Get critical paths."""
        return self._qe.get_critical_paths(top_k)
    
    def agg(self, aggregations: Dict[str, str]) -> pd.Series:
        """Aggregate data."""
        return self._qe.agg(aggregations)
    
    # Delegate DataFrame methods
    def __getattr__(self, name):
        return getattr(self._df, name)
    
    def __len__(self):
        return len(self._df)
    
    def __repr__(self):
        return f"FilterableDataFrame({len(self._df)} rows)"
