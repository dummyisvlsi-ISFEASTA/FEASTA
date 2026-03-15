"""Query helpers over loaded FEASTA data."""

import re
import operator
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import pandas as pd
import numpy as np

from .topology import TopologyBuilder

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

class QueryEngine:
    """Filtering and traversal over a DataFrame plus optional topology."""
    
    def __init__(
        self,
        df: pd.DataFrame,
        topology: Optional[TopologyBuilder] = None,
        name_to_id: Optional[Dict[str, int]] = None,
        id_to_name: Optional[Dict[int, str]] = None
    ):
        self._df = df
        self._topology = topology
        self._name_to_id = name_to_id or {}
        self._id_to_name = id_to_name or {}
    
    def filter(self, **conditions) -> pd.DataFrame:
        """Filter rows with `column__op=value` conditions."""
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
        for op in list(OPERATORS.keys()) + list(STRING_OPERATORS.keys()) + ["in", "nin", "isnull"]:
            suffix = f"__{op}"
            if key.endswith(suffix):
                col = key[:-len(suffix)]
                return col, op
        
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
        
        return series == value
    
    def get(self, name: str) -> Optional[pd.Series]:
        """Return one row by name if present."""
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
    
    def get_fanout(
        self,
        node_name: str,
        depth: int = 1,
        include_properties: bool = True
    ) -> pd.DataFrame:
        """Return the fanout cone from `node_name` up to `depth` hops."""
        if self._topology is None:
            return pd.DataFrame()
        
        if node_name not in self._name_to_id:
            raise KeyError(f"Node '{node_name}' not found")
        
        node_id = self._name_to_id[node_name]
        
        fanout_dict = self._topology.get_fanout_depth(node_id, max_depth=depth)
        
        if not fanout_dict:
            return pd.DataFrame()
        
        result_data = []
        for nid, d in fanout_dict.items():
            name = self._id_to_name.get(nid, f"node_{nid}")
            result_data.append({"Name": name, "_node_id": nid, "Depth": d})
        
        result = pd.DataFrame(result_data)
        
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
        """Return the fanin cone to `node_name` up to `depth` hops."""
        if self._topology is None:
            return pd.DataFrame()
        
        if node_name not in self._name_to_id:
            raise KeyError(f"Node '{node_name}' not found")
        
        node_id = self._name_to_id[node_name]
        
        fanin_dict = self._topology.get_fanin_depth(node_id, max_depth=depth)
        
        if not fanin_dict:
            return pd.DataFrame()
        
        result_data = []
        for nid, d in fanin_dict.items():
            name = self._id_to_name.get(nid, f"node_{nid}")
            result_data.append({"Name": name, "_node_id": nid, "Depth": d})
        
        result = pd.DataFrame(result_data)
        
        if include_properties and self._df is not None:
            result = result.merge(
                self._df.drop(columns=["Depth"], errors="ignore"),
                on="Name",
                how="left",
                suffixes=("", "_prop")
            )
        
        return result.sort_values("Depth")
    
    def get_critical_paths(
        self,
        top_k: int = 10,
        slack_column: str = "SlackWorst_ns",
        max_stages: int = 100
    ) -> List[Dict]:
        """Trace back from the worst-slack endpoints and return up to `top_k` paths."""
        if self._df is None or self._topology is None:
            return []
        
        if slack_column not in self._df.columns:
            return []
        
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

    def get_paths_between(
        self,
        startpoint: str,
        endpoint: str,
        top_k: int = 1,
        max_stages: int = 100
    ) -> List[Dict]:
        """Return up to `top_k` simple paths from `startpoint` to `endpoint`."""
        if self._topology is None:
            return []

        start_id = self._name_to_id.get(startpoint)
        end_id = self._name_to_id.get(endpoint)
        if start_id is None or end_id is None:
            return []

        path_ids = self._enumerate_paths(start_id, end_id, top_k, max_stages)
        results = []
        for ids in path_ids:
            path_names = [self._id_to_name.get(nid, f"node_{nid}") for nid in ids]
            results.append(
                {
                    "startpoint": startpoint,
                    "endpoint": endpoint,
                    "stages": len(ids) - 1,
                    "path": path_names,
                }
            )
        return results
    
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
            
            valid_preds = [
                p for p in predecessors 
                if (p, current) not in self._topology.cycle_breaking_points
                and p not in visited
            ]
            
            if not valid_preds:
                break
            
            next_node = valid_preds[0]
            visited.add(next_node)
            path.insert(0, next_node)
            current = next_node
        
        return path

    def _enumerate_paths(
        self,
        start_id: int,
        end_id: int,
        top_k: int,
        max_stages: int
    ) -> List[List[int]]:
        """Enumerate up to `top_k` simple forward paths from `start_id` to `end_id`."""
        if self._topology is None:
            return []

        if start_id == end_id:
            return [[start_id]]

        paths: List[List[int]] = []
        stack: List[Tuple[int, List[int], Set[int]]] = [(start_id, [start_id], {start_id})]

        while stack and len(paths) < top_k:
            current, path, visited = stack.pop()
            if len(path) - 1 >= max_stages:
                continue

            successors = self._topology.get_fanout(current)
            if len(successors) == 0:
                continue

            next_nodes = []
            for succ in successors:
                if (current, succ) in self._topology.cycle_breaking_points:
                    continue
                if succ in visited:
                    continue
                next_nodes.append(int(succ))

            for succ in reversed(next_nodes):
                next_path = path + [succ]
                if succ == end_id:
                    paths.append(next_path)
                    if len(paths) >= top_k:
                        break
                    continue
                stack.append((succ, next_path, visited | {succ}))

        return paths
    
    def agg(self, aggregations: Dict[str, str]) -> pd.Series:
        """Apply simple pandas aggregations keyed by column name."""
        if self._df is None or self._df.empty:
            return pd.Series()
        
        results = {}
        for col, func in aggregations.items():
            if col in self._df.columns:
                results[f"{col}_{func}"] = self._df[col].agg(func)
        
        return pd.Series(results)
    
    def extract_subgraph(
        self,
        center_node: str,
        hops: int = 2
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Return the nodes and arcs within `hops` of `center_node`."""
        if self._topology is None:
            return pd.DataFrame(), pd.DataFrame()
        
        if center_node not in self._name_to_id:
            raise KeyError(f"Node '{center_node}' not found")
        
        center_id = self._name_to_id[center_node]
        
        fanin = self._topology.get_fanin_depth(center_id, max_depth=hops)
        fanout = self._topology.get_fanout_depth(center_id, max_depth=hops)
        
        all_nodes = set(fanin.keys()) | set(fanout.keys())
        
        if "_node_id" in self._df.columns:
            subgraph_nodes = self._df[self._df["_node_id"].isin(all_nodes)].copy()
        else:
            node_names = {self._id_to_name.get(nid, "") for nid in all_nodes}
            subgraph_nodes = self._df[self._df["Name"].isin(node_names)].copy()

        subgraph_arcs = pd.DataFrame()
        if self._topology is not None:
            src_col = "_source_id" if "_source_id" in self._df.columns else None
            snk_col = "_sink_id" if "_sink_id" in self._df.columns else None
            if src_col and snk_col:
                subgraph_arcs = self._df[
                    self._df[src_col].isin(all_nodes) & self._df[snk_col].isin(all_nodes)
                ].copy()

        return subgraph_nodes, subgraph_arcs

class FilterableDataFrame:
    """Thin wrapper that forwards query helpers to a DataFrame."""
    
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
        """Filter rows."""
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

    def get_paths_between(
        self,
        startpoint: str,
        endpoint: str,
        top_k: int = 1,
        max_stages: int = 100
    ) -> List[Dict]:
        """Get paths between two nodes."""
        return self._qe.get_paths_between(startpoint, endpoint, top_k, max_stages)
    
    def get_critical_paths(self, top_k: int = 10) -> List[Dict]:
        """Get critical paths."""
        return self._qe.get_critical_paths(top_k)
    
    def agg(self, aggregations: Dict[str, str]) -> pd.Series:
        """Aggregate data."""
        return self._qe.agg(aggregations)
    
    def __getattr__(self, name):
        return getattr(self._df, name)
    
    def __len__(self):
        return len(self._df)
    
    def __repr__(self):
        return f"FilterableDataFrame({len(self._df)} rows)"
