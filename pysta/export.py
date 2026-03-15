"""Export FEASTA data into ML-oriented tensor and array formats."""

from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

import pandas as pd
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from torch_geometric.data import Data
    PYGEOM_AVAILABLE = True
except ImportError:
    PYGEOM_AVAILABLE = False

class Normalizer:
    """Column-wise normalization with inverse-transform support."""

    DEFAULT_STRATEGIES = {
        "SlackRise_ns": "none",
        "SlackFall_ns": "none",
        "SlackWorst_ns": "none",
        "SlackMinRise_ns": "none",
        "SlackMinFall_ns": "none",
        "SlackMinWorst_ns": "none",
        
        "SlewRise_ns": {"type": "divide", "divisor": 1.0},
        "SlewFall_ns": {"type": "divide", "divisor": 1.0},

        "Capacitance_pf": "log",
        "DriveResistance_ohm": "log",
        "Area_um2": "log",
        "LeakagePower_pW": "log",
        "TotalPower_pW": "log",
        "FanoutLoad": "log",
        "FaninLoad": "log",
        "PinCount": "log",
        
        "CoordX_um": "minmax",
        "CoordY_um": "minmax",

        "Activity": "none",
        "StaticProbability": "none",

        "LogicDepthFromInput": "minmax",
        "LogicDepthToOutput": "minmax",
    }
    
    def __init__(self, custom_strategies: Optional[Dict] = None):
        self.strategies = self.DEFAULT_STRATEGIES.copy()
        if custom_strategies:
            self.strategies.update(custom_strategies)
        self._params: Dict[str, Dict] = {}
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        columns: List[str]
    ) -> pd.DataFrame:
        """Fit on `columns` and return a normalized copy of `df`."""
        result = df.copy()
        
        for col in columns:
            if col not in df.columns:
                continue
            
            strategy = self.strategies.get(col, "minmax")
            values = df[col].values.astype(np.float32)
            
            nan_mask = np.isnan(values)
            
            if isinstance(strategy, dict):
                strategy_type = strategy.get("type", "none")
            else:
                strategy_type = strategy
            
            if strategy_type == "none":
                self._params[col] = {"type": "none"}
                normalized = values
                
            elif strategy_type == "log":
                min_val = np.nanmin(values)
                shift = max(0, -min_val + 1e-6)
                normalized = np.log1p(values + shift)
                self._params[col] = {"type": "log", "shift": shift}
                
            elif strategy_type == "minmax":
                min_val = np.nanmin(values)
                max_val = np.nanmax(values)
                range_val = max_val - min_val
                if range_val == 0:
                    range_val = 1.0
                normalized = (values - min_val) / range_val
                self._params[col] = {"type": "minmax", "min": min_val, "range": range_val}
                
            elif strategy_type == "zscore":
                mean_val = np.nanmean(values)
                std_val = np.nanstd(values)
                if std_val == 0:
                    std_val = 1.0
                normalized = (values - mean_val) / std_val
                self._params[col] = {"type": "zscore", "mean": mean_val, "std": std_val}
                
            elif strategy_type == "divide":
                if isinstance(strategy, dict):
                    divisor = strategy.get("divisor", 1.0)
                else:
                    divisor = 1.0
                normalized = values / divisor
                self._params[col] = {"type": "divide", "divisor": divisor}
                
            else:
                normalized = values
                self._params[col] = {"type": "none"}

            normalized[nan_mask] = np.nan
            result[col] = normalized
        
        return result
    
    def inverse_transform(
        self,
        values: np.ndarray,
        column: str
    ) -> np.ndarray:
        """Map normalized values for `column` back to the original scale."""
        if column not in self._params:
            return values
        
        params = self._params[column]
        strategy_type = params.get("type", "none")
        
        if strategy_type == "none":
            return values
        
        elif strategy_type == "log":
            shift = params.get("shift", 0)
            return np.expm1(values) - shift
        
        elif strategy_type == "minmax":
            min_val = params.get("min", 0)
            range_val = params.get("range", 1)
            return values * range_val + min_val
        
        elif strategy_type == "zscore":
            mean_val = params.get("mean", 0)
            std_val = params.get("std", 1)
            return values * std_val + mean_val
        
        elif strategy_type == "divide":
            divisor = params.get("divisor", 1)
            return values * divisor
        
        return values
    
    def get_params(self) -> Dict[str, Dict]:
        """Get stored normalization parameters."""
        return self._params.copy()

class TensorBridge:
    """Bridge pandas-backed FEASTA data into tensor-oriented formats."""
    
    def __init__(
        self,
        nodes_df: pd.DataFrame,
        arcs_df: Optional[pd.DataFrame] = None,
        name_to_id: Optional[Dict[str, int]] = None,
        custom_normalization: Optional[Dict] = None
    ):
        """
        Initialize tensor bridge.
        
        Parameters
        ----------
        nodes_df : pd.DataFrame
            Node data.
        arcs_df : pd.DataFrame, optional
            Arc data (for graph export).
        name_to_id : dict, optional
            Name to ID mapping.
        custom_normalization : dict, optional
            Override default normalization strategies.
        """
        self._nodes_df = nodes_df
        self._arcs_df = arcs_df
        self._name_to_id = name_to_id or {}
        
        self._normalizer = Normalizer(custom_normalization)
    
    def to_pytorch_geometric(
        self,
        node_features: List[str],
        edge_weight: str = "Delay",
        edge_features: Optional[List[str]] = None,
        target: Optional[str] = None,
        normalize: bool = True,
        fill_value: float = 0.0
    ):
        """
        Export to PyTorch Geometric Data object.
        
        Parameters
        ----------
        node_features : list
            Column names for node features.
        edge_weight : str, default="Delay"
            Column for edge weights (used in GNN convolution).
        edge_features : list, optional
            Additional edge feature columns.
        target : str, optional
            Column for prediction target (y).
        normalize : bool, default=True
            Apply feature-specific normalization.
        fill_value : float, default=0.0
            Value for filling NaN.
            
        Returns
        -------
        torch_geometric.data.Data
            Graph data object with:
            - x: Node features
            - edge_index: Adjacency
            - edge_weight: Edge weights
            - edge_attr: Additional edge features (if any)
            - y: Target (if specified)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not installed. Run: pip install torch")
        
        if not PYGEOM_AVAILABLE:
            raise ImportError("PyTorch Geometric not installed. Run: pip install torch-geometric")
        
        # Prepare node features
        df = self._nodes_df.copy()
        
        # Filter to available columns
        available_features = [f for f in node_features if f in df.columns]
        if not available_features:
            raise ValueError(f"No valid features found. Available: {list(df.columns)}")
        
        # Normalize if requested
        if normalize:
            df = self._normalizer.fit_transform(df, available_features)
        
        # Build node feature matrix
        x = df[available_features].fillna(fill_value).values.astype(np.float32)
        x = torch.tensor(x, dtype=torch.float)
        
        # Build edge index
        edge_index, edge_weight_tensor, edge_attr_tensor = self._build_edges(
            edge_weight, edge_features, fill_value
        )
        
        # Build target
        y = None
        if target and target in self._nodes_df.columns:
            y_values = self._nodes_df[target].fillna(fill_value).values.astype(np.float32)
            y = torch.tensor(y_values, dtype=torch.float)
        
        # Create Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_weight=edge_weight_tensor,
            y=y
        )
        
        if edge_attr_tensor is not None:
            data.edge_attr = edge_attr_tensor
        
        # Store metadata
        data.feature_names = available_features
        data.normalizer = self._normalizer
        
        return data
    
    def _build_edges(
        self,
        edge_weight_col: str,
        edge_features: Optional[List[str]],
        fill_value: float
    ) -> Tuple[Any, Any, Any]:
        """Build edge tensors from arcs DataFrame."""
        if self._arcs_df is None or self._arcs_df.empty:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            return edge_index, torch.empty(0, dtype=torch.float), None

        arcs = self._arcs_df.copy()

        # Resolve source/sink to integer IDs.
        if "_source_id" in arcs.columns and "_sink_id" in arcs.columns:
            arcs = arcs.rename(columns={"_source_id": "_src", "_sink_id": "_snk"})
        elif "Source" in arcs.columns and "Sink" in arcs.columns:
            arcs["_src"] = arcs["Source"].map(self._name_to_id)
            arcs["_snk"] = arcs["Sink"].map(self._name_to_id)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            return edge_index, torch.empty(0, dtype=torch.float), None

        # Drop rows where either endpoint is unmapped; keeps pairs aligned.
        arcs = arcs.dropna(subset=["_src", "_snk"])
        sources = arcs["_src"].astype(int).values
        sinks = arcs["_snk"].astype(int).values

        edge_index = torch.tensor(np.vstack([sources, sinks]), dtype=torch.long)

        # Edge weights (used by GNN convolution).
        if edge_weight_col and edge_weight_col in arcs.columns:
            weights = arcs[edge_weight_col].fillna(fill_value).values.astype(np.float32)
        else:
            weights = np.ones(len(sources), dtype=np.float32)
        edge_weight = torch.tensor(weights, dtype=torch.float)

        # Optional additional edge features.
        edge_attr = None
        if edge_features:
            available = [f for f in edge_features if f in arcs.columns]
            if available:
                attr_data = arcs[available].fillna(fill_value).values.astype(np.float32)
                edge_attr = torch.tensor(attr_data, dtype=torch.float)

        return edge_index, edge_weight, edge_attr
    
    def to_numpy(
        self,
        features: List[str],
        target: Optional[str] = None,
        normalize: bool = True,
        fill_value: float = 0.0
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Export to NumPy arrays (for scikit-learn, XGBoost).
        
        Parameters
        ----------
        features : list
            Feature column names.
        target : str, optional
            Target column name.
        normalize : bool, default=True
            Apply normalization.
        fill_value : float, default=0.0
            Fill value for NaN.
            
        Returns
        -------
        tuple
            (X, y) where y is None if no target specified.
        """
        df = self._nodes_df.copy()
        
        # Filter to available columns
        available = [f for f in features if f in df.columns]
        if not available:
            raise ValueError(f"No valid features found")
        
        # Normalize if requested
        if normalize:
            df = self._normalizer.fit_transform(df, available)
        
        X = df[available].fillna(fill_value).values.astype(np.float32)
        
        y = None
        if target and target in self._nodes_df.columns:
            y = self._nodes_df[target].fillna(fill_value).values.astype(np.float32)
        
        return X, y
    
    def to_dataframe(
        self,
        features: List[str],
        target: Optional[str] = None,
        normalize: bool = True
    ) -> pd.DataFrame:
        """
        Export normalized DataFrame.
        
        Parameters
        ----------
        features : list
            Feature columns.
        target : str, optional
            Target column (appended if specified).
        normalize : bool, default=True
            Apply normalization.
            
        Returns
        -------
        pd.DataFrame
            Normalized data.
        """
        df = self._nodes_df.copy()
        
        columns = [f for f in features if f in df.columns]
        if target and target in df.columns and target not in columns:
            columns.append(target)
        
        result = df[columns].copy()
        
        # Normalize features (not target)
        if normalize:
            feature_cols = [c for c in columns if c != target]
            result = self._normalizer.fit_transform(result, feature_cols)
        
        return result
    
    def inverse_transform(
        self,
        values: np.ndarray,
        feature_name: str
    ) -> np.ndarray:
        """
        Inverse transform predictions back to original scale.
        
        Parameters
        ----------
        values : np.ndarray
            Normalized values (e.g., model predictions).
        feature_name : str
            Feature name to lookup transform parameters.
            
        Returns
        -------
        np.ndarray
            Values in original scale.
        """
        return self._normalizer.inverse_transform(values, feature_name)
    
    def get_available_features(self) -> List[str]:
        """Get list of available feature columns."""
        if self._nodes_df is None:
            return []
        return list(self._nodes_df.columns)
    
    def get_normalization_params(self) -> Dict[str, Dict]:
        """Get normalization parameters (for saving/loading)."""
        return self._normalizer.get_params()
