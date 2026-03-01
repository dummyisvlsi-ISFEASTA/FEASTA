"""
PySTA Tensor Bridge

Export circuit data to ML-ready formats with feature-specific normalization.
Supports PyTorch Geometric, NumPy, and pandas exports.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

import pandas as pd
import numpy as np

# Optional imports for ML frameworks
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


# =============================================================================
# NORMALIZATION STRATEGIES
# =============================================================================

class Normalizer:
    """
    Feature-specific normalization with inverse transform support.
    
    Strategies:
    - "none": No scaling (for slack - preserve sign and zero boundary)
    - "log": Log scaling log(1 + x) (for power-law distributions)
    - "minmax": Min-Max to [0, 1] (for coordinates)
    - "zscore": Z-score normalization (mean=0, std=1)
    - "divide": Divide by constant
    """
    
    # Default strategies per feature type
    DEFAULT_STRATEGIES = {
        # Timing - preserve sign
        "SlackRise_ns": "none",
        "SlackFall_ns": "none",
        "SlackWorst_ns": "none",
        "SlackMinRise_ns": "none",
        "SlackMinFall_ns": "none",
        "SlackMinWorst_ns": "none",
        
        # Slew - divide by typical clock period (1ns)
        "SlewRise_ns": {"type": "divide", "divisor": 1.0},
        "SlewFall_ns": {"type": "divide", "divisor": 1.0},
        
        # Power-law distributions - log scale
        "Capacitance_pf": "log",
        "DriveResistance_ohm": "log",
        "Area_um2": "log",
        "LeakagePower_pW": "log",
        "TotalPower_pW": "log",
        "FanoutLoad": "log",
        "FaninLoad": "log",
        "PinCount": "log",
        
        # Coordinates - minmax
        "CoordX_um": "minmax",
        "CoordY_um": "minmax",
        
        # Already normalized
        "Activity": "none",
        "StaticProbability": "none",
        
        # Logic depth - relative
        "LogicDepthFromInput": "minmax",
        "LogicDepthToOutput": "minmax",
    }
    
    def __init__(self, custom_strategies: Optional[Dict] = None):
        """
        Initialize normalizer.
        
        Parameters
        ----------
        custom_strategies : dict, optional
            Override default strategies for specific columns.
        """
        self.strategies = self.DEFAULT_STRATEGIES.copy()
        if custom_strategies:
            self.strategies.update(custom_strategies)
        
        # Store parameters for inverse transform
        self._params: Dict[str, Dict] = {}
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        columns: List[str]
    ) -> pd.DataFrame:
        """
        Fit normalizer and transform data.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input data.
        columns : list
            Columns to normalize.
            
        Returns
        -------
        pd.DataFrame
            Normalized data.
        """
        result = df.copy()
        
        for col in columns:
            if col not in df.columns:
                continue
            
            strategy = self.strategies.get(col, "minmax")
            values = df[col].values.astype(np.float32)
            
            # Handle NaN
            nan_mask = np.isnan(values)
            
            if isinstance(strategy, dict):
                strategy_type = strategy.get("type", "none")
            else:
                strategy_type = strategy
            
            if strategy_type == "none":
                # No transformation
                self._params[col] = {"type": "none"}
                normalized = values
                
            elif strategy_type == "log":
                # Log transform: log(1 + x)
                # Handle negative values by shifting
                min_val = np.nanmin(values)
                shift = max(0, -min_val + 1e-6)
                normalized = np.log1p(values + shift)
                self._params[col] = {"type": "log", "shift": shift}
                
            elif strategy_type == "minmax":
                # Min-Max scaling
                min_val = np.nanmin(values)
                max_val = np.nanmax(values)
                range_val = max_val - min_val
                if range_val == 0:
                    range_val = 1.0
                normalized = (values - min_val) / range_val
                self._params[col] = {"type": "minmax", "min": min_val, "range": range_val}
                
            elif strategy_type == "zscore":
                # Z-score normalization
                mean_val = np.nanmean(values)
                std_val = np.nanstd(values)
                if std_val == 0:
                    std_val = 1.0
                normalized = (values - mean_val) / std_val
                self._params[col] = {"type": "zscore", "mean": mean_val, "std": std_val}
                
            elif strategy_type == "divide":
                # Divide by constant
                if isinstance(strategy, dict):
                    divisor = strategy.get("divisor", 1.0)
                else:
                    divisor = 1.0
                normalized = values / divisor
                self._params[col] = {"type": "divide", "divisor": divisor}
                
            else:
                # Default: no transformation
                normalized = values
                self._params[col] = {"type": "none"}
            
            # Restore NaN
            normalized[nan_mask] = np.nan
            result[col] = normalized
        
        return result
    
    def inverse_transform(
        self,
        values: np.ndarray,
        column: str
    ) -> np.ndarray:
        """
        Inverse transform normalized values back to original scale.
        
        Parameters
        ----------
        values : np.ndarray
            Normalized values.
        column : str
            Column name (to lookup transformation params).
            
        Returns
        -------
        np.ndarray
            Original-scale values.
        """
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


# =============================================================================
# TENSOR BRIDGE CLASS
# =============================================================================

class TensorBridge:
    """
    Export circuit data to ML-ready tensor formats.
    
    Features:
    - Feature-specific normalization (preserve slack sign, log-scale powers)
    - Edge weights for GNN convolution
    - PyTorch Geometric Data export
    - NumPy array export
    
    Example:
        >>> bridge = TensorBridge(design)
        >>> data = bridge.to_pytorch_geometric(
        ...     node_features=["SlewRise_ns", "Capacitance_pf", "LogicDepthFromInput"],
        ...     edge_weight="Delay",
        ...     target="SlackWorst_ns"
        ... )
    """
    
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
    
    # =========================================================================
    # PYTORCH GEOMETRIC EXPORT
    # =========================================================================
    
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
            # Empty graph
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_weight = torch.empty(0, dtype=torch.float)
            return edge_index, edge_weight, None
        
        arcs = self._arcs_df
        
        # Get source and sink IDs
        if "_source_id" in arcs.columns and "_sink_id" in arcs.columns:
            sources = arcs["_source_id"].dropna().astype(int).values
            sinks = arcs["_sink_id"].dropna().astype(int).values
        else:
            # Need to map names to IDs
            if "Source" in arcs.columns and "Sink" in arcs.columns:
                sources = arcs["Source"].map(self._name_to_id).dropna().astype(int).values
                sinks = arcs["Sink"].map(self._name_to_id).dropna().astype(int).values
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
                edge_weight = torch.empty(0, dtype=torch.float)
                return edge_index, edge_weight, None
        
        # Filter to valid pairs
        min_len = min(len(sources), len(sinks))
        sources = sources[:min_len]
        sinks = sinks[:min_len]
        
        # Build edge_index
        edge_index = torch.tensor(np.vstack([sources, sinks]), dtype=torch.long)
        
        # Build edge_weight (CRITICAL: used for GNN convolution)
        if edge_weight_col and edge_weight_col in arcs.columns:
            weights = arcs[edge_weight_col].fillna(fill_value).values[:min_len].astype(np.float32)
        else:
            weights = np.ones(min_len, dtype=np.float32)
        
        edge_weight = torch.tensor(weights, dtype=torch.float)
        
        # Build edge_attr (optional additional features)
        edge_attr = None
        if edge_features:
            available = [f for f in edge_features if f in arcs.columns]
            if available:
                attr_data = arcs[available].fillna(fill_value).values[:min_len].astype(np.float32)
                edge_attr = torch.tensor(attr_data, dtype=torch.float)
        
        return edge_index, edge_weight, edge_attr
    
    # =========================================================================
    # NUMPY EXPORT
    # =========================================================================
    
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
    
    # =========================================================================
    # DATAFRAME EXPORT
    # =========================================================================
    
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
    
    # =========================================================================
    # INVERSE TRANSFORM
    # =========================================================================
    
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
    
    # =========================================================================
    # UTILITIES
    # =========================================================================
    
    def get_available_features(self) -> List[str]:
        """Get list of available feature columns."""
        if self._nodes_df is None:
            return []
        return list(self._nodes_df.columns)
    
    def get_normalization_params(self) -> Dict[str, Dict]:
        """Get normalization parameters (for saving/loading)."""
        return self._normalizer.get_params()
