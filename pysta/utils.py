"""
PySTA Utility Functions

Shared utilities for schema validation, type casting, and error handling.
"""

from typing import Dict, List, Optional, Set
import pandas as pd
import numpy as np


# =============================================================================
# SCHEMA DEFINITIONS
# =============================================================================

# Expected columns for each CSV file
# Expected columns for each CSV file
SCHEMA = {
    "network_nodes": {
        "required": ["Name", "InstanceName", "PinName", "Direction"],
        "optional": ["IsPort", "Type", "IsClockNetwork"],
        "types": {
            "Name": str,
            "InstanceName": str,
            "PinName": str,
            "Direction": "category",
            "IsPort": bool,
            "Type": "category",
            "IsClockNetwork": bool
        }
    },
    "network_arcs": {
        "required": ["Source", "Sink"],
        "optional": [
            "NetName", "Connection", "ArcType",
            "Delay_Min_RR", "Delay_Min_RF", "Delay_Min_FR", "Delay_Min_FF",
            "Delay_Max_RR", "Delay_Max_RF", "Delay_Max_FR", "Delay_Max_FF"
        ],
        "types": {
            "Source": str,
            "Sink": str,
            "NetName": str,
            "Connection": "category",
            "ArcType": "category",
            "Delay_Min_RR": np.float32, "Delay_Min_RF": np.float32, "Delay_Min_FR": np.float32, "Delay_Min_FF": np.float32,
            "Delay_Max_RR": np.float32, "Delay_Max_RF": np.float32, "Delay_Max_FR": np.float32, "Delay_Max_FF": np.float32
        }
    },
    "pin_properties": {
        "required": ["FullName"],
        "optional": [
            "Direction", "IsPort", "IsHierarchical", "IsRegisterClock",
            "LibPinName",
            "SlewRise_ns", "SlewFall_ns", "SlewMinRise_ns", "SlewMinFall_ns",
            "SlackRise_ns", "SlackFall_ns", "SlackWorst_ns",
            "SlackMinRise_ns", "SlackMinFall_ns", "SlackMinWorst_ns",
            "IsClock", "ClockNames",
            "Capacitance_pf", "DriveResistance_ohm",
            "Activity", "StaticProbability", "ToggleRate_MHz", "ActivityOrigin",
            "CoordX_um", "CoordY_um"
        ],
        "types": {
            "FullName": str,
            "Direction": "category",
            "IsPort": bool,
            "IsHierarchical": bool,
            "IsRegisterClock": bool,
            "SlewRise_ns": np.float32,
            "SlewFall_ns": np.float32,
            "SlackRise_ns": np.float32,
            "SlackFall_ns": np.float32,
            "SlackWorst_ns": np.float32,
            "IsClock": bool,
            "Capacitance_pf": np.float32,
            "DriveResistance_ohm": np.float32,
            "Activity": np.float32,
            "StaticProbability": np.float32,
            "ToggleRate_MHz": np.float32,
            "CoordX_um": np.float32,
            "CoordY_um": np.float32
        }
    },
    "cell_properties": {
        "required": ["FullInstanceName", "LibertyCell"],
        "optional": [
            "Library",
            "CellType", "IsBuffer", "IsInverter", "IsMemory", "IsMacro", "IsHierarchical",
            "Area_um2",
            "LeakagePower_pW", "SwitchingPower_pW", "InternalPower_pW", "TotalPower_pW",
            "PinCount", "InputPinCount", "OutputPinCount", "BiDirectPinCount",
            "ClockPinCount", "DataPinCount", "AsyncPinCount",
            "FanoutLoad", "FaninLoad",
            "IsCombinational", "IsSequential", "IsClockGating",
            "SetupTime_ns", "HoldTime_ns",
            "TimingArcCount", "HasClockInput", "ClockDomains",
            "Process", "Voltage_V", "Temperature_C"
        ],
        "types": {
            "FullInstanceName": str,
            "LibertyCell": str,
            "Library": str,
            "CellType": "category",
            "IsBuffer": bool,
            "IsInverter": bool,
            "IsMemory": bool,
            "IsMacro": bool,
            "IsHierarchical": bool,
            "Area_um2": np.float32,
            "LeakagePower_pW": np.float32,
            "SwitchingPower_pW": np.float32,
            "InternalPower_pW": np.float32,
            "TotalPower_pW": np.float32,
            "PinCount": np.int32,
            "InputPinCount": np.int32,
            "OutputPinCount": np.int32,
            "FanoutLoad": np.int32,
            "FaninLoad": np.int32
        }
    }
}


# =============================================================================
# CUSTOM EXCEPTIONS
# =============================================================================

class PySTA_Error(Exception):
    """Base exception for PySTA."""
    pass


class SchemaError(PySTA_Error):
    """Raised when CSV schema doesn't match expected format."""
    pass


class LoadError(PySTA_Error):
    """Raised when CSV loading fails."""
    pass


# =============================================================================
# SCHEMA VALIDATION
# =============================================================================

def validate_schema(df: pd.DataFrame, schema_name: str, filepath: str) -> List[str]:
    """
    Validate DataFrame against expected schema.
    
    Returns list of warnings (empty if all good).
    Raises SchemaError if required columns missing.
    """
    if schema_name not in SCHEMA:
        raise ValueError(f"Unknown schema: {schema_name}")
    
    schema = SCHEMA[schema_name]
    warnings = []
    
    # Check required columns
    missing_required = set(schema["required"]) - set(df.columns)
    if missing_required:
        raise SchemaError(
            f"Missing required columns in {filepath}: {missing_required}"
        )
    
    # Check optional columns (warn if missing)
    missing_optional = set(schema["optional"]) - set(df.columns)
    if missing_optional:
        warnings.append(
            f"Missing optional columns in {filepath}: {list(missing_optional)[:5]}..."
        )
    
    return warnings


# =============================================================================
# TYPE CASTING
# =============================================================================

def cast_types(df: pd.DataFrame, schema_name: str) -> pd.DataFrame:
    """
    Cast DataFrame columns to appropriate types.
    Uses float32 for numerics (memory optimization).
    Uses categorical for low-cardinality strings.
    """
    if schema_name not in SCHEMA:
        return df
    
    type_map = SCHEMA[schema_name].get("types", {})
    
    for col, dtype in type_map.items():
        if col not in df.columns:
            continue
        
        try:
            if dtype == "category":
                df[col] = df[col].astype("category")
            elif dtype == bool:
                # Handle various bool representations
                df[col] = df[col].map(
                    lambda x: str(x).lower() in ("true", "1", "yes") if pd.notna(x) else False
                )
            elif dtype in (np.float32, np.float64):
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float32)
            elif dtype in (np.int32, np.int64):
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(np.int32)
            else:
                df[col] = df[col].astype(dtype, errors="ignore")
        except Exception:
            pass  # Keep original type if casting fails
    
    return df


# =============================================================================
# INDEX BUILDING
# =============================================================================

def build_name_index(names: pd.Series) -> tuple:
    """
    Build bidirectional name index for O(1) lookups.
    
    Returns:
        name_to_id: Dict[str, int]
        id_to_name: Dict[int, str]
    """
    unique_names = names.unique()
    name_to_id = {name: idx for idx, name in enumerate(unique_names)}
    id_to_name = {idx: name for idx, name in enumerate(unique_names)}
    
    return name_to_id, id_to_name


# =============================================================================
# ZERO-COPY DATA ASSIGNMENT
# =============================================================================

def merge_by_index(
    master_df: pd.DataFrame,
    property_df: pd.DataFrame,
    name_to_id: Dict[str, int],
    key_column: str = "Name"
) -> pd.DataFrame:
    """
    Zero-copy merge of property data into master DataFrame.
    
    Instead of pd.merge() which creates copies, we:
    1. Map property rows to master index
    2. Assign directly via index
    
    This avoids the 3x memory spike of pd.merge().
    """
    if property_df is None or property_df.empty:
        return master_df
    
    # Get columns to add (exclude key column if it exists in master)
    new_cols = [c for c in property_df.columns if c not in master_df.columns and c != key_column]
    
    if not new_cols:
        return master_df
    
    # Map property rows to master indices
    if key_column in property_df.columns:
        property_indices = property_df[key_column].map(name_to_id)
    else:
        # Assume same order
        property_indices = pd.Series(range(len(property_df)))
    
    # Filter valid indices
    valid_mask = property_indices.notna()
    valid_indices = property_indices[valid_mask].astype(int).values
    
    # Add new columns to master
    for col in new_cols:
        # Initialize with NaN
        master_df[col] = np.nan
        
        # Assign values at valid indices
        if len(valid_indices) > 0:
            values = property_df.loc[valid_mask, col].values
            # Use iloc for index-based assignment
            for i, idx in enumerate(valid_indices):
                if idx < len(master_df):
                    master_df.iloc[idx, master_df.columns.get_loc(col)] = values[i]
    
    return master_df


def fast_merge_by_index(
    master_df: pd.DataFrame,
    property_df: pd.DataFrame,
    key_column: str = "Name"
) -> pd.DataFrame:
    """
    Fast merge using set_index and join (more efficient than loop).
    Still avoids the full pd.merge() overhead.
    """
    if property_df is None or property_df.empty:
        return master_df
    
    # Get columns to add
    new_cols = [c for c in property_df.columns if c not in master_df.columns and c != key_column]
    
    if not new_cols:
        return master_df
    
    # Set index on both DataFrames
    if key_column in master_df.columns and key_column in property_df.columns:
        # Save original index
        original_index = master_df.index.copy()
        
        # Set key as index
        master_indexed = master_df.set_index(key_column, drop=False)
        property_indexed = property_df.set_index(key_column)[new_cols]
        
        # Join (this is more efficient than merge for this case)
        result = master_indexed.join(property_indexed, how="left")
        
        # Restore original index
        result.index = original_index
        
        return result
    
    return master_df


# =============================================================================
# MEMORY UTILITIES
# =============================================================================

def get_memory_usage(df: pd.DataFrame) -> float:
    """Get DataFrame memory usage in MB."""
    return df.memory_usage(deep=True).sum() / (1024 * 1024)


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize DataFrame dtypes to reduce memory usage.
    - float64 -> float32
    - int64 -> int32
    - object with few unique values -> category
    """
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type == np.float64:
            df[col] = df[col].astype(np.float32)
        elif col_type == np.int64:
            df[col] = df[col].astype(np.int32)
        elif col_type == object:
            # Convert to category if few unique values
            n_unique = df[col].nunique()
            if n_unique < 100 and n_unique / len(df) < 0.5:
                df[col] = df[col].astype("category")
    
    return df
