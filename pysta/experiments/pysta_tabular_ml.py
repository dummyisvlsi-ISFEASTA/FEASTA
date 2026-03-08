"""
FEASTA Tabular ML Pipeline: Cell Delay Prediction
==================================================

Preprocesses FEASTA CSV exports into tabular features for classical ML
models (XGBoost, LightGBM, Random Forest) to predict cell propagation delay.

Task:  Given physical and electrical cell-level features, predict the
       propagation delay (ClkToQDelay for sequential, PropagationDelay
       for combinational cells).

Usage:
    python pysta_tabular_ml.py --pysta_path /path/to/csv/dir
"""

import sys
import os
import time
import argparse
import numpy as np
import pandas as pd

# Add PySTA to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pysta import Design


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

# Features (X) — physical + electrical cell properties
FEATURE_COLUMNS = [
    # Physical features
    'Area_um2',                 # Cell area
    'PinCount',                 # Total pin count
    'InputPinCount',            # Number of input pins
    'OutputPinCount',           # Number of output pins
    
    # Electrical features
    'InputCapacitance_pf',      # Input pin capacitance
    'OutputCapacitance_pf',     # Output pin capacitance
    'DriveStrength',            # Drive strength
    'FanoutLoad',               # Fanout load
    'FaninLoad',                # Fanin load
    'LeakagePower_pW',          # Leakage power
    
    # Timing features (inputs, not targets)
    'SetupTime_ns',             # Setup time (for sequential)
    'HoldTime_ns',              # Hold time (for sequential)
    'TimingArcCount',           # Number of timing arcs
    
    # Boolean features (encoded as 0/1)
    'IsBuffer',                 # Buffer cell
    'IsInverter',               # Inverter cell
    'IsCombinational',          # Combinational logic
    'IsSequential',             # Sequential (flip-flop/latch)
    'IsClockGating',            # Clock gating cell
]

# Target (Y) — what we predict
TARGET_COLUMN = 'PropagationDelay_ns'

# Alternate target for sequential cells
TARGET_COLUMN_SEQ = 'ClkToQDelay_ns'


def preprocess_tabular(design, verbose=True):
    """
    Preprocess FEASTA cell_properties CSV into ML-ready tabular data.
    
    Pipeline:
      1. Load cell_properties from Design object
      2. Select feature columns and target
      3. Clean: drop rows with missing target, fill NaN features with 0
      4. Encode booleans as 0/1
      5. Log-transform skewed features (area, power, capacitance)
      6. Return (X, y, feature_names, metadata)
    
    Parameters
    ----------
    design : Design
        PySTA Design object with loaded CSVs
    
    Returns
    -------
    dict with keys:
        'X' : np.ndarray of shape (n_samples, n_features)
        'y' : np.ndarray of shape (n_samples,)
        'feature_names' : list of str
        'df' : pd.DataFrame (the cleaned dataframe)
        'stats' : dict of preprocessing statistics
    """
    if verbose:
        print("\n" + "="*60)
        print("FEASTA Tabular ML Preprocessing")
        print("="*60)
    
    t0 = time.time()
    
    # ----------------------------------------------------------
    # Step 1: Load cell properties
    # ----------------------------------------------------------
    cells_df = design.cells.copy()
    
    if verbose:
        print(f"\n[1] Loaded cell_properties: {len(cells_df)} rows, {len(cells_df.columns)} columns")
    
    # ----------------------------------------------------------
    # Step 2: Build unified target column
    #   - Use PropagationDelay for combinational cells
    #   - Use ClkToQDelay for sequential cells
    # ----------------------------------------------------------
    def get_target(row):
        """Pick the appropriate delay target per cell type."""
        prop_delay = row.get(TARGET_COLUMN, np.nan)
        clk_delay = row.get(TARGET_COLUMN_SEQ, np.nan)
        
        # Try propagation delay first, fall back to ClkToQ
        if pd.notna(prop_delay) and prop_delay > 0:
            return prop_delay
        elif pd.notna(clk_delay) and clk_delay > 0:
            return clk_delay
        return np.nan
    
    cells_df['target_delay'] = cells_df.apply(get_target, axis=1)
    
    # Drop rows with no valid target
    n_before = len(cells_df)
    cells_df = cells_df.dropna(subset=['target_delay'])
    cells_df = cells_df[cells_df['target_delay'] > 0]
    n_after = len(cells_df)
    
    if verbose:
        print(f"\n[2] Target: cell delay (PropagationDelay or ClkToQDelay)")
        print(f"    Valid samples: {n_after} / {n_before} ({n_after/n_before*100:.1f}%)")
    
    if n_after == 0:
        raise ValueError("No valid target values found. Check cell_properties CSV.")
    
    # ----------------------------------------------------------
    # Step 3: Select and clean features
    # ----------------------------------------------------------
    available_features = [f for f in FEATURE_COLUMNS if f in cells_df.columns]
    missing_features = [f for f in FEATURE_COLUMNS if f not in cells_df.columns]
    
    if verbose:
        print(f"\n[3] Feature selection:")
        print(f"    Available: {len(available_features)} / {len(FEATURE_COLUMNS)}")
        if missing_features:
            print(f"    Missing:   {missing_features}")
    
    # Boolean columns: convert true/false strings to 0/1
    bool_cols = ['IsBuffer', 'IsInverter', 'IsCombinational', 'IsSequential', 'IsClockGating']
    for col in bool_cols:
        if col in cells_df.columns:
            cells_df[col] = cells_df[col].map(
                {'true': 1, 'false': 0, True: 1, False: 0, 'True': 1, 'False': 0}
            ).fillna(0).astype(int)
    
    # Numeric columns: convert and fill NaN with 0
    numeric_cols = [c for c in available_features if c not in bool_cols]
    for col in numeric_cols:
        cells_df[col] = pd.to_numeric(cells_df[col], errors='coerce').fillna(0.0)
    
    # ----------------------------------------------------------
    # Step 4: Log-transform skewed features
    # ----------------------------------------------------------
    log_transform_cols = [
        'Area_um2', 'LeakagePower_pW',
        'InputCapacitance_pf', 'OutputCapacitance_pf',
        'FanoutLoad', 'FaninLoad', 'DriveStrength'
    ]
    
    log_features_added = []
    for col in log_transform_cols:
        if col in cells_df.columns:
            log_col = f'{col}_log'
            cells_df[log_col] = np.log1p(np.maximum(cells_df[col].values, 0))
            log_features_added.append(log_col)
    
    if verbose:
        print(f"\n[4] Log-transformed features: {len(log_features_added)}")
        for lf in log_features_added:
            print(f"    + {lf}")
    
    # Final feature list: original features + log-transformed
    final_features = available_features + log_features_added
    
    # ----------------------------------------------------------
    # Step 5: Build X, y arrays
    # ----------------------------------------------------------
    X = cells_df[final_features].values.astype(np.float32)
    y = cells_df['target_delay'].values.astype(np.float32)
    
    # Log-transform target (delay values are often log-normal)
    y_log = np.log1p(y)
    
    stats = {
        'n_samples': len(y),
        'n_features': X.shape[1],
        'target_mean': float(y.mean()),
        'target_std': float(y.std()),
        'target_min': float(y.min()),
        'target_max': float(y.max()),
        'preprocess_time': time.time() - t0,
    }
    
    if verbose:
        print(f"\n[5] Final dataset:")
        print(f"    X shape: {X.shape}")
        print(f"    y shape: {y.shape}")
        print(f"    Target delay range: [{stats['target_min']:.4f}, {stats['target_max']:.4f}] ns")
        print(f"    Target delay mean:  {stats['target_mean']:.4f} ns")
        print(f"    Preprocessing time: {stats['preprocess_time']:.2f}s")
        print("="*60 + "\n")
    
    return {
        'X': X,
        'y': y,
        'y_log': y_log,
        'feature_names': final_features,
        'df': cells_df,
        'stats': stats,
    }


def export_tabular_csv(result, output_path):
    """
    Export preprocessed data to a clean CSV for external ML frameworks.
    
    Parameters
    ----------
    result : dict
        Output from preprocess_tabular()
    output_path : str
        Path to save the CSV
    """
    df_out = pd.DataFrame(result['X'], columns=result['feature_names'])
    df_out['target_delay_ns'] = result['y']
    df_out['target_delay_log'] = result['y_log']
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    df_out.to_csv(output_path, index=False)
    
    print(f"Exported {len(df_out)} samples x {len(result['feature_names'])} features to {output_path}")
    print(f"  File size: {os.path.getsize(output_path) / 1024:.1f} KB")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FEASTA Tabular ML: Preprocess cell properties for delay prediction"
    )
    parser.add_argument("--pysta_path", type=str, required=True,
                        help="Path to directory containing FEASTA CSVs")
    parser.add_argument("--output", type=str, default="tabular_ml_ready.csv",
                        help="Output CSV path (default: tabular_ml_ready.csv)")
    parser.add_argument("--name", type=str, default=None,
                        help="Design name")
    
    args = parser.parse_args()
    
    # Load design
    print(f"Loading FEASTA CSVs from: {args.pysta_path}")
    design = Design(args.pysta_path, name=args.name, lazy_topology=True, verbose=True)
    
    # Preprocess
    result = preprocess_tabular(design, verbose=True)
    
    # Export
    export_tabular_csv(result, args.output)
    
    # Print feature summary for documentation
    print("\n" + "="*60)
    print("FEATURE SUMMARY (for paper)")
    print("="*60)
    print(f"\n{'Feature':<30} {'Type':<15} {'Mean':>10} {'Std':>10}")
    print("-"*65)
    for i, fname in enumerate(result['feature_names']):
        col = result['X'][:, i]
        ftype = "log-scaled" if "_log" in fname else ("boolean" if fname.startswith("Is") else "numeric")
        print(f"{fname:<30} {ftype:<15} {col.mean():>10.4f} {col.std():>10.4f}")
    
    print(f"\n{'Target: delay (ns)':<30} {'continuous':<15} {result['y'].mean():>10.4f} {result['y'].std():>10.4f}")
    print("="*60)
