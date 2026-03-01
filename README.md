# FEASTA: Feature Extraction Framework for Static Timing Analysis

**An open-source feature extraction framework extending OpenSTA for scalable, native access to circuit and timing features for Machine Learning workflows.**

---

## Overview

FEASTA augments the [OpenSTA](https://github.com/parallaxsw/OpenSTA) (v2.7.0) implementation with:

1. **Native C++ Feature Extraction** (`csv/csvWriter.cc`) — Embeds feature extraction directly within the timing engine, bypassing the Tcl interface overhead. Uses internal iterators (`VertexIterator`, `VertexOutEdgeIterator`, `InstancePinIterator`) for O(1) per-object access and single-pass CSV export.

2. **SPEF Coordinate Parser** (`csv/SpefParser.hh`) — Extracts physical pin coordinates (X, Y) from SPEF files and injects them into the pin properties CSV.

3. **Tcl Command Interface** (`app/StaMain.cc`) — Registers custom Tcl commands for invoking the C++ extraction layer from the OpenSTA shell.

4. **PySTA Python Framework** (`pysta/`) — A high-performance Python library that ingests the CSV exports into RAM-resident DataFrames with schema validation, graph topology construction, Django-style query filtering, and ML tensor export.

---

## Repository Structure

```
.
├── app/
│   └── StaMain.cc              # Tcl command registration (FEASTA additions)
├── csv/
│   ├── csvWriter.cc            # Native C++ feature extraction (core of FEASTA)
│   ├── csvWriter.hh            # Header for extraction functions
│   ├── SpefParser.cc           # SPEF coordinate parser implementation
│   ├── SpefParser.hh           # SPEF parser header
│   └── CsvWriter.i             # SWIG interface for Tcl bindings
├── pysta/                      # PySTA Python Framework
│   ├── __init__.py             # Package entry point
│   ├── loader.py               # Design Loader (zero-copy CSV ingestion)
│   ├── query.py                # Query Engine (Django-style filters)
│   ├── topology.py             # Topology Builder (CSR adjacency, BFS)
│   ├── export.py               # ML Bridge (TensorBridge, Normalizer)
│   ├── utils.py                # Schema definitions, validation, utilities
│   ├── experiments/            # Downstream ML task scripts
│   │   ├── pysta_tabular_ml.py # Tabular cell delay prediction (XGBoost)
│   │   ├── pysta_to_dgl.py     # Graph conversion for GNN (DGL)
│   │   ├── benchmark.py        # Performance benchmarking suite
│   │   ├── model.py            # TimingGCN model definition
│   │   ├── train.py            # GNN training script
│   │   └── data_graph.py       # Graph data utilities
│   └── tests/                  # Verification tests
├── examples/                   # Self-contained example designs
│   ├── example1.v              # Small nangate45 design
│   ├── nangate45_slow.lib.gz   # Liberty library
│   └── ...
├── CMakeLists.txt              # Build system (includes csv/ sources)
└── README.md                   # Original OpenSTA README
```

---

## Building FEASTA

FEASTA builds as part of the standard OpenSTA compilation. No separate build step is required — the `csv/csvWriter.cc` source is already integrated into `CMakeLists.txt`.

### Prerequisites

| Dependency | Minimum Version | Purpose |
|------------|----------------|---------|
| CMake | 3.10 | Build system |
| C++ Compiler | C++17 support | GCC 7+ or Clang 5+ |
| Tcl | 8.5+ | Shell interface |
| SWIG | 3.0+ | Tcl bindings generation |
| Flex | 2.6+ | Lexer generation |
| Bison | 3.2+ | Parser generation |
| Eigen3 | 3.x | Matrix operations |
| zlib | — | Compressed file support |
| CUDD | (optional) | BDD package |

**Install on Ubuntu/Debian:**

```bash
sudo apt-get install cmake gcc g++ tcl-dev swig flex bison \
    libeigen3-dev zlib1g-dev libreadline-dev
```

### Compile

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

This produces the `build/sta` binary with all FEASTA commands built in.

### Verify Build

```bash
./build/sta -no_splash -no_init -exit -help
```

---

## Usage

### Step 1: Generate CSV Files (C++ Extraction)

Create a Tcl script that reads your design files and invokes FEASTA extraction commands:

```tcl
# read_and_extract.tcl

# 1. Read design inputs
read_liberty /path/to/library.lib
read_verilog /path/to/netlist.v
link_design <top_module>
read_spef /path/to/parasitics.spef        # Optional: for pin coordinates

# 2. Apply timing constraints
create_clock -name clk -period 10 [get_ports clk]
set_input_delay -clock clk 0 [get_ports -filter "direction == input"]
set_output_delay -clock clk 0 [get_ports -filter "direction == output"]

# 3. Extract features (FEASTA commands)
dump_network_nodes                                              # → network_nodes.csv
dump_network_arcs                                               # → network_arcs.csv
dump_cell_properties cell_properties.csv                        # → cell_properties.csv
dump_pin_properties  pin_properties.csv /path/to/parasitics.spef  # → pin_properties.csv (with coords)
```

Run it:

```bash
./build/sta -no_splash -no_init -exit read_and_extract.tcl
```

This generates four CSV files in the current directory.

### FEASTA Tcl Commands Reference

| Command | Output File | Description |
|---------|-------------|-------------|
| `dump_network_nodes` | `network_nodes.csv` | All pins/ports with structural attributes and clock domain membership |
| `dump_network_arcs` | `network_arcs.csv` | All timing arcs (net + cell) with source/sink and arc delays |
| `dump_cell_properties <file>` | `<file>` | Cell-level attributes: type, area, power, pin counts, timing arc counts |
| `dump_pin_properties <file> [spef]` | `<file>` | Pin-level timing: slew, slack, capacitance, drive resistance, coordinates |
| `dump_pin_coords <spef> <file>` | `<file>` | Standalone pin coordinate extraction from SPEF |
| `get_pin_coords <pin> <spef>` | — | Interactive single-pin coordinate lookup (cached) |

---

### Step 2: Load with PySTA (Python)

#### Prerequisites

```bash
pip install pandas numpy scipy
# Optional for ML tasks:
pip install torch torch-geometric dgl scikit-learn xgboost
```

#### Basic Usage

```python
from pysta import Design

# Load all 4 CSVs from a directory
design = Design("/path/to/csv/directory/", name="MyDesign")

# Access DataFrames
print(f"Nodes: {len(design.nodes)}")
print(f"Arcs:  {len(design.arcs)}")
print(f"Cells: {len(design.cells)}")

# Design summary
print(design.summary())
```

#### Query Engine (Django-Style Filters)

```python
# Find timing violations
violations = design.pins.filter(SlackWorst_ns__lt=0)

# Multi-condition filter
critical = design.pins.filter(
    SlackWorst_ns__lt=0,
    Capacitance_pf__gt=0.5
)

# Supported operators: __lt, __lte, __gt, __gte, __eq, __ne,
#                       __in, __contains, __startswith, __endswith

# O(1) node lookup
node = design.get_node("instance/pin_name")

# Fanout/Fanin traversal
fanout_nodes = design.pins.get_fanout("reg/Q", depth=3)
fanin_nodes  = design.pins.get_fanin("reg/D", depth=3)

# Critical path extraction
paths = design.pins.get_critical_paths(top_k=10)
for p in paths:
    print(f"{p['startpoint']} → {p['endpoint']}: {p['slack']:.3f} ns")
```

#### Topology Builder

```python
# Build graph topology (lazy — built on first access)
design._ensure_topology()
topo = design.topology

# CSR adjacency matrices for O(degree) queries
print(f"Forward adjacency: {topo.forward_adj.shape}")
print(f"Backward adjacency: {topo.backward_adj.shape}")

# Cycle detection (Kahn's algorithm)
has_cycles, breaking_points = topo.detect_cycles()

# Logic depth computation (bidirectional BFS)
depth_in, depth_out = topo.compute_logic_depth(design.nodes)

# Topology statistics
stats = topo.get_stats()
# {'num_nodes', 'num_edges', 'has_cycles', 'max_depth_from_input', ...}
```

#### ML Bridge (Tensor Export)

```python
from pysta import TensorBridge, Normalizer

bridge = TensorBridge(
    nodes_df=design.nodes,
    arcs_df=design.arcs
)

# Export to NumPy (for scikit-learn, XGBoost)
X, y = bridge.to_numpy(
    features=["SlewRise_ns", "Capacitance_pf"],
    target="SlackWorst_ns",
    normalize=True
)

# Export to normalized DataFrame
df = bridge.to_dataframe(
    features=["SlewRise_ns", "Capacitance_pf"],
    normalize=True
)

# Feature-specific normalization with inverse transform
normalizer = Normalizer()
transformed = normalizer.fit_transform(df, columns=["SlewRise_ns"])
original = normalizer.inverse_transform(transformed["SlewRise_ns"].values, column="SlewRise_ns")

# List available features
print(bridge.get_available_features())
```

#### PyTorch Geometric Export (for GNNs)

```python
# Requires: pip install torch torch-geometric
data = bridge.to_pytorch_geometric(
    node_features=["SlewRise_ns", "Capacitance_pf", "CoordX_um", "CoordY_um"],
    target="SlackWorst_ns",
    edge_weight="Delay",
    normalize=True
)
print(f"Nodes: {data.num_nodes}, Edges: {data.num_edges}")
# Ready for GNN training
```

---

## CSV Schema Reference

### network_nodes.csv (Table II)

| Column | Type | Description |
|--------|------|-------------|
| Name | str | Full hierarchical pin/port name |
| InstanceName | str | Parent instance name |
| PinName | str | Pin name within instance |
| Direction | str | `input`, `output`, or `bidirectional` |
| IsPort | bool | `1` for top-level port |
| Type | str | `leaf` or `hierarchical` |
| IsClockNetwork | bool | `1` if part of clock network |

### network_arcs.csv (Table I)

| Column | Type | Description |
|--------|------|-------------|
| Source | str | Source pin/port of the arc |
| Sink | str | Sink pin/port of the arc |
| NetName | str | Connecting net name |
| Delay | float | Arc delay in ns |
| ArcType | str | `net` (interconnect) or `cell` (gate) |
| Delay_Max_RR/RF/FR/FF | float | Per-transition arc delays |

### pin_properties.csv (Table IV)

| Column | Type | Description |
|--------|------|-------------|
| FullName | str | Full hierarchical pin name |
| Direction | str | Pin direction |
| IsHierarchical | bool | `1` for hierarchical pin |
| IsRegisterClock | bool | `1` for register clock pin |
| SlewMaxRise/Fall_ns | float | Max path slew (rise/fall) |
| SlewMinRise/Fall_ns | float | Min path slew (rise/fall) |
| SlackMaxRise/Fall_ns | float | Setup slack (rise/fall) |
| SlackMinRise/Fall_ns | float | Hold slack (rise/fall) |
| Capacitance_pf | float | Total pin capacitance |
| DriveResistance_kOhm | float | Drive resistance |
| Activity | float | Switching activity |
| ToggleRate_MHz | float | Toggle rate |
| CoordX_um / CoordY_um | float | Physical coordinates (from SPEF) |

### cell_properties.csv (Table III)

| Column | Type | Description |
|--------|------|-------------|
| FullInstanceName | str | Full hierarchical instance name |
| LibertyCell | str | Library cell name |
| Library | str | Library name |
| CellType | str | `combinational`, `sequential`, `buffer`, etc. |
| IsBuffer/IsInverter/IsMemory/IsMacro | bool | Cell classification flags |
| Area_um2 | float | Cell area |
| PinCount / InputPinCount / OutputPinCount | int | Pin counts |
| FanoutLoad / FaninLoad | int | Connectivity counts |
| LeakagePower_pW / SwitchingPower_pW / InternalPower_pW / TotalPower_pW | float | Power breakdown |
| SetupTime_ns / HoldTime_ns | float | Sequential cell timing |
| PropagationDelay_ns / ClkToQDelay_ns | float | Cell delays |
| TimingArcCount | int | Number of timing arcs |
| ClockDomains | str | Associated clock domains |

---

## Downstream ML Tasks

### 1. Graph-Based Net Delay Prediction (GNN)

Uses `pysta_to_dgl.py` to convert FEASTA CSVs to DGL heterographs for CircuitNet-style GNN training.

```bash
cd pysta/experiments

# Convert CSVs to DGL graph
python pysta_to_dgl.py \
    --pysta_path /path/to/csv/dir \
    --output_path ./graph/design.bin

# Train TimingGCN model
python train.py \
    --data_path /path/to/csv/dir \
    --checkpoint my_model \
    --iteration 5000
```

**Node features:** Normalized coordinates (X, Y), input capacitance, transition slew  
**Edge target:** Net delay (log-transformed)  
**Graph format:** DGL heterograph with `net_out` and `net_in` edge types

### 2. Tabular Cell Delay Regression (XGBoost)

Uses `pysta_tabular_ml.py` to preprocess cell properties for gradient boosted decision tree models.

```bash
cd pysta/experiments

python pysta_tabular_ml.py \
    --pysta_path /path/to/csv/dir \
    --output tabular_ml_ready.csv
```

**Features:** Area, pin counts, capacitance, drive strength, fanout/fanin, boolean flags (IsBuffer, IsSequential, etc.)  
**Target:** Propagation delay (combinational) or Clock-to-Q delay (sequential)  
**Preprocessing:** Log-scaling for power-law features, boolean encoding, NaN handling

### 3. Performance Benchmarking

```bash
cd pysta/experiments
python benchmark.py --pysta_path /path/to/csv/dir --device auto
```

---

## Quick Start Example

Using the self-contained nangate45 example included in the repository:

```bash
# 1. Build
mkdir build && cd build && cmake .. && make -j$(nproc) && cd ..

# 2. Generate CSVs
cd examples
../build/sta -no_splash -no_init -exit -cmd "
  read_liberty nangate45_slow.lib.gz
  read_verilog example1.v
  link_design top
  create_clock -name clk -period 10 {clk1 clk2}
  set_input_delay -clock clk 0 {in1 sel}
  dump_network_nodes
  dump_network_arcs
  dump_cell_properties cell_properties.csv
  dump_pin_properties pin_properties.csv
"

# 3. Load with PySTA
python3 -c "
import sys; sys.path.insert(0, '..')
from pysta import Design
d = Design('.', name='example', verbose=True)
print(d.summary())
print('Ports:', len(d.pins.filter(IsPort=True)))
"

# 4. Clean up
rm -f network_nodes.csv network_arcs.csv cell_properties.csv pin_properties.csv
cd ..
```

---

## Files Modified from Base OpenSTA

| File | Change Type | Description |
|------|-------------|-------------|
| `app/StaMain.cc` | Modified | Added FEASTA Tcl command registration (`registerNetworkGraphCmds`) |
| `csv/csvWriter.cc` | **New** | Core C++ feature extraction module (~1475 lines) |
| `csv/csvWriter.hh` | **New** | Header for extraction functions |
| `csv/SpefParser.cc` | **New** | SPEF coordinate parser |
| `csv/SpefParser.hh` | **New** | SPEF parser header with inline implementation |
| `csv/CsvWriter.i` | **New** | SWIG interface for Tcl bindings |
| `CMakeLists.txt` | Modified | Added `csv/csvWriter.cc` to `STA_SOURCE`, `csv/` to include paths |
| `pysta/` | **New** | Entire PySTA Python framework (7 modules + experiments) |

---

## License

This project is distributed under the **GNU General Public License v3 (GPLv3)**, consistent with the OpenSTA base project. See [LICENSE](LICENSE) for details.
