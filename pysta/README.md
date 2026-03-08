# PySTA

**Python Static Timing Analysis Interface**

PySTA is a Python framework that bridges OpenSTA's Static Timing Analysis (STA) engine with Machine Learning (ML) workflows. It ingests OpenSTA CSV exports into a structured in-memory representation, enabling efficient queries and direct export to ML-ready tensor formats.

---

## Key Features

| Feature | Description |
|---------|-------------|
| **RAM-Resident Design** | Zero-copy ingestion of CSV exports into indexed Pandas DataFrames with O(1) node lookup |
| **PyTorch Geometric Export** | Direct export to `torch_geometric.data.Data` for graph-based ML tasks |
| **Django-style Query API** | Filtering interface inspired by Django ORM: `pins.filter(SlackWorst_ns__lt=0)` |
| **Topology Analysis** | Sparse CSR matrices for BFS fanin/fanout traversal, cycle detection, and logic depth computation |

---

## Performance

| Operation | Tcl (OpenSTA) | PySTA | Speedup |
|-----------|---------------|-------|---------|
| Find all timing violations | 23 min | 0.21 sec | ~6,600x |
| Filter high capacitance | - | 0.19 sec | - |
| Load 448K-cell design | ~45 sec | 0.30 sec | ~150x |

---

## Prerequisites

PySTA requires CSV exports from FEASTA (the extended OpenSTA). Generate them using the FEASTA Tcl commands:

```tcl
read_liberty design.lib
read_verilog design.v
link_design top
read_sdc design.sdc
read_spef design.spef

write_network_nodes network_nodes.csv
write_network_arcs network_arcs.csv
write_pin_properties pin_properties.csv design.spef
write_cell_properties cell_properties.csv
```

This produces four CSV files:
- `network_nodes.csv` — All pins/nodes with structural attributes
- `network_arcs.csv` — All timing arcs with delay values
- `pin_properties.csv` — Pin-level timing features (slew, slack, capacitance)
- `cell_properties.csv` — Cell-level features (area, power, pin counts)

---

## Quick Start

### Basic Usage

```python
from pysta import Design

# Load design from FEASTA CSV exports
design = Design("./dumps/zipcpu/")
print(design.summary())

# Find timing violations
violations = design.pins.filter(SlackWorst_ns__lt=0)
print(f"Found {len(violations)} violations")

# Get critical paths
paths = design.get_critical_paths(top_k=10)
for p in paths:
    print(f"{p['startpoint']} -> {p['endpoint']}: {p['slack']:.3f} ns")
```

### Export to PyTorch Geometric (GNN)

```python
data = design.to_pytorch_geometric(
    node_features=["SlewRise_ns", "Capacitance_pf", "CoordX_um", "CoordY_um"],
    target="SlackWorst_ns"
)
print(f"Nodes: {data.num_nodes}, Edges: {data.num_edges}")
```

### Query API

```python
# Filter with operators
design.pins.filter(
    SlackWorst_ns__lt=0,           # Less than
    Capacitance_pf__gt=0.5,        # Greater than
    ClockDomains__contains="clk",  # String contains
    CellType__in=["buf", "inv"]    # In list
)

# Connectivity analysis
design.get_fanin("reg/D", depth=3)   # Trace 3 levels back
design.get_fanout("reg/Q", depth=5)  # Trace 5 levels forward

# Subgraph extraction (for GNN training)
nodes, arcs = design.extract_subgraph("critical_reg/D", hops=2)
```

---

## Architecture

```
+------------------------------------------------------------+
|                          PySTA                             |
+------------------------------------------------------------+
|  +-----------+     +-----------+     +-------------+       |
|  |  Loader   |---> |   Query   |---> |TensorBridge |       |
|  |           |     |  Engine   |     |             |       |
|  | CSV Load  |     | Filter    |     | PyTorch     |       |
|  | Indexing  |     | Traversal |     | NumPy       |       |
|  | Validate  |     | Topology  |     | Normalize   |       |
|  +-----------+     +-----------+     +-------------+       |
|                         |                                  |
|                         v                                  |
|  +---------------------------------------------------------+|
|  |                 DESIGN OBJECT                           ||
|  |    nodes (DataFrame) | arcs (DataFrame) | cells        ||
|  +---------------------------------------------------------+|
+------------------------------------------------------------+
```

---

## Project Structure

```
pysta/
|-- __init__.py       # Package entry point
|-- loader.py         # CSV ingestion and schema validation
|-- query.py          # Django-style filter and traversal API
|-- topology.py       # CSR adjacency, cycle detection, BFS depth
|-- export.py         # Tensor bridge (PyTorch Geometric, NumPy)
|-- utils.py          # Schema definitions and shared utilities
```

---

## Application Areas

1. **GNN-based Net Delay Prediction** — Train models to predict net-level delays from structural and physical features
2. **Tabular Cell Power Regression** — Classical regression on cell-level feature vectors (area, pin counts, fanout)
3. **Critical Path Analysis** — Fast fanin/fanout traversal for timing-critical subgraph extraction
4. **Placement-Aware ML** — Leverage physical pin coordinates (CoordX, CoordY) from SPEF for spatial learning

---

## Related Projects

- [OpenSTA](https://github.com/parallaxsw/OpenSTA) — The underlying STA engine
- [CircuitOps](https://github.com/NVlabs/CircuitOps) — OpenROAD-based ML infrastructure (NVIDIA)
- [CircuitNet](https://github.com/circuitnet/CircuitNet) — Open EDA ML dataset and benchmarks
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) — GNN training framework

---
