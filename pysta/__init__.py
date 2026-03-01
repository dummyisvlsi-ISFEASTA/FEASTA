"""
PySTA: Python Static Timing Analysis Interface

A high-performance Python framework for ML-ready Static Timing Analysis.
Load OpenSTA CSV exports, query timing data instantly, and export to PyTorch.

Example:
    >>> from pysta import Design
    >>> design = Design("./dumps/zipcpu/")
    >>> print(design.summary())
    >>> violations = design.pins.filter(SlackWorst_ns__lt=0)
    >>> data = design.to_pytorch_geometric(node_features=["SlewRise_ns", "Capacitance_pf"])
"""

from .loader import Design
from .topology import TopologyBuilder, build_topology
from .query import QueryEngine, FilterableDataFrame
from .export import TensorBridge, Normalizer
from .utils import SchemaError, LoadError

__version__ = "1.0.0"
__all__ = [
    "Design",
    "TopologyBuilder",
    "QueryEngine",
    "FilterableDataFrame",
    "TensorBridge",
    "Normalizer",
    "SchemaError",
    "LoadError",
    "build_topology"
]
