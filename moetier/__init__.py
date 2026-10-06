"""moetier: a minimal standard for tiered MoE expert placement and scheduling."""
from .spec import load, resolve
from .sim import run, prefill, load_trace
__version__ = "0.1.0"
