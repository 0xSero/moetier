"""moetier: a minimal standard for tiered MoE expert placement and scheduling."""
from .spec import load, resolve
try:                                    # sim needs numpy; `moetier probe` is stdlib-only
    from .sim import run, prefill, load_trace
except ImportError:
    pass
__version__ = "0.1.0"
