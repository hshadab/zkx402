"""
zkX402 Python SDK

Privacy-preserving authorization for AI agents using zkML proofs.
"""

from .client import ZKX402Client
from .async_client import AsyncZKX402Client
from .exceptions import (
    ZKX402Error,
    PolicyNotFoundError,
    InvalidInputError,
    ProofGenerationError,
    SimulationError,
)

__version__ = "0.1.0"
__all__ = [
    "ZKX402Client",
    "AsyncZKX402Client",
    "ZKX402Error",
    "PolicyNotFoundError",
    "InvalidInputError",
    "ProofGenerationError",
    "SimulationError",
]
