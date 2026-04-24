# fitness_protocol.py
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable
import pandas as pd


@runtime_checkable
class FitnessEvaluatorProtocol(Protocol):
    """
    Incremental-friendly interface:
      - region fitness at i may depend on genes[i] and genes[i-1]
      - global fitness may depend on the whole genes list (e.g., coverage)
    """

    def compute_region_fitness(self, idx: int, genes: list[int], table: pd.DataFrame) -> float: ...

    def compute_global_fitness(self, genes: list[int], table: pd.DataFrame) -> float: ...

    # optional logging/debug helper (keep if you like)
    def compute_total_fitness(self, genes: list[int], table: pd.DataFrame) -> tuple[dict[str, Any], float]: ...
