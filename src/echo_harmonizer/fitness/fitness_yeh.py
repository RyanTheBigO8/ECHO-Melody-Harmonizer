# fitness/fitness_yeh.py
from __future__ import annotations

from typing import Any, Optional
import pandas as pd

from echo_harmonizer.fitness.fitness_primary import FitnessPrimaryBase
from echo_harmonizer.metrics import compute_CTnCTR, compute_PCS, compute_MCTD


class FitnessEvaluatorYeh(FitnessPrimaryBase):
    TARGET_CTnCTR = 0.74
    TARGET_PCS = 1.42
    TARGET_MCTD = 1.03

    def __init__(
        self,
        harmonization_table: pd.DataFrame,
        *,
        master_table: pd.DataFrame,
        melody_stream,
        weights: Optional[dict[str, float]] = None,
    ):
        super().__init__(harmonization_table, master_table=master_table, weights=weights)
        w = weights or {}
        self.w_ctnctr = float(w.get("w_ctnctr", 1.0))
        self.w_pcs = float(w.get("w_pcs", 1.0))
        self.w_mctd = float(w.get("w_mctd", 1.0))
        self.melody_stream = melody_stream  # needed by metrics.py

    # Yeh fitness is purely global (to match metrics.py exactly)
    def compute_region_fitness(self, idx: int, genes: list[int], table: pd.DataFrame) -> float:
        return 0.0

    def compute_global_fitness(self, genes: list[int], table: pd.DataFrame) -> float:
        ctn = float(compute_CTnCTR(genes, table, self.melody_stream))
        pcs = float(compute_PCS(genes, table, self.melody_stream))
        mctd = float(compute_MCTD(genes, table, self.melody_stream))

        d1 = ctn - self.TARGET_CTnCTR
        d2 = pcs - self.TARGET_PCS
        d3 = mctd - self.TARGET_MCTD

        # maximize == minimize deviation
        return float(
            - self.w_ctnctr * (d1 * d1)
            - self.w_pcs * (d2 * d2)
            - self.w_mctd * (d3 * d3)
        )

    def compute_total_fitness(self, genes: list[int], table: pd.DataFrame) -> tuple[dict[str, Any], float]:
        ctn = float(compute_CTnCTR(genes, table, self.melody_stream))
        pcs = float(compute_PCS(genes, table, self.melody_stream))
        mctd = float(compute_MCTD(genes, table, self.melody_stream))

        d1 = ctn - self.TARGET_CTnCTR
        d2 = pcs - self.TARGET_PCS
        d3 = mctd - self.TARGET_MCTD

        total = float(
            - self.w_ctnctr * (d1 * d1)
            - self.w_pcs * (d2 * d2)
            - self.w_mctd * (d3 * d3)
        )
        summary = {"CTnCTR": ctn, "PCS": pcs, "MCTD": mctd}
        return summary, total
