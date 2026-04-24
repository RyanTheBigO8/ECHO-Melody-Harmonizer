# fitness_atonal.py
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from fitness.fitness_primary import FitnessPrimaryBase


class FitnessEvaluatorAtonal(FitnessPrimaryBase):
    """
    Atonal fitness:
      - overlap (reward)
      - distance + tonal_distance (penalties, optional)
      - repeat penalty (optional)
      - pc_change bonus (optional)
      - sus resolution score (optional)
      - global: coverage (optional)
    """

    def __init__(
        self,
        harmonization_table: pd.DataFrame,
        *,
        master_table: pd.DataFrame,
        chord_dist: Optional[np.ndarray] = None,
        chord_tonal_dist: Optional[np.ndarray] = None,
        weights: Optional[dict[str, float]] = None,
    ):
        super().__init__(
            harmonization_table,
            master_table=master_table,
            chord_dist=chord_dist,
            chord_tonal_dist=chord_tonal_dist,
            weights=weights,
        )
        w = weights or {}
        self.w_sus_resolution = float(w.get("w_sus_resolution", 0.0))

    # --------------------------
    # Local term: sus resolution
    # --------------------------
    def _sus_resolution_score(self, prev_id: int | None, cur_id: int) -> float:
        """
        Signed score:
          +0.1 if prev is sus (some root interpretation) and resolves into cur
          -1.0 if prev is sus and does NOT resolve
           0.0 if prev is not sus (under any interpretation) OR prev_id is None
        """
        if self.w_sus_resolution == 0.0:
            return 0.0
        if prev_id is None:
            return 0.0

        prev_pcs = [int(x) % 12 for x in self.pitch_classes_by_id[int(prev_id)]]
        if not prev_pcs:
            return 0.0

        prev_set = set(prev_pcs)
        cur_mask = int(self.chord_mask_by_id[int(cur_id)])

        any_sus = False
        resolves = False

        def pc_dist(a: int, b: int) -> int:
            return min((b - a) % 12, (a - b) % 12)

        for root in prev_set:
            rel = frozenset(((pc - root) % 12) for pc in prev_set)

            if rel == frozenset({0, 2, 7}):        # sus2
                any_sus = True
                sus_tone = (root + 2) % 12
            elif rel == frozenset({0, 5, 7}):      # sus4
                any_sus = True
                sus_tone = (root + 5) % 12
            else:
                continue

            # target must keep root and fifth
            root_bit = 1 << root
            fifth_bit = 1 << ((root + 7) % 12)
            if (cur_mask & root_bit) == 0 or (cur_mask & fifth_bit) == 0:
                continue

            # target must contain m3 or M3
            m3 = (root + 3) % 12
            M3 = (root + 4) % 12
            has_m3 = (cur_mask & (1 << m3)) != 0
            has_M3 = (cur_mask & (1 << M3)) != 0
            if not (has_m3 or has_M3):
                continue

            # suspended tone moves by 1 or 2 semitones to the third
            if (has_m3 and pc_dist(sus_tone, m3) in (1, 2)) or (has_M3 and pc_dist(sus_tone, M3) in (1, 2)):
                resolves = True
                break

        if not any_sus:
            return 0.0
        return 0.1 if resolves else -1.0

    # --------------------------
    # Protocol methods
    # --------------------------
    def compute_region_terms(self, idx: int, genes: list[int], table: pd.DataFrame) -> dict[str, Any]:
        cur = int(genes[idx])
        prev = int(genes[idx - 1]) if idx > 0 else None

        t: dict[str, Any] = {
            "chord_id": cur,
            "prev_id": prev,
            "overlap": self._overlap_reward(cur, idx),
            "dist": self._distance_penalty(prev, cur),
            "dist_tonal": self._tonal_distance_penalty(prev, cur),
            "repeat": self._repeat_penalty(prev, cur),
            "pc_change": self._pc_change_bonus(prev, cur),
            "sus_resolution": self._sus_resolution_score(prev, cur),
        }
        return t

    def compute_region_fitness(self, idx: int, genes: list[int], table: pd.DataFrame) -> float:
        t = self.compute_region_terms(idx, genes, table)
        total = (
            self.w.w_overlap * t["overlap"]
            - self.w.w_dist * t["dist"]
            - self.w.w_dist_tonal * t["dist_tonal"]
            - self.w.w_repeat * t["repeat"]
            + self.w.w_pc_change * t["pc_change"]
            + self.w_sus_resolution * t["sus_resolution"]
        )
        return float(total)

    def compute_global_fitness(self, genes: list[int], table: pd.DataFrame) -> float:
        # currently only coverage in the base
        return float(self.w.w_coverage * self._coverage_reward(genes))

    def compute_total_fitness(self, genes: list[int], table: pd.DataFrame) -> tuple[dict[str, Any], float]:
        total_region = 0.0

        overlaps: list[float] = []
        dists: list[float] = []
        dist_tonals: list[float] = []
        repeats: list[float] = []
        pc_changes: list[float] = []
        sus_list: list[float] = []

        for i in range(len(genes)):
            t = self.compute_region_terms(i, genes, table)

            overlaps.append(float(t["overlap"]))
            dists.append(float(t["dist"]))
            dist_tonals.append(float(t["dist_tonal"]))
            repeats.append(float(t["repeat"]))
            pc_changes.append(float(t["pc_change"]))
            sus_list.append(float(t["sus_resolution"]))

            total_region += (
                self.w.w_overlap * t["overlap"]
                - self.w.w_dist * t["dist"]
                - self.w.w_dist_tonal * t["dist_tonal"]
                - self.w.w_repeat * t["repeat"]
                + self.w.w_pc_change * t["pc_change"]
                + self.w_sus_resolution * t["sus_resolution"]
            )

        g = self.compute_global_fitness(genes, table)
        total = float(total_region + g)

        terms_summary = {
            "mean_overlap": float(np.mean(overlaps) * self.w.w_overlap) if overlaps else 0.0,
            "mean_dist": float(np.mean(dists) * (-1.0) * self.w.w_dist) if dists else 0.0,
            "mean_dist_tonal": float(np.mean(dist_tonals) * (-1.0) * self.w.w_dist_tonal) if dist_tonals else 0.0,
            "mean_repeat": float(np.mean(repeats) * (-1.0) * self.w.w_repeat) if repeats else 0.0,
            "mean_pc_change": float(np.mean(pc_changes) * self.w.w_pc_change) if pc_changes else 0.0,
            "mean_sus_resolution": float(np.mean(sus_list) * self.w_sus_resolution) if sus_list else 0.0,
            "mean_coverage": float(g),
        }
        return terms_summary, total
