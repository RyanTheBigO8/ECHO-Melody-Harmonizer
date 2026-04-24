# fitness_tonal.py
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from fitness.fitness_primary import FitnessPrimaryBase
from chord_pool.chord_roles import get_role_masks_for_key, get_role_mask_table, bit
from fitness.role_to_function import best_transition_score, mask_has_function


class FitnessEvaluatorTonal(FitnessPrimaryBase):
    """
    Diatonic fitness:
      - overlap (reward)
      - distance / tonal distance (penalties, optional)
      - repeat (penalty, optional)
      - pc_change (bonus, optional)
      - harmonic transition (reward/penalty)
      - leading-tone resolve (reward/penalty)
      - cadence (reward)
      - global: coverage (optional)

    NOTE:
      - sus “resolution” is handled inside role_to_function as a special transition rule,
        so there is NO separate sus term here.
    """

    def __init__(
        self,
        harmonization_table: pd.DataFrame,
        *,
        master_table: pd.DataFrame,
        tonic_pc: int,
        mode: str,  # "major" | "minor"
        role_mask_table: pd.DataFrame | None = None,
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
        self.w_func_transition = float(w.get("w_func_transition", 1.0))
        self.w_leading_tone_resolve = float(w.get("w_leading_tone_resolve", 0.0))
        self.w_cadence = float(w.get("w_cadence", 0.0))

        self.tonic_pc = int(tonic_pc) % 12
        self.mode = str(mode)

        # chord_id -> role mask for this key
        if role_mask_table is None:
            role_mask_table = get_role_mask_table(master_table, force_rebuild=False)

        self.role_mask_by_chord_id: list[int] = get_role_masks_for_key(
            role_mask_table, self.tonic_pc, self.mode)

        # Cadence weights/types by index (if present)
        self.cad_weights = (
            harmonization_table["cadence_weight"].to_numpy(dtype=float)
            if "cadence_weight" in harmonization_table.columns
            else None
        )
        self.cad_types = (
            harmonization_table["cadence_type"].astype(str).to_numpy()
            if "cadence_type" in harmonization_table.columns
            else None
        )

    # --------------------------
    # Extra tonal terms
    # --------------------------

    def _transition_score(self, prev_id: int | None, cur_id: int) -> float:
        if self.w_func_transition == 0.0 or prev_id is None:
            return 0.0
        pm = int(self.role_mask_by_chord_id[int(prev_id)])
        cm = int(self.role_mask_by_chord_id[int(cur_id)])
        return best_transition_score(pm, cm)

    def _leading_tone_resolve_score(self, prev_id: int | None, cur_id: int) -> float:
        """
        Pitch-based micro-bonus (NOT role-based):
        1) Leading tone (scale degree 7) resolves to tonic:
            if prev contains LT pc and cur contains tonic pc => +1.0
        2) Chordal 7th resolves down by semitone into a chord tone:
            if prev is 4-note chord and ANY pc in prev resolves down 1 semitone into cur => +1.0

        This returns a small raw score; scale it via w_leading_tone_resolve.
        """
        if self.w_leading_tone_resolve == 0.0 or prev_id is None:
            return 0.0

        prev_id = int(prev_id)
        cur_id = int(cur_id)

        prev_mask = int(self.chord_mask_by_id[prev_id])
        cur_mask  = int(self.chord_mask_by_id[cur_id])

        if prev_mask == 0 or cur_mask == 0:
            return 0.0

        tonic_pc = int(self.tonic_pc) % 12
        lt_pc = (tonic_pc + 11) % 12  # leading tone in major + harmonic minor
        tonic_bit = 1 << tonic_pc
        lt_bit = 1 << lt_pc

        score = 0.0

        # (1) scale leading-tone -> tonic
        if (prev_mask & lt_bit) != 0 and (cur_mask & tonic_bit) != 0:
            score += 1.0

        # (2) chordal 7th-like downward semitone resolution (very lightweight heuristic)
        # treat "7th chord" as any 4-pc chord
        if prev_mask.bit_count() == 4:
            prev_pcs = self.pitch_classes_by_id[prev_id]
            for pc in prev_pcs:
                pc = int(pc) % 12
                for down in ((pc - 1) % 12, (pc - 2) % 12):
                    if (cur_mask & (1 << down)) != 0:
                        score += 1.0
                        break
                if score > 0.0:
                    break

        return float(score)


    def _cadence_score(self, idx: int, cur_id: int) -> float:
        if self.w_cadence == 0.0 or self.cad_weights is None:
            return 0.0

        w = float(self.cad_weights[idx])
        if w <= 0.0:
            return 0.0

        ctype = ""
        if self.cad_types is not None:
            ctype = str(self.cad_types[idx] or "")

        m = int(self.role_mask_by_chord_id[int(cur_id)])
        if m == 0:
            return 0.0

        # end_of_piece: must be T
        if ctype == "end_of_piece":
            return float(w if mask_has_function(m, {"T"}) else 0.0)

        # otherwise: allow T or D (e.g., half cadence)
        return float(w if mask_has_function(m, {"T", "D"}) else 0.0)

    # --------------------------
    # Protocol methods
    # --------------------------

    def compute_region_terms(self, idx: int, genes: list[int], table: pd.DataFrame) -> dict[str, Any]:
        cur = int(genes[idx])
        prev = int(genes[idx - 1]) if idx > 0 else None

        overlap = self._overlap_reward(cur, idx)
        dist = self._distance_penalty(prev, cur)
        dist_tonal = self._tonal_distance_penalty(prev, cur)
        repeat = self._repeat_penalty(prev, cur)
        pc_change = self._pc_change_bonus(prev, cur)

        func_tr = self._transition_score(prev, cur)
        lt_res = self._leading_tone_resolve_score(prev, cur)
        cad = self._cadence_score(idx, cur)

        return {
            "chord_id": cur,
            "prev_id": prev,
            "overlap": float(overlap),
            "dist": float(dist),
            "dist_tonal": float(dist_tonal),
            "repeat": float(repeat),
            "pc_change": float(pc_change),
            "func_transition": float(func_tr),
            "leading_tone_resolve": float(lt_res),
            "cadence": float(cad),
        }

    def compute_region_fitness(self, idx: int, genes: list[int], table: pd.DataFrame) -> float:
        t = self.compute_region_terms(idx, genes, table)
        total = (
            self.w.w_overlap * t["overlap"]
            - self.w.w_dist * t["dist"]
            - self.w.w_dist_tonal * t["dist_tonal"]
            - self.w.w_repeat * t["repeat"]
            + self.w.w_pc_change * t["pc_change"]
            + self.w_func_transition * t["func_transition"]
            + self.w_leading_tone_resolve * t["leading_tone_resolve"]
            + self.w_cadence * t["cadence"]
        )
        # Start chord correction: better be diatonic triad I
        if idx == 0:
            start_chord_id = int(genes[0])
            start_role_mask = int(self.role_mask_by_chord_id[start_chord_id])
            if (start_role_mask & bit("diat_triad_1")) == 0:
                total -= 1.0  # penalty

        return float(total)

    def compute_global_fitness(self, genes: list[int], table: pd.DataFrame) -> float:
        # only coverage for now (choice 1 denom is computed in base)
        if self.w.w_coverage == 0.0:
            return 0.0
        return float(self.w.w_coverage * self._coverage_reward(genes))

    def compute_total_fitness(self, genes: list[int], table: pd.DataFrame) -> tuple[dict[str, Any], float]:
        total_region = 0.0

        overlaps: list[float] = []
        dists: list[float] = []
        dist_tonals: list[float] = []
        repeats: list[float] = []
        pc_changes: list[float] = []
        func_trs: list[float] = []
        lt_res: list[float] = []
        cads: list[float] = []

        for i in range(len(genes)):
            t = self.compute_region_terms(i, genes, table)

            overlaps.append(t["overlap"])
            dists.append(t["dist"])
            dist_tonals.append(t["dist_tonal"])
            repeats.append(t["repeat"])
            pc_changes.append(t["pc_change"])
            func_trs.append(t["func_transition"])
            lt_res.append(t["leading_tone_resolve"])
            cads.append(t["cadence"])

            total_region += (
                self.w.w_overlap * t["overlap"]
                - self.w.w_dist * t["dist"]
                - self.w.w_dist_tonal * t["dist_tonal"]
                - self.w.w_repeat * t["repeat"]
                + self.w.w_pc_change * t["pc_change"]
                + self.w_func_transition * t["func_transition"]
                + self.w_leading_tone_resolve * t["leading_tone_resolve"]
                + self.w_cadence * t["cadence"]
            )


        g = self.compute_global_fitness(genes, table)
        total = float(total_region + g)

        summary = {
            "mean_overlap": float(np.mean(overlaps) * self.w.w_overlap) if overlaps else 0.0,
            "mean_dist": float(np.mean(dists) * (-1.0) * self.w.w_dist) if dists else 0.0,
            "mean_dist_tonal": float(np.mean(dist_tonals) * (-1.0) * self.w.w_dist_tonal) if dist_tonals else 0.0,
            "mean_repeat": float(np.mean(repeats) * (-1.0) * self.w.w_repeat) if repeats else 0.0,
            "mean_pc_change": float(np.mean(pc_changes) * self.w.w_pc_change) if pc_changes else 0.0,
            "mean_func_transition": float(np.mean(func_trs) * self.w_func_transition) if func_trs else 0.0,
            "mean_leading_tone_resolve": float(np.mean(lt_res) * self.w_leading_tone_resolve) if lt_res else 0.0,
            "mean_coverage": float(g),
        }
        return summary, total
