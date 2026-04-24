# fitness_primary.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from music21 import note as m21note

from chord_pool.chords_master import MASTER_CHORD_REGISTRY


def _pcs_to_mask(pcs: list[int]) -> int:
    m = 0
    for pc in pcs:
        m |= 1 << (int(pc) % 12)
    return m


def _infer_pool_size_from_table(table: pd.DataFrame) -> int:
    """
    Choice (1): pool size = union of acceptable chords across all regions.
    Avoids punishing GA for chords that are impossible anywhere.
    """
    if table is None or "acceptable_chord_ids" not in table.columns:
        return 0
    s: set[int] = set()
    for xs in table["acceptable_chord_ids"].tolist():
        if not xs:
            continue
        for cid in xs:
            s.add(int(cid))
    return len(s)


@dataclass(frozen=True)
class PrimaryWeights:
    # region terms
    w_overlap: float = 1.0
    w_dist: float = 0.0
    w_dist_tonal: float = 0.0
    w_repeat: float = 0.0
    w_pc_change: float = 0.0

    # global terms
    w_coverage: float = 0.0


class FitnessPrimaryBase:
    """
    Helper base:
      - caches chord facts by chord_id
      - precomputes region PC histograms
      - optional distance matrices
      - coverage denominator (choice 1)

    IMPORTANT: assumes chord IDs are contiguous 0..N-1.
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
        self.table = harmonization_table
        self.H = len(harmonization_table)

        # --------------------------
        # Weights
        # --------------------------
        w = weights or {}
        self.w = PrimaryWeights(
            w_overlap=float(w.get("w_overlap", 1.0)),
            w_dist=float(w.get("w_dist", 0.0)),
            w_dist_tonal=float(w.get("w_dist_tonal", 0.0)),
            w_repeat=float(w.get("w_repeat", 0.0)),
            w_pc_change=float(w.get("w_pc_change", 0.0)),
            w_coverage=float(w.get("w_coverage", 0.0)),
        )

        # --------------------------
        # Chord data by id (prefer registry; fallback to table)
        # --------------------------
        if MASTER_CHORD_REGISTRY is not None:
            # Registry is already contiguous and precomputed
            self.N = int(MASTER_CHORD_REGISTRY.N)
            self.pitch_classes_by_id = MASTER_CHORD_REGISTRY.pcs_by_id
            self.chord_mask_by_id = MASTER_CHORD_REGISTRY.mask_by_id
            self.quality_by_id = MASTER_CHORD_REGISTRY.quality_by_id
        else:
            # Fallback: build from master_table (slower)
            ids = master_table["id"].to_numpy(dtype=int)
            max_id = int(ids.max()) if len(ids) else -1
            N = max_id + 1

            if len(ids) != N or set(ids.tolist()) != set(range(N)):
                raise ValueError(
                    "Chord IDs must be contiguous 0..N-1 for direct array indexing.\n"
                    f"Observed: len(ids)={len(ids)} max_id={max_id}.\n"
                    "Fix by rebuilding MASTER_CHORD_TABLE ids contiguously."
                )

            self.N = N

            self.pitch_classes_by_id: list[list[int]] = [[] for _ in range(self.N)]
            self.chord_mask_by_id = np.zeros(self.N, dtype=np.int32)
            self.quality_by_id: list[str] = [""] * self.N

            for _, row in master_table.iterrows():
                cid = int(row["id"])
                pcs = list(row["pitch_classes"])
                self.pitch_classes_by_id[cid] = pcs
                self.chord_mask_by_id[cid] = _pcs_to_mask(pcs)
                self.quality_by_id[cid] = str(row["quality"])

        # --------------------------
        # Distance matrices (optional)
        # --------------------------
        self.D = chord_dist
        self.DT = chord_tonal_dist

        # --------------------------
        # Region pitch-class weights (H x 12)
        # --------------------------
        self.region_pc_weight = np.zeros((self.H, 12), dtype=np.float32)
        if self.H and "region_note_objs" in self.table.columns:
            region_notes_col = self.table["region_note_objs"].tolist()
            for i, notes in enumerate(region_notes_col):
                if not notes:
                    continue
                for n in notes:
                    if isinstance(n, m21note.Note):
                        pc = int(n.pitch.pitchClass) % 12
                        bs = float(getattr(n, "beatStrength", 1.0))
                        self.region_pc_weight[i, pc] += bs

        # --------------------------
        # Coverage denominator (choice 1)
        # --------------------------
        self.coverage_pool_size = _infer_pool_size_from_table(self.table)

    # --------------------------
    # Shared region helpers
    # --------------------------
    def _overlap_reward(self, chord_id: int, region_idx: int) -> float:
        if self.w.w_overlap == 0.0:
            return 0.0
        pcs = self.pitch_classes_by_id[int(chord_id)]
        if not pcs:
            return 0.0
        return float(sum(self.region_pc_weight[region_idx, int(pc) % 12] for pc in pcs))

    def _distance_penalty(self, prev_id: int | None, cur_id: int) -> float:
        if self.w.w_dist == 0.0 or self.D is None or prev_id is None:
            return 0.0
        return float(self.D[int(prev_id), int(cur_id)])

    def _tonal_distance_penalty(self, prev_id: int | None, cur_id: int) -> float:
        if self.w.w_dist_tonal == 0.0 or self.DT is None or prev_id is None:
            return 0.0
        return float(self.DT[int(prev_id), int(cur_id)])

    def _repeat_penalty(self, prev_id: int | None, cur_id: int) -> float:
        if self.w.w_repeat == 0.0 or prev_id is None:
            return 0.0
        return 1.0 if int(prev_id) == int(cur_id) else 0.0

    def _pc_change_bonus(self, prev_id: int | None, cur_id: int) -> float:
        if self.w.w_pc_change == 0.0 or prev_id is None:
            return 0.0
        m_prev = int(self.chord_mask_by_id[int(prev_id)])
        m_cur = int(self.chord_mask_by_id[int(cur_id)])
        removed = (m_prev & ~m_cur).bit_count()
        added = (m_cur & ~m_prev).bit_count()
        return float(max(removed, added))

    # --------------------------
    # Global helper
    # --------------------------
    def _coverage_reward(self, genes: list[int]) -> float:
        if self.w.w_coverage == 0.0:
            return 0.0
        denom = int(self.coverage_pool_size)
        if denom <= 0:
            return 0.0
        used = len(set(int(x) for x in genes))
        return float(used / denom)
