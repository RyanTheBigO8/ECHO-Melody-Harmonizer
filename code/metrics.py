# metrics.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from music21 import note as m21note

from chord_pool.chords_master import MASTER_CHORD_REGISTRY


# ============================================================
# Region indexer: map absolute time -> region index
# ============================================================

def make_region_indexer(table: pd.DataFrame):
    starts = table["region_start"].to_numpy(dtype=float)
    ends = table["region_end"].to_numpy(dtype=float)

    def region_idx(t: float) -> int:
        i = int(np.searchsorted(starts, t, side="right") - 1)
        if i < 0:
            return 0
        if i >= len(starts):
            return len(starts) - 1
        while i + 1 < len(starts) and t >= ends[i]:
            i += 1
        return i

    return region_idx


# ============================================================
# Melody extraction utilities
# ============================================================

@dataclass(frozen=True)
class MelodyArrays:
    starts: np.ndarray  # float
    ends: np.ndarray    # float
    pcs: np.ndarray     # int16
    midis: np.ndarray   # int16
    durs: np.ndarray    # float


def extract_melody_arrays(melody_stream) -> MelodyArrays:
    notes: list[m21note.Note] = list(melody_stream.flatten().notes)
    if not notes:
        zf = np.array([], dtype=float)
        zi = np.array([], dtype=np.int16)
        return MelodyArrays(zf, zf, zi, zi, zf)

    starts = np.array([float(n.offset) for n in notes], dtype=float)
    ends   = np.array([float(n.offset + n.duration.quarterLength) for n in notes], dtype=float)
    pcs    = np.array([int(n.pitch.pitchClass) for n in notes], dtype=np.int16)
    midis  = np.array([int(n.pitch.midi) for n in notes], dtype=np.int16)
    durs   = np.array([float(n.duration.quarterLength) for n in notes], dtype=float)
    return MelodyArrays(starts, ends, pcs, midis, durs)


def active_pc_at_time(arr: MelodyArrays, t: float) -> Optional[int]:
    if len(arr.starts) == 0:
        return None
    i = int(np.searchsorted(arr.starts, t, side="right") - 1)
    if i < 0:
        return None
    if t < arr.ends[i]:
        return int(arr.pcs[i])
    return None


# ============================================================
# Harte tonal centroid (for MCTD)
# ============================================================

def harte_phi_matrix(r1: float = 1.0, r2: float = 1.0, r3: float = 0.5) -> np.ndarray:
    l = np.arange(12, dtype=np.float32)
    Phi = np.zeros((6, 12), dtype=np.float32)

    Phi[0, :] = r1 * np.sin(l * (7.0 * np.pi / 6.0))
    Phi[1, :] = r1 * np.cos(l * (7.0 * np.pi / 6.0))

    Phi[2, :] = r2 * np.sin(l * (3.0 * np.pi / 2.0))
    Phi[3, :] = r2 * np.cos(l * (3.0 * np.pi / 2.0))

    Phi[4, :] = r3 * np.sin(l * (2.0 * np.pi / 3.0))
    Phi[5, :] = r3 * np.cos(l * (2.0 * np.pi / 3.0))

    return Phi


_PHI = harte_phi_matrix()


def tonal_centroid_from_pcs(pcs: Iterable[int]) -> np.ndarray:
    c = np.zeros(12, dtype=np.float32)
    for pc in pcs:
        c[int(pc) % 12] = 1.0
    norm1 = float(np.sum(c))
    if norm1 <= 0:
        return np.zeros(6, dtype=np.float32)
    return (_PHI @ c) / norm1


def tonal_centroid_for_note_pc(pc: int) -> np.ndarray:
    return _PHI[:, int(pc) % 12].astype(np.float32)


# ============================================================
# Yeh-style metrics: CTnCTR, PCS, MCTD
# ============================================================

_PCS_POS = {0, 3, 4, 7, 8, 9}
_PCS_ZERO = {5}

def genes_to_pcs_seq(genes: list[int]) -> list[tuple[int, ...]]:
    pcs_by_id = MASTER_CHORD_REGISTRY.pcs_by_id
    return [tuple(int(x) % 12 for x in pcs_by_id[int(cid)]) for cid in genes]


def compute_CTnCTR_pcs(
    chords_pcs: list[Optional[tuple[int, ...]]],
    table: pd.DataFrame,
    melody_stream,
) -> float:
    ridx = make_region_indexer(table)
    arr = extract_melody_arrays(melody_stream)
    if len(arr.starts) == 0:
        return 1.0

    nc = 0
    nn = 0
    np_proper = 0

    for i in range(len(arr.starts)):
        t0 = float(arr.starts[i])
        pc = int(arr.pcs[i])

        cpcs0 = chords_pcs[ridx(t0)] if chords_pcs else None
        chord0 = set(int(x) % 12 for x in (cpcs0 or ()))

        if pc in chord0:
            nc += 1
            continue

        nn += 1
        if i + 1 >= len(arr.starts):
            continue

        # step-wise?
        if abs(int(arr.midis[i + 1]) - int(arr.midis[i])) > 2:
            continue

        # resolves *to a chord tone* at the next note onset time
        t1 = float(arr.starts[i + 1])
        pc1 = int(arr.pcs[i + 1])

        cpcs1 = chords_pcs[ridx(t1)] if chords_pcs else None
        chord1 = set(int(x) % 12 for x in (cpcs1 or ()))

        if pc1 in chord1:
            np_proper += 1

    denom = nc + nn
    return float((nc + np_proper) / denom) if denom > 0 else 1.0


def compute_PCS_pcs(
    chords_pcs: list[Optional[tuple[int, ...]]],
    table: pd.DataFrame,
    melody_stream,
    *,
    step_qL: float = 0.25,
) -> float:
    ridx = make_region_indexer(table)
    arr = extract_melody_arrays(melody_stream)
    if len(arr.starts) == 0:
        return 0.0

    t_end = float(np.max(arr.ends)) if len(arr.ends) else 0.0
    if t_end <= 0:
        return 0.0

    H = len(chords_pcs)
    total = 0.0
    n = 0
    t = 0.0
    half = 0.5 * step_qL

    while t < t_end:
        tm = t + half
        mel_pc = active_pc_at_time(arr, tm)
        if mel_pc is not None:
            cpcs = chords_pcs[ridx(tm)] if chords_pcs else None
            chord_pcs = (cpcs or ())

            s = 0
            for cpc in chord_pcs:
                interval = (int(mel_pc) - int(cpc)) % 12
                if interval in _PCS_POS:
                    s += 1
                elif interval in _PCS_ZERO:
                    s += 0
                else:
                    s -= 1

            if H > 0:
                total += float(s) / float(H)
            n += 1
        t += step_qL

    return float(total / n) if n > 0 else 0.0


def compute_MCTD_pcs(chords_pcs: list[Optional[tuple[int, ...]]], table: pd.DataFrame, melody_stream) -> float:
    ridx = make_region_indexer(table)
    arr = extract_melody_arrays(melody_stream)
    if len(arr.starts) == 0:
        return 0.0

    # cache centroids for unique chords
    centroid_cache: dict[tuple[int, ...], np.ndarray] = {}

    num = 0.0
    den = 0.0

    for i in range(len(arr.starts)):
        w = float(arr.durs[i])
        if w <= 0:
            continue

        t0 = float(arr.starts[i])
        pc = int(arr.pcs[i])

        cpcs = chords_pcs[ridx(t0)] if chords_pcs else None
        chord_key = tuple(sorted(set(int(x) % 12 for x in (cpcs or ()))))

        if chord_key not in centroid_cache:
            centroid_cache[chord_key] = tonal_centroid_from_pcs(chord_key)

        z_note = tonal_centroid_for_note_pc(pc)
        z_chord = centroid_cache[chord_key]
        dist = float(np.linalg.norm(z_note - z_chord))

        num += w * dist
        den += w

    return float(num / den) if den > 0 else 0.0


# ============================================================
# Additional Wrappers
# ============================================================

def compute_CTnCTR(genes: list[int], table: pd.DataFrame, melody_stream) -> float:
    return compute_CTnCTR_pcs(genes_to_pcs_seq(genes), table, melody_stream)
    

def compute_PCS(
    genes: list[int],
    table: pd.DataFrame,
    melody_stream,
    *,
    step_qL: float = 0.25,
) -> float:
    return compute_PCS_pcs(genes_to_pcs_seq(genes), table, melody_stream, step_qL=step_qL)


def compute_MCTD(genes: list[int], table: pd.DataFrame, melody_stream) -> float:
    return compute_MCTD_pcs(genes_to_pcs_seq(genes), table, melody_stream)


def evaluate_solution_metrics(genes: list[int], table: pd.DataFrame, melody_stream) -> dict[str, float]:
    return {
        "CTnCTR": compute_CTnCTR(genes, table, melody_stream),
        "PCS": compute_PCS(genes, table, melody_stream),
        "MCTD": compute_MCTD(genes, table, melody_stream),
    }
