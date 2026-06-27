# chord_dist.py
from __future__ import annotations
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd


### Cache path for distance matrices
CACHE_DIR = Path(__file__).resolve().parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

DIST_NPY_PATH  = CACHE_DIR / "CHORD_DIST_MATRIX.npy"
TONAL_DIST_NPY_PATH = CACHE_DIR / "CHORD_TONAL_DIST_MATRIX.npy"


### [DISTANCE FUNCTION 1]: Custom stepwise distance
def chord_distance_steps(pcs1: List[int], pcs2: List[int]) -> float:
    if not pcs1 or not pcs2:
        return 0.0
    total = 0.0
    for pc1 in pcs1:
        best = min(min((pc2 - pc1) % 12, (pc1 - pc2) % 12) for pc2 in pcs2)
        total += best
    return float(total / 2.0)


def load_or_build_distance_matrix(df: pd.DataFrame, force_rebuild: bool = False) -> np.ndarray:
    if (not force_rebuild) and DIST_NPY_PATH.exists():
        D = np.load(DIST_NPY_PATH)
        if D.shape == (len(df), len(df)):
            return D

    pcs_list = df["pitch_classes"].tolist()
    n = len(pcs_list)
    D = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        for j in range(n):
            D[i, j] = chord_distance_steps(pcs_list[i], pcs_list[j])

    np.save(DIST_NPY_PATH, D)
    return D

### [DISTANCE FUNCTION 2]: Tonal distance (not used currently)
def _harte_phi_matrix(r1: float = 1.0, r2: float = 1.0, r3: float = 0.5) -> np.ndarray:
    """
    Φ is a 6x12 matrix. Column k is ?_k as defined by Harte et al. (2006).
    Angles:
      - fifths:       k * 7?/6
      - minor thirds: k * 3?/2
      - major thirds: k * 2?/3

    Arguments:
        - r1, r2, r3: scaling factors for each interval class
    """
    l = np.arange(12, dtype=np.float32)
    Phi = np.zeros((6, 12), dtype=np.float32)

    # Circle of 'fifths' (7 semitones)
    Phi[0, :] = r1 * np.sin(l * (7.0 * np.pi / 6.0))
    Phi[1, :] = r1 * np.cos(l * (7.0 * np.pi / 6.0))

    # Circle of 'minor thirds' (3 semitones)
    Phi[2, :] = r2 * np.sin(l * (3.0 * np.pi / 2.0))
    Phi[3, :] = r2 * np.cos(l * (3.0 * np.pi / 2.0))

    # Circle of 'major thirds' (4 semitones)
    Phi[4, :] = r3 * np.sin(l * (2.0 * np.pi / 3.0))
    Phi[5, :] = r3 * np.cos(l * (2.0 * np.pi / 3.0))

    return Phi

_PHI = _harte_phi_matrix()

def tonal_centroid_from_pcs(pcs: list[int]) -> np.ndarray:
    """
    Build a 12D chroma vector c from pcs (binary), L1-normalize, then ζ = (1/||c||1) Φ c.
    """
    c = np.zeros(12, dtype=np.float32)
    for pc in pcs:
        c[int(pc) % 12] = 1.0

    norm1 = float(np.sum(c))
    if norm1 <= 0:
        return np.zeros(6, dtype=np.float32)

    return (_PHI @ c) / norm1  # ζ in R^6

def load_or_build_tonal_distance_matrix(df: pd.DataFrame, force_rebuild: bool = False) -> np.ndarray:
    """
    D_tonal[i, j] = ||ζ_i - ζ_j||_2 where ζ is Harte tonal centroid for the chord pcs.
    """
    if (not force_rebuild) and TONAL_DIST_NPY_PATH.exists():
        return np.load(TONAL_DIST_NPY_PATH)

    pcs_list = df["pitch_classes"].tolist()
    Z = np.stack([tonal_centroid_from_pcs(pcs) for pcs in pcs_list], axis=0)  # (N,6)

    # pairwise euclidean distance
    # (N,1,6) - (1,N,6) -> (N,N,6) -> norm -> (N,N)
    diff = Z[:, None, :] - Z[None, :, :]
    D = np.sqrt(np.sum(diff * diff, axis=2)).astype(np.float32)

    np.save(TONAL_DIST_NPY_PATH, D)
    return D