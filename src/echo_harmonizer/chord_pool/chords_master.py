# chords_master.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from functools import lru_cache
from typing import Dict, List

import numpy as np
import pandas as pd
from music21 import pitch as m21pitch, chord as m21chord, interval


# ============================================================
# Canonical roots (includes enharmonics)
# ============================================================

ROOT_NAMES = [
    "C",  "B#",
    "C#", "D-",
    "D",
    "D#", "E-",
    "E",  "F-",
    "F",  "E#",
    "F#", "G-",
    "G",
    "G#", "A-",
    "A",
    "A#", "B-",
    "B",  "C-",
]

def _norm_name(name: str) -> str:
    """music21 uses '-' for flats (E-). Normalize to 'b' style (Eb)."""
    return name.replace("-", "b")


# ============================================================
# Chord formulas (interval names for spelling)
# NOTE: For it6 we keep order: b6 - 1 - #4 (do NOT invert)
# ============================================================

CHORD_INTERVALS: Dict[str, List[str]] = {
    # triads
    "maj":  ["P1", "M3", "P5"],
    "min":  ["P1", "m3", "P5"],
    "sus2": ["P1", "M2", "P5"],
    "sus4": ["P1", "P4", "P5"],
    "aug":  ["P1", "M3", "A5"],
    "dim":  ["P1", "m3", "d5"],

    # sevenths
    "maj7":     ["P1", "M3", "P5", "M7"],
    "7":        ["P1", "M3", "P5", "m7"],
    "min7":     ["P1", "m3", "P5", "m7"],
    "halfdim7": ["P1", "m3", "d5", "m7"],
    "dim7":     ["P1", "m3", "d5", "d7"],

    # Italian Augmented 6th (ANCHOR = tonic)
    # Keep stored note order: b6, 1, #4
    "it6":      ["m6", "P1", "A4"],
}

SYMBOL_SUFFIX: Dict[str, str] = {
    # triads
    "maj":  "",
    "min":  "m",
    "sus2": "sus2",
    "sus4": "sus4",
    "aug":  "aug",
    "dim":  "dim",

    # sevenths
    "maj7":     "maj7",
    "7":        "7",
    "min7":     "m7",
    "halfdim7": "m7b5",
    "dim7":     "dim7",

    # it6
    "it6":      "It6",
}

SEVENTH_QUALITIES = {"maj7", "7", "min7", "halfdim7", "dim7"}


# ============================================================
# Cache paths
# ============================================================

CACHE_DIR = Path(__file__).resolve().parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

TABLE_PKL_PATH = CACHE_DIR / "MASTER_CHORD_TABLE.pkl"
REGISTRY_NPZ_PATH = CACHE_DIR / "MASTER_CHORD_REGISTRY.npz"

# ============================================================
# Registry (fast runtime view)
# ============================================================

@dataclass(frozen=True)
class ChordRegistry:
    N: int
    pcs_by_id: list[list[int]]        # Python lists (ragged)
    mask_by_id: np.ndarray            # (N,) int32 bitmask to represent pcs
    root_pc_by_id: np.ndarray         # (N,) int16
    quality_by_id: list[str]          # (N,) str
    is_seventh_by_id: np.ndarray      # (N,) bool


def _pcs_to_mask(pcs: list[int]) -> int:
    m = 0
    for pc in pcs:
        m |= 1 << (int(pc) % 12)
    return m


# ============================================================
# Build MASTER_CHORD_TABLE
# ============================================================

def build_master_chord_table() -> pd.DataFrame:
    """
    Returns DataFrame with columns:
      - id: int (0..N-1 contiguous)
      - root: str
      - root_pc: int
      - quality: str
      - is_seventh: bool
      - symbol: str
      - pitch_names: list[str]         (keeps the defined interval order)
      - pitch_classes: list[int]       (keeps the defined interval order)
      - pcs_tuple: tuple[int,...]      (sorted unique; for set-ish lookups)
    """
    rows = []
    cid = 0

    for root_name in ROOT_NAMES:
        root_pitch = m21pitch.Pitch(root_name)
        root_pc = int(root_pitch.pitchClass) % 12

        for quality, int_names in CHORD_INTERVALS.items():
            chord_pitches = [interval.Interval(iv).transposePitch(root_pitch) for iv in int_names]

            pitch_names = [_norm_name(p.name) for p in chord_pitches]
            pitch_classes = [int(p.pitchClass) % 12 for p in chord_pitches]
            pcs_tuple = tuple(sorted(set(pitch_classes)))

            rows.append({
                "id": cid,
                "root": _norm_name(root_name),
                "root_pc": int(root_pc),
                "quality": quality,
                "is_seventh": bool(quality in SEVENTH_QUALITIES),
                "symbol": f"{_norm_name(root_name)}{SYMBOL_SUFFIX[quality]}",
                "pitch_names": pitch_names,
                "pitch_classes": pitch_classes,
                "pcs_tuple": pcs_tuple,
            })
            cid += 1

    return pd.DataFrame(rows)


def load_or_build_master_table(force_rebuild: bool = False) -> pd.DataFrame:
    if (not force_rebuild) and TABLE_PKL_PATH.exists():
        df = pd.read_pickle(TABLE_PKL_PATH)
        return df

    df = build_master_chord_table()
    df.to_pickle(TABLE_PKL_PATH)
    df.to_csv(TABLE_PKL_PATH.with_suffix(".csv"), index=False)
    return df


# ============================================================
# Build / load registry cache
# ============================================================

def build_chord_registry(df: pd.DataFrame) -> ChordRegistry:
    ids = df["id"].to_numpy(dtype=int)
    max_id = int(ids.max()) if len(ids) else -1

    # Enforce contiguity for direct indexing everywhere
    if max_id + 1 != len(ids) or set(ids.tolist()) != set(range(max_id + 1)):
        raise ValueError("MASTER_CHORD_TABLE ids must be contiguous 0..N-1.")

    N = max_id + 1

    pcs_by_id: list[list[int]] = [None] * N  # type: ignore
    mask_by_id = np.zeros(N, dtype=np.int32)
    root_pc_by_id = np.zeros(N, dtype=np.int16)
    quality_by_id: list[str] = [""] * N
    is_seventh_by_id = np.zeros(N, dtype=bool)

    for _, r in df.iterrows():
        cid = int(r["id"])
        pcs = list(r["pitch_classes"])
        pcs_by_id[cid] = pcs
        mask_by_id[cid] = np.int32(_pcs_to_mask(pcs))
        root_pc_by_id[cid] = np.int16(int(r["root_pc"]) % 12)
        quality_by_id[cid] = str(r["quality"])
        is_seventh_by_id[cid] = bool(r["is_seventh"])

    return ChordRegistry(
        N=N,
        pcs_by_id=pcs_by_id,
        mask_by_id=mask_by_id,
        root_pc_by_id=root_pc_by_id,
        quality_by_id=quality_by_id,
        is_seventh_by_id=is_seventh_by_id,
    )


def _save_registry_npz(reg: ChordRegistry) -> None:
    """
    Save arrays to a single .npz.
    pcs_by_id is ragged -> store as object array of small int arrays.
    """
    pcs_obj = np.empty(reg.N, dtype=object)
    for i in range(reg.N):
        pcs_obj[i] = np.array(reg.pcs_by_id[i], dtype=np.int16)

    qual_obj = np.array(reg.quality_by_id, dtype=object)

    np.savez_compressed(
        REGISTRY_NPZ_PATH,
        N=np.array([reg.N], dtype=np.int32),
        pcs_by_id=pcs_obj,
        mask_by_id=reg.mask_by_id.astype(np.int32),
        root_pc_by_id=reg.root_pc_by_id.astype(np.int16),
        quality_by_id=qual_obj,
        is_seventh_by_id=reg.is_seventh_by_id.astype(np.bool_),
    )


def _load_registry_npz() -> ChordRegistry:
    z = np.load(REGISTRY_NPZ_PATH, allow_pickle=True)
    N = int(z["N"][0])

    pcs_obj = z["pcs_by_id"]
    pcs_by_id: list[list[int]] = []
    for i in range(N):
        pcs_by_id.append(list(np.array(pcs_obj[i], dtype=np.int16).tolist()))

    quality_by_id = [str(x) for x in z["quality_by_id"].tolist()]

    return ChordRegistry(
        N=N,
        pcs_by_id=pcs_by_id,
        mask_by_id=np.array(z["mask_by_id"], dtype=np.int32),
        root_pc_by_id=np.array(z["root_pc_by_id"], dtype=np.int16),
        quality_by_id=quality_by_id,
        is_seventh_by_id=np.array(z["is_seventh_by_id"], dtype=bool),
    )


def load_or_build_registry(df: pd.DataFrame, force_rebuild: bool = False) -> ChordRegistry:
    if (not force_rebuild) and REGISTRY_NPZ_PATH.exists():
        reg = _load_registry_npz()
        # rebuild if mismatch (e.g., new quality added changes N)
        if reg.N == len(df):
            return reg

    reg = build_chord_registry(df)
    _save_registry_npz(reg)
    return reg


# ============================================================
# Optional: chord object on demand (not persisted)
# ============================================================

@lru_cache(maxsize=2048)
def chord_obj_from_pitch_names(pitch_names_tuple: tuple[str, ...]) -> m21chord.Chord:
    pitches = [m21pitch.Pitch(n) for n in pitch_names_tuple]
    return m21chord.Chord(pitches)


def get_chord_obj(chord_id: int) -> m21chord.Chord:
    row = CHORD_BY_ID.loc[int(chord_id)]
    return chord_obj_from_pitch_names(tuple(row["pitch_names"]))


# ============================================================
# Public globals
# ============================================================

MASTER_CHORD_TABLE = load_or_build_master_table()

CHORD_BY_ID = MASTER_CHORD_TABLE.set_index("id")

ID_BY_SYMBOL = {str(r.symbol): int(r.id) for _, r in MASTER_CHORD_TABLE.iterrows()}
ID_BY_ROOT_QUALITY = {(str(r.root), str(r.quality)): int(r.id) for _, r in MASTER_CHORD_TABLE.iterrows()}

from collections import defaultdict

IDS_BY_PCS_TUPLE = defaultdict(list)
for _, r in MASTER_CHORD_TABLE.iterrows():
    IDS_BY_PCS_TUPLE[tuple(r.pcs_tuple)].append(int(r.id))
IDS_BY_PCS_TUPLE = dict(IDS_BY_PCS_TUPLE)

MASTER_CHORD_REGISTRY = load_or_build_registry(MASTER_CHORD_TABLE)


# %% Debug / test code
if __name__ == "__main__":
    from IPython.display import display
    print("N chords:", len(MASTER_CHORD_TABLE))
    display(MASTER_CHORD_TABLE.tail(50))
    print("Registry N:", MASTER_CHORD_REGISTRY.N)

# %%