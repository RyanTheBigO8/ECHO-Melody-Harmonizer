# chord_roles.py
from __future__ import annotations
from typing import Optional
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from music21 import pitch as m21pitch, interval as m21interval, scale as m21scale

from echo_harmonizer.key_context import KeyContext

# ----------------------------
# Cache path for role mask table
# ----------------------------

CACHE_DIR = Path(__file__).resolve().parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

ROLE_MASK_TABLE_PKL_PATH = CACHE_DIR / "ROLE_MASK_TABLE.pkl"

# ----------------------------
# Role vocabulary
# ----------------------------

def build_role_vocab() -> tuple[list[str], dict[str, int]]:
    names: list[str] = []

    for d in range(1, 8): names.append(f"diat_triad_{d}")
    for d in range(1, 8): names.append(f"diat_7th_{d}")
    for d in range(1, 8): names.append(f"parallel_triad_{d}")
    for d in range(1, 8): names.append(f"parallel_7th_{d}")

    for d in range(1, 8): names.append(f"V/{d}")
    for d in range(1, 8): names.append(f"V7/{d}")
    for d in range(1, 8): names.append(f"vii/{d}")
    for d in range(1, 8): names.append(f"vii7/{d}")

    for d in range(1, 8): names.append(f"sus2/{d}")
    for d in range(1, 8): names.append(f"sus4/{d}")

    for d in range(1, 8): names.append(f"CM_M3-/{d}")
    for d in range(1, 8): names.append(f"CM_m3-/{d}")
    for d in range(1, 8): names.append(f"CM_m3+/{d}")
    for d in range(1, 8): names.append(f"CM_M3+/{d}")

    names.append("N6")
    names.append("It6")

    rid = {n: i for i, n in enumerate(names)}
    return names, rid

ROLE_NAMES, ROLE_ID = build_role_vocab()

def bit(role_name: str) -> int:
    return 1 << ROLE_ID[role_name]

def decode_mask(mask: int) -> list[str]:
    out: list[str] = []
    m = int(mask)
    while m:
        lsb = m & -m
        idx = lsb.bit_length() - 1
        if 0 <= idx < len(ROLE_NAMES):
            out.append(ROLE_NAMES[idx])
        m ^= lsb
    return out

ROLE_TABLE = pd.DataFrame({
    "role_id": list(range(len(ROLE_NAMES))),
    "role_name": ROLE_NAMES,
})

# ----------------------------
# Role groups (single source of truth)
# ----------------------------

GROUP_MASKS: dict[str, int] = {}

m = 0
for d in range(1, 8):
    m |= bit(f"diat_triad_{d}")
GROUP_MASKS["diat_triads"] = m

m = 0
for d in range(1, 8):
    m |= bit(f"diat_7th_{d}")
GROUP_MASKS["diat_7ths"] = m

m = 0
for d in range(1, 8):
    m |= bit(f"parallel_triad_{d}")
GROUP_MASKS["parallel_triads"] = m

m = 0
for d in range(1, 8):
    m |= bit(f"parallel_7th_{d}")
GROUP_MASKS["parallel_7ths"] = m

m = 0
for d in range(1, 8):
    m |= bit(f"V/{d}")
    m |= bit(f"V7/{d}")
    m |= bit(f"vii/{d}")
    m |= bit(f"vii7/{d}")
GROUP_MASKS["secondary_dominants"] = m

m = 0
for d in range(1, 8):
    m |= bit(f"sus2/{d}")
    m |= bit(f"sus4/{d}")
GROUP_MASKS["sus_chords"] = m

m = 0
for d in range(1, 8):
    m |= bit(f"CM_M3-/{d}")
    m |= bit(f"CM_m3-/{d}")
    m |= bit(f"CM_m3+/{d}")
    m |= bit(f"CM_M3+/{d}")
GROUP_MASKS["chromatic_mediants"] = m

GROUP_MASKS["neapolitan"] = bit("N6")
GROUP_MASKS["it6"] = bit("It6")

def group_mask(name: str) -> int:
    return int(GROUP_MASKS.get(name, 0))

def allowed_mask_from_groups(enabled_groups: list[str] | set[str] | tuple[str, ...]) -> int:
    m = 0
    for g in enabled_groups:
        m |= group_mask(str(g))
    return int(m)

# ----------------------------
# Canonical tonic spellings per (tonic_pc, mode)
# NOTE: ROLE_MASK_TABLE columns are (tonic_pc, mode), so enharmonic keys are merged.
# We pick one canonical spelling for each pc per mode here.
# ----------------------------

_CANON_TONIC_MAJOR: dict[int, str] = {
    0: "C",
    1: "Db",
    2: "D",
    3: "Eb",
    4: "E",
    5: "F",
    6: "Gb",
    7: "G",
    8: "Ab",
    9: "A",
    10: "Bb",
    11: "B",
}

_CANON_TONIC_MINOR: dict[int, str] = {
    0: "C",
    1: "C#",
    2: "D",
    3: "Eb",
    4: "E",
    5: "F",
    6: "F#",
    7: "G",
    8: "G#",
    9: "A",
    10: "Bb",
    11: "B",
}

def _canon_tonic_name(tonic_pc: int, mode: str) -> str:
    tonic_pc = int(tonic_pc) % 12
    if mode == "major":
        return _CANON_TONIC_MAJOR[tonic_pc]
    if mode == "minor":
        return _CANON_TONIC_MINOR[tonic_pc]
    raise ValueError(f"Invalid mode: {mode}")

def _norm_name(name: str) -> str:
    """music21 uses '-' for flats (E-). Normalize to 'b' style (Eb)."""
    return name.replace("-", "b")

# ----------------------------
# Small helpers
# ----------------------------

def _pcs_set(pitch_classes: list[int]) -> frozenset[int]:
    return frozenset(int(x) % 12 for x in pitch_classes)

def _has_nondiatonic(pcs: frozenset[int], diat_set: frozenset[int]) -> bool:
    return any(pc not in diat_set for pc in pcs)

def _is_subset(pcs: frozenset[int], allowed: frozenset[int]) -> bool:
    return all(pc in allowed for pc in pcs)

def _scale_pitches(tonic_name: str, mode: str, *, natural_minor: bool = True) -> list[m21pitch.Pitch]:
    tonic = m21pitch.Pitch(str(tonic_name))

    if mode == "major":
        sc = m21scale.MajorScale(tonic)
    elif mode == "minor":
        sc = m21scale.MinorScale(tonic) if natural_minor else m21scale.HarmonicMinorScale(tonic)
    else:
        raise ValueError(f"Invalid mode: {mode}")

    deg7 = []
    for d in range(1, 8):
        p = sc.pitchFromDegree(d)
        deg7.append(m21pitch.Pitch(p.nameWithOctave))
    return deg7

def _parallel_mode(mode: str) -> str:
    return "minor" if mode == "major" else "major"

def _chord_tones_from_scale(deg_pitches: list[m21pitch.Pitch], degree: int, size: int) -> list[m21pitch.Pitch]:
    """
    Triad: degrees 1,3,5 of the scale built on `degree`.
    7th:   degrees 1,3,5,7 of the scale built on `degree`.
    Uses modulo degree indexing (pitch classes only matter).
    """
    i = (int(degree) - 1) % 7
    idxs = [i, (i + 2) % 7, (i + 4) % 7] if size == 3 else [i, (i + 2) % 7, (i + 4) % 7, (i + 6) % 7]
    return [deg_pitches[j] for j in idxs]

# ----------------------------
# Build role masks for ONE key (deduped by canonical spelling)
# ----------------------------

def build_role_masks(master_table: pd.DataFrame, kc: KeyContext) -> list[int]:
    """
    Compute role masks for ONE key context, but assign each role ONLY to the
    chord ID whose ROOT SPELLING matches the canonical spelling for this key/role.
    Enharmonic duplicates remain unassigned => they won't enter non-atonal pools.
    """
    # ----- size / contiguity -----
    ids = master_table["id"].to_numpy(dtype=int)
    max_id = int(ids.max()) if len(ids) else -1
    N = max_id + 1

    # ----- fast lookup tables from master_table -----
    # (root, quality) -> chord_id
    id_by_root_quality: dict[tuple[str, str], int] = {}
    # chord_id -> pcs_set, quality, root, root_pc
    pcs_by_id: list[frozenset[int]] = [frozenset()] * N
    quality_by_id: list[str] = [""] * N
    root_by_id: list[str] = [""] * N
    root_pc_by_id = np.zeros(N, dtype=np.int16)

    for _, r in master_table.iterrows():
        cid = int(r["id"])
        root = str(r["root"])
        q = str(r["quality"])
        id_by_root_quality[(root, q)] = cid
        pcs_by_id[cid] = _pcs_set(r["pitch_classes"])
        quality_by_id[cid] = q
        root_by_id[cid] = root
        root_pc_by_id[cid] = np.int16(int(r["root_pc"]) % 12)

    def _get_id(root_name: str, quality: str) -> int | None:
        return id_by_root_quality.get((_norm_name(root_name), quality))

    masks = [0] * N

    # canonical tonic spelling for this (tonic_pc, mode)
    tonic_name = _canon_tonic_name(kc.tonic_pc, kc.mode)

    # main / parallel scales in correct spelling
    main_deg = _scale_pitches(tonic_name, kc.mode)  # 7 pitches
    par_mode = _parallel_mode(kc.mode)
    par_deg = _scale_pitches(tonic_name, par_mode)

    main_deg_name = {d: _norm_name(main_deg[d - 1].name) for d in range(1, 8)}
    par_deg_name  = {d: _norm_name(par_deg[d - 1].name)  for d in range(1, 8)}

    # ----------------------------
    # (A) Diatonic triads / 7ths (exact degrees, correct spelling)
    # ----------------------------
    diat_triad_ids: set[int] = set()
    par_triad_ids: set[int] = set()

    for d in range(1, 8):
        root = main_deg_name[d]

        # diat triad: find chord by pcs-set via matching root+quality in master
        # easiest: try each triad quality; only one will exist with this root+pcs-set
        # (because master_table includes fixed qualities)
        triad_candidates = ("maj", "min", "dim", "aug", "sus2", "sus4")
        triad_tones = _chord_tones_from_scale(main_deg, d, size=3)
        triad_pcs = frozenset(int(p.pitchClass) % 12 for p in triad_tones)

        triad_id: int | None = None
        for q in triad_candidates:
            cid = _get_id(root, q)
            if cid is None:
                continue
            if pcs_by_id[cid] == triad_pcs:
                triad_id = cid
                break

        if triad_id is not None:
            masks[triad_id] |= bit(f"diat_triad_{d}")
            diat_triad_ids.add(triad_id)

        # diat 7th
        seventh_candidates = ("maj7", "7", "min7", "halfdim7", "dim7")
        seventh_tones = _chord_tones_from_scale(main_deg, d, size=4)
        seventh_pcs = frozenset(int(p.pitchClass) % 12 for p in seventh_tones)

        seventh_id: int | None = None
        for q in seventh_candidates:
            cid = _get_id(root, q)
            if cid is None:
                continue
            if pcs_by_id[cid] == seventh_pcs:
                seventh_id = cid
                break

        if seventh_id is not None:
            masks[seventh_id] |= bit(f"diat_7th_{d}")

    # ----------------------------
    # (B) Parallel triads / 7ths
    # ----------------------------
    for d in range(1, 8):
        root = par_deg_name[d]

        triad_candidates = ("maj", "min", "dim", "aug", "sus2", "sus4")
        triad_tones = _chord_tones_from_scale(par_deg, d, size=3)
        triad_pcs = frozenset(int(p.pitchClass) % 12 for p in triad_tones)

        triad_id: int | None = None
        for q in triad_candidates:
            cid = _get_id(root, q)
            if cid is None:
                continue
            if pcs_by_id[cid] == triad_pcs:
                triad_id = cid
                break

        if triad_id is not None:
            masks[triad_id] |= bit(f"parallel_triad_{d}")
            par_triad_ids.add(triad_id)

        seventh_candidates = ("maj7", "7", "min7", "halfdim7", "dim7")
        seventh_tones = _chord_tones_from_scale(par_deg, d, size=4)
        seventh_pcs = frozenset(int(p.pitchClass) % 12 for p in seventh_tones)

        seventh_id: int | None = None
        for q in seventh_candidates:
            cid = _get_id(root, q)
            if cid is None:
                continue
            if pcs_by_id[cid] == seventh_pcs:
                seventh_id = cid
                break

        if seventh_id is not None:
            masks[seventh_id] |= bit(f"parallel_7th_{d}")

    # ----------------------------
    # (C) Applied chords (V/x, V7/x, vii/x, vii7/x)
    # Rule: only assign if chord contains ANY non-diatonic pc vs MAIN key.
    # ----------------------------
    # Determine diatonic degree "mode" (for deciding the target key spelling)
    # Simple rule: if the diatonic triad on that degree is major/aug => target is major, else minor.
    deg_mode: dict[int, str] = {}
    for d in range(1, 8):
        # find the triad id we assigned for diat_triad_d (if any)
        root = main_deg_name[d]
        triad_id = None
        for q in ("maj", "min", "dim", "aug"):
            cid = _get_id(root, q)
            if cid is not None and (masks[cid] & bit(f"diat_triad_{d}")) != 0:
                triad_id = cid
                break
        if triad_id is None:
            deg_mode[d] = "major"
        else:
            q = quality_by_id[triad_id]
            deg_mode[d] = "major" if q in ("maj", "aug") else "minor"

    for d in range(1, 8):
        target_tonic = m21pitch.Pitch(main_deg_name[d])
        target_mode = deg_mode[d]  # "major" or "minor" (harmonic minor for minor)

        # Dominant root = P5 above target tonic
        V_root = target_tonic.transpose("P5")
        V_root_name = _norm_name(V_root.name)

        # Leading-tone root = m2 below target tonic
        vii_root = target_tonic.transpose("-m2")
        vii_root_name = _norm_name(vii_root.name)

        # V/x
        cid = _get_id(V_root_name, "maj")
        if cid is not None and _has_nondiatonic(pcs_by_id[cid], kc.diat_set):
            masks[cid] |= bit(f"V/{d}")

        # V7/x
        cid = _get_id(V_root_name, "7")
        if cid is not None and _has_nondiatonic(pcs_by_id[cid], kc.diat_set):
            masks[cid] |= bit(f"V7/{d}")

        # vii/x
        cid = _get_id(vii_root_name, "dim")
        if cid is not None and _has_nondiatonic(pcs_by_id[cid], kc.diat_set):
            masks[cid] |= bit(f"vii/{d}")

        # vii7/x (allow either halfdim7 or dim7)
        cid = _get_id(vii_root_name, "halfdim7")
        if cid is not None and _has_nondiatonic(pcs_by_id[cid], kc.diat_set):
            masks[cid] |= bit(f"vii7/{d}")

        cid = _get_id(vii_root_name, "dim7")
        if cid is not None and _has_nondiatonic(pcs_by_id[cid], kc.diat_set):
            masks[cid] |= bit(f"vii7/{d}")

    # ----------------------------
    # (D) Sus chords: root must be a main-key degree root AND all pcs must be diatonic (main key).
    # ----------------------------
    for d in range(1, 8):
        root = main_deg_name[d]

        cid = _get_id(root, "sus2")
        if cid is not None and _is_subset(pcs_by_id[cid], kc.diat_set):
            masks[cid] |= bit(f"sus2/{d}")

        cid = _get_id(root, "sus4")
        if cid is not None and _is_subset(pcs_by_id[cid], kc.diat_set):
            masks[cid] |= bit(f"sus4/{d}")

    # ----------------------------
    # (E) Chromatic mediants (root spelled from the referenced DEGREE pitch)
    # Only assign if chord is NOT already a diatonic/parallel triad.
    # Assign to both maj and min triads if they exist.
    # ----------------------------
    triad_block = diat_triad_ids | par_triad_ids

    for d in range(1, 8):
        base = m21pitch.Pitch(main_deg_name[d])

        # tag -> interval string
        cm_cases = [
            ("CM_M3-", "-M3"),
            ("CM_m3-", "-m3"),
            ("CM_m3+", "m3"),
            ("CM_M3+", "M3"),
        ]

        for tag, iv in cm_cases:
            root_pitch = base.transpose(iv)
            root = _norm_name(root_pitch.name)

            for q in ("maj", "min"):
                cid = _get_id(root, q)
                if cid is None:
                    continue
                if cid in triad_block:
                    continue
                masks[cid] |= bit(f"{tag}/{d}")

    # ----------------------------
    # (F) Specials: N6 and It6 (spelled from tonic)
    # ----------------------------
    tonic_pitch = m21pitch.Pitch(_norm_name(tonic_name))

    # N6: bII major triad (m2 above tonic, spelled as b2, not #1)
    bII = tonic_pitch.transpose("m2")
    bII_name = _norm_name(bII.name)
    cid = _get_id(bII_name, "maj")
    if cid is not None:
        masks[cid] |= bit("N6")

    # It6: anchored at tonic (quality "it6" in your master table)
    cid = _get_id(_norm_name(tonic_name), "it6")
    if cid is not None:
        masks[cid] |= bit("It6")

    return masks

# ----------------------------
# Build role mask table for ALL keys
# ----------------------------

def build_role_mask_table(master_table: pd.DataFrame) -> pd.DataFrame:
    """
    Build role masks for ALL 24 keys and return as a DataFrame:

      index: chord_id (0..N-1)
      columns: MultiIndex (tonic_pc, mode)
      values: Python int bitmask (dtype=object)
    """
    ids = master_table["id"].to_numpy(dtype=int)
    max_id = int(ids.max()) if len(ids) else -1
    N = max_id + 1

    cols = pd.MultiIndex.from_product(
        [range(12), ("major", "minor")],
        names=["tonic_pc", "mode"],
    )
    out = pd.DataFrame(index=pd.RangeIndex(N, name="chord_id"), columns=cols, dtype=object)

    for tonic_pc in range(12):
        for mode in ("major", "minor"):
            kc = KeyContext(tonic_pc, mode)
            masks = build_role_masks(master_table, kc)
            out[(tonic_pc, mode)] = masks

    return out

def load_or_build_role_mask_table(master_table: pd.DataFrame, *, force_rebuild: bool = False) -> pd.DataFrame:
    """
    Load the single cached role-mask table if present and compatible; otherwise rebuild.
    Compatibility check: row count must match chord count.
    """
    if (not force_rebuild) and ROLE_MASK_TABLE_PKL_PATH.exists():
        try:
            df = pd.read_pickle(ROLE_MASK_TABLE_PKL_PATH)
            ids = master_table["id"].to_numpy(dtype=int)
            max_id = int(ids.max()) if len(ids) else -1
            N = max_id + 1
            if isinstance(df, pd.DataFrame) and len(df) == N:
                return df
        except Exception:
            pass

    df = build_role_mask_table(master_table)
    df.to_pickle(ROLE_MASK_TABLE_PKL_PATH)
    df.to_csv(ROLE_MASK_TABLE_PKL_PATH.with_suffix(".csv"))
    return df

def get_role_masks_for_key(role_mask_table: pd.DataFrame, tonic_pc: int, mode: str) -> list[int]:
    """
    Convenience: fetch the role masks vector for one key from the table.
    """
    tonic_pc = int(tonic_pc) % 12
    col = (tonic_pc, mode)
    if col not in role_mask_table.columns:
        raise KeyError(f"Key {col} not found in role_mask_table columns.")
    return [int(x) for x in role_mask_table[col].tolist()]

# ----------------------------
# Visualization helper
# ----------------------------

def role_names_for_key(master_table: pd.DataFrame, masks_for_key: list[int]) -> pd.DataFrame:
    df = master_table.copy()
    df["role_mask"] = [int(x) for x in masks_for_key]
    df["role_names"] = [decode_mask(int(x)) for x in masks_for_key]
    df["n_roles"] = df["role_names"].apply(len)
    return df


# ============================================================
# Public globals
# ============================================================
_ROLE_MASK_TABLE: Optional[pd.DataFrame] = None

def get_role_mask_table(
    master_table: pd.DataFrame,
    *,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """
    Lazy-load and cache ROLE_MASK_TABLE in memory (module-global).
    This avoids disk IO at import time and plays nicely with multiprocessing.
    """
    global _ROLE_MASK_TABLE
    if _ROLE_MASK_TABLE is None or force_rebuild:
        _ROLE_MASK_TABLE = load_or_build_role_mask_table(master_table, force_rebuild=force_rebuild)
    return _ROLE_MASK_TABLE


# %% Debug / test code
if __name__ == "__main__":
    from echo_harmonizer.chord_pool.chords_master import MASTER_CHORD_TABLE
    role_table = get_role_mask_table(MASTER_CHORD_TABLE)

    masks_cmaj = get_role_masks_for_key(role_table, 2, "major")  # C major
    df_cmaj = role_names_for_key(MASTER_CHORD_TABLE, masks_cmaj)

    from IPython.display import display
    display(df_cmaj)

# %%
