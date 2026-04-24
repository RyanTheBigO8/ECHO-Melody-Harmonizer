# build_pool.py
from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
from music21 import key as m21key
from music21 import meter, note as m21note

from melody.preprocess_melody import Melody
from chord_pool.chords_master import MASTER_CHORD_TABLE, MASTER_CHORD_REGISTRY
from chord_pool.chord_roles import get_role_masks_for_key, allowed_mask_from_groups, get_role_mask_table, bit
from config import ChordPolicyConfig, Density, KeySpec
from chord_pool.pool_policy import effective_enabled_groups


# ----------------------------
# Spelling preference + dedupe
# ----------------------------

def _key_sig_bias(tonic: str, mode: str) -> int:
    """
    Return key signature bias:
      < 0 : prefer flats
      = 0 : prefer naturals (no accidental preference)
      > 0 : prefer sharps

    Uses music21 Key.sharps (flats are negative).
    """
    tonic_m21 = str(tonic).replace("b", "-")
    k = m21key.Key(tonic_m21, str(mode))
    return int(k.sharps)


def _root_spelling_score(root_name: str, key_sig: int) -> int:
    """
    Higher is better.

    Primary objective: fewer accidentals wins (natural > single accidental > double, etc.)
    Secondary tie-break: match key signature direction (sharp vs flat) if accidentals count ties.
    """
    r = str(root_name)

    n_sharp = r.count("#")
    n_flat = r.count("b")
    accidental_count = n_sharp + n_flat  # B## => 2, Ebb => 2, etc.

    # tie-break preference among equal accidental_count
    if key_sig > 0:
        # prefer sharps over flats
        pref = n_sharp - n_flat
    elif key_sig < 0:
        # prefer flats over sharps
        pref = n_flat - n_sharp
    else:
        # neutral: don't bias # vs b
        pref = 0

    # Big base so accidental_count dominates tie-break.
    return 1000 - 10 * accidental_count + pref


def _dedupe_pool_by_preference(pool: pd.DataFrame, *, key_sig: int) -> pd.DataFrame:
    """
    Remove enharmonic duplicates by keeping ONE representative per pitch-class chord.

    Equivalence key:
      (root_pc, quality, pcs_tuple)

    Chooser:
      - prefer flats or sharps according to prefer_flats
      - tie-breaker: smaller chord id

    Returns a new DataFrame (sorted by id).
    """
    if pool.empty:
        return pool

    chosen_idx: dict[tuple[int, str, tuple[int, ...]], int] = {}

    for idx, r in pool.iterrows():
        k = (
            int(r["root_pc"]) % 12,
            str(r["quality"]),
            tuple(r["pcs_tuple"]),
        )

        if k not in chosen_idx:
            chosen_idx[k] = idx
            continue

        prev = chosen_idx[k]
        cur_score = _root_spelling_score(str(r["root"]), key_sig)
        prev_score = _root_spelling_score(str(pool.at[prev, "root"]), key_sig)

        if cur_score > prev_score:
            chosen_idx[k] = idx
        elif cur_score == prev_score:
            # deterministic tie-breaker: smaller chord id wins
            if int(r["id"]) < int(pool.at[prev, "id"]):
                chosen_idx[k] = idx

    out = pool.loc[list(chosen_idx.values())].copy()
    out.sort_values("id", inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out


# ----------------------------
# Atonal pool: old root-union logic
# ----------------------------

def build_chord_pool_atonal(
    melody: Melody,
    master_table: pd.DataFrame,
    top_k_keys: int | None = None,
    include_sevenths: bool = False,
) -> pd.DataFrame:
    allowed_root_pcs: set[int] = set()
    key_df = melody.global_key_candidates
    if top_k_keys is not None:
        key_df = key_df.head(top_k_keys)

    for _, row in key_df.iterrows():
        k = m21key.Key(row["tonic"], row["mode"])

        # diatonic scale degree pitch classes
        for p in k.pitches:
            allowed_root_pcs.add(int(p.pitchClass) % 12)

        # raised 7 in minor (harmonic minor)
        if row["mode"] == "minor":
            natural_7 = k.pitchFromDegree(7)
            raised_7 = natural_7.transpose(1)
            allowed_root_pcs.add(int(raised_7.pitchClass) % 12)

    pool = master_table[master_table["root_pc"].isin(sorted(allowed_root_pcs))].copy()

    if not include_sevenths:
        pool = pool[~pool["is_seventh"]].copy()

    # Dedupe enharmonic spellings to keep search space small and spelling consistent.
    # Use the best key candidate (index 0) as the spelling anchor.
    if not melody.global_key_candidates.empty:
        anchor = melody.global_key_candidates.iloc[0]
        key_sig = _key_sig_bias(str(anchor["tonic"]), str(anchor["mode"]))
        pool = _dedupe_pool_by_preference(pool, key_sig=key_sig)

    return pool



# ----------------------------
# Tonal/chromatic pool: role-mask filtering
# ----------------------------

def build_chord_pool_by_roles(
    master_table: pd.DataFrame,
    role_masks_for_key: list[int],
    *,
    allowed_role_mask: int,
) -> pd.DataFrame:
    '''
    Extract rows from master_table where the chord's role mask for the given key
    has ANY bit in allowed_role_mask.
    '''
    ids = master_table["id"].to_numpy(dtype=int)
    keep = [(int(role_masks_for_key[cid]) & int(allowed_role_mask)) != 0 for cid in ids]
    pool = master_table[keep].copy()

    return pool


# ----------------------------
# Acceptability evaluation (copied from your old chord_pool.py)
# ----------------------------

def _pcs_to_mask(pcs: list[int]) -> int:
    m = 0
    for pc in pcs:
        m |= 1 << (int(pc) % 12)
    return m


def _precompute_point_melody_state(
    grid: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    H = len(grid)
    active_pc = np.full(H, -1, dtype=np.int16)
    next_pc   = np.full(H, -1, dtype=np.int16)
    semitone  = np.zeros(H, dtype=bool)

    offsets = grid["offset"].to_numpy(dtype=float)
    r_ends  = grid["region_end"].to_numpy(dtype=float)
    region_notes = grid["region_note_objs"].tolist()

    for i in range(H):
        t = float(offsets[i])
        end = float(r_ends[i])
        notes = region_notes[i] or []

        active_idx = None
        for j, n in enumerate(notes):
            s = float(n.offset)
            e = float(n.offset + n.duration.quarterLength)
            if s <= t < e:
                active_idx = j
                break

        if active_idx is None:
            continue

        a = notes[active_idx]
        active_pc[i] = int(a.pitch.pitchClass)

        a_start = float(a.offset)
        a_midi = int(a.pitch.midi)

        for k in range(active_idx + 1, len(notes)):
            b = notes[k]
            b_start = float(b.offset)
            if b_start <= a_start:
                continue
            if b_start >= end:
                break
            next_pc[i] = int(b.pitch.pitchClass)
            semitone[i] = (abs(int(b.pitch.midi) - a_midi) == 1)
            break

    return active_pc, next_pc, semitone


def get_acceptable_chords_per_point(
    chord_pool: pd.DataFrame,
    grid: pd.DataFrame,
) -> list[list[int]]:
    chord_ids = chord_pool["id"].to_numpy(dtype=np.int32)
    chord_masks = MASTER_CHORD_REGISTRY.mask_by_id[chord_ids]
    is_seventh  = MASTER_CHORD_REGISTRY.is_seventh_by_id[chord_ids]

    active_pc, next_pc, semitone_step = _precompute_point_melody_state(grid)
    acceptable_lists: list[list[int]] = []

    for i in range(len(grid)):
        mel_pc = int(active_pc[i])

        if mel_pc < 0:
            acceptable_lists.append(chord_ids.tolist())
            continue

        mel_bit = 1 << mel_pc

        ok_sev = (chord_masks & mel_bit) != 0

        dissonant_mask = (1 << ((mel_pc + 1) % 12)) | (1 << ((mel_pc - 1) % 12))
        no_raw_dissonance = (chord_masks & dissonant_mask) == 0

        if bool(semitone_step[i]) and int(next_pc[i]) >= 0:
            next_bit = 1 << int(next_pc[i])
            resolves_to_chord_tone = (chord_masks & next_bit) != 0
        else:
            resolves_to_chord_tone = np.zeros(len(chord_masks), dtype=bool)

        ok_nonsev = no_raw_dissonance | resolves_to_chord_tone

        ok = np.where(is_seventh, ok_sev, ok_nonsev)
        acceptable_lists.append(chord_ids[ok].tolist())

    return acceptable_lists


# ----------------------------
# Wrapper
# ----------------------------

def prepare_harmonization(
    melody: Melody,
    *,
    policy_cfg: ChordPolicyConfig,
    density: Density = "mid",
    selected_key: KeySpec | None = None,          # required for non-atonal
    top_k_keys_atonal: int | None = None,         # internal knob for atonal
    master_table: pd.DataFrame = MASTER_CHORD_TABLE,
    role_mask_table: pd.DataFrame | None = None,  # can pass preloaded
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      chord_pool, harmonization_table

    policy_cfg.enabled_groups:
      - if None => pool_policy defaults
      - else => treat as "these groups ON" for that preset (policy enforces required)
    """
    # 1) Build harmonization grid (cadence weights applied inside melody)
    grid = melody.get_harmonization_grid(density=density, apply_cadence=True)

    # 2) Determine effective enabled groups for this policy
    enabled_groups = effective_enabled_groups(policy_cfg.name, policy_cfg.enabled_groups)

    # 3) Build chord pool
    if policy_cfg.name == "atonal":
        include_sevenths = ("sevenths" in enabled_groups)
        pool = build_chord_pool_atonal(
            melody,
            master_table,
            top_k_keys=top_k_keys_atonal,
            include_sevenths=include_sevenths,
        )
    else:
        if selected_key is None:
            raise ValueError("selected_key is required for non-atonal chord policies.")

        if role_mask_table is None:
            role_mask_table = get_role_mask_table(master_table, force_rebuild=False)

        # Get role masks for this key
        masks = get_role_masks_for_key(role_mask_table, selected_key.tonic_pc, selected_key.mode)

        allowed_mask = allowed_mask_from_groups(enabled_groups)

        pool = build_chord_pool_by_roles(
            master_table,
            masks,
            allowed_role_mask=allowed_mask,
        )

    if pool.empty:
        raise ValueError(f"Chord pool is empty for policy={policy_cfg.name} enabled_groups={sorted(enabled_groups)}")

    # 4) Acceptable chord IDs per region
    acceptable = get_acceptable_chords_per_point(pool, grid)
    # Special check: ensure region 0 has at least one diatonic tonic triad if not atonal
    # if policy_cfg.name != "atonal":
    #     tonic_triad_bit = bit("diat_triad_1")

    #     # 'masks' is your role_masks_for_key (list[int] indexed by chord_id)
    #     acceptable0 = [cid for cid in acceptable[0] if (int(masks[int(cid)]) & tonic_triad_bit) != 0]
    #     # replace acceptable[0] if acceptable0 is non-empty
    #     if acceptable0:
    #         acceptable[0] = acceptable0

    #     # if not acceptable0:
    #     #     raise ValueError(
    #     #         "Start-chord constraint failed: region 0 has no acceptable diat_triad_1 options. "
    #     #         "Try relaxing density, acceptability rules, or chord policy."
    #     #     )
        
    grid = grid.copy()
    grid["acceptable_chord_ids"] = acceptable

    # Safety check: no region should be empty
    for i, opts in enumerate(acceptable):
        if not opts:
            raise ValueError(f"acceptable_chord_ids at region {i} is empty. Consider relaxing policy or density.")

    return pool, grid


# %% Debug / Test
if __name__ == "__main__":
    import argparse
    import pandas as pd

    from melody.preprocess_melody import Melody
    from config import ChordPolicyConfig, KeySpec

    parser = argparse.ArgumentParser()
    parser.add_argument("melody_path", type=str, help="Path to melody file (midi/musicxml/etc.)")
    parser.add_argument("--policy", type=str, default="chromatic", choices=["diatonic", "chromatic", "atonal"])
    parser.add_argument("--density", type=str, default="mid", choices=["dense", "mid", "sparse"])
    parser.add_argument("--key-index", type=int, default=0, help="Key candidate index (tonal only)")
    args = parser.parse_args()

    mel = Melody(args.melody_path)

    selected_key = None
    if args.policy != "atonal":
        cands = mel.global_key_candidates
        if cands is None or cands.empty:
            raise ValueError("No global_key_candidates found in melody.")
        row = cands.iloc[int(args.key_index)]
        selected_key = KeySpec(
            tonic=str(row["tonic"]),
            mode=str(row["mode"]),
            tonic_pc=int(row["tonic_pc"]),
            confidence=float(row["confidence"]),
        )

    pool, table = prepare_harmonization(
        mel,
        policy_cfg=ChordPolicyConfig(name=args.policy, enabled_groups=None),
        density=args.density,
        selected_key=selected_key,
    )

    print("\n=== POOL ===")
    print(f"pool_size={len(pool)}  policy={args.policy}  density={args.density}")
    print(pool.head(20).to_string(index=False))

    print("\n=== TABLE (full) ===")
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_colwidth", 200)
    print(table.to_string(index=True))

    # quick sanity checks
    if "acceptable_chord_ids" in table.columns:
        sizes = table["acceptable_chord_ids"].apply(lambda xs: 0 if not xs else len(xs))
        print("\n=== ACCEPTABLE SIZES ===")
        print(sizes.describe())
        bad = sizes.index[sizes == 0].tolist()
        if bad:
            print("\nWARNING: regions with 0 acceptable chords:", bad)
        else:
            print("\nOK: all regions have >= 1 acceptable chord.")

# %%
