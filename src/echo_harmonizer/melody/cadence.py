# cadence.py
import pandas as pd
from music21 import note

# ----------------------------
# Density-invariant note access
# ----------------------------

_EPS = 1e-6

def _onsets_in_region(row):
    """
    Return only notes/rests whose *start* is inside [region_start, region_end).
    This prevents sustained notes from previous regions from polluting motifs.
    """
    start = float(row["region_start"])
    end = float(row["region_end"])
    out = []
    for n in row.get("region_note_objs", []) or []:
        off = float(getattr(n, "offset", 0.0))
        if (off + _EPS) >= start and off < (end - _EPS):
            out.append(n)
    return out

def _first_row_with_onsets(rows: pd.DataFrame):
    rows = rows.sort_values("region_start")
    for _, r in rows.iterrows():
        if _onsets_in_region(r):
            return r
    return None

def _last_onset_before_time(grid, t):
    """
    Find the last note/rest onset at or before time t (in quarterLength units),
    scanning all region_note_objs. Works regardless of grid density.
    """
    best = None
    best_off = -1e18
    for row in grid.itertuples(index=False):
        for n in getattr(row, "region_note_objs", []) or []:
            off = float(getattr(n, "offset", 0.0))
            if off <= (t + _EPS) and off > best_off:
                best_off = off
                best = n
    return best, best_off


def _event_ends_at(n, off):
    """Return end time (offset + duration) for a note/rest object."""
    dur = float(getattr(n, "quarterLength", 0.0))
    return float(off) + dur


# --- INTERNAL HELPERS ---

def _robust_avg(scores):
    """
    Average only the best half (ceil) of scores.
    This prevents one expected "cadentially different" phrase from tanking
    an otherwise good segmentation.
    """
    if not scores:
        return 0.0
    scores = sorted(scores, reverse=True)
    k = max(1, (len(scores) + 1) // 2)  # ceil half
    return sum(scores[:k]) / k

def _check_boundary_strength(grid, end_measure_m):
    """
    Boundary strength at the end of measure end_measure_m.
    Uses the last onset before boundary and checks if it ends "properly"
    (rest, or long sustain to boundary).
    """
    # boundary time = maximum region_end among rows of that measure
    rows = grid[grid["measure"] == end_measure_m]
    if rows.empty:
        return 0.0

    boundary_t = float(rows["region_end"].max())

    n, off = _last_onset_before_time(grid, boundary_t)
    if n is None:
        return 0.0

    end_t = _event_ends_at(n, off)
    dur = float(getattr(n, "quarterLength", 0.0))

    score = 0.0

    # (2) endings: rest is strong
    if getattr(n, "isRest", False):
        score += 2.0

    # long note that reaches boundary = strong cadence-like closure
    # (tolerance because music21 offsets can be floaty)
    reaches = end_t >= (boundary_t - 0.05)
    if reaches:
        score += 1.0
        if dur >= 1.5:   # tweak thresholds as you like
            score += 1.0
        if dur >= 3.0:
            score += 0.5

    return score

def _extract_motif(grid, measure_start_m, max_notes=6, span_measures=2):
    """
    Extract onset-based motif starting from the FIRST real onset inside the chunk.
    This avoids the 'measure starts with empty region' bug (e.g., pickup/rest).
    """
    m2 = measure_start_m + span_measures - 1
    rows = grid[(grid["measure"] >= measure_start_m) & (grid["measure"] <= m2)]
    if rows.empty:
        return []

    rows = rows.sort_values("region_start")

    motif = []
    for _, row in rows.iterrows():
        for n in _onsets_in_region(row):
            if getattr(n, "isRest", False):
                motif.append(("R", float(getattr(n, "quarterLength", 0.0))))
            else:
                p = getattr(n, "pitch", None)
                motif.append((int(getattr(p, "pitchClass", 0)),
                              float(getattr(n, "quarterLength", 0.0))))
            if len(motif) >= max_notes:
                return motif
    return motif

def _compare_motifs(grid, measure_A, measure_B, pattern_len=8, span_measures=2):
    """Compares two onset-based sequences. Returns 0.0 to ~3.0."""
    motif_A = _extract_motif(grid, measure_A, max_notes=pattern_len, span_measures=span_measures)
    motif_B = _extract_motif(grid, measure_B, max_notes=pattern_len, span_measures=span_measures)

    if not motif_A or not motif_B:
        return 0.0

    check_len = min(len(motif_A), len(motif_B))
    if check_len == 0:
        return 0.0

    matches = 0.0
    for i in range(check_len):
        pitch_A, dur_A = motif_A[i]
        pitch_B, dur_B = motif_B[i]
        if pitch_A == pitch_B:
            matches += 1.0
        if abs(dur_A - dur_B) < 0.25:
            matches += 0.5

    return (matches / check_len) * 2.0

def _rhythm_signature_at_measure(grid, m, k=8, span_measures=2):
    """
    Pitch-agnostic onset fingerprint starting from the FIRST real onset in the chunk.
    """
    m2 = m + span_measures - 1
    rows = grid[(grid["measure"] >= m) & (grid["measure"] <= m2)]
    if rows.empty:
        return ()

    rows = rows.sort_values("region_start")

    # pick t0 as the first onset time in the span (not just measure boundary)
    t0 = None
    for _, row in rows.iterrows():
        ons = _onsets_in_region(row)
        if ons:
            t0 = float(getattr(ons[0], "offset", float(row["region_start"])))
            break
    if t0 is None:
        return ()

    ons_rel = []
    for _, row in rows.iterrows():
        for n in _onsets_in_region(row):
            ons_rel.append(float(getattr(n, "offset", 0.0)) - t0)

    return tuple(round(x, 3) for x in ons_rel[:k])

def _sig_similarity(a, b, tol=0.10):
    """Return [0..1] similarity; penalize length mismatch."""
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    denom = max(len(a), len(b))
    if denom == 0:
        return 0.0
    ok = 0
    for x, y in zip(a[:n], b[:n]):
        if abs(x - y) <= tol:
            ok += 1
    return ok / denom

def _evaluate_segmentation(grid, chunk_size, total_measures):
    """
    Score = (Boundary Strength * 1.0) + (Motif Similarity * 2.0) + (Rhythm Consistency * 1.0)

    Rhythm Consistency:
      - compute a pitch-agnostic onset "signature" at the start of each chunk
      - reward segmentations where adjacent chunk-start signatures are similar
    """
    boundary_scores = []
    similarity_scores = []
    rhythm_scores = []

    chunks = []
    for start_m in range(1, total_measures + 1, chunk_size):
        if start_m > total_measures:
            break
        end_m = min(start_m + chunk_size - 1, total_measures)
        chunks.append((start_m, end_m))

    # A. Boundaries (end of each chunk)
    for _, end_m in chunks:
        boundary_scores.append(_check_boundary_strength(grid, end_m))

    # B. Motivic Similarity (start of chunk N vs start of chunk N+1)
    if len(chunks) > 1:
        for i in range(len(chunks) - 1):
            curr_chunk = chunks[i]
            next_chunk = chunks[i + 1]
            score = _compare_motifs(grid, curr_chunk[0], next_chunk[0], pattern_len=8, span_measures=2)
            similarity_scores.append(score)
    else:
        similarity_scores.append(0.0)

    # C. Rhythm Consistency (compare rhythm signatures at chunk starts)
    if len(chunks) > 1:
        sigs = [_rhythm_signature_at_measure(grid, start_m, k=8, span_measures=2) for start_m, _ in chunks]
        for a, b in zip(sigs, sigs[1:]):
            rhythm_scores.append(_sig_similarity(a, b, tol=0.10))
    else:
        rhythm_scores.append(0.0)

    avg_boundary = sum(boundary_scores) / len(boundary_scores) if boundary_scores else 0.0

    # key change: robust similarity aggregation
    avg_sim = _robust_avg(similarity_scores)
    avg_rhythm = _robust_avg(rhythm_scores)

    # weights (tune if needed)
    return (avg_boundary * 2.0) + (avg_sim * 1.0) + (avg_rhythm * 1.0)

def _mark_cadence_points(grid, chunk_size, total_measures):
    """
    Dumb marking rule:
    - For each split boundary (every chunk_size measures), mark ALL harmonization
      points in that boundary measure as cadence points.
    - Do not overwrite end_of_piece.
    """
    for m_num in range(chunk_size, total_measures + 1, chunk_size):
        # skip final measure: end_of_piece already marked
        if m_num == total_measures:
            continue

        rows = grid[grid["measure"] == m_num]
        if rows.empty:
            continue

        strength = _check_boundary_strength(grid, m_num)
        if strength < 0.5:
            continue

        for idx in rows.index:
            # don't overwrite something stronger
            if grid.at[idx, "cadence_type"] == "end_of_piece":
                continue
            if grid.at[idx, "cadence_weight"] < 10.0:
                grid.at[idx, "cadence_weight"] = max(grid.at[idx, "cadence_weight"], float(strength))
                grid.at[idx, "cadence_type"] = f"{chunk_size}bar_structure"


# ----------------------------
# Apply Cadence Weights to Grid
# ----------------------------

def apply_cadence_weights(grid: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates cadence weights using Competitive Segmentation.
    Now supports 'Monolith' structures (no internal cadences) if subdivisions are weak.
    """
    if grid.empty:
        return grid

    # 1. Initialize Columns
    if "cadence_weight" not in grid.columns:
        grid["cadence_weight"] = 0.0
    if "cadence_type" not in grid.columns:
        grid["cadence_type"] = ""

    # 2. Mark End of Piece (Always a cadence)
    # mark all rows of the final measure as end_of_piece too
    last_m = int(grid["measure"].max())
    for idx in grid[grid["measure"] == last_m].index:
        grid.at[idx, "cadence_weight"] = max(grid.at[idx, "cadence_weight"], 3.0)
        grid.at[idx, "cadence_type"] = "end_of_piece"
    
    total_measures = int(grid["measure"].max())
    
    # 3. Find Candidates
    # Standard divisors (e.g. 12 -> 3, 4, 6)
    candidates = [i for i in range(3, total_measures // 2 + 1) if total_measures % i == 0]
    
    # --- FIX: Add TOTAL LENGTH as a candidate ---
    # This represents the hypothesis: "The piece is just one long phrase."
    # If internal splits are weak, this candidate will win.
    candidates.append(total_measures)
    
    # Fallback if nothing divides cleanly (e.g. 13 bars), still check 4 vs Total
    if len(candidates) == 1: # Only total_measures is there
        candidates.insert(0, 4) # Add 4 as a fallback option to test against
        
    print(f"[Cadence] Evaluating structures: {candidates} for length {total_measures}")

    best_chunk_size = -1
    best_fitness = -1.0
    
    # 4. Competitive Evaluation
    for chunk_size in candidates:
        score = _evaluate_segmentation(grid, chunk_size, total_measures)
        print(f"  > Structure {chunk_size}-bar: Fitness = {score:.2f}")
        
        # We generally prefer splits over monoliths if scores are close/tied.
        # But if Monolith scores significantly higher, it wins.
        if score > best_fitness:
            best_fitness = score
            best_chunk_size = chunk_size
    
    # 5. Apply Winner
    print(f"[Cadence] WINNER: {best_chunk_size}-bar phrases")
    
    # If the winner is the Monolith (total length), we effectively do nothing
    # because the end is already marked. But we call the function to be safe.
    _mark_cadence_points(grid, best_chunk_size, total_measures)
    
    return grid

# %% TESTING CODE (can be removed in production) %%
if __name__ == "__main__":
    import argparse
    import pandas as pd

    from echo_harmonizer.melody.preprocess_melody import Melody

    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="Path to MIDI/MusicXML input melody")
    ap.add_argument("--density", default="mid", choices=["sparse", "mid", "dense"])
    args = ap.parse_args()

    mel = Melody(args.path)
    grid = mel.get_harmonization_grid(density=args.density)

    cols = [c for c in ["measure", "local_offset", "region_start", "region_end", "region_notes",
                        "cadence_weight", "cadence_type"] if c in grid.columns]

    print("\n=== Grid (cadence-marked) ===")
    print(grid[cols].to_string(index=False))

    print("\n=== Cadence points only ===")
    print(grid.loc[grid["cadence_weight"] > 0, cols].to_string(index=False))

# %%
