# inspect_sol.py
from __future__ import annotations

import argparse
from pathlib import Path
import os, shutil, pickle, copy, subprocess
from tracemalloc import start
from typing import Any, Optional, Iterable

import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd

from music21 import stream, chord as m21chord, pitch as m21pitch, interval, harmony, expressions, meter

from echo_harmonizer.melody.preprocess_melody import Melody
from echo_harmonizer.chord_pool.chords_master import MASTER_CHORD_TABLE
from echo_harmonizer.metrics import evaluate_solution_metrics

# ============================================================
# Rendering utilities
# ============================================================
def midi_to_wav(midi_path: Path, wav_path: Path, *, sf2_path: Path) -> None:
    from midi2audio import FluidSynth  # local import so script still runs without it

    if not Path(sf2_path).exists():
        print(f"[Export] Skipping WAV (missing sf2): {sf2_path}")
        return

    if shutil.which("fluidsynth") is None:
        print("[Export] Skipping WAV (fluidsynth executable not found on PATH).")
        return

    FluidSynth(sound_font=str(sf2_path)).midi_to_audio(str(midi_path), str(wav_path))

def _active_melody_midi_at_time(melody_part, t: float) -> int | None:
    # Simple scan is OK for offline export; if you want faster later, we can index arrays.
    for n in melody_part.flatten().notes:
        s = float(n.offset)
        e = float(n.offset + n.duration.quarterLength)
        if s <= t < e:
            return int(n.pitch.midi)
    return None

def _push_chord_below_melody(ch, melody_midi: int) -> None:
    """
    Transpose chord down by octaves until its top note is strictly below melody_midi.
    """
    if melody_midi is None:
        return
    # music21 chord pitches are sortable by midi
    while True:
        top = max(int(p.midi) for p in ch.pitches)
        if top <= int(melody_midi):
            break
        ch.transpose(-12, inPlace=True)

# ============================================================
# Load solutions + meta
# ============================================================

def load_snapshot(pkl_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    Returns:
      meta: dict
      solutions: list of dicts:
        {"name": str, "genes": list[int], "fitness": float|None}
    """
    obj = pickle.load(open(pkl_path, "rb"))

    meta: dict[str, Any] = {}
    sols: list[dict[str, Any]] = []

    # New "top" format: dict with solutions + meta
    if isinstance(obj, dict) and "solutions" in obj and isinstance(obj["solutions"], list):
        meta = dict(obj.get("meta", {}) or {})
        for i, item in enumerate(obj["solutions"]):
            if isinstance(item, dict) and "genes" in item:
                sols.append({
                    "name": f"{pkl_path.stem}_idx{i:03d}",
                    "genes": list(item["genes"]),
                    "fitness": item.get("total_fitness", item.get("fitness", None)),
                })
        return meta, sols

    # New "best" format: dict with best_genes + meta
    if isinstance(obj, dict) and "best_genes" in obj:
        meta = dict(obj.get("meta", {}) or {})
        sols.append({
            "name": f"{pkl_path.stem}",
            "genes": list(obj["best_genes"]),
            "fitness": obj.get("best_total_fitness", None),
        })
        return meta, sols

    raise ValueError(f"Unrecognized pickle format in {pkl_path}")


def require_meta(meta: dict[str, Any], pkl_path: Path) -> tuple[str, np.ndarray, np.ndarray]:
    """
    Enforce Option-A requirement: snapshot must contain melody_path + region arrays.
    """
    melody_path = meta.get("melody_path", None)
    starts = meta.get("region_start", None)
    ends = meta.get("region_end", None)

    if not melody_path or starts is None or ends is None:
        raise ValueError(
            "This snapshot does not contain the required 'meta' context.\n"
            "Re-run GA after adding save_context to snapshots (Option A).\n"
            f"Missing from {pkl_path.name}: melody_path / region_start / region_end."
        )

    starts_arr = np.array(starts, dtype=float)
    ends_arr = np.array(ends, dtype=float)
    if len(starts_arr) == 0 or len(ends_arr) == 0 or len(starts_arr) != len(ends_arr):
        raise ValueError("Invalid region_start/region_end in meta.")

    return str(melody_path), starts_arr, ends_arr


def resolve_melody_path(melody_path: str) -> Path:
    path = Path(melody_path)
    if path.exists():
        return path

    normalized = melody_path.replace("\\", "/")
    marker = "test/my-dataset/"
    if marker in normalized:
        candidate = ROOT / "data" / "my-dataset" / normalized.split(marker, 1)[1]
        if candidate.exists():
            return candidate

    candidate = ROOT / melody_path
    if candidate.exists():
        return candidate

    return path


# ============================================================
# Minimal harmonization table (only what we need)
# ============================================================

def build_table_from_regions(starts: np.ndarray, ends: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({
        "region_start": starts.astype(float),
        "region_end": ends.astype(float),
    })


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
# Export
# ============================================================

CHORD_INTERVALS_EXPORT = {
    "maj":  ["P1", "M3", "P5"],
    "min":  ["P1", "m3", "P5"],
    "sus2": ["P1", "M2", "P5"],
    "sus4": ["P1", "P4", "P5"],
    "aug":  ["P1", "M3", "A5"],
    "dim":  ["P1", "m3", "d5"],
    "maj7":     ["P1", "M3", "P5", "M7"],
    "7":        ["P1", "M3", "P5", "m7"],
    "min7":     ["P1", "m3", "P5", "m7"],
    "halfdim7": ["P1", "m3", "d5", "m7"],
    "dim7":     ["P1", "m3", "d5", "d7"],
    "it6":      ["P1", "M3", "A6"],  # ?????“â™¯4 if root is ??
}

def build_root_position_chord(root: str, quality: str, root_octave: int = 3) -> m21chord.Chord:
    int_names = CHORD_INTERVALS_EXPORT[str(quality)]
    rp = m21pitch.Pitch(str(root).replace("b", "-"))
    rp.octave = root_octave

    pitches = []
    prev_midi = None
    for iv in int_names:
        p = interval.Interval(iv).transposePitch(rp)
        if prev_midi is not None:
            while p.midi <= prev_midi:
                p.octave += 1
        pitches.append(p)
        prev_midi = p.midi

    return m21chord.Chord(pitches)


def export_solutions_combined(
    melody_part: stream.Part,
    table: pd.DataFrame,
    solutions: list[dict[str, Any]],
    out_mid: Path,
    out_mxl: Path,
    root_octave: int = 3,
) -> None:
    by_id = MASTER_CHORD_TABLE.set_index("id")

    if len(table) == 0:
        raise ValueError("harmonization_table is empty")

    piece_len = float(table["region_end"].iloc[-1])
    if piece_len <= 0:
        piece_len = float(melody_part.highestTime)

    mel_out = stream.Part()
    chd_out = stream.Part()

    mel_elems = list(melody_part.flatten().notesAndRests)

    for j, sol in enumerate(solutions):
        genes = sol["genes"]
        base = j * piece_len

        # Insert time signatures
        ts = melody_part.recurse().getElementsByClass(meter.TimeSignature).first()
        if ts is not None:
            mel_out.insert(base, copy.deepcopy(ts))
            chd_out.insert(base, copy.deepcopy(ts))

        # Insert melody notes/rests
        for el in mel_elems:
            el2 = copy.deepcopy(el)
            mel_out.insert(base + float(el.offset), el2)

        H = min(len(genes), len(table))
        for i in range(H):
            start = base + float(table["region_start"].iloc[i])
            end = base + float(table["region_end"].iloc[i])
            dur = max(0.0, end - start)
            
            # Insert chord notes
            chord_id = int(genes[i])
            chord_row = by_id.loc[chord_id]
            ch = build_root_position_chord(chord_row["root"], chord_row["quality"], root_octave=root_octave)

            mid_t = (start + end) * 0.5
            mel_midi = _active_melody_midi_at_time(melody_part, base + float(mid_t))
            if mel_midi is not None:
                _push_chord_below_melody(ch, mel_midi)

            ch.duration.quarterLength = dur
            chd_out.insert(start, ch)

            # Insert chord symbols
            sym = str(chord_row.get("symbol", ""))  # your MASTER_CHORD_TABLE already has this
            if sym:
                # Special-case augmented-6th labels (ChordSymbol can?™t parse these reliably)
                if sym in {"it6", "it+6", "itaug6", "italian6"}:
                    chd_out.insert(start, expressions.TextExpression("It+6"))
                try:
                    cs = harmony.ChordSymbol(sym)
                    chd_out.insert(start, cs)
                except Exception:
                    pass

    mel_out.coreElementsChanged()
    chd_out.coreElementsChanged()

    sc = stream.Score()
    sc.insert(0, mel_out)
    sc.insert(0, chd_out)

    if not mel_out.hasMeasures():
        mel_out.makeMeasures(inPlace=True)
    if not chd_out.hasMeasures():
        chd_out.makeMeasures(inPlace=True)

    sc.write("midi", fp=str(out_mid))
    # midi_to_wav(out_mid, out_mid.with_suffix(".wav"), sf2_path=Path("./../gm.sf2"))
    sc.write("musicxml", fp=str(out_mxl))


# ============================================================
# Main
# ============================================================

def main(pkl_path: str) -> None:
    pkl_path = Path(pkl_path)

    meta, solutions = load_snapshot(pkl_path)
    melody_path, region_start, region_end = require_meta(meta, pkl_path)
    table = build_table_from_regions(region_start, region_end)

    # Keep exports isolated by snapshot so existing files are not deleted.
    out_dir = ROOT / "export" / pkl_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    melody = Melody(resolve_melody_path(melody_path))
    melody_part = melody.melody_stream

    sym_by_id = MASTER_CHORD_TABLE.set_index("id")["symbol"].to_dict()

    print(f"[Inspect] Snapshot: {pkl_path.name}")
    print(f"[Inspect] melody_path: {melody_path}")
    print(f"[Inspect] regions: {len(table)}")
    print(f"[Inspect] solutions: {len(solutions)}")

    # Metrics per solution
    metrics_rows = []
    for sol in solutions:
        genes = sol["genes"]
        m = evaluate_solution_metrics(genes, table, melody_part)
        sol["metrics"] = m
        metrics_rows.append({"name": sol["name"], **m})

    for sol in solutions:
        name = sol["name"]
        fitness = sol.get("fitness", None)
        genes = sol["genes"]
        chord_names = [sym_by_id.get(int(cid), f"id{cid}") for cid in genes]

        m = sol.get("metrics", {})
        print("\n" + "=" * 80)
        print(f"[Inspect] {name}  fitness={fitness}")
        print(f"  Metrics: CTnCTR={m.get('CTnCTR', float('nan')):.6f}  PCS={m.get('PCS', float('nan')):.6f}  MCTD={m.get('MCTD', float('nan')):.6f}")
        print("  " + " | ".join(chord_names))

    if metrics_rows:
        dfm = pd.DataFrame(metrics_rows)
        out_csv = out_dir / f"{pkl_path.stem}_metrics.csv"
        dfm.to_csv(out_csv, index=False)
        print(f"\n[Inspect] Wrote metrics CSV: {out_csv}")

        if len(dfm) > 1:
            mean = dfm[["CTnCTR", "PCS", "MCTD"]].mean()
            std = dfm[["CTnCTR", "PCS", "MCTD"]].std()
            print("\n" + "-" * 80)
            print(f"[Inspect] Aggregate over {len(dfm)} solutions:")
            print(f"  CTnCTR meanÂ±std = {mean['CTnCTR']:.6f} Â± {std['CTnCTR']:.6f}")
            print(f"  PCS    meanÂ±std = {mean['PCS']:.6f} Â± {std['PCS']:.6f}")
            print(f"  MCTD   meanÂ±std = {mean['MCTD']:.6f} Â± {std['MCTD']:.6f}")

    base_name = pkl_path.stem
    out_mid = out_dir / f"{base_name}_COMBINED.mid"
    out_mxl = out_dir / f"{base_name}_COMBINED.musicxml"

    export_solutions_combined(
        melody_part=melody_part,
        table=table,
        solutions=solutions,
        out_mid=out_mid,
        out_mxl=out_mxl,
        root_octave=3,
    )

    print(f"\n[Inspect] Exported combined MIDI: {out_mid}")
    print(f"[Inspect] Exported combined MusicXML: {out_mxl}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect/export GA solutions from a self-contained snapshot .pkl")
    parser.add_argument("pkl_path", type=str, help="Path to GA snapshot .pkl (best_*.pkl or top_*.pkl)")
    args = parser.parse_args()
    main(args.pkl_path)
