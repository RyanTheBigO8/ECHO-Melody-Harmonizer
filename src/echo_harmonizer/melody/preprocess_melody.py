# melody.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal
from bisect import bisect_right

import pandas as pd
from music21 import converter, note, chord, stream, meter, analysis
from music21 import pitch as m21pitch

from echo_harmonizer.config import KeySpec, Density
from echo_harmonizer.melody import cadence


class Melody:
    def __init__(self, filepath: str):
        self.filepath = filepath

        # Parse once
        self.score: stream.Score = converter.parse(filepath)
        parts = self.score.parts
        
        # Get melody part (assumed to be the first part)
        self.melody_part: stream.Part = parts[0]
        self.melody_part_flat: stream.Part = self.melody_part.flatten()

        # Build melody stream (monophonic)
        self.melody_stream: stream.Part = self.extract_melody()

        # Get chord part (if any)
        self.chord_part: Optional[stream.Part] = parts[1] if len(parts) > 1 else None
        self.chord_offsets: Optional[list[float]] = None
        self.chord_events: Optional[list[tuple[float, tuple[int, ...]]]] = None
        
        if self.chord_part is not None:
            offs, evs = self.extract_chord_offsets_and_events(self.chord_part)
            self.chord_offsets = offs or None # set to None if empty
            self.chord_events = evs or None # set to None if empty
        else:
            self.chord_events = None
        
        # Time signature
        self.time_signature: meter.TimeSignature = self.get_time_signature()

        # Measures once
        self._ensure_measures()
        self.num_measures: int = self.get_measure_count()

        # Beat strength info
        self.beat_info: pd.DataFrame = self.get_beat_strength()

        # Key analysis
        self._global_key_candidates: Optional[pd.DataFrame] = None
        self.selected_key: Optional[KeySpec] = None

    # ----------------------------
    # Internal helpers
    # ----------------------------
    def _ensure_measures(self) -> None:
        if not self.melody_stream.hasMeasures():
            self.melody_stream.makeMeasures(inPlace=True)

    # ----------------------------
    # Core extraction
    # ----------------------------

    def extract_melody(self) -> stream.Part:
        mel = stream.Part()
        for el in self.melody_part_flat.notesAndRests:
            off = float(el.offset)

            if isinstance(el, note.Rest):
                mel.coreInsert(off, el)
            elif isinstance(el, note.Note):
                mel.coreInsert(off, el)
            elif isinstance(el, chord.Chord):
                top_note = el.sortAscending()[-1]
                mel.coreInsert(off, top_note)

        mel.coreElementsChanged()
        return mel

    # ----------------------------
    # Metadata
    # ----------------------------

    def get_time_signature(self) -> meter.TimeSignature:
        ts_list = list(self.melody_part_flat.getElementsByClass(meter.TimeSignature))
        if ts_list:
            ts = ts_list[0]
            print(f"[Melody] Time signature (from score): {ts.ratioString}")
        else:
            ts = self.melody_part_flat.bestTimeSignature()
            print(f"[Melody] Time signature (estimated): {ts.ratioString}")

        self.melody_stream.insert(0.0, ts)
        return ts

    def get_measure_count(self) -> int:
        self._ensure_measures()
        num_measures = len(self.melody_stream.getElementsByClass("Measure"))
        print(f"[Melody] Number of measures: {num_measures}")
        return num_measures

    def get_beat_strength(self) -> pd.DataFrame:
        self._ensure_measures()
        rows = []
        for n in self.melody_stream.recurse().notes:
            rows.append({
                "note": n.nameWithOctave,
                "measure": int(n.measureNumber),
                "beat": n.beat,
                "beat_strength": float(n.beatStrength),
            })
        return pd.DataFrame(rows)

    # ----------------------------
    # Key analysis (lazy)
    # ----------------------------

    @property
    def global_key_candidates(self) -> pd.DataFrame:
        if self._global_key_candidates is None:
            self._global_key_candidates = self.get_global_key_candidates()
        return self._global_key_candidates

    def get_global_key_candidates(self, min_confidence: float = 0.5) -> pd.DataFrame:
        ka = analysis.discrete.BellmanBudge()
        primary_key = ka.getSolution(self.melody_stream)
        all_candidates = [primary_key] + list(primary_key.alternateInterpretations)

        results = []
        for k in all_candidates:
            cc = float(getattr(k, "correlationCoefficient", 0.0))
            if cc >= min_confidence:
                tonic_name = k.tonic.name.replace("-", "b")
                tonic_pc = int(k.tonic.pitchClass) % 12
                results.append({
                    "key": f"{tonic_name} {k.mode}",
                    "tonic": tonic_name,
                    "tonic_pc": tonic_pc,
                    "mode": k.mode,
                    "confidence": round(cc, 4),
                })

        if not results:
            k = primary_key
            tonic_name = k.tonic.name.replace("-", "b")
            tonic_pc = int(k.tonic.pitchClass) % 12
            results.append({
                "key": f"{tonic_name} {k.mode}",
                "tonic": tonic_name,
                "tonic_pc": tonic_pc,
                "mode": k.mode,
                "confidence": round(float(k.correlationCoefficient), 4),
            })

        return pd.DataFrame(results)

    def select_key(self, option: Optional[int] = None) -> KeySpec:
        df = self.global_key_candidates
        if df.empty:
            raise ValueError("global_key_candidates is empty")

        idx = 0 if option is None else int(option)
        r = df.iloc[idx]
        ks = KeySpec(
            tonic=str(r["tonic"]),
            mode=str(r["mode"]),
            tonic_pc=int(r["tonic_pc"]) % 12,
            confidence=float(r["confidence"]),
        )
        self.selected_key = ks
        return ks

    # ----------------------------
    # Harmonization Grid (density-aware)
    # ----------------------------

    # Hardcoded presets per TS for clarity.
    # Offsets are local offsets in quarterLength from measure start.
    _OFFSETS_BY_TS_DENSITY: dict[str, dict[str, tuple[float, ...]]] = {
        "2/4": {
            "dense": (0.0, 1.0),
            "mid":   (0.0,),
            "sparse":(0.0,),
        },
        "3/4": {
            "dense": (0.0, 1.0, 2.0),
            "mid":   (0.0,),
            "sparse":(0.0,),
        },
        "4/4": {
            "dense": (0.0, 1.0, 2.0, 3.0),
            "mid":   (0.0, 2.0),
            "sparse":(0.0,),
        },
        "5/4": {
            "dense": (0.0, 1.0, 2.0, 3.0, 4.0),
            "mid":   (0.0, 3.0),
            "sparse":(0.0,),
        },
        "6/4": {
            "dense": (0.0, 1.0, 2.0, 3.0, 4.0, 5.0),
            "mid":   (0.0, 3.0),
            "sparse":(0.0,),
        },
        "6/8": {
            # 6/8 bar = 3.0 quarterLength; eighth = 0.5
            "dense": (0.0, 0.5, 1.0, 1.5, 2.0, 2.5),
            "mid":   (0.0, 1.5),  # dotted-quarter beats
            "sparse":(0.0,),
        },
        "9/8": {
            # 9/8 bar = 4.5 quarterLength; eighth = 0.5
            "dense": (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0),
            "mid":   (0.0, 1.5, 3.0),
            "sparse":(0.0,),
        },
        "12/8": {
            # 12/8 bar = 6.0 quarterLength; eighth = 0.5
            "dense": (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5),
            "mid":   (0.0, 1.5, 3.0, 4.5),
            "sparse":(0.0, 3.0),
        },
    }

    def get_strong_beat_offsets(self, density: Density = "mid") -> list[float]:
        ratio = self.time_signature.ratioString
        d = str(density)

        if ratio in self._OFFSETS_BY_TS_DENSITY and d in self._OFFSETS_BY_TS_DENSITY[ratio]:
            return list(self._OFFSETS_BY_TS_DENSITY[ratio][d])

        # Fallback: sparse -> 0, mid -> {0, bar/2}, dense -> every beat
        bar_len = float(self.time_signature.barDuration.quarterLength)

        if d == "sparse":
            return [0.0]
        if d == "mid":
            return [0.0, bar_len / 2.0]

        # dense fallback: approximate beats by beatDuration if possible
        beat_len = float(self.time_signature.beatDuration.quarterLength)
        if beat_len <= 0:
            return [0.0, bar_len / 2.0]
        offs = []
        x = 0.0
        while x < bar_len - 1e-9:
            offs.append(float(x))
            x += beat_len
        return offs or [0.0]
    
    def extract_chord_offsets_and_events(
        self, chord_part: stream.Part
    ) -> tuple[list[float], list[tuple[float, tuple[int, ...]]]]:
        """
        Scan chord_part once and return:
        (1) offsets: list[float]  -- absolute quarterLength onsets for harmonization points
        (2) events:  list[(float, pcs_tuple)] -- absolute onset + sorted unique pitch classes

        Filtering / cleaning (important for offsets):
        - keep onsets that are chord.Chord OR have >=2 notes at the same onset
        - dedupe with tolerance
        - clip to [0, piece_end)
        - ensure 0.0 exists if any event exists
        """
        flat = chord_part.flatten()
        eps = 1e-6
        piece_end = float(self.score.highestTime)

        onset_counts: dict[float, int] = {}
        pcs_by_onset: dict[float, set[int]] = {}
        chord_onsets: set[float] = set()

        for el in flat.recurse().notes:
            off = float(el.offset)
            if off < 0.0 or off >= (piece_end - eps):
                continue

            if isinstance(el, chord.Chord):
                chord_onsets.add(off)
                k = int(len(el.pitches))
                onset_counts[off] = max(onset_counts.get(off, 0), k)
                pcs = {int(p.pitchClass) % 12 for p in el.pitches}
                if pcs:
                    pcs_by_onset.setdefault(off, set()).update(pcs)
            else:  # note.Note
                onset_counts[off] = onset_counts.get(off, 0) + 1
                if getattr(el, "pitch", None) is not None:
                    pcs_by_onset.setdefault(off, set()).add(int(el.pitch.pitchClass) % 12)

        if not onset_counts:
            return [], []

        # sort raw onsets
        raw = sorted(onset_counts.keys())

        # dedupe with tolerance
        offs: list[float] = []
        for t in raw:
            if not offs or abs(t - offs[-1]) > eps:
                offs.append(float(t))

        # keep block-chord-like onsets
        filtered = [t for t in offs if (t in chord_onsets) or (onset_counts.get(t, 0) >= 2)]

        # ensure 0.0 exists if any chord event exists
        if filtered and abs(filtered[0] - 0.0) > eps:
            filtered = [0.0] + filtered

        # build events aligned to filtered offsets (only if pcs exist)
        events: list[tuple[float, tuple[int, ...]]] = []
        for t in filtered:
            pcs = pcs_by_onset.get(float(t), set())
            if pcs:
                events.append((float(t), tuple(sorted(pcs))))

        return filtered, events


    def build_harmonization_grid(
        self,
        density: Density = "mid",
        *,
        chord_offsets: Optional[list[float]] = None,
    ) -> pd.DataFrame:
        measures = list(self.melody_stream.getElementsByClass("Measure"))

        points: list[dict[str, object]] = []
        piece_end = float(self.melody_stream.highestTime)

        # Decide offsets source
        if chord_offsets is None:
            chord_offsets = self.chord_offsets  # may still be None

        if chord_offsets is None:
            # default per-measure offsets
            strong_local_offsets = self.get_strong_beat_offsets(density=density)
            for m in measures:
                m_off = float(m.offset)
                for local_off in strong_local_offsets:
                    points.append({
                        "measure": int(m.number),
                        "local_offset": float(local_off),
                        "offset": m_off + float(local_off),
                    })
        else:
            # absolute offsets override
            eps = 1e-6
            raw = [float(x) for x in chord_offsets if x == x]  # drop NaN
            raw = [x for x in raw if x >= 0.0 and x < (piece_end - eps)]
            if not raw:
                return pd.DataFrame()

            raw.sort()

            offs: list[float] = []
            for x in raw:
                if not offs or abs(x - offs[-1]) > eps:
                    offs.append(float(x))

            if offs and abs(offs[0] - 0.0) > eps:
                offs = [0.0] + offs

            m_starts = [float(m.offset) for m in measures]
            m_nums = [int(m.number) for m in measures]

            for off in offs:
                idx = bisect_right(m_starts, float(off)) - 1
                if idx < 0:
                    idx = 0
                m_start = float(m_starts[idx])
                points.append({
                    "measure": int(m_nums[idx]),
                    "local_offset": float(off - m_start),
                    "offset": float(off),
                })

        if not points:
            return pd.DataFrame()

        points.sort(key=lambda d: float(d["offset"]))
        for i, p in enumerate(points):
            start = float(p["offset"])
            end = float(points[i + 1]["offset"]) if i + 1 < len(points) else piece_end
            p["region_start"] = start
            p["region_end"] = end

        # Melody note arrays for region slicing
        notes = list(self.melody_stream.flatten().notes)
        n_starts = [float(n.offset) for n in notes]
        n_ends = [float(n.offset + n.duration.quarterLength) for n in notes]
        i_note = 0
        N = len(notes)

        # Ground-truth chord events iterator (if chord track exists)
        chord_events: list[tuple[float, tuple[int, ...]]] = []
        k = 0
        last_pcs: Optional[tuple[int, ...]] = None

        if self.chord_part is not None:
            # Prefer cached events if you stored them; otherwise compute on demand.
            chord_events = list(self.chord_events or [])
            if chord_events == []:
                _offs, chord_events = self.extract_chord_offsets_and_events(self.chord_part)
            chord_events.sort(key=lambda x: x[0])

        # Single pass over regions: attach gt chord pcs + melody notes
        for p in points:
            start = float(p["region_start"])
            end = float(p["region_end"])

            # ---- gt_chord_pcs (carry-forward; last chord in region wins) ----
            if chord_events:
                while k < len(chord_events) and chord_events[k][0] < start:
                    last_pcs = chord_events[k][1]
                    k += 1
                while k < len(chord_events) and chord_events[k][0] < end:
                    last_pcs = chord_events[k][1]
                    k += 1
                p["gt_chord_pcs"] = last_pcs
            else:
                p["gt_chord_pcs"] = None

            # ---- existing melody note collection ----
            while i_note < N and n_ends[i_note] <= start:
                i_note += 1

            region_note_objs = []
            j = i_note
            while j < N and n_starts[j] < end:
                if n_ends[j] > start:
                    region_note_objs.append(notes[j])
                j += 1

            p["region_note_objs"] = region_note_objs
            p["region_notes"] = [n.nameWithOctave for n in region_note_objs]

        return pd.DataFrame(points, columns=[
            "measure",
            "local_offset",
            "offset",
            "region_start",
            "region_end",
            "region_notes",
            "region_note_objs",
            "gt_chord_pcs",
        ])

    def get_harmonization_grid(
        self,
        density: Density = "mid",
        *,
        apply_cadence: bool = True,
        chord_offsets: Optional[list[float]] = None,
    ) -> pd.DataFrame:
        if chord_offsets is None:
            chord_offsets = self.chord_offsets # could still be None
        grid = self.build_harmonization_grid(density=density, chord_offsets=chord_offsets)
        if apply_cadence and not grid.empty:
            grid = cadence.apply_cadence_weights(grid)
        return grid

    @property
    def harmonization_grid(self) -> pd.DataFrame:
        return self.get_harmonization_grid(density="mid", apply_cadence=True)


# %% Debug / test code
if __name__ == "__main__":
    melody_path = "./../test/doremi.musicxml"
    melody = Melody(melody_path)
    print("Melody loaded from:", melody_path)
    print("Time signature:", melody.time_signature.ratioString)
    print("Number of measures:", melody.num_measures)
    print("\nTop key candidates:")
    print(melody.global_key_candidates)
    print("\nHarmonization grid (sparse density):")
    grid = melody.get_harmonization_grid(density="sparse", apply_cadence=True)
    print(grid)

# %%
