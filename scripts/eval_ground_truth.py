# eval_ground_truth.py
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd

from echo_harmonizer.melody.preprocess_melody import Melody
from echo_harmonizer.metrics import compute_CTnCTR_pcs, compute_PCS_pcs, compute_MCTD_pcs


def main(path: str, step_qL: float = 0.25) -> None:
    mel = Melody(path)

    # IMPORTANT: keep cadence off for "ground truth" evaluation unless you have a reason otherwise
    table = mel.get_harmonization_grid(apply_cadence=False)

    if "gt_chord_pcs" not in table.columns:
        raise ValueError("No gt_chord_pcs column found. Did the input include a chord part?")

    chords_pcs = table["gt_chord_pcs"].tolist()

    # if the piece begins with no chord, early regions may be None
    # (optional) forward-fill so every region has something:
    last = None
    for i, x in enumerate(chords_pcs):
        if x is not None:
            last = x
        else:
            chords_pcs[i] = last

    ctn = compute_CTnCTR_pcs(chords_pcs, table, mel.melody_stream)
    pcs = compute_PCS_pcs(chords_pcs, table, mel.melody_stream, step_qL=step_qL)
    mctd = compute_MCTD_pcs(chords_pcs, table, mel.melody_stream)

    print(f"[GT] CTnCTR={ctn:.6f}  PCS={pcs:.6f}  MCTD={mctd:.6f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=str, help="Path to melody+chords MIDI/MusicXML")
    args = ap.parse_args()
    main(args.path)
