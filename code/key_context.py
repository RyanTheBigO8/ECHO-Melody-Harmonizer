# key_context.py
from __future__ import annotations
from typing import Literal, Optional

Mode = Literal["major", "minor"]

_MAJOR_SCALE = (0, 2, 4, 5, 7, 9, 11)
_HARM_MINOR  = (0, 2, 3, 5, 7, 8, 11)
_NAT_MINOR   = (0, 2, 3, 5, 7, 8, 10)

def _scale_pcs(tonic_pc: int, mode: Mode) -> list[int]:
    tonic_pc %= 12
    rel = _MAJOR_SCALE if mode == "major" else _HARM_MINOR
    return [int((tonic_pc + x) % 12) for x in rel]

def _parallel_scale_pcs(tonic_pc: int, mode: Mode) -> list[int]:
    tonic_pc %= 12
    # parallel minor uses natural minor; parallel major uses major
    rel = _NAT_MINOR if mode == "major" else _MAJOR_SCALE
    return [int((tonic_pc + x) % 12) for x in rel]

def _triad_set_for_degree(scale: list[int], degree: int) -> frozenset[int]:
    i = (degree - 1) % 7
    pcs = {scale[i], scale[(i + 2) % 7], scale[(i + 4) % 7]}
    return frozenset(int(x) % 12 for x in pcs)

def _seventh_set_for_degree(scale: list[int], degree: int) -> frozenset[int]:
    i = (degree - 1) % 7
    pcs = {scale[i], scale[(i + 2) % 7], scale[(i + 4) % 7], scale[(i + 6) % 7]}
    return frozenset(int(x) % 12 for x in pcs)


class KeyContext:
    """
    Stores all key-dependent lookup tables for ONE key.
    Construct on demand: KeyContext(0, "major") for C major.
    """

    def __init__(self, tonic_pc: int, mode: Mode):
        self.tonic_pc = int(tonic_pc) % 12
        self.mode: Mode = mode

        # Main key scale: major or harmonic minor
        self.diat_scale: list[int] = _scale_pcs(self.tonic_pc, self.mode)
        self.diat_set: frozenset[int] = frozenset(self.diat_scale)
        self.degree_pc: list[int] = self.diat_scale[:]  # degree 1..7 at indices 0..6

        # Parallel key scale (major <-> harmonic minor)
        self.par_scale: list[int] = _parallel_scale_pcs(self.tonic_pc, self.mode)
        self.par_set: frozenset[int] = frozenset(self.par_scale)

        # Exact diatonic/parallel triad & seventh sets by degree
        self.diat_triad_by_deg: list[frozenset[int]] = [_triad_set_for_degree(self.diat_scale, d) for d in range(1, 8)]
        self.diat_7th_by_deg:   list[frozenset[int]] = [_seventh_set_for_degree(self.diat_scale, d) for d in range(1, 8)]
        self.par_triad_by_deg:  list[frozenset[int]] = [_triad_set_for_degree(self.par_scale, d) for d in range(1, 8)]
        self.par_7th_by_deg:    list[frozenset[int]] = [_seventh_set_for_degree(self.par_scale, d) for d in range(1, 8)]

        # Fast exact-set -> degree maps
        self.diat_triad_map = {self.diat_triad_by_deg[d - 1]: d for d in range(1, 8)}
        self.diat_7th_map   = {self.diat_7th_by_deg[d - 1]: d for d in range(1, 8)}
        self.par_triad_map  = {self.par_triad_by_deg[d - 1]: d for d in range(1, 8)}
        self.par_7th_map    = {self.par_7th_by_deg[d - 1]: d for d in range(1, 8)}

        # Applied chord roots per target degree:
        # V/x root is a fifth above target degree root
        # vii/x root is a semitone below target degree root
        self.V_root_of_deg: list[int] = [int((self.degree_pc[d - 1] + 7) % 12) for d in range(1, 8)]
        self.vii_root_of_deg: list[int] = [int((self.degree_pc[d - 1] + 11) % 12) for d in range(1, 8)]

        # Root->degree maps for quick applied lookup
        self.V_degree_by_root = {self.V_root_of_deg[d - 1]: d for d in range(1, 8)}
        self.vii_degree_by_root = {self.vii_root_of_deg[d - 1]: d for d in range(1, 8)}

        # Sus chords: root must be degree root (same mapping)
        self.degree_by_root = {self.degree_pc[d - 1]: d for d in range(1, 8)}

        # Chromatic mediant roots relative to each degree root
        # We store a mapping: root_pc -> list of (tag, degree)
        self.cm_by_root: dict[int, list[tuple[str, int]]] = {pc: [] for pc in range(12)}
        for d in range(1, 8):
            base = self.degree_pc[d - 1]
            self.cm_by_root[(base - 4) % 12].append(("CM_M3-", d))
            self.cm_by_root[(base - 3) % 12].append(("CM_m3-", d))
            self.cm_by_root[(base + 3) % 12].append(("CM_m3+", d))
            self.cm_by_root[(base + 4) % 12].append(("CM_M3+", d))

        # Specials
        self.bII_pc: int = int((self.tonic_pc + 1) % 12)
        self.it6_set: frozenset[int] = frozenset({self.tonic_pc, (self.tonic_pc + 6) % 12, (self.tonic_pc + 8) % 12})
