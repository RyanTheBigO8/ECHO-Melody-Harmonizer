# role_to_function.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

from echo_harmonizer.chord_pool.chord_roles import ROLE_NAMES

Func = Literal["T", "S", "D", "CM", "UNK"]
RoleGroup = Literal["DIAT", "PAR", "APPLIED", "SUS", "CM", "N6", "IT6", "OTHER"]


@dataclass(frozen=True)
class RoleInfo:
    role_id: int
    name: str
    group: RoleGroup
    degree: Optional[int]  # 1..7 when applicable, else None
    func: Func


# ----------------------------
# Base function transition scores (edit freely)
# ----------------------------

# This is intentionally small/simple. You can tune later.
BASE_FUNC_SCORE: dict[tuple[Func, Func], float] = {
    ("T", "T"): 0.2,
    ("T", "S"): 0.4,
    ("T", "D"): 0.3,
    ("S", "T"): 0.2,
    ("S", "S"): 0.2,
    ("S", "D"): 0.5,
    ("D", "T"): 0.8,
    ("D", "S"): -0.5,
    ("D", "D"): 0.0,

    # CM defaults (chromatic fitness can treat these differently if desired)
    ("CM", "T"): 0.0,
    ("CM", "S"): 0.0,
    ("CM", "D"): 0.0,
    ("T", "CM"): 0.0,
    ("S", "CM"): 0.0,
    ("D", "CM"): 0.0,
    ("CM", "CM"): 0.2,

    ("UNK", "UNK"): -0.5,
}


def _func_score(a: Func, b: Func) -> float:
    return float(BASE_FUNC_SCORE.get((a, b), -0.5))


# ----------------------------
# Role parsing + role->function
# ----------------------------

def _parse_degree_suffix(name: str) -> Optional[int]:
    # supports "..._3" and ".../3"
    if "/" in name:
        try:
            return int(name.split("/")[-1])
        except Exception:
            return None
    if "_" in name:
        tail = name.split("_")[-1]
        try:
            return int(tail)
        except Exception:
            return None
    return None


def _degree_to_func(deg: int) -> Func:
    # very standard (editable):
    # T: I, iii, vi
    # S: ii, IV
    # D: V, vii
    if deg in (1, 3, 6):
        return "T"
    if deg in (2, 4):
        return "S"
    if deg in (5, 7):
        return "D"
    return "UNK"


def role_info_from_name(role_id: int, name: str) -> RoleInfo:
    n = str(name)
    HIDE_7TH_DEGS = {1, 3, 6}

    if n.startswith("diat_triad_"):
        deg = _parse_degree_suffix(n)
        func = _degree_to_func(deg or 0)
        return RoleInfo(role_id, n, "DIAT", deg, func)
    
    if n.startswith("diat_7th_"):
        deg = _parse_degree_suffix(n)
        func = "UNK" if (deg in HIDE_7TH_DEGS) else _degree_to_func(deg or 0)
        return RoleInfo(role_id, n, "DIAT", deg, func)

    if n.startswith("parallel_triad_"):
        deg = _parse_degree_suffix(n)
        func = _degree_to_func(deg or 0)
        return RoleInfo(role_id, n, "PAR", deg, func)
    
    if n.startswith("parallel_7th_"):
        deg = _parse_degree_suffix(n)
        func = "UNK" if (deg in HIDE_7TH_DEGS) else _degree_to_func(deg or 0)
        return RoleInfo(role_id, n, "PAR", deg, func)
    
    if n.startswith("V/") or n.startswith("V7/") or n.startswith("vii/") or n.startswith("vii7/"):
        # ?œapplied to degree x??
        deg = _parse_degree_suffix(n)
        return RoleInfo(role_id, n, "APPLIED", deg, "D")

    if n.startswith("sus2/") or n.startswith("sus4/"):
        deg = _parse_degree_suffix(n)
        return RoleInfo(role_id, n, "SUS", deg, "S")

    if n.startswith("CM_"):
        deg = _parse_degree_suffix(n)
        return RoleInfo(role_id, n, "CM", deg, "CM")

    if n == "N6":
        return RoleInfo(role_id, n, "N6", None, "S")

    if n == "It6":
        return RoleInfo(role_id, n, "IT6", None, "S")

    return RoleInfo(role_id, n, "OTHER", None, "UNK")


# Precompute RoleInfo by role_id (fast, no decode strings at runtime beyond lookup)
ROLE_INFO_BY_ID: list[RoleInfo] = [role_info_from_name(i, nm) for i, nm in enumerate(ROLE_NAMES)]


def iter_role_ids(mask: int):
    """Yield role_ids in ascending order (LSB to MSB)."""
    m = int(mask)
    while m:
        lsb = m & -m
        rid = lsb.bit_length() - 1
        yield rid
        m ^= lsb

def mask_has_function(mask: int, funcs: set[Func]) -> bool:
    """
    True iff mask contains ANY role whose mapped function is in funcs.
    Uses ROLE_INFO_BY_ID (no string scanning).
    """
    want = set(funcs)
    for rid in iter_role_ids(mask):
        if ROLE_INFO_BY_ID[int(rid)].func in want:
            return True
    return False


# ----------------------------
# Special tuning rules (including SUS resolution)
# ----------------------------

def transition_score_for_role_pair(prev_role_id: int, cur_role_id: int) -> float:
    """
    Computes score for a single prev_role -> cur_role pair:
      stage1: base function transition score
      stage2: tuning adjustments / overrides (may return -100)
    """
    prev = ROLE_INFO_BY_ID[int(prev_role_id)]
    cur = ROLE_INFO_BY_ID[int(cur_role_id)]

    # ---- stage1: function transition ----
    base = _func_score(prev.func, cur.func)

    # ---- stage2: tuning adjustments / overrides ----

    # (A) diat/par mixture tuning for ANY function transition
    if prev.group in ("DIAT", "PAR") and cur.group in ("DIAT", "PAR"):
        if prev.group == "PAR" and cur.group == "PAR":
            base += -0.2
        elif prev.group != cur.group:
            base += +0.1
        # diat->diat : +0

    # (B) Applied chords must resolve to their target degree (treat target as temporary tonic)
    # V/x, V7/x, vii/x, vii7/x -> diat_y: reward ONLY if y == x, else -10
    if prev.group == "APPLIED":
        if cur.group not in ("DIAT"):
            return -10.0
        if prev.degree is None or cur.degree is None or prev.degree != cur.degree:
            return -10.0
        # ?œapplied dominant resolves to temporary tonic??
        return _func_score("D", "T") + 0.2  # deterministic, simple

    # (C) N6 and It6 must go to diat_V (degree 5), otherwise -10
    if prev.group == "N6":
        if cur.group not in ("DIAT"):
            return -10.0
        if cur.degree != 5:
            return -10.0
        return _func_score("S", "D") + 0.3

    if prev.group == "IT6":
        if cur.group not in ("DIAT"):
            return -10.0
        if cur.degree != 5:
            return -10.0
        return _func_score("S", "D") + 0.3

    # (D) diat_x -> CM_*/y only allowed if x == y, else -10
    if cur.group == "CM":
        if (prev.group != "DIAT"):
            return -10.0
        if prev.degree is None or cur.degree is None or prev.degree != cur.degree:
            return -10.0
        # allow with no extra bonus
        return _func_score(prev.func, "CM")

    # (E) SUS resolution rule purely by roles:
    # sus2/x or sus4/x -> diat_y (triad or 7th): +0.1 if x==y else -10
    if prev.group == "SUS":
        if cur.group not in ("DIAT"):
            return -10.0
        if prev.degree is None or cur.degree is None or prev.degree != cur.degree:
            return -10.0
        return base + 0.1

    return float(base)


# ----------------------------
# Chord-to-chord best transition (Option A tie-break)
# ----------------------------

def best_transition_score(prev_role_mask: int, cur_role_mask: int) -> float:
    """
    Return ONLY the best score (float).

    Tie-break Option A (deterministic):
      - iterate prev roles ascending, cur roles ascending
      - update only when score > best_score
      => first max wins deterministically.
    """
    pm = int(prev_role_mask)
    cm = int(cur_role_mask)

    if pm == 0 or cm == 0:
        return 0.0

    best = -1e18
    for pr in iter_role_ids(pm):
        for cr in iter_role_ids(cm):
            s = transition_score_for_role_pair(pr, cr)
            if s > best:
                best = s

    return float(best)
