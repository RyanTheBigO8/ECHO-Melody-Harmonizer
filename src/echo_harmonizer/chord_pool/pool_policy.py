# pool_policy.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Optional

from echo_harmonizer.chord_pool.chord_roles import bit

ChordPolicyName = Literal["diatonic", "chromatic", "atonal"]

# ----------------------------
# Policy group specs (for UI + defaults)
# ----------------------------

@dataclass(frozen=True)
class GroupSpec:
    key: str
    label: str
    default_on: bool
    required: bool


_POLICY_GROUPS: dict[ChordPolicyName, list[GroupSpec]] = {
    "diatonic": [
        GroupSpec("diat_triads", "Diatonic triads (I..vii)  [REQUIRED]", True, True),
        GroupSpec("diat_7ths", "Diatonic 7ths (I7..vii7)", True, False),
        GroupSpec("sus_chords", "Suspended chords (sus2/sus4)", True, False),
    ],
    "chromatic": [
        GroupSpec("diat_triads", "Diatonic triads (I..vii)  [REQUIRED]", True, True),
        GroupSpec("diat_7ths", "Diatonic 7ths (I7..vii7)", True, False),
        GroupSpec("sus_chords", "Suspended chords (sus2/sus4)", True, False),
        GroupSpec("parallel_triads", "Parallel-key triads", True, False),
        GroupSpec("parallel_7ths", "Parallel-key 7ths", True, False),
        GroupSpec("secondary_dominants", "Secondary dominants / leading-tone (V/x, vii/x, etc.)", True, False),
        GroupSpec("chromatic_mediants", "Chromatic mediants", True, False),
        GroupSpec("neapolitan", "Neapolitan (N6)", True, False),
        GroupSpec("it6", "Italian augmented 6th (It+6)", True, False),
    ],
    "atonal": [
        GroupSpec("triads", "Triads (6 qualities)  [REQUIRED]", True, True),
        GroupSpec("sevenths", "7th chords (5 qualities)", True, False),
    ],
}


def list_policy_groups(policy: ChordPolicyName) -> list[GroupSpec]:
    """UI helper: main.py uses this to print options and map indices -> group keys."""
    return _POLICY_GROUPS[policy][:]


def effective_enabled_groups(policy: ChordPolicyName, user_enabled: Optional[Iterable[str]]) -> set[str]:
    """
    Apply policy defaults if user_enabled is None; otherwise treat user_enabled as the set of groups ON.
    Always enforce required groups ON.
    Return the final set of enabled group keys.
    """
    specs = _POLICY_GROUPS[policy]
    allowed = {s.key for s in specs}
    required = {s.key for s in specs if s.required}

    if user_enabled is None:
        enabled = {s.key for s in specs if s.default_on}
    else:
        enabled = {str(x) for x in user_enabled if str(x) in allowed}

    enabled |= required
    return enabled
