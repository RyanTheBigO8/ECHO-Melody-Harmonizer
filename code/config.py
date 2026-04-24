# config.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

# ----------------------------
# Public CLI-facing types
# ----------------------------

Style = Literal["diatonic", "chromatic", "non_functional", "atonal", "yeh"]
StyleArg = Literal["diatonic", "chromatic", "non-functional", "non_functional", "atonal", "yeh", "auto"]

Density = Literal["dense", "mid", "sparse"]
DensityArg = Literal["dense", "mid", "sparse", "auto"]


# ----------------------------
# Role toggles for tonal/chromatic chord policies
# ----------------------------

RoleGroup = Literal[
    "diat_triads",
    "diat_7ths",
    "sus_chords",
    "parallel_triads",
    "parallel_7ths",
    "secondary_dominants",
    "chromatic_mediants",
    "neapolitan",
    "it6",
]

ROLE_GROUP_LABELS: list[tuple[RoleGroup, str]] = [
    ("diat_triads", "Diatonic triads (I..vii)  [REQUIRED]"),
    ("diat_7ths", "Diatonic 7ths (I7..vii7)"),
    ("sus_chords", "Suspended chords (sus2/sus4)"),
    ("parallel_triads", "Parallel-key triads"),
    ("parallel_7ths", "Parallel-key 7ths"),
    ("secondary_dominants", "Secondary dominants / leading-tone (V/x, vii/x, etc.)"),
    ("chromatic_mediants", "Chromatic mediants"),
    ("neapolitan", "Neapolitan (N6)"),
    ("it6", "Italian augmented 6th (It+6)"),
]


# ----------------------------
# Key spec (keep for compatibility with melody.py)
# ----------------------------

@dataclass(frozen=True)
class KeySpec:
    tonic: str
    mode: str                 # "major" | "minor"
    tonic_pc: int
    confidence: float


# ----------------------------
# GA / run configs
# ----------------------------

@dataclass(frozen=True)
class GAConfig:
    pop_size: int = 1000
    max_gens: int = 200
    elite_frac: float = 0.10

    pc_start: float = 0.90
    pc_end: float = 0.90
    pm_start: float = 0.10
    pm_end: float = 0.10

    warmup_frac: float = 0.15
    early_stop: bool = False
    stagnation_gens: int = 30
    stagnation_improve: float = 0.03

    schedule_mode: Literal["linear", "adaptive"] = "linear"
    adaptive_window: int = 10
    adaptive_target_improve: float = 0.10


@dataclass(frozen=True)
class RunConfig:
    # Logging
    log_every: int = 10
    log_best_terms: bool = True
    use_tqdm: bool = True

    # Saving
    save_best_every: int = 20
    save_top_frac: float = 0.05
    save_top_every: int = 50
    save_final_top: bool = True

    # Animation tracing
    trace_pop_fitness: bool = True
    trace_every: int = 1

    # Threads (main may override)
    n_threads: int = 1


# ----------------------------
# Chord policy config
# ----------------------------

ChordPolicyName = Literal["diatonic", "chromatic", "atonal"]

@dataclass(frozen=True)
class ChordPolicyConfig:
    """
    User preference only.

    - name: which policy preset to load (implemented in pool_policy.py)
    - enabled_groups: which role groups the user wants enabled when that preset is loaded.
      Interpretation is up to pool_policy.py (and/or main.py).
      If None: pool_policy.py uses its own defaults for that preset.
    """
    name: ChordPolicyName
    enabled_groups: Optional[tuple[str, ...]] = None


# ----------------------------
# Presets (high-level wiring)
# ----------------------------

@dataclass(frozen=True)
class PresetConfig:
    """
    High-level preset:
      - style: which harmonization style user asked for
      - density: harmonization grid density
      - chord_policy: name of the chord policy to load
      - fitness: name of the fitness implementation to load
      - weights: default weights for that fitness
    """
    style: Style
    density: Density
    chord_policy: ChordPolicyName
    fitness: str
    weights: dict[str, float]


# ----------------------------
# Arg parsing
# ----------------------------

def parse_style_arg(s: str | None) -> Style:
    """
    CLI accepts:
      - 'auto' -> chromatic
      - 'non-functional' -> non_functional
    """
    if s is None or s == "auto":
        return "chromatic"
    if s == "non-functional":
        return "non_functional"
    if str(s).lower() == "yeh":
        return "yeh"
    if s in ("diatonic", "chromatic", "atonal", "non_functional", "yeh"):
        return s  # type: ignore
    raise ValueError(f"Invalid style: {s}")


def parse_density_arg(s: str | None) -> Density:
    if s is None or s == "auto":
        return "mid"
    if s in ("dense", "mid", "sparse"):
        return s  # type: ignore
    raise ValueError(f"Invalid density: {s}")


# ----------------------------
# Default weights per fitness
# ----------------------------

DEFAULT_WEIGHTS_ATONAL = {
    "w_overlap": 0.8,
    "w_dist": 0.6,
    "w_dist_tonal": 0.5,
    "w_repeat": 0.5,
    "w_pc_change": 0.1,
    "w_sus_resolution": 0.20,
    "w_coverage": 0.8,
}

DEFAULT_WEIGHTS_DIATONIC = {
    "w_overlap": 1.0,
    "w_dist": 0.3,
    "w_dist_tonal": 0.0,
    "w_repeat": 0.1,
    "w_pc_change": 0.0,
    "w_func_transition": 0.8,
    "w_leading_tone_resolve": 0.2,
    "w_cadence": 0.75,
    "w_coverage": 0.5,
}

DEFAULT_WEIGHTS_CHROMATIC = {
    **DEFAULT_WEIGHTS_DIATONIC,
}

DEFAULT_WEIGHTS_NON_FUNCTIONAL = {
    **DEFAULT_WEIGHTS_DIATONIC,
    "w_func_transition": 0.1,
}

DEFAULT_WEIGHTS_YEH = {
    # Yeh targets (fitness_yeh reads w_* only; targets are constants in the evaluator)
    "w_ctnctr": 1.0,
    "w_pcs": 1.0,
    "w_mctd": 1.0,

    # Optional: explicitly turn off base terms if you later reuse them somewhere
    "w_overlap": 0.0,
    "w_dist": 0.0,
    "w_dist_tonal": 0.0,
    "w_repeat": 0.0,
    "w_pc_change": 0.0,
    "w_coverage": 0.0,
}


# ----------------------------
# Preset registry
# ----------------------------

PRESETS: dict[Style, PresetConfig] = {
    "diatonic": PresetConfig(
        style="diatonic",
        density="mid",
        chord_policy="diatonic",
        fitness="fitness_tonal",
        weights=DEFAULT_WEIGHTS_DIATONIC,
    ),
    "chromatic": PresetConfig(
        style="chromatic",
        density="mid",
        chord_policy="chromatic",
        fitness="fitness_tonal",
        weights=DEFAULT_WEIGHTS_CHROMATIC,
    ),
    "non_functional": PresetConfig(
        style="non_functional",
        density="mid",
        chord_policy="chromatic",
        fitness="fitness_tonal",
        weights=DEFAULT_WEIGHTS_NON_FUNCTIONAL,
    ),
    "atonal": PresetConfig(
        style="atonal",
        density="mid",
        chord_policy="atonal",
        fitness="fitness_atonal",
        weights=DEFAULT_WEIGHTS_ATONAL,
    ),
    "yeh": PresetConfig(
        style="yeh",
        density="mid",
        chord_policy="diatonic",
        fitness="fitness_yeh",
        weights=DEFAULT_WEIGHTS_YEH,
    ),
}
