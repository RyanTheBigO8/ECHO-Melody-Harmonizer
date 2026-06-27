# main.py
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional
import time
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd

from echo_harmonizer.melody.preprocess_melody import Melody

from echo_harmonizer.config import (
    PRESETS,
    GAConfig,
    RunConfig,
    ChordPolicyConfig,
    KeySpec,
    parse_density_arg,
    parse_style_arg,
)
from echo_harmonizer.chord_pool.pool_policy import list_policy_groups
from echo_harmonizer.chord_pool.build_pool import prepare_harmonization
from echo_harmonizer.ga import HarmonizeGA

from echo_harmonizer.fitness.fitness_tonal import FitnessEvaluatorTonal
from echo_harmonizer.fitness.fitness_atonal import FitnessEvaluatorAtonal
from echo_harmonizer.fitness.fitness_yeh import FitnessEvaluatorYeh

from echo_harmonizer.chord_pool.chords_master import MASTER_CHORD_TABLE
from echo_harmonizer.chord_pool.chord_dist import load_or_build_distance_matrix, load_or_build_tonal_distance_matrix
# Precompute/load chord distance matrices
CHORD_DIST_MATRIX = load_or_build_distance_matrix(MASTER_CHORD_TABLE)
CHORD_TONAL_DISTANCE_MATRIX = load_or_build_tonal_distance_matrix(MASTER_CHORD_TABLE)

# plotting (assumed to exist, as requested)
from echo_harmonizer.plot_history import (
    plot_single_trial_fitness_curve,
    plot_single_trial_term_curves,
    plot_multi_trial_fitness_curve,
    animate_population_fitness_scatter,
)

# ----------------------------
# CLI
# ----------------------------

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="GA melody harmonization")

    p.add_argument("melody_path", type=str, help="Input melody filepath")

    p.add_argument(
        "--style",
        type=str,
        default="auto",
        help="diatonic|chromatic|non-functional|atonal|yeh|auto (default: auto)",
    )
    p.add_argument(
        "--density",
        type=str,
        default="auto",
        help="dense|mid|sparse|auto (default: auto)",
    )
    p.add_argument(
        "--auto",
        action="store_true",
        help="Auto presets for everything (style, density, key, role groups). No prompts.",
    )
    p.add_argument(
        "-n",
        "--n-runs",
        type=int,
        default=1,
        help="Number of GA runs (default: 1). If >1, averages across trials.",
    )
    p.add_argument(
        "-t",
        "--threads",
        type=int,
        default=1,
        help="Threads for region fitness recomputation (default: 1)",
    )

    return p

# ----------------------------
# Simple console UI
# ----------------------------

def _print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def _print_user_prefs(*, melody_path: str, style: str, density: str, auto: bool, n_runs: int, threads: int) -> None:
    _print_header("User preferences")
    print(f"melody_path : {melody_path}")
    print(f"style       : {style}")
    print(f"density     : {density}")
    print(f"--auto      : {auto}")
    print(f"n_runs      : {n_runs}")
    print(f"threads     : {threads}")

def _print_key_candidates(df: pd.DataFrame) -> None:
    _print_header("Detected global key candidates")
    if df.empty:
        print("(no candidates)")
        return

    lines = []
    for i, r in df.reset_index(drop=True).iterrows():
        tonic = str(r.get("tonic", ""))
        mode = str(r.get("mode", ""))
        conf = float(r.get("confidence", 0.0))
        tonic_pc = r.get("tonic_pc", None)
        tonic_pc_s = f", tonic_pc={int(tonic_pc)}" if tonic_pc is not None and tonic_pc == tonic_pc else ""
        lines.append(f"[{i}] {tonic} {mode} (confidence={conf:.3f}{tonic_pc_s})")
    print("\n".join(lines))

def _prompt_key_index(df: pd.DataFrame) -> int:
    # Enter/'auto' => 0
    while True:
        s = input("Choose key index (0-based), or 'auto' (Enter = auto): ").strip().lower()
        if s == "" or s == "auto":
            return 0
        try:
            k = int(s)
            if 0 <= k < len(df):
                return k
        except Exception:
            pass
        print(f"Invalid. Enter an integer 0..{len(df)-1}, or press Enter.")

def _keyspec_from_candidate_row(row: pd.Series) -> KeySpec:
    tonic = str(row["tonic"])
    mode = str(row["mode"])
    conf = float(row.get("confidence", 0.0))

    if "tonic_pc" in row and row["tonic_pc"] == row["tonic_pc"]:
        tonic_pc = int(row["tonic_pc"])
    else:
        # minimal fallback if tonic_pc missing
        from music21 import pitch as m21pitch
        tonic_pc = int(m21pitch.Pitch(tonic.replace("b", "-")).pitchClass)

    return KeySpec(tonic=tonic, mode=mode, tonic_pc=int(tonic_pc) % 12, confidence=conf)

def _print_policy_groups(policy_name: str) -> None:
    _print_header(f"Chord-role groups for policy '{policy_name}'")
    specs = list_policy_groups(policy_name)  # type: ignore[arg-type]
    for i, s in enumerate(specs):
        req = " (required)" if s.required else ""
        default = " [default ON]" if s.default_on else " [default OFF]"
        print(f"[{i}] {s.label}{req}{default}")

def _prompt_enabled_groups(policy_name: str) -> Optional[tuple[str, ...]]:
    """
    Return:
      None => use policy defaults
      tuple(keys...) => treat as enabled groups ON (required enforced by pool_policy)
    """
    specs = list_policy_groups(policy_name)  # type: ignore[arg-type]

    while True:
        s = input("Enter indices to ENABLE (e.g. 0 2 4), or 'auto' (Enter = auto): ").strip().lower()
        if s == "" or s == "auto":
            return None

        toks = [t for t in s.replace(",", " ").split() if t]
        try:
            idxs = sorted(set(int(t) for t in toks))
        except Exception:
            print("Invalid. Enter integers like: 0 2 4  (or press Enter).")
            continue

        if any(i < 0 or i >= len(specs) for i in idxs):
            print(f"Out of range. Valid indices: 0..{len(specs)-1}")
            continue

        return tuple(specs[i].key for i in idxs)
    
def compute_pop_size(table: pd.DataFrame, *, min_pop: int = 200, max_pop: int = 2000) -> int:
    H = len(table)
    if H <= 0 or "acceptable_chord_ids" not in table.columns:
        return int(min_pop)

    acceptable = table["acceptable_chord_ids"].tolist()
    A = float(np.mean([len(x) for x in acceptable])) if acceptable else 0.0

    eff = max(1.0, float(H) * float(A))
    pop = int(min_pop + 30.0 * (eff ** 0.5))

    # keep even for pairing
    if pop % 2 == 1:
        pop += 1

    return int(max(min_pop, min(max_pop, pop)))


def make_run_output_dir(melody_path: str, n_runs: int) -> Path:
    melody_stem = Path(melody_path).stem
    timestamp = datetime.now().strftime("%m-%d_%H-%M-%S")
    if n_runs > 1:
        timestamp = f"{timestamp}_n{n_runs}"
    base_dir = ROOT / "output" / melody_stem / timestamp

    out_dir = base_dir
    suffix = 1
    while out_dir.exists():
        out_dir = base_dir.with_name(f"{base_dir.name}_{suffix:02d}")
        suffix += 1

    out_dir.mkdir(parents=True, exist_ok=False)
    return out_dir


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    args = _build_argparser().parse_args()

    style = parse_style_arg(args.style)
    density = parse_density_arg(args.density)

    n_runs = max(1, int(args.n_runs))
    threads = max(1, int(args.threads))

    _print_user_prefs(
        melody_path=args.melody_path,
        style=style,
        density=density,
        auto=bool(args.auto),
        n_runs=n_runs,
        threads=threads,
    )

    preset = PRESETS[style]
    _print_header("Resolved preset wiring")
    print(f"preset.style       : {preset.style}")
    print(f"preset.density     : {preset.density}  (CLI override: {density})")
    print(f"preset.chord_policy: {preset.chord_policy}")
    print(f"preset.fitness     : {preset.fitness}")

    # Load melody
    melody = Melody(Path(args.melody_path))

    key_df: pd.DataFrame = melody.global_key_candidates
    _print_key_candidates(key_df)

    # Set output directory. Timestamped runs avoid overwriting previous outputs.
    out_dir = make_run_output_dir(args.melody_path, n_runs)

    # Key selection (non-atonal only)
    selected_key: Optional[KeySpec] = None
    if style != "atonal":
        if key_df.empty:
            raise ValueError("No global key candidates found.")
        key_idx = 0 if args.auto else _prompt_key_index(key_df)
        selected_key = _keyspec_from_candidate_row(key_df.iloc[key_idx])
        _print_header("Selected key")
        print(asdict(selected_key))

    # Role group toggles
    policy_name = preset.chord_policy
    _print_policy_groups(policy_name)
    enabled_groups = None if args.auto else _prompt_enabled_groups(policy_name)

    policy_cfg = ChordPolicyConfig(name=policy_name, enabled_groups=enabled_groups)
    _print_header("Chord policy config")
    print(asdict(policy_cfg))

    # Prepare pool + table
    pool, table = prepare_harmonization(
        melody,
        policy_cfg=policy_cfg,
        density=density,
        selected_key=selected_key,
        master_table=MASTER_CHORD_TABLE,
        role_mask_table=None,
    )
    auto_pop = compute_pop_size(table, min_pop=200, max_pop=2000)

    _print_header("Prepared harmonization")
    print(f"grid regions: {len(table)}")
    print(f"pool chords : {len(pool)}")
    print(f"pop size    : {auto_pop}")
    if selected_key is not None:
        print(f"key         : {selected_key.tonic} {selected_key.mode}")

    # Save harmonization table as CSV for reference
    id2name = dict(zip(pool["id"].astype(int), pool["symbol"].astype(str)))  # or "name" if you use that column
    cols = [c for c in ("measure","offset","region_end","region_note_names","acceptable_chord_ids") if c in table.columns]
    df = table[cols].copy()
    df["acceptable_chords"] = df["acceptable_chord_ids"].apply(lambda xs: [id2name.get(int(i), str(i)) for i in (xs or [])])
    df.drop(columns=["acceptable_chord_ids"], inplace=True)
    df.to_csv(out_dir / "harmonization_table.csv", index=False)

    # Fitness evaluator
    if preset.fitness == "fitness_tonal":
        if selected_key is None:
            raise ValueError("fitness_tonal requires selected_key.")
        evaluator = FitnessEvaluatorTonal(
            table,
            master_table=MASTER_CHORD_TABLE,
            tonic_pc=selected_key.tonic_pc,
            mode=selected_key.mode,
            chord_dist=CHORD_DIST_MATRIX,
            chord_tonal_dist=CHORD_TONAL_DISTANCE_MATRIX,
            weights=preset.weights,
        )
    elif preset.fitness == "fitness_atonal":
        evaluator = FitnessEvaluatorAtonal(
            table,
            master_table=MASTER_CHORD_TABLE,
            chord_dist=CHORD_DIST_MATRIX,
            chord_tonal_dist=CHORD_TONAL_DISTANCE_MATRIX,
            weights=preset.weights,
        )
    elif preset.fitness == "fitness_yeh":
        evaluator = FitnessEvaluatorYeh(
            table,
            master_table=MASTER_CHORD_TABLE,
            melody_stream=melody.melody_stream,
            weights=preset.weights,
        )
    else:
        raise ValueError(f"Unknown preset.fitness: {preset.fitness}")
    
    regions = table[["region_start", "region_end"]].to_numpy(dtype=float)
    save_context = {
        "melody_path": str(Path(args.melody_path)),
        "region_start": regions[:, 0].tolist(),
        "region_end": regions[:, 1].tolist(),
    }
    ga = HarmonizeGA(
        evaluator=evaluator,
        harmonization_table=table,
        seed=42,
        n_threads=threads,
        save_context=save_context,
    )

    ga_cfg = GAConfig()
    run_cfg = RunConfig(n_threads=threads)

    # Shared kwargs (no parameter soup at call sites)
    base_run_kwargs = dict(
        pop_size=auto_pop,
        max_gens=ga_cfg.max_gens,
        elite_frac=ga_cfg.elite_frac,
        pc_start=ga_cfg.pc_start,
        pc_end=ga_cfg.pc_end,
        pm_start=ga_cfg.pm_start,
        pm_end=ga_cfg.pm_end,
        warmup_frac=ga_cfg.warmup_frac,
        early_stop=ga_cfg.early_stop,
        stagnation_gens=ga_cfg.stagnation_gens,
        stagnation_improve=ga_cfg.stagnation_improve,
        schedule_mode=ga_cfg.schedule_mode,
        adaptive_window=ga_cfg.adaptive_window,
        adaptive_target_improve=ga_cfg.adaptive_target_improve,
        use_tqdm=run_cfg.use_tqdm,
        log_every=run_cfg.log_every,
        log_best_terms=run_cfg.log_best_terms,
        save_best_every=0,
        save_top_frac=run_cfg.save_top_frac,
        save_top_every=0,
        save_final_top=run_cfg.save_final_top,
        trace_pop_fitness=run_cfg.trace_pop_fitness,
        trace_every=run_cfg.trace_every,
    )

    if n_runs == 1:
        run_kwargs = dict(base_run_kwargs)
        run_kwargs["out_dir"] = out_dir

        t0 = time.process_time()
        best, log_df, extras = ga.run(**run_kwargs)
        t_end = time.process_time() - t0

        _print_header("GA result (single run)")
        print(f"ga_cputime_sec         : {t_end:.3f}")
        print(f"best_total_fitness     : {best.total_fitness:.6f}")
        print(f"mean_region_fitness    : {best.total_fitness / max(1, len(table)):.6f}")
        print(f"out_dir                : {out_dir}")

        # Save log CSV
        log_df.to_csv(out_dir / "log.csv", index=False)

        # Plots + GIF
        print("\nSaving plots and animations...")
        plot_single_trial_fitness_curve(log_df, save_path=str(out_dir / "fitness_anytime.png"))
        plot_single_trial_term_curves(log_df, save_path=str(out_dir / "fitness_terms.png"))

        pop_hist = extras.get("pop_fitness_history", None)
        if pop_hist:
            animate_population_fitness_scatter(pop_hist, save_path=str(out_dir / "pop_fitness.gif"))
        
        print("✅Done.")

    else:
        # For multi-run: no saving/trace spam; early_stop off
        run_kwargs = dict(base_run_kwargs)
        run_kwargs.update(
            dict(
                pop_size=ga_cfg.pop_size,
                use_tqdm=True,
                out_dir=out_dir,
                trace_pop_fitness=False,
                early_stop=False,
            )
        )

        t0 = time.process_time()
        # df_avg = ga.run_n_trials(
        #     n_trials=n_runs,
        #     run_kwargs=run_kwargs,
        #     base_seed=0,
        # )
        anytime_df, metrics_df = ga.run_n_trials_analysis(
            n_trials=n_runs,
            run_kwargs=run_kwargs,
            base_seed=0,
            melody_stream=melody.melody_stream,  # or omit if evaluator has .melody_stream
        )
        t_end = time.process_time() - t0

        _print_header("GA result (multi-run average)")
        print(f"total_cputime_sec      : {t_end:.3f}  (for {n_runs} runs)")
        anytime_df.to_csv(out_dir / "avg_curve.csv", index=False)

        print("\nSaving averaged plots...")
        plot_multi_trial_fitness_curve(anytime_df, save_path=str(out_dir / "fitness_anytime_multi.png"))
        print("✅Done.")

        
if __name__ == "__main__":
    main()
