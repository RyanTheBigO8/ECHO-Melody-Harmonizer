# ga.py
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import heapq
import pickle
import random

import numpy as np
import pandas as pd

from fitness.fitness_protocol import FitnessEvaluatorProtocol

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

try:
    from colorama import Fore, Style
except Exception:  # pragma: no cover
    Fore = Style = None


# ============================================================
# Data container
# ============================================================

@dataclass
class Individual:
    genes: list[int]
    region_fitness: list[float]
    global_fitness: float
    total_fitness: float


# ============================================================
# Helpers for saving
# ============================================================

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _save_pickle(obj: Any, path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


# ============================================================
# GA driver class
# ============================================================

class HarmonizeGA:
    """
    Genetic algorithm search for harmonization.

    One-sided dependency:
      region fitness at i depends on genes[i] and genes[i-1].
      Therefore, if gene k changes => affected regions {k, k+1}.
    """

    def __init__(
        self,
        evaluator: FitnessEvaluatorProtocol,
        harmonization_table: pd.DataFrame,
        *,
        save_context: dict[str, Any] | None = None,
        seed: int = 0,
        n_threads: int = 1,
    ):
        self.evaluator = evaluator
        self.table = harmonization_table
        self.H = len(harmonization_table)
        self.save_context = save_context or {}
        self.rng = random.Random(seed)

        self.n_threads = int(n_threads)
        self._executor: Optional[ThreadPoolExecutor] = None
        if self.n_threads > 1:
            self._executor = ThreadPoolExecutor(max_workers=self.n_threads)

        print(f"[GA] Using {self.n_threads} thread(s) for fitness recomputation.")

        if "acceptable_chord_ids" not in self.table.columns:
            raise ValueError("harmonization_table must contain column 'acceptable_chord_ids'")

        self.acceptable: list[list[int]] = self.table["acceptable_chord_ids"].tolist()
        for i, opts in enumerate(self.acceptable):
            if not opts:
                raise ValueError(f"acceptable_chord_ids at region {i} is empty; GA cannot initialize.")

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    # ----------------------------
    # Scheduling
    # ----------------------------

    @staticmethod
    def _schedule_linear_with_warmup(gen: int, max_gens: int, start: float, end: float, warmup_gens: int) -> float:
        if gen < warmup_gens:
            return start
        if max_gens <= warmup_gens + 1:
            return end
        t = (gen - warmup_gens) / (max_gens - warmup_gens - 1)
        t = max(0.0, min(1.0, t))
        return (1.0 - t) * start + t * end

    @staticmethod
    def _schedule_adaptive_with_warmup(gen: int, start: float, end: float, warmup_gens: int, progress: float) -> float:
        if gen < warmup_gens:
            return start
        progress = max(0.0, min(1.0, progress))
        return (1.0 - progress) * start + progress * end

    @staticmethod
    def _tournament_k(pop_size: int) -> int:
        if pop_size > 1500:
            return 4
        if pop_size > 1000:
            return 3
        return 2

    # ----------------------------
    # Population init
    # ----------------------------

    def init_population(self, pop_size: int) -> list[Individual]:
        pop: list[Individual] = []
        full_idxs = list(range(self.H))

        for _ in range(int(pop_size)):
            genes = [self.rng.choice(opts) for opts in self.acceptable]

            # compute region fitnesses
            pairs = self._compute_region_fitness_many(genes, full_idxs)
            rf = [0.0] * self.H
            for i, val in pairs:
                rf[i] = float(val)

            region_total = float(sum(rf))
            g = self._compute_global_fitness(genes)

            pop.append(
                Individual(
                    genes=genes,
                    region_fitness=rf,
                    global_fitness=g,
                    total_fitness=region_total + g,
                )
            )

        return pop

    # ----------------------------
    # Selection
    # ----------------------------

    def select_one(self, pop: list[Individual], k: int) -> Individual:
        best = pop[self.rng.randrange(len(pop))]
        for _ in range(k - 1):
            cand = pop[self.rng.randrange(len(pop))]
            if cand.total_fitness > best.total_fitness:
                best = cand
        return best

    # ----------------------------
    # Incremental recomputation
    # ----------------------------

    def _compute_global_fitness(self, genes: list[int]) -> float:
        return float(self.evaluator.compute_global_fitness(genes, self.table))

    def _compute_region_fitness_many(self, genes: list[int], indices: list[int]) -> list[tuple[int, float]]:
        if not indices:
            return []

        if self._executor is None or len(indices) <= 1:
            out: list[tuple[int, float]] = []
            for i in indices:
                out.append((int(i), float(self.evaluator.compute_region_fitness(int(i), genes, self.table))))
            return out

        # reuse executor
        idxs = [int(i) for i in indices]
        vals = list(self._executor.map(lambda j: float(self.evaluator.compute_region_fitness(j, genes, self.table)), idxs))
        return list(zip(idxs, vals))

    def _recompute_affected_inplace(self, ind: Individual, affected: set[int]) -> None:
        idxs = sorted(i for i in affected if 0 <= i < self.H)

        old_global = float(ind.global_fitness)
        region_total = float(ind.total_fitness - old_global)

        if idxs:
            old_vals = {i: float(ind.region_fitness[i]) for i in idxs}
            new_pairs = self._compute_region_fitness_many(ind.genes, idxs)
            for i, new in new_pairs:
                old = old_vals[i]
                ind.region_fitness[i] = float(new)
                region_total += (float(new) - float(old))

        new_global = self._compute_global_fitness(ind.genes)
        ind.global_fitness = float(new_global)
        ind.total_fitness = float(region_total + new_global)

    # ----------------------------
    # Crossover + Mutation
    # ----------------------------

    def crossover_single_point(self, p1: Individual, p2: Individual, pc: float) -> tuple[Individual, Individual, set[int]]:
        if self.H < 2 or self.rng.random() >= pc:
            c1 = Individual(p1.genes[:], p1.region_fitness[:], p1.global_fitness, p1.total_fitness)
            c2 = Individual(p2.genes[:], p2.region_fitness[:], p2.global_fitness, p2.total_fitness)
            return c1, c2, set()

        cut = self.rng.randrange(1, self.H)

        c1_genes = p1.genes[:cut] + p2.genes[cut:]
        # Force the first gene to come from p1 to avoid excessive disruption
        # c1_genes[0] = p1.genes[0]
        c2_genes = p2.genes[:cut] + p1.genes[cut:]

        c1_rf = p1.region_fitness[:cut] + p2.region_fitness[cut:]
        c2_rf = p2.region_fitness[:cut] + p1.region_fitness[cut:]

        c1_region_total = float(sum(c1_rf))
        c2_region_total = float(sum(c2_rf))
        c1_global = self._compute_global_fitness(c1_genes)
        c2_global = self._compute_global_fitness(c2_genes)

        c1 = Individual(c1_genes, c1_rf, c1_global, c1_region_total + c1_global)
        c2 = Individual(c2_genes, c2_rf, c2_global, c2_region_total + c2_global)

        return c1, c2, {cut, cut + 1}

    def mutate_uniform_reset(self, child: Individual, pm_gene: float) -> set[int]:
        affected: set[int] = set()
        for i in range(self.H):
            if self.rng.random() < pm_gene:
                child.genes[i] = self.rng.choice(self.acceptable[i])
                affected.update({i, i + 1})
        return affected

    # ----------------------------
    # Survivor selection
    # ----------------------------

    @staticmethod
    def _top_k(pool: list[Individual], k: int) -> list[Individual]:
        return heapq.nlargest(k, pool, key=lambda ind: ind.total_fitness)

    def survivor_selection(self, parents: list[Individual], offspring: list[Individual], elite_frac: float) -> list[Individual]:
        pop_size = len(parents)
        pool = parents + offspring

        elite_k = max(1, int(pop_size * float(elite_frac)))
        elites = self._top_k(pool, elite_k)

        k = self._tournament_k(pop_size)
        next_pop = elites[:]
        while len(next_pop) < pop_size:
            next_pop.append(self.select_one(pool, k))
        return next_pop

    # ----------------------------
    # Saving snapshots
    # ----------------------------

    def _save_best(self, pop: list[Individual], gen: int, out_dir: Path) -> None:
        best = max(pop, key=lambda ind: ind.total_fitness)
        payload = {
            "gen": int(gen),
            "best_total_fitness": float(best.total_fitness),
            "best_mean_region_fitness": float(best.total_fitness) / float(self.H),
            "best_genes": best.genes[:],
            "meta": dict(self.save_context),
        }
        _save_pickle(payload, out_dir / f"best_gen_{gen:04d}.pkl")

    def _save_top_frac(self, pop: list[Individual], gen: int, out_dir: Path, frac: float) -> None:
        k = max(1, int(len(pop) * float(frac)))
        top = self._top_k(pop, k)
        payload = {
            "gen": gen,
            "frac": float(frac),
            "meta": dict(self.save_context),   # <-- add
            "solutions": [{
                "total_fitness": float(ind.total_fitness),
                "mean_region_fitness": float(ind.total_fitness) / self.H,
                "genes": ind.genes[:],
            } for ind in top],
        }
        _save_pickle(payload, out_dir / f"top_{int(frac*100):02d}pct_gen_{gen:04d}.pkl")

    # ----------------------------
    # Optional tracing for animation
    # ----------------------------

    def _population_fitness_array(self, pop: list[Individual]) -> np.ndarray:
        return np.array([ind.total_fitness / float(self.H) for ind in pop], dtype=np.float32)

    # ----------------------------
    # Main GA loop
    # ----------------------------

    def run(
        self,
        pop_size: int,
        max_gens: int,
        *,
        elite_frac: float = 0.20,
        pc_start: float = 0.90,
        pc_end: float = 0.20,
        pm_start: float = 0.50,
        pm_end: float = 0.20,
        warmup_frac: float = 0.10,
        early_stop: bool = True,
        stagnation_gens: int = 30,
        stagnation_improve: float = 0.05,

        use_tqdm: bool = True,

        log_every: int = 10,
        log_best_terms: bool = False,

        out_dir: str | Path | None = None,
        save_best_every: int = 10,
        save_top_frac: float = 0.10,
        save_top_every: int = 50,
        save_final_top: bool = True,

        schedule_mode: str = "linear",  # "linear" | "adaptive"
        adaptive_window: int = 10,
        adaptive_target_improve: float = 0.10,

        trace_pop_fitness: bool = False,
        trace_every: int = 1,
    ) -> tuple[Individual, pd.DataFrame, dict[str, Any]]:

        warmup_gens = max(1, int(max_gens * float(warmup_frac)))
        pop = self.init_population(pop_size)

        out_path: Optional[Path] = None
        if out_dir is not None:
            out_path = Path(out_dir)
            _ensure_dir(out_path)

        log_rows: list[dict[str, Any]] = []
        best_avg: Optional[float] = None
        stagnant = 0

        pop_fitness_history: list[np.ndarray] = []
        avg_hist: list[float] = []

        it = range(int(max_gens))
        if use_tqdm and tqdm is not None:
            it = tqdm(it, desc="GA", total=int(max_gens), dynamic_ncols=True)

        try:
            for gen in it:
                fits = [ind.total_fitness for ind in pop]
                avg_fit = float(np.mean(fits))
                best = max(pop, key=lambda ind: ind.total_fitness)

                best_term_stats: dict[str, Any] = {}
                if log_best_terms:
                    best_term_stats, _ = self.evaluator.compute_total_fitness(best.genes, self.table)

                # schedule pc/pm
                avg_hist.append(avg_fit)
                if schedule_mode == "linear":
                    pc = self._schedule_linear_with_warmup(gen, max_gens, pc_start, pc_end, warmup_gens)
                    pm = self._schedule_linear_with_warmup(gen, max_gens, pm_start, pm_end, warmup_gens)
                elif schedule_mode == "adaptive":
                    if gen < warmup_gens or len(avg_hist) <= adaptive_window:
                        prog = 0.0
                    else:
                        base = avg_hist[-adaptive_window - 1]
                        cur = avg_hist[-1]
                        rel = 0.0 if base == 0.0 else (cur - base) / abs(base)
                        prog = max(0.0, min(1.0, rel / float(adaptive_target_improve)))
                    pc = self._schedule_adaptive_with_warmup(gen, pc_start, pc_end, warmup_gens, prog)
                    pm = self._schedule_adaptive_with_warmup(gen, pm_start, pm_end, warmup_gens, prog)
                else:
                    raise ValueError("schedule_mode must be 'linear' or 'adaptive'")

                row: dict[str, Any] = {
                    "gen": int(gen),
                    "pc": float(pc),
                    "pm_gene": float(pm),
                    "avg_fitness": float(avg_fit) / float(self.H),
                    "best_fitness": float(best.total_fitness) / float(self.H),
                }
                # dynamically include whatever keys evaluator provides (tonal/atonal differ)
                if log_best_terms:
                    for k, v in best_term_stats.items():
                        row[f"best_{k}"] = float(v) if isinstance(v, (int, float, np.floating)) else v

                log_rows.append(row)

                if use_tqdm and tqdm is not None:
                    it.set_postfix(avg=f"{avg_fit:.3f}", best=f"{best.total_fitness:.3f}", pc=f"{pc:.2f}", pm=f"{pm:.2f}")
                elif gen % log_every == 0 or gen == max_gens - 1:
                    print(f"[GA] gen={gen:4d}  avg={avg_fit:.3f}  best={best.total_fitness:.3f}  pc={pc:.2f}  pm={pm:.2f}  stagnant={stagnant}/{stagnation_gens}")

                if trace_pop_fitness and (gen % int(trace_every) == 0):
                    pop_fitness_history.append(self._population_fitness_array(pop))

                if out_path is not None:
                    if save_best_every > 0 and gen % int(save_best_every) == 0:
                        self._save_best(pop, gen, out_path)
                    if save_top_every > 0 and gen % int(save_top_every) == 0:
                        self._save_top_frac(pop, gen, out_path, float(save_top_frac))

                # stagnation logic (avg fitness relative improvement)
                if early_stop:
                    if best_avg is None or avg_fit >= best_avg * (1.0 + float(stagnation_improve)):
                        best_avg = avg_fit
                        stagnant = 0
                    else:
                        stagnant += 1
                        if stagnant >= int(stagnation_gens):
                            it.close()
                            msg = f"Terminating: avg fitness failed to improve by {stagnation_improve*100:.0f}% for {stagnation_gens} generations."
                            print("[GA] " + Fore.BLUE + msg + Style.RESET_ALL)
                            break

                # --- Parent selection ---
                k = self._tournament_k(pop_size)
                num_pairs = pop_size // 2
                pairs = [(self.select_one(pop, k), self.select_one(pop, k)) for _ in range(num_pairs)]

                # --- Reproduction ---
                offspring: list[Individual] = []
                for p1, p2 in pairs:
                    c1, c2, affected_cut = self.crossover_single_point(p1, p2, pc)

                    affected1 = set(affected_cut)
                    affected2 = set(affected_cut)

                    affected1 |= self.mutate_uniform_reset(c1, pm)
                    affected2 |= self.mutate_uniform_reset(c2, pm)

                    if affected1:
                        self._recompute_affected_inplace(c1, affected1)
                    if affected2:
                        self._recompute_affected_inplace(c2, affected2)

                    offspring.append(c1)
                    offspring.append(c2)

                offspring = offspring[:pop_size]
                pop = self.survivor_selection(pop, offspring, elite_frac)

            # final saves
            if out_path is not None and save_final_top:
                self._save_top_frac(pop, gen, out_path, float(save_top_frac))
                self._save_best(pop, gen, out_path)

            best = max(pop, key=lambda ind: ind.total_fitness)

            extras: dict[str, Any] = {}
            if out_path is not None:
                extras["out_dir"] = out_path
            if trace_pop_fitness:
                extras["pop_fitness_history"] = pop_fitness_history

            return best, pd.DataFrame(log_rows), extras

        finally:
            # if user forgets, we still don’t leak threads
            self.close()

    # ============================================================
    # Multiple trials averaged
    # ============================================================

    def run_n_trials(self, n_trials: int, run_kwargs: dict[str, Any], base_seed: int = 0) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []

        for t in range(int(n_trials)):
            ga_t = HarmonizeGA(
                evaluator=self.evaluator,
                harmonization_table=self.table,
                seed=int(base_seed) + int(t),
                n_threads=int(getattr(self, "n_threads", 1)),
            )

            kwargs = dict(run_kwargs)
            kwargs["out_dir"] = None
            kwargs["trace_pop_fitness"] = False
            kwargs["early_stop"] = False

            _best, log_df, _extras = ga_t.run(**kwargs)

            for _, r in log_df[["gen", "avg_fitness", "best_fitness"]].iterrows():
                rows.append(
                    {
                        "trial": int(t),
                        "gen": int(r["gen"]),
                        "avg_fitness": float(r["avg_fitness"]),
                        "best_fitness": float(r["best_fitness"]),
                    }
                )

        df = pd.DataFrame(rows)
        out = df.groupby("gen", as_index=False).agg(
            avg_mean_fitness=("avg_fitness", "mean"),
            avg_best_fitness=("best_fitness", "mean"),
        )
        return out
    
    def run_n_trials_analysis(
        self,
        n_trials: int,
        run_kwargs: dict[str, Any],
        *,
        base_seed: int = 0,
        melody_stream=None,
        pcs_step_qL: float = 0.25,
        print_table: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Like run_n_trials(), but also computes Yeh-style metrics (CTnCTR, PCS, MCTD)
        and CPU time for the best solution of each trial.

        Returns:
        anytime_df: averaged (over trials) mean/best fitness curve by generation
        metrics_df: per-trial summary (fitness + metrics + cpu time)
        """
        import time

        # Local import to avoid pulling in music21/metrics unless needed.
        from metrics import compute_CTnCTR, compute_PCS, compute_MCTD

        mel = melody_stream
        if mel is None:
            mel = getattr(self.evaluator, "melody_stream", None)
        if mel is None:
            raise ValueError(
                "melody_stream is required to compute CTnCTR/PCS/MCTD. "
                "Pass melody_stream=..., or ensure evaluator.melody_stream exists."
            )

        anytime_rows: list[dict[str, Any]] = []
        trial_rows: list[dict[str, Any]] = []

        for t in range(int(n_trials)):
            seed = int(base_seed) + int(t)
            ga_t = HarmonizeGA(
                evaluator=self.evaluator,
                harmonization_table=self.table,
                seed=seed,
                n_threads=int(getattr(self, "n_threads", 1)),
            )

            kwargs = dict(run_kwargs)
            kwargs["out_dir"] = None
            kwargs["trace_pop_fitness"] = False

            t0 = time.process_time()
            best, log_df, _extras = ga_t.run(**kwargs)
            cpu_s = float(time.process_time() - t0)

            # anytime curve
            for _, r in log_df[["gen", "avg_fitness", "best_fitness"]].iterrows():
                anytime_rows.append(
                    {
                        "trial": int(t),
                        "gen": int(r["gen"]),
                        "avg_fitness": float(r["avg_fitness"]),
                        "best_fitness": float(r["best_fitness"]),
                    }
                )

            # per-trial best metrics
            genes = best.genes
            ctnctr = float(compute_CTnCTR(genes, self.table, mel))
            pcs = float(compute_PCS(genes, self.table, mel, step_qL=float(pcs_step_qL)))
            mctd = float(compute_MCTD(genes, self.table, mel))

            trial_rows.append(
                {
                    "trial": int(t),
                    "seed": int(seed),
                    "cpu_s": cpu_s,
                    "best_fitness": float(best.total_fitness) / float(self.H),
                    "CTnCTR": ctnctr,
                    "PCS": pcs,
                    "MCTD": mctd,
                }
            )

        # Return the averaged anytime plot (same schema as run_n_trials)
        df_any = pd.DataFrame(anytime_rows)
        anytime_df = df_any.groupby("gen", as_index=False).agg(
            avg_mean_fitness=("avg_fitness", "mean"),
            avg_best_fitness=("best_fitness", "mean"),
        )

        metrics_df = pd.DataFrame(trial_rows)

        if print_table:
            cols = ["cpu_s", "best_fitness", "CTnCTR", "PCS", "MCTD"]

            mean = metrics_df[cols].mean(numeric_only=True)
            std  = metrics_df[cols].std(numeric_only=True, ddof=1)  # sample std

            mean_row = {"trial": "mean", "seed": "-",
                        **{c: float(mean[c]) for c in cols}}
            std_row  = {"trial": "std",  "seed": "-",
                        **{c: float(std[c])  for c in cols}}

            out_df = pd.concat([metrics_df, pd.DataFrame([mean_row, std_row])], ignore_index=True)

            pd.set_option("display.max_columns", None)
            pd.set_option("display.width", 140)
            print("\n=== Trial summary (best solution per trial) ===")
            print(out_df.to_string(index=False))

        return anytime_df, metrics_df

# # ga.py
# from __future__ import annotations

# from concurrent.futures import ThreadPoolExecutor
# from dataclasses import dataclass
# from pathlib import Path
# from typing import Any, Optional

# import heapq
# import pickle
# import random

# import numpy as np
# import pandas as pd
# import time

# from fitness.fitness_protocol import FitnessEvaluatorProtocol

# try:
#     from tqdm import tqdm
# except Exception:  # pragma: no cover
#     tqdm = None

# try:
#     from colorama import Fore, Style
# except Exception:  # pragma: no cover
#     Fore = Style = None


# # ============================================================
# # Data container
# # ============================================================

# @dataclass
# class Individual:
#     genes: list[int]
#     region_fitness: list[float]
#     global_fitness: float
#     total_fitness: float


# # ============================================================
# # Helpers for saving
# # ============================================================

# def _ensure_dir(p: Path) -> None:
#     p.mkdir(parents=True, exist_ok=True)

# def _save_pickle(obj: Any, path: Path) -> None:
#     with open(path, "wb") as f:
#         pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


# # ============================================================
# # GA driver class
# # ============================================================

# class HarmonizeGA:
#     """
#     Genetic algorithm search for harmonization.

#     One-sided dependency:
#       region fitness at i depends on genes[i] and genes[i-1].
#       Therefore, if gene k changes => affected regions {k, k+1}.
#     """

#     def __init__(
#         self,
#         evaluator: FitnessEvaluatorProtocol,
#         harmonization_table: pd.DataFrame,
#         *,
#         save_context: dict[str, Any] | None = None,
#         seed: int = 0,
#         n_threads: int = 1,
#     ):
#         self.evaluator = evaluator
#         self.table = harmonization_table
#         self.H = len(harmonization_table)
#         self.save_context = save_context or {}
#         self.rng = random.Random(seed)

#         self.n_threads = int(n_threads)
#         self._executor: Optional[ThreadPoolExecutor] = None
#         if self.n_threads > 1:
#             self._executor = ThreadPoolExecutor(max_workers=self.n_threads)

#         print(f"[GA] Using {self.n_threads} thread(s) for fitness recomputation.")

#         if "acceptable_chord_ids" not in self.table.columns:
#             raise ValueError("harmonization_table must contain column 'acceptable_chord_ids'")

#         self.acceptable: list[list[int]] = self.table["acceptable_chord_ids"].tolist()
#         for i, opts in enumerate(self.acceptable):
#             if not opts:
#                 raise ValueError(f"acceptable_chord_ids at region {i} is empty; GA cannot initialize.")

#     def close(self) -> None:
#         if self._executor is not None:
#             self._executor.shutdown(wait=True)
#             self._executor = None

#     # ----------------------------
#     # Scheduling
#     # ----------------------------

#     @staticmethod
#     def _schedule_linear_with_warmup(gen: int, max_gens: int, start: float, end: float, warmup_gens: int) -> float:
#         if gen < warmup_gens:
#             return start
#         if max_gens <= warmup_gens + 1:
#             return end
#         t = (gen - warmup_gens) / (max_gens - warmup_gens - 1)
#         t = max(0.0, min(1.0, t))
#         return (1.0 - t) * start + t * end

#     @staticmethod
#     def _schedule_adaptive_with_warmup(gen: int, start: float, end: float, warmup_gens: int, progress: float) -> float:
#         if gen < warmup_gens:
#             return start
#         progress = max(0.0, min(1.0, progress))
#         return (1.0 - progress) * start + progress * end

#     @staticmethod
#     def _tournament_k(pop_size: int) -> int:
#         if pop_size > 1500:
#             return 4
#         if pop_size > 1000:
#             return 3
#         return 2

#     # ----------------------------
#     # Population init
#     # ----------------------------

#     def init_population(self, pop_size: int) -> list[Individual]:
#         pop: list[Individual] = []
#         full_idxs = list(range(self.H))

#         for _ in range(int(pop_size)):
#             genes = [self.rng.choice(opts) for opts in self.acceptable]

#             # compute region fitnesses
#             pairs = self._compute_region_fitness_many(genes, full_idxs)
#             rf = [0.0] * self.H
#             for i, val in pairs:
#                 rf[i] = float(val)

#             region_total = float(sum(rf))
#             g = self._compute_global_fitness(genes)

#             pop.append(
#                 Individual(
#                     genes=genes,
#                     region_fitness=rf,
#                     global_fitness=g,
#                     total_fitness=region_total + g,
#                 )
#             )

#         return pop

#     # ----------------------------
#     # Selection
#     # ----------------------------

#     def select_one(self, pop: list[Individual], k: int) -> Individual:
#         best = pop[self.rng.randrange(len(pop))]
#         for _ in range(k - 1):
#             cand = pop[self.rng.randrange(len(pop))]
#             if cand.total_fitness > best.total_fitness:
#                 best = cand
#         return best

#     # ----------------------------
#     # Incremental recomputation
#     # ----------------------------

#     def _compute_global_fitness(self, genes: list[int]) -> float:
#         return float(self.evaluator.compute_global_fitness(genes, self.table))

#     def _compute_region_fitness_many(self, genes: list[int], indices: list[int]) -> list[tuple[int, float]]:
#         if not indices:
#             return []

#         if self._executor is None or len(indices) <= 1:
#             out: list[tuple[int, float]] = []
#             for i in indices:
#                 out.append((int(i), float(self.evaluator.compute_region_fitness(int(i), genes, self.table))))
#             return out

#         # reuse executor
#         idxs = [int(i) for i in indices]
#         vals = list(self._executor.map(lambda j: float(self.evaluator.compute_region_fitness(j, genes, self.table)), idxs))
#         return list(zip(idxs, vals))

#     def _recompute_affected_inplace(self, ind: Individual, affected: set[int]) -> None:
#         idxs = sorted(i for i in affected if 0 <= i < self.H)

#         old_global = float(ind.global_fitness)
#         region_total = float(ind.total_fitness - old_global)

#         if idxs:
#             old_vals = {i: float(ind.region_fitness[i]) for i in idxs}
#             new_pairs = self._compute_region_fitness_many(ind.genes, idxs)
#             for i, new in new_pairs:
#                 old = old_vals[i]
#                 ind.region_fitness[i] = float(new)
#                 region_total += (float(new) - float(old))

#         new_global = self._compute_global_fitness(ind.genes)
#         ind.global_fitness = float(new_global)
#         ind.total_fitness = float(region_total + new_global)

#     # ----------------------------
#     # Crossover + Mutation
#     # ----------------------------

#     def crossover_single_point(self, p1: Individual, p2: Individual, pc: float) -> tuple[Individual, Individual, set[int]]:
#         if self.H < 2 or self.rng.random() >= pc:
#             c1 = Individual(p1.genes[:], p1.region_fitness[:], p1.global_fitness, p1.total_fitness)
#             c2 = Individual(p2.genes[:], p2.region_fitness[:], p2.global_fitness, p2.total_fitness)
#             return c1, c2, set()

#         cut = self.rng.randrange(1, self.H)

#         c1_genes = p1.genes[:cut] + p2.genes[cut:]
#         # Force the first gene to come from p1 to avoid excessive disruption
#         # c1_genes[0] = p1.genes[0]
#         c2_genes = p2.genes[:cut] + p1.genes[cut:]

#         c1_rf = p1.region_fitness[:cut] + p2.region_fitness[cut:]
#         c2_rf = p2.region_fitness[:cut] + p1.region_fitness[cut:]

#         c1_region_total = float(sum(c1_rf))
#         c2_region_total = float(sum(c2_rf))
#         c1_global = self._compute_global_fitness(c1_genes)
#         c2_global = self._compute_global_fitness(c2_genes)

#         c1 = Individual(c1_genes, c1_rf, c1_global, c1_region_total + c1_global)
#         c2 = Individual(c2_genes, c2_rf, c2_global, c2_region_total + c2_global)

#         return c1, c2, {cut, cut + 1}

#     def mutate_uniform_reset(self, child: Individual, pm_gene: float) -> set[int]:
#         affected: set[int] = set()
#         for i in range(self.H):
#             if self.rng.random() < pm_gene:
#                 child.genes[i] = self.rng.choice(self.acceptable[i])
#                 affected.update({i, i + 1})
#         return affected

#     # ----------------------------
#     # Survivor selection
#     # ----------------------------

#     @staticmethod
#     def _top_k(pool: list[Individual], k: int) -> list[Individual]:
#         return heapq.nlargest(k, pool, key=lambda ind: ind.total_fitness)

#     def survivor_selection(self, parents: list[Individual], offspring: list[Individual], elite_frac: float) -> list[Individual]:
#         pop_size = len(parents)
#         pool = parents + offspring

#         elite_k = max(1, int(pop_size * float(elite_frac)))
#         elites = self._top_k(pool, elite_k)

#         k = self._tournament_k(pop_size)
#         next_pop = elites[:]
#         while len(next_pop) < pop_size:
#             next_pop.append(self.select_one(pool, k))
#         return next_pop

#     # ----------------------------
#     # Saving snapshots
#     # ----------------------------

#     def _save_best(self, pop: list[Individual], gen: int, out_dir: Path) -> None:
#         best = max(pop, key=lambda ind: ind.total_fitness)
#         payload = {
#             "gen": int(gen),
#             "best_total_fitness": float(best.total_fitness),
#             "best_mean_region_fitness": float(best.total_fitness) / float(self.H),
#             "best_genes": best.genes[:],
#             "meta": dict(self.save_context),
#         }
#         _save_pickle(payload, out_dir / f"best_gen_{gen:04d}.pkl")

#     def _save_top_frac(self, pop: list[Individual], gen: int, out_dir: Path, frac: float) -> None:
#         k = max(1, int(len(pop) * float(frac)))
#         top = self._top_k(pop, k)
#         payload = {
#             "gen": gen,
#             "frac": float(frac),
#             "meta": dict(self.save_context),   # <-- add
#             "solutions": [{
#                 "total_fitness": float(ind.total_fitness),
#                 "mean_region_fitness": float(ind.total_fitness) / self.H,
#                 "genes": ind.genes[:],
#             } for ind in top],
#         }
#         _save_pickle(payload, out_dir / f"top_{int(frac*100):02d}pct_gen_{gen:04d}.pkl")

#     # ----------------------------
#     # Optional tracing for animation
#     # ----------------------------

#     def _population_fitness_array(self, pop: list[Individual]) -> np.ndarray:
#         return np.array([ind.total_fitness / float(self.H) for ind in pop], dtype=np.float32)

#     # ----------------------------
#     # Main GA loop
#     # ----------------------------

#     def run(
#         self,
#         pop_size: int,
#         max_gens: int,
#         *,
#         elite_frac: float = 0.20,
#         pc_start: float = 0.90,
#         pc_end: float = 0.20,
#         pm_start: float = 0.40,
#         pm_end: float = 0.05,
#         warmup_frac: float = 0.10,
#         early_stop: bool = True,
#         stagnation_gens: int = 30,
#         stagnation_improve: float = 0.05,

#         use_tqdm: bool = True,

#         log_every: int = 10,
#         log_best_terms: bool = False,

#         out_dir: str | Path | None = None,
#         save_best_every: int = 10,
#         save_top_frac: float = 0.10,
#         save_top_every: int = 50,
#         save_final_top: bool = True,

#         schedule_mode: str = "linear",  # "linear" | "adaptive"
#         adaptive_window: int = 10,
#         adaptive_target_improve: float = 0.10,

#         trace_pop_fitness: bool = False,
#         trace_every: int = 1,
#         profile: bool = False,
#     ) -> tuple[Individual, pd.DataFrame, dict[str, Any]]:

#         stats = {
#             "init": 0.0,
#             "selection": 0.0,
#             "crossover": 0.0,
#             "mutation": 0.0,
#             "evaluation": 0.0,
#             "survivor": 0.0,
#         }

#         t0_init = time.perf_counter()
#         warmup_gens = max(1, int(max_gens * float(warmup_frac)))
#         pop = self.init_population(pop_size)
#         if profile:
#             stats["init"] += (time.perf_counter() - t0_init)

#         out_path: Optional[Path] = None
#         if out_dir is not None:
#             out_path = Path(out_dir)
#             _ensure_dir(out_path)

#         log_rows: list[dict[str, Any]] = []
#         best_avg: Optional[float] = None
#         stagnant = 0

#         pop_fitness_history: list[np.ndarray] = []
#         avg_hist: list[float] = []

#         it = range(int(max_gens))
#         if use_tqdm and tqdm is not None:
#             it = tqdm(it, desc="GA", total=int(max_gens), dynamic_ncols=True)

#         try:
#             for gen in it:
#                 fits = [ind.total_fitness for ind in pop]
#                 avg_fit = float(np.mean(fits))
#                 best = max(pop, key=lambda ind: ind.total_fitness)

#                 best_term_stats: dict[str, Any] = {}
#                 if log_best_terms:
#                     best_term_stats, _ = self.evaluator.compute_total_fitness(best.genes, self.table)

#                 # schedule pc/pm
#                 avg_hist.append(avg_fit)
#                 if schedule_mode == "linear":
#                     pc = self._schedule_linear_with_warmup(gen, max_gens, pc_start, pc_end, warmup_gens)
#                     pm = self._schedule_linear_with_warmup(gen, max_gens, pm_start, pm_end, warmup_gens)
#                 elif schedule_mode == "adaptive":
#                     if gen < warmup_gens or len(avg_hist) <= adaptive_window:
#                         prog = 0.0
#                     else:
#                         base = avg_hist[-adaptive_window - 1]
#                         cur = avg_hist[-1]
#                         rel = 0.0 if base == 0.0 else (cur - base) / abs(base)
#                         prog = max(0.0, min(1.0, rel / float(adaptive_target_improve)))
#                     pc = self._schedule_adaptive_with_warmup(gen, pc_start, pc_end, warmup_gens, prog)
#                     pm = self._schedule_adaptive_with_warmup(gen, pm_start, pm_end, warmup_gens, prog)
#                 else:
#                     raise ValueError("schedule_mode must be 'linear' or 'adaptive'")

#                 row: dict[str, Any] = {
#                     "gen": int(gen),
#                     "pc": float(pc),
#                     "pm_gene": float(pm),
#                     "avg_fitness": float(avg_fit) / float(self.H),
#                     "best_fitness": float(best.total_fitness) / float(self.H),
#                 }
#                 # dynamically include whatever keys evaluator provides (tonal/atonal differ)
#                 if log_best_terms:
#                     for k, v in best_term_stats.items():
#                         row[f"best_{k}"] = float(v) if isinstance(v, (int, float, np.floating)) else v

#                 log_rows.append(row)

#                 if use_tqdm and tqdm is not None:
#                     it.set_postfix(avg=f"{avg_fit:.3f}", best=f"{best.total_fitness:.3f}", pc=f"{pc:.2f}", pm=f"{pm:.2f}")
#                 elif gen % log_every == 0 or gen == max_gens - 1:
#                     print(f"[GA] gen={gen:4d}  avg={avg_fit:.3f}  best={best.total_fitness:.3f}  pc={pc:.2f}  pm={pm:.2f}  stagnant={stagnant}/{stagnation_gens}")

#                 if trace_pop_fitness and (gen % int(trace_every) == 0):
#                     pop_fitness_history.append(self._population_fitness_array(pop))

#                 if out_path is not None:
#                     if save_best_every > 0 and gen % int(save_best_every) == 0:
#                         self._save_best(pop, gen, out_path)
#                     if save_top_every > 0 and gen % int(save_top_every) == 0:
#                         self._save_top_frac(pop, gen, out_path, float(save_top_frac))

#                 # stagnation logic (avg fitness relative improvement)
#                 if early_stop:
#                     if best_avg is None or avg_fit >= best_avg * (1.0 + float(stagnation_improve)):
#                         best_avg = avg_fit
#                         stagnant = 0
#                     else:
#                         stagnant += 1
#                         if stagnant >= int(stagnation_gens):
#                             if hasattr(it, "close"):
#                                 it.close()
#                             msg = f"Terminating: avg fitness failed to improve by {stagnation_improve*100:.0f}% for {stagnation_gens} generations."
#                             print("[GA] " + Fore.BLUE + msg + Style.RESET_ALL)
#                             break

#                 # --- Parent selection ---
#                 t0_sel = time.perf_counter()
#                 k = self._tournament_k(pop_size)
#                 num_pairs = pop_size // 2
#                 pairs = [(self.select_one(pop, k), self.select_one(pop, k)) for _ in range(num_pairs)]
#                 if profile:
#                     stats["selection"] += (time.perf_counter() - t0_sel)

#                 # --- Reproduction ---
#                 offspring: list[Individual] = []
#                 for p1, p2 in pairs:
#                     t0_cx = time.perf_counter()
#                     c1, c2, affected_cut = self.crossover_single_point(p1, p2, pc)
#                     if profile:
#                         stats["crossover"] += (time.perf_counter() - t0_cx)

#                     t0_mut = time.perf_counter()
#                     affected1 = set(affected_cut)
#                     affected2 = set(affected_cut)

#                     affected1 |= self.mutate_uniform_reset(c1, pm)
#                     affected2 |= self.mutate_uniform_reset(c2, pm)
#                     if profile:
#                         stats["mutation"] += (time.perf_counter() - t0_mut)

#                     t0_eval = time.perf_counter()
#                     if affected1:
#                         self._recompute_affected_inplace(c1, affected1)
#                     if affected2:
#                         self._recompute_affected_inplace(c2, affected2)
#                     if profile:
#                         stats["evaluation"] += (time.perf_counter() - t0_eval)

#                     offspring.append(c1)
#                     offspring.append(c2)

#                 t0_surv = time.perf_counter()
#                 offspring = offspring[:pop_size]
#                 pop = self.survivor_selection(pop, offspring, elite_frac)
#                 if profile:
#                     stats["survivor"] += (time.perf_counter() - t0_surv)

#             # final saves
#             if out_path is not None and save_final_top:
#                 self._save_top_frac(pop, gen, out_path, float(save_top_frac))
#                 self._save_best(pop, gen, out_path)

#             best = max(pop, key=lambda ind: ind.total_fitness)

#             extras: dict[str, Any] = {}
#             if out_path is not None:
#                 extras["out_dir"] = out_path
#             if trace_pop_fitness:
#                 extras["pop_fitness_history"] = pop_fitness_history
#             if profile:
#                 extras["profile_stats"] = stats

#             return best, pd.DataFrame(log_rows), extras

#         finally:
#             # if user forgets, we still don’t leak threads
#             self.close()

#     # ============================================================
#     # Multiple trials averaged
#     # ============================================================

#     def run_n_trials(self, n_trials: int, run_kwargs: dict[str, Any], base_seed: int = 0) -> pd.DataFrame:
#         rows: list[dict[str, Any]] = []

#         for t in range(int(n_trials)):
#             ga_t = HarmonizeGA(
#                 evaluator=self.evaluator,
#                 harmonization_table=self.table,
#                 seed=int(base_seed) + int(t),
#                 n_threads=int(getattr(self, "n_threads", 1)),
#             )

#             kwargs = dict(run_kwargs)
#             kwargs["out_dir"] = None
#             kwargs["trace_pop_fitness"] = False
#             kwargs["early_stop"] = False

#             _best, log_df, _extras = ga_t.run(**kwargs)

#             for _, r in log_df[["gen", "avg_fitness", "best_fitness"]].iterrows():
#                 rows.append(
#                     {
#                         "trial": int(t),
#                         "gen": int(r["gen"]),
#                         "avg_fitness": float(r["avg_fitness"]),
#                         "best_fitness": float(r["best_fitness"]),
#                     }
#                 )

#         df = pd.DataFrame(rows)
#         out = df.groupby("gen", as_index=False).agg(
#             avg_mean_fitness=("avg_fitness", "mean"),
#             avg_best_fitness=("best_fitness", "mean"),
#         )
#         return out
    
#     def run_n_trials_analysis(
#         self,
#         n_trials: int,
#         run_kwargs: dict[str, Any],
#         *,
#         base_seed: int = 0,
#         melody_stream=None,
#         pcs_step_qL: float = 0.25,
#         print_table: bool = True,
#     ) -> tuple[pd.DataFrame, pd.DataFrame]:
#         """
#         Like run_n_trials(), but also computes Yeh-style metrics (CTnCTR, PCS, MCTD)
#         and CPU time for the best solution of each trial.

#         Returns:
#         anytime_df: averaged (over trials) mean/best fitness curve by generation
#         metrics_df: per-trial summary (fitness + metrics + cpu time)
#         """
#         import time

#         # Local import to avoid pulling in music21/metrics unless needed.
#         from metrics import compute_CTnCTR, compute_PCS, compute_MCTD

#         mel = melody_stream
#         if mel is None:
#             mel = getattr(self.evaluator, "melody_stream", None)
#         if mel is None:
#             raise ValueError(
#                 "melody_stream is required to compute CTnCTR/PCS/MCTD. "
#                 "Pass melody_stream=..., or ensure evaluator.melody_stream exists."
#             )

#         anytime_rows: list[dict[str, Any]] = []
#         trial_rows: list[dict[str, Any]] = []

#         for t in range(int(n_trials)):
#             seed = int(base_seed) + int(t)
#             ga_t = HarmonizeGA(
#                 evaluator=self.evaluator,
#                 harmonization_table=self.table,
#                 seed=seed,
#                 n_threads=int(getattr(self, "n_threads", 1)),
#             )

#             kwargs = dict(run_kwargs)
#             kwargs["out_dir"] = None
#             kwargs["trace_pop_fitness"] = False

#             t0 = time.process_time()
#             best, log_df, _extras = ga_t.run(**kwargs)
#             cpu_s = float(time.process_time() - t0)

#             # anytime curve
#             for _, r in log_df[["gen", "avg_fitness", "best_fitness"]].iterrows():
#                 anytime_rows.append(
#                     {
#                         "trial": int(t),
#                         "gen": int(r["gen"]),
#                         "avg_fitness": float(r["avg_fitness"]),
#                         "best_fitness": float(r["best_fitness"]),
#                     }
#                 )

#             # per-trial best metrics
#             genes = best.genes
#             ctnctr = float(compute_CTnCTR(genes, self.table, mel))
#             pcs = float(compute_PCS(genes, self.table, mel, step_qL=float(pcs_step_qL)))
#             mctd = float(compute_MCTD(genes, self.table, mel))

#             trial_rows.append(
#                 {
#                     "trial": int(t),
#                     "seed": int(seed),
#                     "cpu_s": cpu_s,
#                     "best_fitness": float(best.total_fitness) / float(self.H),
#                     "CTnCTR": ctnctr,
#                     "PCS": pcs,
#                     "MCTD": mctd,
#                 }
#             )

#         # Return the averaged anytime plot (same schema as run_n_trials)
#         df_any = pd.DataFrame(anytime_rows)
#         anytime_df = df_any.groupby("gen", as_index=False).agg(
#             avg_mean_fitness=("avg_fitness", "mean"),
#             avg_best_fitness=("best_fitness", "mean"),
#         )

#         metrics_df = pd.DataFrame(trial_rows)

#         if print_table:
#             cols = ["cpu_s", "best_fitness", "CTnCTR", "PCS", "MCTD"]

#             mean = metrics_df[cols].mean(numeric_only=True)
#             std  = metrics_df[cols].std(numeric_only=True, ddof=1)  # sample std

#             mean_row = {"trial": "mean", "seed": "-",
#                         **{c: float(mean[c]) for c in cols}}
#             std_row  = {"trial": "std",  "seed": "-",
#                         **{c: float(std[c])  for c in cols}}

#             out_df = pd.concat([metrics_df, pd.DataFrame([mean_row, std_row])], ignore_index=True)

#             pd.set_option("display.max_columns", None)
#             pd.set_option("display.width", 140)
#             print("\n=== Trial summary (best solution per trial) ===")
#             print(out_df.to_string(index=False))

#         return anytime_df, metrics_df

