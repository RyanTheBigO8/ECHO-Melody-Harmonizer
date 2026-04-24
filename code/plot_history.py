# plot_history.py
from __future__ import annotations

from typing import Optional, Sequence
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def animate_population_fitness_scatter(
    pop_fitness_history: Sequence[np.ndarray],
    save_path: str,
    fps: int = 12,
) -> None:
    """
    Animation:
      x-axis: population index (0..pop_size-1)
      y-axis: individual's total fitness (already normalized if you stored it that way)
      frames: generations
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    if not pop_fitness_history:
        raise ValueError("pop_fitness_history is empty. Run GA with trace_pop_fitness=True.")

    pop_size = int(pop_fitness_history[0].shape[0])
    x = np.arange(pop_size, dtype=int)

    all_vals = np.concatenate([np.asarray(a, dtype=float) for a in pop_fitness_history])
    y_min = float(np.min(all_vals))
    y_max = float(np.max(all_vals))
    pad = 0.05 * (y_max - y_min + 1e-9)
    y_min -= pad
    y_max += pad

    fig, ax = plt.subplots()
    ax.set_xlim(-1, pop_size)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("population_id")
    ax.set_ylabel("fitness_value")

    line, = ax.plot([], [], marker="o", linestyle="None")

    def init():
        line.set_data([], [])
        ax.set_title("Fitness by population_id (gen 0)")
        return (line,)

    def update(frame: int):
        y = np.asarray(pop_fitness_history[frame], dtype=float)
        line.set_data(x, y)
        ax.set_title(f"Fitness by population_id (gen {frame})")
        return (line,)

    anim = FuncAnimation(fig, update, frames=len(pop_fitness_history), init_func=init, interval=1000 / fps, blit=True)

    if save_path.lower().endswith(".gif"):
        anim.save(save_path, writer="pillow", fps=fps)
    else:
        anim.save(save_path, fps=fps)

    plt.close(fig)


def _require_cols(df: pd.DataFrame, cols: Sequence[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def plot_single_trial_fitness_curve(
    log_df: pd.DataFrame,
    *,
    title: str = "GA fitness over generations",
    show: bool = False,
    save_path: Optional[str] = None,
) -> None:
    """
    Plots:
      - best_fitness
      - avg_fitness
    """
    _require_cols(log_df, ["gen", "best_fitness", "avg_fitness"])

    x = log_df["gen"].to_numpy(dtype=int)
    y_best = log_df["best_fitness"].to_numpy(dtype=float)
    y_avg = log_df["avg_fitness"].to_numpy(dtype=float)

    plt.figure()
    plt.plot(x, y_best, label="best_fitness")
    plt.plot(x, y_avg, label="avg_fitness")
    plt.xlabel("generation")
    plt.ylabel("fitness (per-region normalized)")
    plt.title(title)
    plt.legend()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def plot_single_trial_term_curves(
    log_df: pd.DataFrame,
    *,
    title: str = "Best individual's mean term contributions",
    show: bool = False,
    save_path: Optional[str] = None,
) -> None:
    """
    Plots ALL columns that look like mean term contributions.

    Convention used by ga.py logging:
      - evaluator returns keys like "mean_overlap", "mean_dist", ...
      - ga.py stores them as columns: "best_mean_overlap", "best_mean_dist", ...

    If none exist, it falls back to plotting any numeric "best_*" columns
    except "best_fitness" itself.
    """
    _require_cols(log_df, ["gen"])

    # preferred: "best_mean_*"
    term_cols = [c for c in log_df.columns if c.startswith("best_mean_")]

    # fallback: any numeric "best_*" columns (excluding best_fitness)
    if not term_cols:
        candidate = [c for c in log_df.columns if c.startswith("best_") and c != "best_fitness"]
        term_cols = [c for c in candidate if np.issubdtype(log_df[c].dtype, np.number)]

    if not term_cols:
        raise ValueError(
            "No term columns found to plot. "
            "Expected columns like 'best_mean_overlap', or numeric 'best_*' columns."
        )

    x = log_df["gen"].to_numpy(dtype=int)

    plt.figure(figsize=(10, 6))
    for c in term_cols:
        y = log_df[c].to_numpy(dtype=float)
        plt.plot(x, y, label=c)

    plt.xlabel("generation")
    plt.ylabel("term value (already weighted)")
    plt.title(title)

    # legend outside
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def plot_multi_trial_fitness_curve(
    avg_df: pd.DataFrame,
    *,
    title: str = "Average GA curves across trials",
    show: bool = False,
    save_path: Optional[str] = None,
) -> None:
    """
    Expects output from ga.run_n_trials():
      columns:
        - gen
        - avg_mean_fitness
        - avg_best_fitness
    """
    _require_cols(avg_df, ["gen", "avg_mean_fitness", "avg_best_fitness"])

    x = avg_df["gen"].to_numpy(dtype=int)
    y_best = avg_df["avg_best_fitness"].to_numpy(dtype=float)
    y_mean = avg_df["avg_mean_fitness"].to_numpy(dtype=float)

    plt.figure()
    plt.plot(x, y_best, label="avg_best_fitness")
    plt.plot(x, y_mean, label="avg_mean_fitness")
    plt.xlabel("generation")
    plt.ylabel("fitness")
    plt.title(title)
    plt.legend()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
