"""
tuning.py
=========
REVAC (Relevance Estimation and Value Calibration) hyperparameter tuner
for the outer DrugEA parameters.

Background
----------
REVAC (Nannen & Eiben, 2006) maintains a population of N candidate
hyperparameter vectors and iteratively improves their distribution by:
  1. Evaluating each candidate (run the EA, measure quality).
  2. Selecting the top μ by quality.
  3. Fitting a kernel density estimate (KDE) to the selected pool,
     dimension-by-dimension.
  4. Sampling N−μ new candidates from the KDE.
  5. Repeating until the evaluation budget is exhausted.

Relevance estimation
--------------------
A parameter is RELEVANT if the selected pool converges to a narrow
region (low normalised std ≈ low sensitivity of quality to variation
in that direction).  IRRELEVANT parameters stay spread across their
range — the EA doesn't care about them.

Parameters tuned
----------------
    mu           population size                    [10, 100]   int
    lambda_ratio offspring ratio  λ/μ               [0.5, 4.0]  float
    sigma_share  niche radius for fitness sharing   [0.1, 0.8]  float
    alpha        sharing function exponent           [0.5, 2.0]  float
    tournament_k tournament size                    [2, 7]      int

Quality metric
--------------
Best feasible fitness achieved by the archive over n_gens generations
(lower = tighter binding = better).  If no feasible solution is found,
the run is penalised with quality=999.  Averaged over n_seeds EA runs
for robustness.

Usage
-----
    cd ~/drugea
    python tuning.py                   # default: budget=50, n_gens=30
    python tuning.py --budget 80 --n-gens 50 --n-seeds 2
    python tuning.py --pop-rev 30 --sel-rev 15 --rounds 4

Outputs
-------
    data/revac_results.csv             log of every evaluation
    surrogate/revac_convergence.png    quality-vs-evaluation plot
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from main import run_ea, EAConfig

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ROOT       = Path(__file__).parent
RESULTS_CSV = _ROOT / "data" / "revac_results.csv"
CONV_PNG    = _ROOT / "surrogate" / "revac_convergence.png"

# ---------------------------------------------------------------------------
# Hyperparameter search space
# ---------------------------------------------------------------------------

@dataclass
class HParam:
    """One hyperparameter dimension."""
    name:   str
    lo:     float
    hi:     float
    is_int: bool = False


HPARAM_SPACE: List[HParam] = [
    HParam("mu",           10.0, 100.0, is_int=True),
    HParam("lambda_ratio",  0.5,   4.0),
    HParam("sigma_share",   0.1,   0.8),
    HParam("alpha",         0.5,   2.0),
    HParam("tournament_k",  2.0,   7.0, is_int=True),
]

N_DIMS = len(HPARAM_SPACE)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_N_POP    = 20    # hyperparameter population size  N
DEFAULT_N_SEL    = 10    # survivors per REVAC round        μ_rev
DEFAULT_N_GENS   = 30    # EA generations per evaluation
DEFAULT_N_SEEDS  = 1     # EA seeds per evaluation (averaged)
DEFAULT_BUDGET   = 50    # total EA evaluations

QUALITY_PENALTY  = 999.0  # returned when no feasible solution found
KDE_MIN_BW_FRAC  = 0.10   # minimum KDE bandwidth as fraction of param range


# ---------------------------------------------------------------------------
# Theta ↔ EAConfig
# ---------------------------------------------------------------------------

def theta_to_config(
    theta: np.ndarray,
    n_gens: int = DEFAULT_N_GENS,
    seed: Optional[int] = 42,
) -> EAConfig:
    """Convert a raw hyperparameter vector to a silent EAConfig."""
    mu           = int(round(float(np.clip(theta[0], 10, 100))))
    lambda_ratio = float(np.clip(theta[1], 0.5, 4.0))
    sigma_share  = float(np.clip(theta[2], 0.1, 0.8))
    alpha        = float(np.clip(theta[3], 0.5, 2.0))
    tournament_k = int(round(float(np.clip(theta[4], 2, 7))))
    lambda_      = max(2, int(round(mu * lambda_ratio)))

    return EAConfig(
        mu=mu,
        lambda_=lambda_,
        n_gens=n_gens,
        sigma_share=sigma_share,
        alpha=alpha,
        tournament_k=tournament_k,
        seed=seed,
        log_every=n_gens + 1,   # no per-gen lines
        verbose=False,           # suppress all stdout from run_ea
    )


def random_theta(rng: np.random.Generator) -> np.ndarray:
    """Sample one random hyperparameter vector uniformly from the search space."""
    theta = np.zeros(N_DIMS)
    for i, hp in enumerate(HPARAM_SPACE):
        if hp.is_int:
            theta[i] = float(rng.integers(int(hp.lo), int(hp.hi) + 1))
        else:
            theta[i] = rng.uniform(hp.lo, hp.hi)
    return theta


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_theta(
    theta: np.ndarray,
    n_gens: int = DEFAULT_N_GENS,
    seeds: Optional[List[int]] = None,
) -> float:
    """
    Evaluate one hyperparameter vector by running the EA.

    Runs the EA once per seed, averages the best feasible fitness.
    Returns QUALITY_PENALTY if no feasible solution is found.
    """
    if seeds is None:
        seeds = [42]

    qualities = []
    for seed in seeds:
        cfg = theta_to_config(theta, n_gens=n_gens, seed=seed)
        try:
            results = run_ea(cfg)
            best = results["best_fitness"]
            if best is None or not results["best_chromosome"].feasible:
                best = QUALITY_PENALTY
        except Exception:
            best = QUALITY_PENALTY
        qualities.append(best)

    return float(np.mean(qualities))


# ---------------------------------------------------------------------------
# KDE variation operator  (REVAC step 3-4)
# ---------------------------------------------------------------------------

def _kde_sample(values: np.ndarray, hp: HParam, rng: np.random.Generator) -> float:
    """
    Sample one value from a Gaussian KDE of `values`, clipped to [lo, hi].

    Bandwidth by Silverman's rule (h = 1.06 σ n^{-0.2}), lower-bounded
    by KDE_MIN_BW_FRAC × (hi − lo) to prevent collapse on small pools.
    """
    n   = len(values)
    bw  = 1.06 * float(np.std(values)) * (n ** -0.2)
    bw  = max(bw, KDE_MIN_BW_FRAC * (hp.hi - hp.lo))

    base   = values[rng.integers(0, n)]
    sample = float(np.clip(base + rng.normal(0.0, bw), hp.lo, hp.hi))

    if hp.is_int:
        sample = float(np.clip(round(sample), hp.lo, hp.hi))
    return sample


def sample_offspring(
    selected: np.ndarray,
    rng: np.random.Generator,
    n_offspring: int,
) -> np.ndarray:
    """
    Generate n_offspring new hyperparameter vectors by per-dimension KDE
    sampling from the selected (top-μ_rev) pool.

    Dimensions are treated independently — this is the marginal-distribution
    estimation at the core of REVAC.
    """
    offspring = np.zeros((n_offspring, N_DIMS))
    for j, hp in enumerate(HPARAM_SPACE):
        pool = selected[:, j]
        for k in range(n_offspring):
            offspring[k, j] = _kde_sample(pool, hp, rng)
    return offspring


# ---------------------------------------------------------------------------
# Relevance analysis
# ---------------------------------------------------------------------------

def relevance_analysis(candidates: np.ndarray, qualities: np.ndarray, n_sel: int) -> None:
    """
    Estimate parameter relevance from the final selected pool.

    Normalised std of selected values relative to search width:
        < 0.20  → HIGH relevance   (selection pressured this dimension)
        0.20-0.40 → MEDIUM
        > 0.40  → low relevance    (EA insensitive to this parameter)
    """
    top_k    = candidates[np.argsort(qualities)[:n_sel]]
    best_row = candidates[np.argmin(qualities)]

    print("\nParameter relevance (from final selected pool)")
    print("─" * 58)
    print(f"  {'Parameter':>14}   {'Best value':>10}   "
          f"{'Norm. std':>9}   Relevance")
    print("  " + "─" * 54)

    for j, hp in enumerate(HPARAM_SPACE):
        vals     = top_k[:, j]
        best_val = best_row[j]
        if hp.is_int:
            best_val = round(best_val)
        norm_std = float(np.std(vals)) / (hp.hi - hp.lo)

        if norm_std < 0.20:
            label = "HIGH  ★"
        elif norm_std < 0.40:
            label = "MEDIUM"
        else:
            label = "low"

        if hp.is_int:
            print(f"  {hp.name:>14}   {int(best_val):>10}   "
                  f"{norm_std:>9.3f}   {label}")
        else:
            print(f"  {hp.name:>14}   {best_val:>10.4f}   "
                  f"{norm_std:>9.3f}   {label}")


# ---------------------------------------------------------------------------
# Convergence plot
# ---------------------------------------------------------------------------

def plot_convergence(history: List[dict]) -> None:
    """Save quality-vs-evaluation scatter + running-best line to CONV_PNG."""
    evals   = [h["eval_id"] for h in history]
    quality = [h["quality"] for h in history]

    running_best, best = [], float("inf")
    for q in quality:
        if q < QUALITY_PENALTY and q < best:
            best = q
        running_best.append(best)

    fig, ax = plt.subplots(figsize=(9, 4))
    feasible_mask  = [q < QUALITY_PENALTY for q in quality]
    infeasible_mask = [not m for m in feasible_mask]

    ax.scatter(
        [e for e, m in zip(evals, feasible_mask)  if m],
        [q for q, m in zip(quality, feasible_mask)  if m],
        s=20, alpha=0.55, color="steelblue", label="feasible run",
    )
    ax.scatter(
        [e for e, m in zip(evals, infeasible_mask) if m],
        [min(q, 30) for q, m in zip(quality, infeasible_mask) if m],
        s=20, alpha=0.4, color="lightcoral", marker="x",
        label="no feasible solution",
    )
    finite_run_best = [b for b in running_best if b < float("inf")]
    if finite_run_best:
        ax.plot(evals, running_best, color="crimson", lw=2, label="best so far")

    ax.set_xlabel("Evaluation #")
    ax.set_ylabel("Best feasible fitness (lower = better)")
    ax.set_title("REVAC Convergence — DrugEA Hyperparameter Tuning")
    ax.legend(fontsize=9)
    fig.tight_layout()
    CONV_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(CONV_PNG, dpi=150)
    plt.close(fig)
    print(f"Convergence plot  → {CONV_PNG}")


# ---------------------------------------------------------------------------
# Main REVAC loop
# ---------------------------------------------------------------------------

def run_revac(
    n_pop:   int = DEFAULT_N_POP,
    n_sel:   int = DEFAULT_N_SEL,
    n_gens:  int = DEFAULT_N_GENS,
    n_seeds: int = DEFAULT_N_SEEDS,
    n_rounds: int = 3,
    seed:    int = 0,
) -> tuple[np.ndarray, float]:
    """
    Run REVAC and return (best_theta, best_quality).

    Total EA evaluations ≈ n_pop  +  n_rounds × (n_pop − n_sel)
    """
    rng         = np.random.default_rng(seed)
    eval_seeds  = list(range(n_seeds))
    n_offspring = n_pop - n_sel

    total_evals = n_pop + n_rounds * n_offspring

    print("=" * 65)
    print("REVAC Hyperparameter Tuning — DrugEA")
    print("=" * 65)
    print(f"N={n_pop}  μ_rev={n_sel}  rounds={n_rounds}  "
          f"n_gens_per_eval={n_gens}  seeds={eval_seeds}")
    print(f"Total evaluations: ~{total_evals}  "
          f"(≈{total_evals * n_gens * 0.4 / 60:.0f} min estimated)")
    print()
    print(f"  {'Parameter':>14}   {'Range':>18}   Type")
    print("  " + "─" * 40)
    for hp in HPARAM_SPACE:
        kind = "int" if hp.is_int else "float"
        print(f"  {hp.name:>14}   [{hp.lo:.1f}, {hp.hi:.1f}]{' ':>8}   {kind}")
    print("─" * 65)

    history: List[dict] = []
    eval_id = 0

    def _log_eval(theta, q, dt, round_idx=0):
        nonlocal eval_id
        eval_id += 1
        cfg = theta_to_config(theta, n_gens)
        print(f"  [{eval_id:>3d}]  "
              f"μ={cfg.mu:<3d}  λ={cfg.lambda_:<3d}  "
              f"σ_sh={theta[2]:.3f}  α={theta[3]:.2f}  "
              f"k={cfg.tournament_k}  "
              f"→ q={q:.4f}  ({dt:.1f}s)")
        history.append({
            "eval_id":      eval_id,
            "quality":      q,
            "round":        round_idx,
            "mu":           cfg.mu,
            "lambda_":      cfg.lambda_,
            "lambda_ratio": theta[1],
            "sigma_share":  theta[2],
            "alpha":        theta[3],
            "tournament_k": cfg.tournament_k,
        })

    # ------------------------------------------------------------------
    # 1. Initial population
    # ------------------------------------------------------------------
    print(f"\nInitial population ({n_pop} random candidates) ...")
    candidates = np.array([random_theta(rng) for _ in range(n_pop)])
    qualities  = np.full(n_pop, float("inf"))

    for i in range(n_pop):
        t0           = time.time()
        q            = evaluate_theta(candidates[i], n_gens=n_gens, seeds=eval_seeds)
        qualities[i] = q
        _log_eval(candidates[i], q, time.time() - t0, round_idx=0)

    # ------------------------------------------------------------------
    # 2. REVAC rounds
    # ------------------------------------------------------------------
    for round_idx in range(1, n_rounds + 1):
        idx_sorted = np.argsort(qualities)
        best_q     = qualities[idx_sorted[0]]
        print(f"\nREVAC round {round_idx}/{n_rounds}  "
              f"— best quality so far: {best_q:.4f}")

        selected  = candidates[idx_sorted[:n_sel]]
        offspring = sample_offspring(selected, rng, n_offspring)

        print(f"  ({n_offspring} new candidates via KDE sampling)")
        new_qualities = np.zeros(n_offspring)
        for k in range(n_offspring):
            t0              = time.time()
            q               = evaluate_theta(offspring[k], n_gens=n_gens, seeds=eval_seeds)
            new_qualities[k] = q
            _log_eval(offspring[k], q, time.time() - t0, round_idx=round_idx)

        # Merge & keep top n_pop
        all_c     = np.vstack([candidates, offspring])
        all_q     = np.concatenate([qualities, new_qualities])
        top_idx   = np.argsort(all_q)[:n_pop]
        candidates = all_c[top_idx]
        qualities  = all_q[top_idx]

    # ------------------------------------------------------------------
    # 3. Report
    # ------------------------------------------------------------------
    best_idx   = int(np.argmin(qualities))
    best_theta = candidates[best_idx]
    best_q     = qualities[best_idx]
    best_cfg   = theta_to_config(best_theta, n_gens=100)   # full run config

    print("\n" + "=" * 65)
    print(f"REVAC complete  —  best quality: {best_q:.4f}")
    print("=" * 65)
    print(f"\nBest hyperparameter setting:")
    print(f"  μ            = {best_cfg.mu}")
    print(f"  λ            = {best_cfg.lambda_}  (λ/μ = {best_theta[1]:.3f})")
    print(f"  σ_share      = {best_cfg.sigma_share:.4f}")
    print(f"  α            = {best_cfg.alpha:.4f}")
    print(f"  tournament_k = {best_cfg.tournament_k}")

    relevance_analysis(candidates, qualities, n_sel)

    # Save CSV
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(RESULTS_CSV, index=False)
    print(f"\nResults log       → {RESULTS_CSV}")

    plot_convergence(history)

    print(f"\nTo run the best config for 100 generations:")
    print(f"  python main.py "
          f"--pop {best_cfg.mu} "
          f"--lambda_ {best_cfg.lambda_} "
          f"--share {best_cfg.sigma_share:.4f} "
          f"--gens 100 "
          f"--seed 42")

    return best_theta, best_q


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="REVAC hyperparameter tuning for DrugEA",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--budget",   type=int, default=DEFAULT_BUDGET,
                   help="Total EA evaluations")
    p.add_argument("--n-gens",   type=int, default=DEFAULT_N_GENS,
                   help="EA generations per evaluation")
    p.add_argument("--n-seeds",  type=int, default=DEFAULT_N_SEEDS,
                   help="EA seeds per evaluation (averaged)")
    p.add_argument("--pop-rev",  type=int, default=DEFAULT_N_POP,
                   help="REVAC population size N")
    p.add_argument("--sel-rev",  type=int, default=DEFAULT_N_SEL,
                   help="REVAC selection size μ")
    p.add_argument("--rounds",   type=int, default=None,
                   help="REVAC rounds (derived from budget if omitted)")
    p.add_argument("--seed",     type=int, default=0,
                   help="REVAC random seed")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    n_pop       = args.pop_rev
    n_sel       = args.sel_rev
    n_offspring = max(1, n_pop - n_sel)

    if args.rounds is not None:
        n_rounds = args.rounds
    else:
        remaining = args.budget - n_pop
        n_rounds  = max(1, remaining // n_offspring)

    run_revac(
        n_pop=n_pop,
        n_sel=n_sel,
        n_gens=args.n_gens,
        n_seeds=args.n_seeds,
        n_rounds=n_rounds,
        seed=args.seed,
    )
