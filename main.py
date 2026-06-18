"""
main.py
=======
The DrugEA evolutionary loop.

Ties together all modules in the correct order:
  representation.py  →  Chromosome, decode, random_population
  fitness.py         →  evaluate, evaluate_population
  operators.py       →  mutate, crossover
  selection.py       →  apply_fitness_sharing, select_parents, mu_plus_lambda,
                         ElitistArchive

One generation
--------------
1.  apply_fitness_sharing(population)        shared fitness F'(i)
2.  select_parents(population, n_pairs)      tournament selection
3.  crossover(p1, p2)  +  mutate(child)      produce λ offspring
4.  evaluate_population(offspring)           score offspring
5.  mu_plus_lambda(parents, offspring, μ)    retain best μ
6.  archive.update(survivors, generation)    update elitist archive

Usage
-----
    python main.py                  # run with default config
    python main.py --gens 200       # custom generation count
    python main.py --pop 50 --seed 7

Config
------
All hyperparameters are collected in the EAConfig dataclass so they
can be passed around cleanly and later fed into tuning.py (REVAC).
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from representation import Chromosome, decode, random_population
from fitness import evaluate, evaluate_population, EvalResult
from operators import mutate, crossover
from selection import (
    apply_fitness_sharing,
    select_parents,
    mu_plus_lambda,
    ElitistArchive,
    SIGMA_SHARE,
    ALPHA,
    TOURNAMENT_K,
)

# ---------------------------------------------------------------------------
# EA configuration
# ---------------------------------------------------------------------------

@dataclass
class EAConfig:
    """
    All hyperparameters for one EA run.

    These are the outer parameters that REVAC will tune in Step 6.
    The inner parameters (σ_i, w_j, tournament k) are self-adapted
    inside the chromosome and do not appear here.

    Attributes
    ----------
    mu          : population size (number of survivors per generation)
    lambda_     : number of offspring produced per generation
    n_gens      : number of generations to run
    cx_operator : crossover operator ("uniform" or "single_point")
    sigma_share : niche radius for fitness sharing
    alpha       : sharing function exponent
    tournament_k: tournament size for parent selection
    seed        : random seed for reproducibility (None = random)
    log_every   : print progress every this many generations
    """
    mu:           int   = 30
    lambda_:      int   = 30
    n_gens:       int   = 100
    cx_operator:  str   = "uniform"
    sigma_share:  float = SIGMA_SHARE
    alpha:        float = ALPHA
    tournament_k: int   = TOURNAMENT_K
    seed:         Optional[int] = 42
    log_every:    int   = 10
    verbose:      bool  = True


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _population_stats(population: List[Chromosome], generation: int) -> dict:
    """Compute summary statistics for one generation."""
    fitnesses   = [c.fitness for c in population if c.fitness is not None]
    n_feasible  = sum(c.feasible for c in population)
    n_viol      = [c.n_violations for c in population]

    stats = {
        "generation":   generation,
        "n_feasible":   n_feasible,
        "best_fitness": min(fitnesses) if fitnesses else float("inf"),
        "mean_fitness": float(np.mean(fitnesses)) if fitnesses else float("inf"),
        "worst_fitness":max(fitnesses) if fitnesses else float("inf"),
        "mean_violations": float(np.mean(n_viol)),
    }
    return stats


def _log(stats: dict, archive: ElitistArchive) -> None:
    """Print one line of generation statistics."""
    arc_str = (
        f"{archive.best.fitness:.4f}" if archive.best is not None else "none"
    )
    print(
        f"Gen {stats['generation']:>4d} | "
        f"feasible {stats['n_feasible']:>3d} | "
        f"best {stats['best_fitness']:>8.4f} | "
        f"mean {stats['mean_fitness']:>8.4f} | "
        f"archive {arc_str}"
    )


# ---------------------------------------------------------------------------
# Core EA loop
# ---------------------------------------------------------------------------

def run_ea(config: EAConfig) -> dict:
    """
    Run the full evolutionary algorithm and return results.

    Parameters
    ----------
    config : EAConfig with all hyperparameters.

    Returns
    -------
    dict with keys:
        "best_chromosome" : best feasible Chromosome found (or best overall)
        "best_smiles"     : SMILES string of the best molecule
        "best_fitness"    : fitness value of the best molecule
        "archive"         : ElitistArchive with full history
        "history"         : list of per-generation stats dicts
        "population"      : final population
    """
    rng = np.random.default_rng(config.seed)

    # ------------------------------------------------------------------
    # 0. Initialise
    # ------------------------------------------------------------------
    if config.verbose:
        print("=" * 65)
        print("DrugEA — Evolutionary Drug Candidate Search")
        print("=" * 65)
        print(f"μ={config.mu}  λ={config.lambda_}  gens={config.n_gens}  "
              f"seed={config.seed}  cx={config.cx_operator}")
        print(f"σ_share={config.sigma_share}  α={config.alpha}  "
              f"tournament_k={config.tournament_k}")
        print("-" * 65)

    t_start = time.time()

    population = random_population(
        size=config.mu,
        seed=int(rng.integers(0, 2**31)),
    )

    evaluate_population(population)

    archive = ElitistArchive()
    archive.update(population, generation=0)

    history: List[dict] = []
    stats = _population_stats(population, generation=0)
    history.append(stats)
    if config.verbose:
        _log(stats, archive)

    # ------------------------------------------------------------------
    # 1. Generational loop
    # ------------------------------------------------------------------
    for gen in range(1, config.n_gens + 1):

        # Step 1: shared fitness (applied before selection)
        shared_fitness = apply_fitness_sharing(
            population,
            sigma_share=config.sigma_share,
            alpha=config.alpha,
        )

        # Step 2: parent selection
        n_pairs = config.lambda_ // 2         # each pair produces 2 children
        pairs = select_parents(
            population,
            n_pairs=n_pairs,
            shared_fitness=shared_fitness,
            k=config.tournament_k,
            rng=rng,
        )

        # Step 3: crossover + mutation → offspring
        offspring: List[Chromosome] = []
        for p1, p2 in pairs:
            c1, c2 = crossover(p1, p2, rng, operator=config.cx_operator)
            offspring.append(mutate(c1, rng))
            offspring.append(mutate(c2, rng))

        # Handle odd lambda_
        if config.lambda_ % 2 == 1:
            extra_parent = population[int(rng.integers(0, config.mu))]
            offspring.append(mutate(extra_parent.copy(), rng))

        # Step 4: evaluate offspring
        evaluate_population(offspring)

        # Step 5: (μ+λ) survivor selection + archive update
        population = mu_plus_lambda(
            population,
            offspring,
            mu=config.mu,
            archive=archive,
            generation=gen,
        )

        # Step 6: log
        stats = _population_stats(population, generation=gen)
        history.append(stats)

        if config.verbose and (gen % config.log_every == 0 or gen == config.n_gens):
            _log(stats, archive)

    # ------------------------------------------------------------------
    # 2. Final report
    # ------------------------------------------------------------------
    t_elapsed = time.time() - t_start

    # Best molecule: prefer archive (best feasible ever); fallback to best overall
    if archive.best is not None:
        best_chrom = archive.best
    else:
        best_chrom = min(population, key=lambda c: c.fitness or float("inf"))

    _, best_smiles = decode(best_chrom)

    if config.verbose:
        # Re-evaluate to get full EvalResult for the report
        result: EvalResult = evaluate(best_chrom)

        print("-" * 65)
        print(f"Run complete in {t_elapsed:.1f}s  ({config.n_gens} generations)")
        print()
        print("Best molecule found:")
        print(f"  SMILES       : {best_smiles}")
        print(f"  Fitness      : {best_chrom.fitness:.4f}")
        print(f"  Feasible     : {best_chrom.feasible}")
        print(f"  ΔG_bind      : {result.dg_bind:.3f} kcal/mol")
        print(f"  MW           : {result.properties['MW']:.1f} Da")
        print(f"  logP         : {result.properties['logP']:.2f}")
        print(f"  HBD / HBA    : {result.properties['HBD']} / {result.properties['HBA']}")
        print(f"  SA score     : {result.properties['SA']:.2f}")
        print(f"  PAINS alerts : {result.properties['PAINS']}")
        print(f"  n_violations : {result.n_violations}")
        print()
        print(f"Archive history ({len(archive.history)} improvements):")
        for gen_idx, fit in archive.history:
            print(f"  gen {gen_idx:>4d}  →  fitness {fit:.4f}")
        print("=" * 65)

    return {
        "best_chromosome": best_chrom,
        "best_smiles":     best_smiles,
        "best_fitness":    best_chrom.fitness,
        "archive":         archive,
        "history":         history,
        "population":      population,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> EAConfig:
    parser = argparse.ArgumentParser(description="DrugEA — evolutionary drug search")
    parser.add_argument("--pop",    type=int,   default=30,       help="Population size μ")
    parser.add_argument("--lambda_",type=int,   default=30,       help="Offspring count λ")
    parser.add_argument("--gens",   type=int,   default=100,      help="Number of generations")
    parser.add_argument("--cx",     type=str,   default="uniform",help="Crossover operator")
    parser.add_argument("--share",  type=float, default=SIGMA_SHARE, help="Niche radius σ_share")
    parser.add_argument("--seed",   type=int,   default=42,       help="Random seed")
    parser.add_argument("--log",    type=int,   default=10,       help="Log every N gens")
    args = parser.parse_args()

    return EAConfig(
        mu=args.pop,
        lambda_=args.lambda_,
        n_gens=args.gens,
        cx_operator=args.cx,
        sigma_share=args.share,
        seed=args.seed,
        log_every=args.log,
    )


if __name__ == "__main__":
    config = _parse_args()
    results = run_ea(config)