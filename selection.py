"""
selection.py
============
Step 4 of the DrugEA project.

Defines all selection mechanisms used by the EA:

PARENT SELECTION:
  - tournament_select()      : feasibility-aware tournament (Lecture 16)

DIVERSITY PRESERVATION:
  - tanimoto_distance()      : pairwise Tanimoto distance between two molecules
  - compute_sharing_matrix() : full N×N pairwise distance matrix
  - apply_fitness_sharing()  : adjust fitness values using sharing function F'(i)

SURVIVOR SELECTION:
  - ElitistArchive           : stores the best feasible molecule ever found
  - mu_plus_lambda()         : (μ+λ) survivor selection with archive reinjection

CONVENIENCE:
  - select_parents()         : select μ parent pairs for offspring generation

Lecture 16 connections
----------------------
- Feasibility-aware tournament: implements the dominance hierarchy where
  f(x)=0 (feasible) always beats f(x)>0 (infeasible), regardless of fitness.
- Fitness sharing: F'(i) = F(i) / Σ_j sh(d(i,j)) where sh is a power-law
  function with niche radius σ_share.  Applied in phenotype space using
  Tanimoto chemical similarity as the distance metric.
- (μ+λ) selection: best μ individuals from μ parents + λ offspring, ranked
  by feasibility first, fitness second.  Elitist archive ensures the best
  feasible molecule is never permanently lost.

Order of operations in one EA generation
-----------------------------------------
1. apply_fitness_sharing(population)   →  shared fitness F'(i) computed
2. tournament_select(population)       →  parents chosen by shared fitness
3. [crossover + mutate in operators.py]→  λ offspring produced
4. evaluate_population(offspring)      →  offspring fitness computed
5. mu_plus_lambda(parents, offspring)  →  best μ survivors retained
6. archive.update(population)          →  archive updated with new best
"""

from __future__ import annotations

import random
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

_MORGAN_GEN = GetMorganGenerator(radius=2, fpSize=2048)

def _morgan_fp(mol: Chem.Mol):
    return _MORGAN_GEN.GetFingerprint(mol)

from representation import Chromosome, decode, N_CONSTRAINTS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SIGMA_SHARE: float = 0.3   # niche radius  (Tanimoto > 0.7 = same scaffold)
ALPHA: float = 1.0         # sharing function exponent (linear, Lecture 16)
TOURNAMENT_K: int = 3      # default tournament size


# ---------------------------------------------------------------------------
# Morgan fingerprint helper
# ---------------------------------------------------------------------------

def _morgan_fp(mol):
    return _MORGAN_GEN.GetFingerprint(mol)


def tanimoto_distance(chrom_a: Chromosome, chrom_b: Chromosome) -> float:
    """
    Compute the Tanimoto distance between two chromosomes in phenotype space.

    d(i,j) = 1 − Tanimoto(fp_i, fp_j)

    Tanimoto similarity = 1  →  identical molecules  →  d = 0
    Tanimoto similarity = 0  →  no overlap           →  d = 1

    If either chromosome decodes to an invalid molecule, returns 1.0
    (maximum distance — treat as completely dissimilar).

    Parameters
    ----------
    chrom_a, chrom_b : two Chromosome objects.

    Returns
    -------
    float in [0, 1].
    """
    mol_a, _ = decode(chrom_a)
    mol_b, _ = decode(chrom_b)

    if mol_a is None or mol_b is None:
        return 1.0

    fp_a = _morgan_fp(mol_a)
    fp_b = _morgan_fp(mol_b)

    similarity = DataStructs.TanimotoSimilarity(fp_a, fp_b)
    return 1.0 - float(similarity)


# ---------------------------------------------------------------------------
# Fitness sharing  (Lecture 16)
# ---------------------------------------------------------------------------

def _sharing_function(d: float, sigma_share: float, alpha: float) -> float:
    """
    sh(d) = 1 − (d / σ_share)^α    if d ≤ σ_share
           = 0                       otherwise

    Parameters
    ----------
    d           : distance between two individuals.
    sigma_share : niche radius.
    alpha       : exponent controlling the shape of the sharing curve.
    """
    if d > sigma_share:
        return 0.0
    return 1.0 - (d / sigma_share) ** alpha


def compute_sharing_matrix(
    population: List[Chromosome],
    sigma_share: float = SIGMA_SHARE,
    alpha: float = ALPHA,
) -> np.ndarray:
    """
    Compute the full N×N matrix of sharing function values sh(d(i,j)).

    sh(i,j) = sharing_function(tanimoto_distance(i,j))

    The diagonal sh(i,i) = 1 because d(i,i) = 0 ≤ σ_share always.

    Parameters
    ----------
    population  : list of N Chromosome objects.
    sigma_share : niche radius (default SIGMA_SHARE = 0.3).
    alpha       : sharing exponent (default ALPHA = 1.0).

    Returns
    -------
    np.ndarray of shape (N, N) with sh values in [0, 1].
    """
    N = len(population)
    sh_matrix = np.zeros((N, N), dtype=float)

    for i in range(N):
        for j in range(i, N):
            if i == j:
                sh_matrix[i][j] = 1.0
            else:
                d = tanimoto_distance(population[i], population[j])
                s = _sharing_function(d, sigma_share, alpha)
                sh_matrix[i][j] = s
                sh_matrix[j][i] = s   # symmetric

    return sh_matrix


def apply_fitness_sharing(
    population: List[Chromosome],
    sigma_share: float = SIGMA_SHARE,
    alpha: float = ALPHA,
) -> np.ndarray:
    """
    Compute shared fitness F'(i) for every individual in the population.

    F'(i) = F(i) / Σ_j sh(d(i,j))

    where F(i) = chrom.fitness (must be evaluated before calling this).

    Because we are MINIMISING fitness, the shared fitness is:
        F'(i) = F(i) * Σ_j sh(d(i,j))
    (multiply instead of divide — crowded individuals are penalised by
    having their fitness increased, which is worse in minimisation).

    Molecules in densely populated chemical regions get their fitness
    value increased (penalised), creating selection pressure to spread
    across structurally distinct scaffolds.

    Parameters
    ----------
    population  : list of N evaluated Chromosome objects.
    sigma_share : niche radius.
    alpha       : sharing function exponent.

    Returns
    -------
    np.ndarray of shape (N,) containing the shared fitness F'(i).

    Note: this does NOT modify chrom.fitness in-place — the raw fitness
    is preserved so it can be used by the elitist archive.  The returned
    array is used only during parent selection.
    """
    N = len(population)
    raw_fitness = np.array([c.fitness for c in population], dtype=float)

    sh_matrix = compute_sharing_matrix(population, sigma_share, alpha)
    niche_counts = sh_matrix.sum(axis=1)   # Σ_j sh(d(i,j)) for each i

    # Minimisation: multiply by niche count to penalise crowded individuals
    shared_fitness = raw_fitness * niche_counts

    return shared_fitness


# ---------------------------------------------------------------------------
# Parent selection: feasibility-aware tournament  (Lecture 16)
# ---------------------------------------------------------------------------

def tournament_select(
    population: List[Chromosome],
    shared_fitness: np.ndarray,
    k: int = TOURNAMENT_K,
    rng: Optional[np.random.Generator] = None,
) -> Chromosome:
    """
    Select one parent via feasibility-aware tournament selection.

    Tournament rules (lexicographic priority):
      1. Any feasible individual (n_violations == 0) beats any infeasible one,
         regardless of fitness value.
      2. Among feasible individuals: the one with LOWER shared fitness wins
         (we are minimising).
      3. Among infeasible individuals: the one with FEWER violated constraints
         wins; ties broken by lower shared fitness.

    This implements the dominance hierarchy from Lecture 16:
    feasibility is treated as a hard prerequisite; fitness is secondary.

    Parameters
    ----------
    population     : list of N Chromosome objects (all evaluated).
    shared_fitness : np.ndarray of shape (N,) from apply_fitness_sharing().
    k              : tournament size (number of candidates sampled).
    rng            : numpy random Generator (uses random module if None).

    Returns
    -------
    The winning Chromosome (a reference, not a copy — caller should copy
    before modifying).
    """
    if rng is None:
        indices = random.sample(range(len(population)), min(k, len(population)))
    else:
        indices = rng.choice(len(population), size=min(k, len(population)),
                             replace=False).tolist()

    candidates = [(population[i], shared_fitness[i], i) for i in indices]

    # Separate feasible and infeasible
    feasible   = [(c, f, i) for c, f, i in candidates if c.n_violations == 0]
    infeasible = [(c, f, i) for c, f, i in candidates if c.n_violations  > 0]

    if feasible:
        # Rule 1+2: best feasible by shared fitness (lower = better)
        winner, _, _ = min(feasible, key=lambda x: x[1])
    else:
        # Rule 3: fewest violations, then lower shared fitness
        winner, _, _ = min(infeasible, key=lambda x: (x[0].n_violations, x[1]))

    return winner


# ---------------------------------------------------------------------------
# Elitist archive
# ---------------------------------------------------------------------------

class ElitistArchive:
    """
    Stores the single best feasible Chromosome found at any point in the run.

    If the current population loses all feasible individuals (e.g. after
    a bad generation), the archive individual is reinjected, replacing the
    worst individual in the population.

    This is the elitism mechanism from Lecture 16, repurposed for
    constrained single-objective optimisation: once feasibility is
    discovered, it is never permanently lost.

    Attributes
    ----------
    best     : the best feasible Chromosome seen so far (or None).
    history  : list of (generation, fitness) tuples — best feasible per gen.
    """

    def __init__(self):
        self.best: Optional[Chromosome] = None
        self.history: List[Tuple[int, float]] = []

    def update(self, population: List[Chromosome], generation: int = 0) -> bool:
        """
        Scan the population and update the archive if a better feasible
        individual is found.

        Parameters
        ----------
        population : list of evaluated Chromosome objects.
        generation : current generation number (for logging).

        Returns
        -------
        True if the archive was updated (new best found), False otherwise.
        """
        feasible = [c for c in population if c.feasible and c.fitness is not None]
        if not feasible:
            return False

        best_in_pop = min(feasible, key=lambda c: c.fitness)

        if self.best is None or best_in_pop.fitness < self.best.fitness:
            self.best = best_in_pop.copy()
            self.history.append((generation, self.best.fitness))
            return True

        return False

    def reinject(self, population: List[Chromosome]) -> List[Chromosome]:
        """
        If the population contains no feasible individuals and the archive
        is non-empty, replace the worst individual with the archive best.

        Parameters
        ----------
        population : list of evaluated Chromosome objects (modified in-place).

        Returns
        -------
        The (possibly modified) population list.
        """
        if self.best is None:
            return population

        has_feasible = any(c.feasible for c in population)
        if has_feasible:
            return population

        # Find and replace the worst individual (highest fitness = worst)
        worst_idx = max(
            range(len(population)),
            key=lambda i: population[i].fitness if population[i].fitness is not None
                          else float("inf")
        )
        population[worst_idx] = self.best.copy()
        return population

    def __repr__(self) -> str:
        if self.best is None:
            return "ElitistArchive(empty)"
        _, smi = decode(self.best)
        return (
            f"ElitistArchive(best_fitness={self.best.fitness:.4f}, "
            f"smiles={smi!r})"
        )


# ---------------------------------------------------------------------------
# Survivor selection: (μ+λ)  (Lecture 16)
# ---------------------------------------------------------------------------

def mu_plus_lambda(
    parents: List[Chromosome],
    offspring: List[Chromosome],
    mu: int,
    archive: Optional[ElitistArchive] = None,
    generation: int = 0,
) -> List[Chromosome]:
    """
    (μ+λ) survivor selection: retain the best μ individuals from the
    combined pool of μ parents + λ offspring.

    Ranking criterion (lexicographic):
      1. Feasible individuals ranked before infeasible ones.
      2. Among feasible: lower fitness is better (minimisation).
      3. Among infeasible: fewer violations first, then lower fitness.

    After selection:
      - Archive is updated with the best feasible individual.
      - If no feasible individual exists in the new population, the archive
        individual is reinjected (replacing the worst).

    Parameters
    ----------
    parents    : list of μ parent Chromosomes (all evaluated).
    offspring  : list of λ offspring Chromosomes (all evaluated).
    mu         : number of survivors to retain.
    archive    : ElitistArchive instance (optional but recommended).
    generation : current generation number (for archive logging).

    Returns
    -------
    List of μ survivor Chromosomes.
    """
    combined = parents + offspring

    def rank_key(c: Chromosome):
        fitness = c.fitness if c.fitness is not None else float("inf")
        if c.n_violations == 0:
            return (0, fitness)          # feasible first, then by fitness
        return (1, c.n_violations, fitness)   # infeasible: by violations then fitness

    combined.sort(key=rank_key)
    survivors = combined[:mu]

    # Update and potentially reinject from elitist archive
    if archive is not None:
        archive.update(survivors, generation)
        survivors = archive.reinject(survivors)

    return survivors


# ---------------------------------------------------------------------------
# Convenience: select μ parent pairs for offspring generation
# ---------------------------------------------------------------------------

def select_parents(
    population: List[Chromosome],
    n_pairs: int,
    shared_fitness: np.ndarray,
    k: int = TOURNAMENT_K,
    rng: Optional[np.random.Generator] = None,
) -> List[Tuple[Chromosome, Chromosome]]:
    """
    Select n_pairs of parents via tournament selection.

    Each parent in a pair is selected independently by tournament.
    Parents within a pair may be the same individual (self-mating is
    possible but statistically unlikely in a large population).

    Parameters
    ----------
    population     : list of evaluated Chromosome objects.
    n_pairs        : number of (parent1, parent2) pairs to produce.
    shared_fitness : pre-computed shared fitness array from apply_fitness_sharing().
    k              : tournament size.
    rng            : numpy random Generator.

    Returns
    -------
    List of n_pairs tuples (parent1, parent2).
    """
    pairs = []
    for _ in range(n_pairs):
        p1 = tournament_select(population, shared_fitness, k, rng)
        p2 = tournament_select(population, shared_fitness, k, rng)
        pairs.append((p1, p2))
    return pairs


# ---------------------------------------------------------------------------
# Quick self-test  (run with: python selection.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from representation import random_population
    from fitness import evaluate_population

    rng = np.random.default_rng(42)
    print("=" * 60)
    print("selection.py self-test")
    print("=" * 60)

    # Build and evaluate a population of 20
    pop = random_population(size=20, seed=42)
    evaluate_population(pop)

    n_feasible = sum(c.feasible for c in pop)
    print(f"Population size    : {len(pop)}")
    print(f"Feasible           : {n_feasible}/{len(pop)}")
    print(f"Raw fitness range  : [{min(c.fitness for c in pop):.3f}, "
          f"{max(c.fitness for c in pop):.3f}]")
    print()

    # 1. Tanimoto distance
    d = tanimoto_distance(pop[0], pop[1])
    print(f"Tanimoto distance(pop[0], pop[1]) = {d:.4f}  (in [0,1])")
    d_self = tanimoto_distance(pop[0], pop[0])
    print(f"Tanimoto distance(pop[0], pop[0]) = {d_self:.4f}  (should be 0.0)")
    assert d_self == 0.0, "Self-distance should be 0!"
    print()

    # 2. Fitness sharing
    print("Computing sharing matrix (20×20)...")
    shared = apply_fitness_sharing(pop, sigma_share=SIGMA_SHARE, alpha=ALPHA)
    print(f"Shared fitness range: [{shared.min():.3f}, {shared.max():.3f}]")
    print(f"Raw vs shared (first 5):")
    for i in range(5):
        print(f"  [{i}] raw={pop[i].fitness:.4f}  shared={shared[i]:.4f}")
    print()

    # 3. Tournament selection
    print("Tournament selection (k=3), 10 trials:")
    selections = []
    for _ in range(10):
        winner = tournament_select(pop, shared, k=3, rng=rng)
        selections.append(winner)
    n_feasible_selected = sum(c.feasible for c in selections)
    print(f"  Feasible winners: {n_feasible_selected}/10  "
          f"(should be ≥ {n_feasible}/20 proportion)")
    mean_sel_fitness = np.mean([c.fitness for c in selections])
    mean_pop_fitness = np.mean([c.fitness for c in pop])
    print(f"  Mean selected fitness : {mean_sel_fitness:.4f}")
    print(f"  Mean population fitness: {mean_pop_fitness:.4f}")
    print(f"  Selection pressure     : "
          f"{'✓ lower is better' if mean_sel_fitness < mean_pop_fitness else '~ no pressure'}")
    print()

    # 4. Elitist archive
    archive = ElitistArchive()
    updated = archive.update(pop, generation=0)
    print(f"Archive update: {updated}  →  {archive}")
    print()

    # 5. mu+lambda selection
    from operators import mutate, crossover
    parents = pop
    offspring = []
    for p1, p2 in zip(pop[:10], pop[10:]):
        c1, c2 = crossover(p1, p2, rng)
        offspring.extend([mutate(c1, rng), mutate(c2, rng)])
    evaluate_population(offspring)

    survivors = mu_plus_lambda(parents, offspring, mu=20, archive=archive, generation=1)
    n_feas_surv = sum(c.feasible for c in survivors)
    print(f"(μ+λ) selection:")
    print(f"  Parents    : {len(parents)}  Offspring  : {len(offspring)}")
    print(f"  Survivors  : {len(survivors)}  Feasible   : {n_feas_surv}")
    best = min(survivors, key=lambda c: c.fitness)
    print(f"  Best survivor fitness : {best.fitness:.4f}  feasible={best.feasible}")
    print(f"  Archive after gen 1   : {archive}")
    print()

    # 6. select_parents convenience function
    pairs = select_parents(pop, n_pairs=5, shared_fitness=shared, k=3, rng=rng)
    print(f"select_parents: {len(pairs)} pairs selected")
    for i, (p1, p2) in enumerate(pairs):
        _, s1 = decode(p1); _, s2 = decode(p2)
        print(f"  pair {i}: {s1!r}  ×  {s2!r}")
    print()

    # 7. Verify archive reinject works
    # Force a scenario where no survivor is feasible
    fake_pop = random_population(size=5, seed=99)
    for c in fake_pop:
        c.fitness = 999.0
        c.n_violations = 6    # all infeasible
    pre_reinject = [c.feasible for c in fake_pop]
    fake_pop = archive.reinject(fake_pop)
    post_reinject = [c.feasible for c in fake_pop]
    print(f"Archive reinject test:")
    print(f"  Before: {sum(pre_reinject)} feasible  →  After: {sum(post_reinject)} feasible")
    assert sum(post_reinject) >= 1 or archive.best is None, "Reinject failed!"
    print("  Reinject ✓")
    print()

    print("All assertions passed. selection.py is ready.")