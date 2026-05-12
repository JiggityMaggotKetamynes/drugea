"""
operators.py
============
Step 3 of the DrugEA project.

Defines all variation operators acting on Chromosome objects:

MUTATION (on the fragment-selection genes b_i):
  - fragment_substitution()  : random resetting   (Lecture 15)
  - fragment_creep()         : integer creep       (Lecture 15)
  - insertion()              : add a random token  (variable-length)
  - deletion()               : remove a token      (variable-length)

SELF-ADAPTIVE STEP-SIZE UPDATE (Lecture 15, uncorrelated n step-sizes):
  - update_sigma()           : mutate σ_i before applying mutation

RECOMBINATION:
  - uniform_crossover()      : each gene drawn independently from parent 1 or 2
  - single_point_crossover() : prefix from one parent, suffix from the other

COMBINED:
  - mutate()                 : applies all mutation operators in sequence
  - crossover()              : dispatcher that calls one recombination operator

All operators:
  - take Chromosome objects and return NEW Chromosome objects (no in-place mutation)
  - invalidate fitness and n_violations on offspring (set to None / N_CONSTRAINTS)
  - guarantee chemically valid offspring because Group SELFIES makes any
    token sequence valid by construction

Design notes
------------
Operator probabilities
  p_substitution : probability of random resetting at each position i
                   = σ_i  (the self-adapted step size)
  p_creep        : probability of creep mutation at each position i
                   = 0.5 * σ_i  (half the substitution probability)
  p_insertion    : probability of inserting a token at any step  = P_INSERT
  p_deletion     : probability of deleting a token at any step   = P_DELETE

Self-adaptation learning rates (Lecture 15):
  τ'  = 1 / sqrt(2 * K)          global learning rate
  τ   = 1 / sqrt(2 * sqrt(K))    individual learning rate
  ε_0 = SIGMA_MIN = 0.01          lower boundary for σ_i
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

from representation import (
    Chromosome,
    N_CONSTRAINTS,
    VOCAB_SIZE,
    K_MIN,
    K_MAX,
    SIGMA_MIN,
    SIGMA_MAX,
    SIGMA_INIT,
    W_MIN,
)

# ---------------------------------------------------------------------------
# Operator hyperparameters
# ---------------------------------------------------------------------------

P_INSERT: float = 0.1    # probability of inserting a token per mutation call
P_DELETE: float = 0.1    # probability of deleting a token per mutation call
P_CREEP_SCALE: float = 0.5   # creep probability = P_CREEP_SCALE * σ_i
CREEP_DELTA: int = 1     # step size for integer creep (b_i ± CREEP_DELTA)
W_ETA: float = 0.2       # learning rate for penalty weight mutation


# ---------------------------------------------------------------------------
# Helper: invalidate cached fitness on a chromosome
# ---------------------------------------------------------------------------

def _invalidate(chrom: Chromosome) -> Chromosome:
    """Reset fitness cache after any structural change."""
    chrom.fitness = None
    chrom.n_violations = N_CONSTRAINTS
    return chrom


# ---------------------------------------------------------------------------
# Self-adaptive σ update  (Lecture 15, uncorrelated n step-sizes)
# ---------------------------------------------------------------------------

def update_sigma(
    sigma: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Mutate the per-fragment step-size vector σ using the uncorrelated
    n step-sizes scheme from Lecture 15.

    Formula
    -------
        τ'       = 1 / sqrt(2 * K)
        τ        = 1 / sqrt(2 * sqrt(K))
        σ_i' = σ_i * exp(τ' * N(0,1) + τ * N_i(0,1))
        σ_i' = clip(σ_i', SIGMA_MIN, SIGMA_MAX)

    The global perturbation τ' * N(0,1) is shared across all positions;
    the individual perturbation τ * N_i(0,1) is independent per position.

    Parameters
    ----------
    sigma : current step-size array, shape (K,).
    rng   : numpy random Generator.

    Returns
    -------
    New sigma array of the same shape with updated values.
    """
    K = len(sigma)
    tau_prime = 1.0 / math.sqrt(2.0 * K)
    tau       = 1.0 / math.sqrt(2.0 * math.sqrt(K))

    global_noise     = rng.standard_normal()                  # shared
    individual_noise = rng.standard_normal(size=K)            # per-position

    new_sigma = sigma * np.exp(
        tau_prime * global_noise + tau * individual_noise
    )
    return np.clip(new_sigma, SIGMA_MIN, SIGMA_MAX)


# ---------------------------------------------------------------------------
# Self-adaptive penalty weight update  (Lecture 17)
# ---------------------------------------------------------------------------

def update_weights(
    w: np.ndarray,
    rng: np.random.Generator,
    eta: float = W_ETA,
) -> np.ndarray:
    """
    Mutate the per-constraint penalty weight vector w using log-normal
    perturbation (Lecture 17 self-adaptation of penalty parameters).

    Formula
    -------
        w_j' = w_j * exp(η * N(0,1))
        w_j' = max(w_j', W_MIN)

    Parameters
    ----------
    w   : current weight array, shape (N_CONSTRAINTS,).
    rng : numpy random Generator.
    eta : learning rate (default W_ETA = 0.2).

    Returns
    -------
    New weight array of the same shape.
    """
    noise = rng.standard_normal(size=len(w))
    new_w = w * np.exp(eta * noise)
    return np.maximum(new_w, W_MIN)


# ---------------------------------------------------------------------------
# Mutation operator 1: fragment substitution (random resetting, Lecture 15)
# ---------------------------------------------------------------------------

def fragment_substitution(
    chrom: Chromosome,
    rng: np.random.Generator,
) -> Chromosome:
    """
    For each fragment position i, replace b_i with a uniform random draw
    from the full vocabulary with probability σ_i.

    This is the large-step exploration operator — it can move the molecule
    discontinuously across chemical space by replacing entire building blocks.
    Probability of mutation at each position is the self-adapted σ_i.

    Parameters
    ----------
    chrom : parent Chromosome (not modified).
    rng   : numpy random Generator.

    Returns
    -------
    New Chromosome with some fragment genes replaced.
    """
    offspring = chrom.copy()

    for i in range(offspring.K):
        if rng.random() < offspring.sigma[i]:
            offspring.b[i] = rng.integers(0, VOCAB_SIZE)

    return _invalidate(offspring)


# ---------------------------------------------------------------------------
# Mutation operator 2: fragment creep (integer creep, Lecture 15)
# ---------------------------------------------------------------------------

def fragment_creep(
    chrom: Chromosome,
    rng: np.random.Generator,
    delta: int = CREEP_DELTA,
) -> Chromosome:
    """
    For each fragment position i, shift b_i by ±delta (mod VOCAB_SIZE)
    with probability P_CREEP_SCALE * σ_i.

    This is the small-step exploitation operator.  Moving b_i by ±1 selects
    a structurally adjacent fragment in the vocabulary, implementing the
    neighborhood structure N: S → 2^S from the CSP formulation (Lecture 13).

    Parameters
    ----------
    chrom : parent Chromosome (not modified).
    rng   : numpy random Generator.
    delta : step size for the integer shift (default 1).

    Returns
    -------
    New Chromosome with some genes shifted by ±delta.
    """
    offspring = chrom.copy()

    for i in range(offspring.K):
        p_creep = P_CREEP_SCALE * float(offspring.sigma[i])
        if rng.random() < p_creep:
            direction = rng.choice([-delta, +delta])
            offspring.b[i] = (int(offspring.b[i]) + direction) % VOCAB_SIZE

    return _invalidate(offspring)


# ---------------------------------------------------------------------------
# Mutation operator 3: insertion (variable-length)
# ---------------------------------------------------------------------------

def insertion(
    chrom: Chromosome,
    rng: np.random.Generator,
    p: float = P_INSERT,
) -> Chromosome:
    """
    With probability p, insert a randomly selected token at a random position.
    K → K+1, clamped to K_MAX.

    Parameters
    ----------
    chrom : parent Chromosome (not modified).
    rng   : numpy random Generator.
    p     : insertion probability (default P_INSERT = 0.1).

    Returns
    -------
    New Chromosome, possibly one token longer.
    """
    offspring = chrom.copy()

    if rng.random() < p and offspring.K < K_MAX:
        pos       = rng.integers(0, offspring.K + 1)   # insert before pos
        new_token = rng.integers(0, VOCAB_SIZE)
        new_sigma_val = SIGMA_INIT                      # initialise σ for new position

        offspring.b     = np.insert(offspring.b,     pos, new_token)
        offspring.sigma = np.insert(offspring.sigma, pos, new_sigma_val)
        # w stays the same shape (N_CONSTRAINTS,) — not per-fragment

    return _invalidate(offspring)


# ---------------------------------------------------------------------------
# Mutation operator 4: deletion (variable-length)
# ---------------------------------------------------------------------------

def deletion(
    chrom: Chromosome,
    rng: np.random.Generator,
    p: float = P_DELETE,
) -> Chromosome:
    """
    With probability p, remove a randomly selected token.
    K → K-1, clamped to K_MIN.

    Parameters
    ----------
    chrom : parent Chromosome (not modified).
    rng   : numpy random Generator.
    p     : deletion probability (default P_DELETE = 0.1).

    Returns
    -------
    New Chromosome, possibly one token shorter.
    """
    offspring = chrom.copy()

    if rng.random() < p and offspring.K > K_MIN:
        pos = rng.integers(0, offspring.K)
        offspring.b     = np.delete(offspring.b,     pos)
        offspring.sigma = np.delete(offspring.sigma, pos)

    return _invalidate(offspring)


# ---------------------------------------------------------------------------
# Combined mutate()
# ---------------------------------------------------------------------------

def mutate(
    chrom: Chromosome,
    rng: np.random.Generator,
) -> Chromosome:
    """
    Apply the full mutation pipeline to a Chromosome.

    Order of operations (all produce new objects, no in-place modification):
      1. Update σ_i  (self-adaptive step sizes — must happen BEFORE mutation)
      2. Update w_j  (self-adaptive penalty weights)
      3. Fragment substitution  (large-step, probability = σ_i)
      4. Fragment creep         (small-step, probability = P_CREEP_SCALE * σ_i)
      5. Insertion              (variable-length, probability = P_INSERT)
      6. Deletion               (variable-length, probability = P_DELETE)

    The σ_i update must precede its use in steps 3–4, as specified in
    Lecture 15: "the strategy parameters are updated first, then used."

    Parameters
    ----------
    chrom : parent Chromosome (not modified).
    rng   : numpy random Generator.

    Returns
    -------
    Mutated offspring Chromosome with fitness invalidated.
    """
    offspring = chrom.copy()

    # Step 1: update strategy parameters
    offspring.sigma = update_sigma(offspring.sigma, rng)
    offspring.w     = update_weights(offspring.w, rng)

    # Step 2: apply mutation operators
    offspring = fragment_substitution(offspring, rng)
    offspring = fragment_creep(offspring, rng)
    offspring = insertion(offspring, rng)
    offspring = deletion(offspring, rng)

    return _invalidate(offspring)


# ---------------------------------------------------------------------------
# Recombination operator 1: uniform crossover  (Lecture 15)
# ---------------------------------------------------------------------------

def uniform_crossover(
    parent1: Chromosome,
    parent2: Chromosome,
    rng: np.random.Generator,
    p_swap: float = 0.5,
) -> Tuple[Chromosome, Chromosome]:
    """
    Uniform crossover on the fragment-selection genes.

    Each gene position i in child1 is drawn from parent1 or parent2
    with probability p_swap.  child2 receives the complement.

    For variable-length chromosomes, we operate on the length of the
    shorter parent and append the tail of the longer parent to the
    child that did not inherit that parent's prefix.

    This is chemically equivalent to fragment merging — combining
    building blocks from two active compounds into one hybrid molecule.

    Parameters
    ----------
    parent1, parent2 : two parent Chromosomes (not modified).
    rng              : numpy random Generator.
    p_swap           : probability of swapping at each position (default 0.5).

    Returns
    -------
    (child1, child2) — two offspring Chromosomes.
    """
    K1, K2   = parent1.K, parent2.K
    K_short  = min(K1, K2)
    K_long   = max(K1, K2)
    longer   = parent1 if K1 >= K2 else parent2
    shorter  = parent2 if K1 >= K2 else parent1

    # Initialise children from parents
    b1     = shorter.b.copy()
    b2     = shorter.b.copy()
    sig1   = shorter.sigma.copy()
    sig2   = shorter.sigma.copy()

    # Uniform gene swap over shared length
    mask = rng.random(size=K_short) < p_swap      # True → take from longer
    b1[mask]   = longer.b[:K_short][mask]
    b2[~mask]  = longer.b[:K_short][~mask]
    sig1[mask]   = longer.sigma[:K_short][mask]
    sig2[~mask]  = longer.sigma[:K_short][~mask]

    # Append tail of longer parent to each child with probability 0.5
    if K_long > K_short:
        tail_b   = longer.b[K_short:]
        tail_sig = longer.sigma[K_short:]
        if rng.random() < 0.5:
            b1   = np.concatenate([b1,   tail_b])
            sig1 = np.concatenate([sig1, tail_sig])
        else:
            b2   = np.concatenate([b2,   tail_b])
            sig2 = np.concatenate([sig2, tail_sig])

    # Clamp lengths
    b1,   sig1 = b1[:K_MAX],   sig1[:K_MAX]
    b2,   sig2 = b2[:K_MAX],   sig2[:K_MAX]
    if len(b1) < K_MIN:
        pad = rng.integers(0, VOCAB_SIZE, size=K_MIN - len(b1))
        b1  = np.concatenate([b1, pad])
        sig1 = np.concatenate([sig1, np.full(len(pad), SIGMA_INIT)])
    if len(b2) < K_MIN:
        pad = rng.integers(0, VOCAB_SIZE, size=K_MIN - len(b2))
        b2  = np.concatenate([b2, pad])
        sig2 = np.concatenate([sig2, np.full(len(pad), SIGMA_INIT)])

    # Blend penalty weights (average of parents)
    w_child = (parent1.w + parent2.w) / 2.0

    child1 = _invalidate(Chromosome(b=b1, sigma=sig1, w=w_child.copy()))
    child2 = _invalidate(Chromosome(b=b2, sigma=sig2, w=w_child.copy()))

    return child1, child2


# ---------------------------------------------------------------------------
# Recombination operator 2: single-point crossover
# ---------------------------------------------------------------------------

def single_point_crossover(
    parent1: Chromosome,
    parent2: Chromosome,
    rng: np.random.Generator,
) -> Tuple[Chromosome, Chromosome]:
    """
    Single-point crossover on the fragment-selection genes.

    A cut point is chosen uniformly in [1, min(K1,K2)-1].
    child1 = prefix of parent1 + suffix of parent2.
    child2 = prefix of parent2 + suffix of parent1.

    Preserves prefix chemistry from one parent and suffix chemistry
    from the other — useful when the vocabulary has a meaningful
    ordering (e.g. scaffold tokens early, R-group tokens late).

    Parameters
    ----------
    parent1, parent2 : two parent Chromosomes (not modified).
    rng              : numpy random Generator.

    Returns
    -------
    (child1, child2) — two offspring Chromosomes.
    """
    K_short = min(parent1.K, parent2.K)

    if K_short < 2:
        # Cannot cut — return copies of parents
        return parent1.copy(), parent2.copy()

    cut = int(rng.integers(1, K_short))   # cut point in [1, K_short-1]

    b1   = np.concatenate([parent1.b[:cut],     parent2.b[cut:]])
    b2   = np.concatenate([parent2.b[:cut],     parent1.b[cut:]])
    sig1 = np.concatenate([parent1.sigma[:cut], parent2.sigma[cut:]])
    sig2 = np.concatenate([parent2.sigma[:cut], parent1.sigma[cut:]])

    # Clamp lengths to [K_MIN, K_MAX]
    for b, sig in [(b1, sig1), (b2, sig2)]:
        pass  # length is sum of two slices — always ≥ K_short ≥ 2 ≥ K_MIN

    b1,   sig1 = b1[:K_MAX],   sig1[:K_MAX]
    b2,   sig2 = b2[:K_MAX],   sig2[:K_MAX]

    w_child = (parent1.w + parent2.w) / 2.0

    child1 = _invalidate(Chromosome(b=b1, sigma=sig1, w=w_child.copy()))
    child2 = _invalidate(Chromosome(b=b2, sigma=sig2, w=w_child.copy()))

    return child1, child2


# ---------------------------------------------------------------------------
# Crossover dispatcher
# ---------------------------------------------------------------------------

def crossover(
    parent1: Chromosome,
    parent2: Chromosome,
    rng: np.random.Generator,
    operator: str = "uniform",
) -> Tuple[Chromosome, Chromosome]:
    """
    Dispatch to the specified recombination operator.

    Parameters
    ----------
    parent1, parent2 : two parent Chromosomes.
    rng              : numpy random Generator.
    operator         : "uniform" or "single_point".

    Returns
    -------
    (child1, child2)
    """
    if operator == "uniform":
        return uniform_crossover(parent1, parent2, rng)
    elif operator == "single_point":
        return single_point_crossover(parent1, parent2, rng)
    else:
        raise ValueError(f"Unknown crossover operator: {operator!r}")


# ---------------------------------------------------------------------------
# Quick self-test  (run with: python operators.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from representation import random_chromosome, random_population, decode
    from fitness import evaluate

    rng = np.random.default_rng(42)
    print("=" * 60)
    print("operators.py self-test")
    print("=" * 60)

    # 1. Sigma update
    chrom = random_chromosome(rng=rng)
    old_sigma = chrom.sigma.copy()
    new_sigma = update_sigma(chrom.sigma, rng)
    print(f"σ update  old: {old_sigma.round(3)}  →  new: {new_sigma.round(3)}")
    assert np.all(new_sigma >= SIGMA_MIN), "σ below minimum!"
    assert np.all(new_sigma <= SIGMA_MAX), "σ above maximum!"
    print("  σ bounds respected ✓")
    print()

    # 2. Fragment substitution
    chrom = random_chromosome(rng=rng)
    offspring = fragment_substitution(chrom, rng)
    print(f"Substitution  parent K={chrom.K}  offspring K={offspring.K}")
    print(f"  parent b   : {chrom.b}")
    print(f"  offspring b: {offspring.b}")
    assert offspring.fitness is None, "fitness not invalidated!"
    print("  fitness invalidated ✓")
    print()

    # 3. Fragment creep
    offspring_c = fragment_creep(chrom, rng)
    diffs = np.abs(offspring_c.b.astype(int) - chrom.b.astype(int))
    print(f"Creep  max gene shift: {diffs.max()}  (should be 0 or 1)")
    assert diffs.max() <= 1 or diffs.max() == VOCAB_SIZE - 1, "creep step too large!"
    print("  creep step size ✓")
    print()

    # 4. Insertion / deletion
    long_chrom = random_chromosome(k_min=K_MIN, k_max=K_MAX-1, rng=rng)
    ins = insertion(long_chrom, rng, p=1.0)   # force insertion
    print(f"Insertion  K: {long_chrom.K} → {ins.K}  (expected +1 or K_MAX cap)")
    assert ins.K <= K_MAX, "insertion exceeded K_MAX!"

    fat_chrom = random_chromosome(k_min=K_MIN+1, k_max=K_MAX, rng=rng)
    del_ = deletion(fat_chrom, rng, p=1.0)   # force deletion
    print(f"Deletion   K: {fat_chrom.K} → {del_.K}  (expected -1 or K_MIN floor)")
    assert del_.K >= K_MIN, "deletion violated K_MIN!"
    print()

    # 5. Full mutate() pipeline
    chrom = random_chromosome(rng=rng)
    _, smi_before = decode(chrom)
    offspring = mutate(chrom, rng)
    _, smi_after = decode(offspring)
    print(f"mutate()  before: {smi_before!r}  →  after: {smi_after!r}")
    assert offspring.fitness is None, "mutate did not invalidate fitness!"
    print("  fitness invalidated ✓")
    print()

    # 6. Uniform crossover
    p1 = random_chromosome(rng=rng)
    p2 = random_chromosome(rng=rng)
    c1, c2 = uniform_crossover(p1, p2, rng)
    print(f"Uniform crossover  p1.K={p1.K}  p2.K={p2.K}  c1.K={c1.K}  c2.K={c2.K}")
    assert K_MIN <= c1.K <= K_MAX, f"c1 K out of range: {c1.K}"
    assert K_MIN <= c2.K <= K_MAX, f"c2 K out of range: {c2.K}"
    _, s1 = decode(c1); _, s2 = decode(c2)
    print(f"  child1 SMILES: {s1!r}")
    print(f"  child2 SMILES: {s2!r}")
    print()

    # 7. Single-point crossover
    c3, c4 = single_point_crossover(p1, p2, rng)
    print(f"Single-point crossover  c3.K={c3.K}  c4.K={c4.K}")
    _, s3 = decode(c3); _, s4 = decode(c4)
    print(f"  child3 SMILES: {s3!r}")
    print(f"  child4 SMILES: {s4!r}")
    print()

    # 8. Validity check across 500 mutation offspring
    pop = random_population(size=100, seed=0)
    n_valid = 0
    for c in pop:
        off = mutate(c, rng)
        mol, _ = decode(off)
        if mol is not None:
            n_valid += 1
    print(f"Validity after mutate(): {n_valid}/100 valid  (expected 100)")
    assert n_valid == 100, "SELFIES validity guarantee violated after mutation!"

    # 9. Validity check across 500 crossover offspring
    n_valid_cx = 0
    for i in range(0, len(pop)-1, 2):
        c1, c2 = uniform_crossover(pop[i], pop[i+1], rng)
        for c in (c1, c2):
            mol, _ = decode(c)
            if mol is not None:
                n_valid_cx += 1
    print(f"Validity after crossover(): {n_valid_cx}/100 valid  (expected 100)")
    assert n_valid_cx == 100, "SELFIES validity guarantee violated after crossover!"
    
    print()
    print("All assertions passed. operators.py is ready.")