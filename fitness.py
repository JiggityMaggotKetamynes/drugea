"""
fitness.py
==========
Step 2 of the DrugEA project.

Defines:
  - EvalResult           : named tuple returned by evaluate()
  - MockSurrogate        : RDKit-property-based ΔG_bind approximation
                           (placeholder until the real PyTorch MLP is trained)
  - compute_violations() : compute all six pharmacological constraint violations
  - evaluate()           : Chromosome → EvalResult  (the EA's fitness oracle)

Fitness function
----------------
    eval(x) = −ΔG_bind(x) + Σ_j  w_j · v_j(x)

where the six constraint violation functions v_j(x) are:

    v1(x) = max(0, MW(x)   − 500)²                           [molecular weight]
    v2(x) = max(0, logP(x) − 5)²  + max(0, −2 − logP(x))²   [lipophilicity]
    v3(x) = max(0, HBD(x)  − 5)²                             [H-bond donors]
    v4(x) = max(0, HBA(x)  − 10)²                            [H-bond acceptors]
    v5(x) = max(0, SA(x)   − 6)²                             [synthetic access.]
    v6(x) = Σ_t  𝟙[PAINS alert t present]                    [ADMET / toxicity]

A molecule is FEASIBLE iff n_violations == 0  (all six constraints satisfied).
Among feasible molecules, lower eval() == tighter binding == better.

Mock surrogate
--------------
The real surrogate is a PyTorch MLP trained on ChEMBL EGFR binding data
(see surrogate/model.py and surrogate/train.py — Step 2b, deferred).

The mock surrogate approximates ΔG_bind from RDKit molecular properties
using a heuristic formula calibrated so that:
  - drug-like molecules (MW≈350, logP≈2) score around −8 to −10 kcal/mol
  - non-drug-like molecules score closer to 0 or positive
  - the score has enough variance to create meaningful selection pressure

The interface is identical to the real surrogate:
    surrogate.predict(mol) → float   (ΔG_bind in kcal/mol)

Swapping in the real PyTorch model requires changing ONE line in this file.

SA score
--------
RDKit's SA (Synthetic Accessibility) score is in the Contrib folder.
We try to import it; if unavailable we fall back to a simple approximation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

# Local import
from representation import Chromosome, decode, N_CONSTRAINTS

# ---------------------------------------------------------------------------
# SA score import (RDKit contrib — may not be present in all installations)
# ---------------------------------------------------------------------------

try:
    from rdkit.Chem import RDConfig
    import os, sys
    sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
    import sascorer
    _SA_AVAILABLE = True
except Exception:
    _SA_AVAILABLE = False
    print(
        "[fitness] WARNING: RDKit SA scorer not found. "
        "Using ring-complexity approximation instead."
    )


def _sa_score(mol: Chem.Mol) -> float:
    """Return SA score in [1, 10].  Lower = easier to synthesise."""
    if _SA_AVAILABLE:
        return sascorer.calculateScore(mol)
    # Fallback: penalise ring complexity and molecular size.
    n_rings = rdMolDescriptors.CalcNumRings(mol)
    n_heavy = mol.GetNumHeavyAtoms()
    # Rough linear approximation; calibrated so typical drug ≈ 3–4
    return min(10.0, 1.0 + 0.15 * n_rings + 0.02 * n_heavy)


# ---------------------------------------------------------------------------
# PAINS / ADMET filter catalog
# ---------------------------------------------------------------------------

_pains_params = FilterCatalogParams()
_pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
_PAINS_CATALOG = FilterCatalog(_pains_params)


def _count_pains_alerts(mol: Chem.Mol) -> int:
    """Return the number of PAINS structural alerts present in mol."""
    return len(_PAINS_CATALOG.GetMatches(mol))


# ---------------------------------------------------------------------------
# EvalResult
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    """
    All information produced by a single fitness evaluation.

    Attributes
    ----------
    fitness      : scalar eval(x) = −ΔG_bind + Σ w_j·v_j  (minimise)
    dg_bind      : predicted binding free energy ΔG_bind (kcal/mol)
    feasible     : True iff all six constraints are satisfied
    n_violations : number of violated constraints (0 = feasible)
    violations   : array of six raw violation magnitudes v_j(x)
    properties   : dict of computed molecular properties
    uncertainty  : surrogate prediction uncertainty (std dev); 0.0 for mock
    """
    fitness:      float
    dg_bind:      float
    feasible:     bool
    n_violations: int
    violations:   np.ndarray        # shape (N_CONSTRAINTS,)
    properties:   dict
    uncertainty:  float = 0.0


# ---------------------------------------------------------------------------
# Violation functions
# ---------------------------------------------------------------------------

def compute_violations(mol: Chem.Mol) -> Tuple[np.ndarray, dict]:
    """
    Compute the six pharmacological constraint violation magnitudes.

    Parameters
    ----------
    mol : a valid RDKit Mol object.

    Returns
    -------
    violations : np.ndarray shape (6,)  — raw violation magnitudes v_j(x)
    props      : dict of the underlying molecular property values
    """
    # --- compute molecular properties via RDKit ---
    mw    = Descriptors.MolWt(mol)
    logp  = Descriptors.MolLogP(mol)
    hbd   = Lipinski.NumHDonors(mol)
    hba   = Lipinski.NumHAcceptors(mol)
    sa    = _sa_score(mol)
    pains = _count_pains_alerts(mol)

    props = {
        "MW":    mw,
        "logP":  logp,
        "HBD":   hbd,
        "HBA":   hba,
        "SA":    sa,
        "PAINS": pains,
    }

    # --- violation functions ---
    # v1: molecular weight  (Lipinski: MW ≤ 500 Da)
    v1 = max(0.0, mw - 500.0) ** 2

    # v2: lipophilicity  (Lipinski: logP ∈ [−2, 5])
    v2 = max(0.0, logp - 5.0) ** 2 + max(0.0, -2.0 - logp) ** 2

    # v3: H-bond donors  (Lipinski: HBD ≤ 5)
    v3 = max(0.0, hbd - 5.0) ** 2

    # v4: H-bond acceptors  (Lipinski: HBA ≤ 10)
    v4 = max(0.0, hba - 10.0) ** 2

    # v5: synthetic accessibility  (SA ≤ 6)
    v5 = max(0.0, sa - 6.0) ** 2

    # v6: ADMET / PAINS alerts  (binary count, matches Lecture 13 formulation)
    v6 = float(pains)

    violations = np.array([v1, v2, v3, v4, v5, v6], dtype=float)
    return violations, props


def count_violated(violations: np.ndarray) -> int:
    """Return the number of constraints with non-zero violation magnitude."""
    return int(np.sum(violations > 0.0))


# ---------------------------------------------------------------------------
# Mock surrogate
# ---------------------------------------------------------------------------

class MockSurrogate:
    """
    Heuristic binding affinity surrogate.

    Approximates ΔG_bind from RDKit molecular descriptors.
    Interface is identical to the real PyTorch surrogate so the swap
    in surrogate/model.py requires no changes to evaluate().

    Formula
    -------
    We reward:
      - MW in the sweet spot [250, 450]  →  larger penalty for deviation
      - logP near 2.5  →  optimal membrane permeability
      - aromatic ring count 2–3  →  typical drug pharmacophore
      - HBD/HBA in Lipinski range  →  bioavailability

    The output is scaled so drug-like molecules produce ΔG ≈ −8 to −10
    kcal/mol, matching the range of real EGFR inhibitor binding energies.
    Uncertainty is always 0.0 (the real MLP reports MC-Dropout std dev).
    """

    def predict(self, mol: Chem.Mol) -> Tuple[float, float]:
        """
        Parameters
        ----------
        mol : valid RDKit Mol.

        Returns
        -------
        (dg_bind, uncertainty) where uncertainty = 0.0 for the mock.
        """
        mw   = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd  = Lipinski.NumHDonors(mol)
        hba  = Lipinski.NumHAcceptors(mol)
        n_ar = rdMolDescriptors.CalcNumAromaticRings(mol)

        # MW component: peak reward at MW=350, decays quadratically
        mw_score = -max(0.0, 1.0 - ((mw - 350.0) / 150.0) ** 2)

        # logP component: peak reward at logP=2.5
        logp_score = -max(0.0, 1.0 - ((logp - 2.5) / 2.5) ** 2)

        # Aromatic ring component: reward 2–3 rings
        ar_score = -max(0.0, 1.0 - ((n_ar - 2.5) / 1.5) ** 2)

        # HBD/HBA component: reward moderate H-bonding capacity
        hb_score = -max(0.0, 1.0 - (min(hbd + hba, 10) / 10.0))

        # Combine with weights calibrated to ΔG range of −8 to −10 kcal/mol
        dg = (
              4.0 * mw_score
            + 2.5 * logp_score
            + 2.0 * ar_score
            + 1.5 * hb_score
        )

        # Add small deterministic noise based on atom count for variance
        n_heavy = mol.GetNumHeavyAtoms()
        noise = 0.3 * math.sin(n_heavy * 1.7)   # deterministic, not random
        dg += noise

        return float(dg), 0.0   # (dg_bind, uncertainty)


# ---------------------------------------------------------------------------
# Module-level surrogate instance
# ---------------------------------------------------------------------------
# To swap in the real PyTorch model, replace this one line:
#   from surrogate.model import RealSurrogate
#   SURROGATE = RealSurrogate.load("surrogate/checkpoints/best.pt")

SURROGATE = MockSurrogate()


# ---------------------------------------------------------------------------
# evaluate()
# ---------------------------------------------------------------------------

def evaluate(chrom: Chromosome) -> EvalResult:
    """
    Evaluate a Chromosome and return a fully populated EvalResult.

    Pipeline
    --------
    1. Decode the chromosome to an RDKit Mol.
    2. Compute pharmacological property violations v_j(x).
    3. Predict ΔG_bind using SURROGATE.predict(mol).
    4. Compute the unified fitness:
           fitness = −ΔG_bind + Σ_j  w_j · v_j(x)
    5. Return EvalResult with all intermediate values cached.

    Invalid molecules
    -----------------
    If decode() returns None (extremely rare with SELFIES), a penalty
    EvalResult is returned with very high fitness and all constraints
    marked as violated.  This should never occur in normal operation.

    Parameters
    ----------
    chrom : Chromosome to evaluate.

    Returns
    -------
    EvalResult (see dataclass definition above).
    """
    mol, smiles = decode(chrom)

    # --- handle the (extremely rare) invalid molecule case ---
    if mol is None:
        penalty = 1e6
        violations = np.full(N_CONSTRAINTS, penalty, dtype=float)
        return EvalResult(
            fitness=penalty,
            dg_bind=0.0,
            feasible=False,
            n_violations=N_CONSTRAINTS,
            violations=violations,
            properties={"SMILES": "INVALID"},
            uncertainty=0.0,
        )

    # --- compute constraint violations ---
    violations, props = compute_violations(mol)
    props["SMILES"] = smiles

    # --- predict binding affinity ---
    dg_bind, uncertainty = SURROGATE.predict(mol)

    # --- unified fitness function ---
    # fitness = −ΔG_bind  +  Σ_j  w_j · v_j(x)
    # chrom.w has shape (N_CONSTRAINTS,); violations has shape (N_CONSTRAINTS,)
    penalty_term = float(np.dot(chrom.w, violations))
    fitness = -dg_bind + penalty_term

    n_viol = count_violated(violations)
    feasible = (n_viol == 0)

    return EvalResult(
        fitness=fitness,
        dg_bind=dg_bind,
        feasible=feasible,
        n_violations=n_viol,
        violations=violations,
        properties=props,
        uncertainty=uncertainty,
    )


# ---------------------------------------------------------------------------
# Convenience: evaluate a full population in-place
# ---------------------------------------------------------------------------

def evaluate_population(population: list) -> list:
    """
    Evaluate every Chromosome in `population`, updating .fitness and
    .n_violations in-place, and return the list of EvalResults.

    Parameters
    ----------
    population : list of Chromosome objects.

    Returns
    -------
    List of EvalResult objects in the same order as the population.
    """
    results = []
    for chrom in population:
        result = evaluate(chrom)
        chrom.fitness = result.fitness
        chrom.n_violations = result.n_violations
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Quick self-test  (run with: python fitness.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from representation import random_population

    print("=" * 60)
    print("fitness.py self-test")
    print("=" * 60)
    print(f"SA scorer available : {_SA_AVAILABLE}")
    print()

    # 1. Evaluate a single random chromosome.
    from representation import random_chromosome
    rng = np.random.default_rng(42)
    chrom = random_chromosome(rng=rng)
    result = evaluate(chrom)

    print("Single chromosome evaluation:")
    print(f"  SMILES       : {result.properties['SMILES']}")
    print(f"  MW           : {result.properties['MW']:.1f} Da")
    print(f"  logP         : {result.properties['logP']:.2f}")
    print(f"  HBD / HBA    : {result.properties['HBD']} / {result.properties['HBA']}")
    print(f"  SA score     : {result.properties['SA']:.2f}")
    print(f"  PAINS alerts : {result.properties['PAINS']}")
    print(f"  ΔG_bind      : {result.dg_bind:.3f} kcal/mol")
    print(f"  Violations   : {result.violations}")
    print(f"  n_violations : {result.n_violations}")
    print(f"  Feasible     : {result.feasible}")
    print(f"  Fitness      : {result.fitness:.4f}")
    print()

    # 2. Evaluate a known drug-like molecule (Ibuprofen).
    from rdkit import Chem
    ibu = Chem.MolFromSmiles("CC(C)Cc1ccc(cc1)C(C)C(=O)O")
    from representation import encode
    chrom_ibu = encode(ibu)
    if chrom_ibu is not None:
        result_ibu = evaluate(chrom_ibu)
        print("Ibuprofen (encoded → decoded) evaluation:")
        print(f"  SMILES       : {result_ibu.properties['SMILES']}")
        print(f"  MW           : {result_ibu.properties['MW']:.1f} Da")
        print(f"  logP         : {result_ibu.properties['logP']:.2f}")
        print(f"  ΔG_bind      : {result_ibu.dg_bind:.3f} kcal/mol")
        print(f"  n_violations : {result_ibu.n_violations}")
        print(f"  Feasible     : {result_ibu.feasible}")
        print(f"  Fitness      : {result_ibu.fitness:.4f}")
        print()

    # 3. Evaluate a population of 20 and report statistics.
    pop = random_population(size=20, seed=7)
    results = evaluate_population(pop)

    fitnesses    = [r.fitness for r in results]
    n_feasible   = sum(r.feasible for r in results)
    mean_fitness = np.mean(fitnesses)
    mean_viol    = np.mean([r.n_violations for r in results])

    print("Population of 20 — summary statistics:")
    print(f"  Feasible individuals : {n_feasible}/20")
    print(f"  Mean fitness         : {mean_fitness:.4f}")
    print(f"  Mean n_violations    : {mean_viol:.2f}")
    print(f"  Best fitness         : {min(fitnesses):.4f}")
    print(f"  Worst fitness        : {max(fitnesses):.4f}")
    print()

    # 4. Sanity check: feasibility-aware ranking.
    feasible_results  = [r for r in results if r.feasible]
    infeasible_results = [r for r in results if not r.feasible]
    print(f"Feasibility breakdown:")
    print(f"  Feasible   : {len(feasible_results)}")
    print(f"  Infeasible : {len(infeasible_results)}")
    if feasible_results:
        best_feasible = min(feasible_results, key=lambda r: r.fitness)
        print(f"  Best feasible SMILES  : {best_feasible.properties['SMILES']}")
        print(f"  Best feasible fitness : {best_feasible.fitness:.4f}")

    print()
    print("fitness.py self-test complete.")