"""
representation.py
=================
Step 1 of the DrugEA project.

Defines:
  - FRAGMENT_VOCABULARY : the curated Group-SELFIES token library (|V| fragments)
  - Chromosome          : dataclass holding b, sigma, w gene arrays
  - decode()            : Chromosome -> (RDKit Mol, SMILES string)  [always valid]
  - random_chromosome() : sample a random valid Chromosome
  - encode()            : RDKit Mol -> Chromosome  (for seeding from known drugs)
  - get_neighbors()     : return the creep-neighborhood of a Chromosome (for CSP N)

Design note
-----------
Molecules are represented as sequences of Group-SELFIES tokens.  Every token
sequence decodes to a chemically valid molecule by construction — no
sanitisation failures, no rejection sampling.  This directly addresses the
validity problem raised in the literature review and by Professor Francesca.

The chromosome layout is:
    ⟨ b_1 … b_K  |  σ_1 … σ_K  |  w_1 … w_m ⟩
where
    b_i  ∈ {0 … |V|-1}   integer  fragment-selection gene
    σ_i  ∈ (0, 1]        float    per-fragment self-adaptive mutation step size
    w_j  ∈ [w_min, ∞)    float    per-constraint self-adaptive penalty weight
    m = 6                          number of pharmacological constraints
    K                              variable (K_MIN ≤ K ≤ K_MAX)
"""

from __future__ import annotations

import random
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import selfies as sf
from rdkit import Chem
from rdkit.Chem import RWMol

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

K_MIN: int = 2          # minimum number of fragments in a chromosome
K_MAX: int = 8          # maximum number of fragments in a chromosome
N_CONSTRAINTS: int = 6  # MW, logP, HBD, HBA, SA, ADMET  (matches fitness.py)
SIGMA_INIT: float = 0.5 # initial per-fragment mutation step size
SIGMA_MIN: float = 0.01 # lower boundary for σ_i  (ε_0 in the lecture notes)
SIGMA_MAX: float = 1.0  # upper boundary for σ_i
W_INIT: float = 1.0     # initial penalty weight for every constraint
W_MIN: float = 0.1      # lower boundary for w_j

# ---------------------------------------------------------------------------
# Fragment vocabulary
# ---------------------------------------------------------------------------
# Each entry is a valid SELFIES string representing one drug-like fragment.
# The vocabulary is intentionally diverse: it covers the ring systems,
# heteroatom-containing rings, and functional-group linkers that account for
# the majority of oral drug scaffolds (Bemis-Murcko analysis of approved drugs).
#
# Design rationale
# ----------------
# - Aromatic carbocycles  : benzene, naphthalene
# - N-containing aromatics: pyridine, pyrimidine, pyrazine, imidazole,
#                           pyrazole, indole, benzimidazole, quinoline
# - O/S-containing rings  : furan, thiophene, benzofuran, benzothiophene
# - Saturated N-rings     : piperidine, piperazine, morpholine, pyrrolidine
# - Functional linkers    : amide, sulfonamide, urea, ether, ester
# - Simple carbons        : methyl, ethyl, isopropyl (for R-group decoration)
#
# All strings are validated below at import time — any token that fails
# sf.decoder() is removed with a warning so the vocabulary is always clean.

_RAW_VOCABULARY: List[str] = [
    # --- aromatic carbocycles ---
    "[C][=C][C][=C][C][=C][Ring1][=A]",                        # benzene
    "[C][=C][C][=C][C][=C][Ring1][=A][C][=C][C][=C][Ring2][Ring1][=A]",  # naphthalene

    # --- N-containing aromatics ---
    "[N][=C][C][=C][C][=C][Ring1][=A]",                        # pyridine
    "[N][=C][N][=C][C][=C][Ring1][=A]",                        # pyrimidine
    "[N][=C][C][=N][C][=C][Ring1][=A]",                        # pyrazine
    "[C][=C][N][=C][N][Ring1][=A]",                            # imidazole
    "[C][=C][N][=N][C][Ring1][=A]",                            # pyrazole
    "[C][=C][C][=C][C][NH1][C][=C][Ring2][Ring1][=A]",         # indole
    "[C][=C][C][=C][C][NH1][C][=N][Ring2][Ring1][=A]",         # benzimidazole
    "[C][=C][C][=C][C][=C][C][=N][Ring2][Ring1][=A]",          # quinoline

    # --- O/S-containing aromatics ---
    "[C][=C][O][C][=C][Ring1][=A]",                            # furan
    "[C][=C][S][C][=C][Ring1][=A]",                            # thiophene
    "[C][=C][C][=C][C][O][C][=C][Ring2][Ring1][=A]",           # benzofuran
    "[C][=C][C][=C][C][S][C][=C][Ring2][Ring1][=A]",           # benzothiophene

    # --- saturated N-heterocycles ---
    "[C][C][C][C][C][NH1][Ring1][=A]",                         # piperidine
    "[C][C][N][C][C][N][Ring1][=A]",                           # piperazine
    "[C][C][O][C][C][N][Ring1][=A]",                           # morpholine
    "[C][C][C][C][NH1][Ring1][=A]",                            # pyrrolidine
    "[C][C][N][C][C][O][Ring1][=A]",                           # oxazoline

    # --- functional-group linkers ---
    "[C][=O][NH1]",                                            # amide bond
    "[S][=O][=O][NH1]",                                        # sulfonamide
    "[C][=O][NH1][C][=O]",                                     # urea
    "[C][O][C]",                                               # ether
    "[C][=O][O][C]",                                           # ester
    "[C][#N]",                                                 # nitrile
    "[C][F]",                                                  # fluoromethyl
    "[C][Cl]",                                                 # chloromethyl

    # --- simple carbon R-group decorators ---
    "[C]",                                                     # methyl
    "[C][C]",                                                  # ethyl
    "[C][C][C]",                                               # propyl
    "[C][Branch1][C][C][C]",                                   # isopropyl
    "[C][C][C][C]",                                            # butyl
    "[O][C]",                                                  # methoxy
    "[O][C][C]",                                               # ethoxy
    "[NH2]",                                                   # amine
    "[NH1][C]",                                                # methylamine
    "[C][=O][OH1]",                                            # carboxylic acid
    "[C][OH1]",                                                # alcohol
    "[C][C][OH1]",                                             # ethanol linker
]


def _validate_vocabulary(raw: List[str]) -> List[str]:
    """
    Validate each SELFIES token by attempting to decode it.
    Tokens that produce None or raise an exception are silently dropped.
    Returns the cleaned vocabulary list.
    """
    clean: List[str] = []
    for token in raw:
        try:
            smi = sf.decoder(token)
            if smi and Chem.MolFromSmiles(smi) is not None:
                clean.append(token)
            else:
                print(f"[representation] WARNING: dropped invalid token: {token!r}")
        except Exception as exc:
            print(f"[representation] WARNING: exception on token {token!r}: {exc}")
    return clean


# Build and validate the vocabulary once at import time.
FRAGMENT_VOCABULARY: List[str] = _validate_vocabulary(_RAW_VOCABULARY)
VOCAB_SIZE: int = len(FRAGMENT_VOCABULARY)

if VOCAB_SIZE < 10:
    raise RuntimeError(
        f"Vocabulary too small after validation ({VOCAB_SIZE} tokens). "
        "Check your selfies installation."
    )


def vocab_index_to_selfies(index: int) -> str:
    """Return the SELFIES token at position `index` in the vocabulary."""
    return FRAGMENT_VOCABULARY[index % VOCAB_SIZE]


def chromosome_to_selfies(b: np.ndarray) -> str:
    """
    Concatenate the SELFIES tokens selected by the integer gene array b.
    Returns a single SELFIES string representing the full molecule.
    """
    return "".join(vocab_index_to_selfies(i) for i in b)


# ---------------------------------------------------------------------------
# Chromosome dataclass
# ---------------------------------------------------------------------------

@dataclass
class Chromosome:
    """
    A single individual in the EA population.

    Attributes
    ----------
    b : np.ndarray, dtype=int, shape (K,)
        Fragment-selection genes.  b[i] is an index into FRAGMENT_VOCABULARY.
    sigma : np.ndarray, dtype=float, shape (K,)
        Per-fragment self-adaptive mutation step sizes σ_i ∈ (SIGMA_MIN, SIGMA_MAX].
    w : np.ndarray, dtype=float, shape (N_CONSTRAINTS,)
        Per-constraint self-adaptive penalty weights w_j ≥ W_MIN.
    fitness : Optional[float]
        Cached fitness value; None means not yet evaluated.
    n_violations : int
        Number of violated pharmacological constraints (0 = feasible).
    """
    b: np.ndarray                        # shape (K,)  int
    sigma: np.ndarray                    # shape (K,)  float
    w: np.ndarray                        # shape (N_CONSTRAINTS,)  float
    fitness: Optional[float] = field(default=None, compare=False)
    n_violations: int = field(default=N_CONSTRAINTS, compare=False)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def K(self) -> int:
        """Number of fragments (chromosome length)."""
        return len(self.b)

    @property
    def feasible(self) -> bool:
        """True iff all pharmacological constraints are satisfied."""
        return self.n_violations == 0

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def copy(self) -> "Chromosome":
        """Return a deep copy of this chromosome."""
        return Chromosome(
            b=self.b.copy(),
            sigma=self.sigma.copy(),
            w=self.w.copy(),
            fitness=self.fitness,
            n_violations=self.n_violations,
        )

    def to_dict(self) -> dict:
        """Serialise to a plain dict (useful for logging / REVAC)."""
        return {
            "b": self.b.tolist(),
            "sigma": self.sigma.tolist(),
            "w": self.w.tolist(),
            "fitness": self.fitness,
            "n_violations": self.n_violations,
            "K": self.K,
        }

    def __repr__(self) -> str:
        smi = decode(self)[1]
        return (
            f"Chromosome(K={self.K}, fitness={self.fitness}, "
            f"feasible={self.feasible}, smiles={smi!r})"
        )


# ---------------------------------------------------------------------------
# decode()
# ---------------------------------------------------------------------------

def decode(chrom: Chromosome) -> Tuple[Optional[Chem.Mol], str]:
    """
    Decode a Chromosome to an RDKit Mol object and a canonical SMILES string.

    Pipeline
    --------
    1. Concatenate the SELFIES tokens selected by chrom.b  →  one SELFIES string
    2. Call sf.decoder() to convert SELFIES → SMILES
    3. Call Chem.MolFromSmiles() to get the RDKit Mol

    Validity guarantee
    ------------------
    Group SELFIES guarantees that step 2 always produces a syntactically
    valid SMILES string.  Step 3 can still return None in rare edge cases
    (e.g. empty SELFIES after concatenation), which is handled gracefully.

    Returns
    -------
    (mol, smiles) where mol is None only if the SELFIES string is empty.
    """
    selfies_str = chromosome_to_selfies(chrom.b)

    if not selfies_str:
        return None, ""

    try:
        smiles = sf.decoder(selfies_str)
    except Exception:
        # Should not happen with valid SELFIES tokens, but be defensive.
        return None, ""

    if not smiles:
        return None, ""

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        # Extremely rare with SELFIES; return empty rather than crash.
        return None, ""

    canonical = Chem.MolToSmiles(mol)
    return mol, canonical


# ---------------------------------------------------------------------------
# random_chromosome()
# ---------------------------------------------------------------------------

def random_chromosome(
    k_min: int = K_MIN,
    k_max: int = K_MAX,
    rng: Optional[np.random.Generator] = None,
) -> Chromosome:
    """
    Sample a random valid Chromosome of length K ~ Uniform[k_min, k_max].

    Parameters
    ----------
    k_min, k_max : fragment count bounds (inclusive).
    rng          : optional numpy random Generator for reproducibility.

    Initialisation
    --------------
    b     : uniform random integers in [0, VOCAB_SIZE)
    sigma : all initialised to SIGMA_INIT  (= 0.5)
    w     : all initialised to W_INIT      (= 1.0)
    """
    if rng is None:
        rng = np.random.default_rng()

    K = int(rng.integers(k_min, k_max + 1))
    b = rng.integers(0, VOCAB_SIZE, size=K)
    sigma = np.full(K, SIGMA_INIT, dtype=float)
    w = np.full(N_CONSTRAINTS, W_INIT, dtype=float)

    return Chromosome(b=b, sigma=sigma, w=w)


# ---------------------------------------------------------------------------
# encode()
# ---------------------------------------------------------------------------

def encode(mol: Chem.Mol) -> Optional[Chromosome]:
    """
    Convert an existing RDKit Mol into a Chromosome.

    Used to seed the initial population with known drug-like molecules
    (e.g. approved drugs, known actives from ChEMBL).

    Strategy
    --------
    1. Convert the molecule to a canonical SMILES string.
    2. Convert SMILES → SELFIES via sf.encoder().
    3. Tokenise the SELFIES string into individual tokens.
    4. For each token, find the closest match in FRAGMENT_VOCABULARY by
       exact match first, then fall back to a random vocabulary index.
       This is lossy — the re-decoded molecule will differ from the input
       if the input contains tokens not in the vocabulary.  That is
       acceptable: encode() is only used for population seeding, and the
       EA will quickly diverge from the seed anyway.

    Returns None if the molecule cannot be converted.
    """
    if mol is None:
        return None

    smiles = Chem.MolToSmiles(mol)
    if not smiles:
        return None

    try:
        selfies_str = sf.encoder(smiles)
    except Exception:
        return None

    if not selfies_str:
        return None

    # Tokenise the SELFIES string into a list of individual tokens.
    tokens: List[str] = list(sf.split_selfies(selfies_str))

    if not tokens:
        return None

    # Map each token to a vocabulary index.
    vocab_set = {tok: idx for idx, tok in enumerate(FRAGMENT_VOCABULARY)}
    b_list: List[int] = []
    for tok in tokens:
        if tok in vocab_set:
            b_list.append(vocab_set[tok])
        else:
            # Token not in vocabulary — assign a random index.
            b_list.append(random.randint(0, VOCAB_SIZE - 1))

    # Clamp length to [K_MIN, K_MAX].
    if len(b_list) < K_MIN:
        # Pad with random tokens.
        b_list += [random.randint(0, VOCAB_SIZE - 1) for _ in range(K_MIN - len(b_list))]
    elif len(b_list) > K_MAX:
        b_list = b_list[:K_MAX]

    K = len(b_list)
    b = np.array(b_list, dtype=int)
    sigma = np.full(K, SIGMA_INIT, dtype=float)
    w = np.full(N_CONSTRAINTS, W_INIT, dtype=float)

    return Chromosome(b=b, sigma=sigma, w=w)


# ---------------------------------------------------------------------------
# get_neighbors()
# ---------------------------------------------------------------------------

def get_neighbors(chrom: Chromosome, delta: int = 1) -> List[Chromosome]:
    """
    Return the creep-neighborhood N(chrom) of a Chromosome.

    This is the direct implementation of the neighborhood structure
    N: S → 2^S from the CSP formulation (Lecture 13).

    Each neighbor differs from chrom by exactly one fragment-selection
    gene b_i shifted by ±delta (mod VOCAB_SIZE).  For a chromosome of
    length K, this yields 2·K neighbors.

    Parameters
    ----------
    chrom : the center chromosome.
    delta : step size for integer creep (default 1).

    Returns
    -------
    List of Chromosome objects, each a copy of chrom with one gene changed.
    """
    neighbors: List[Chromosome] = []
    for i in range(chrom.K):
        for direction in (-delta, +delta):
            neighbor = chrom.copy()
            neighbor.b[i] = (int(neighbor.b[i]) + direction) % VOCAB_SIZE
            neighbor.fitness = None       # invalidate cached fitness
            neighbor.n_violations = N_CONSTRAINTS
            neighbors.append(neighbor)
    return neighbors


# ---------------------------------------------------------------------------
# Convenience: random population
# ---------------------------------------------------------------------------

def random_population(
    size: int,
    k_min: int = K_MIN,
    k_max: int = K_MAX,
    seed: Optional[int] = None,
) -> List[Chromosome]:
    """
    Generate a list of `size` random Chromosomes.

    Parameters
    ----------
    size  : population size μ.
    k_min : minimum fragment count per individual.
    k_max : maximum fragment count per individual.
    seed  : optional integer seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    return [random_chromosome(k_min, k_max, rng) for _ in range(size)]


# ---------------------------------------------------------------------------
# Quick self-test  (run with: python representation.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Vocabulary size : {VOCAB_SIZE} tokens")
    print(f"K range         : [{K_MIN}, {K_MAX}]")
    print(f"N constraints   : {N_CONSTRAINTS}")
    print()

    # 1. Sample a random chromosome and decode it.
    rng = np.random.default_rng(42)
    chrom = random_chromosome(rng=rng)
    mol, smi = decode(chrom)
    print(f"Random chromosome  : {chrom.to_dict()}")
    print(f"Decoded SMILES     : {smi}")
    print(f"RDKit Mol valid    : {mol is not None}")
    print()

    # 2. Test encode() round-trip on aspirin.
    aspirin = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
    chrom2 = encode(aspirin)
    if chrom2 is not None:
        mol2, smi2 = decode(chrom2)
        print(f"Aspirin encode → decode SMILES : {smi2}")
        print(f"Aspirin chromosome K           : {chrom2.K}")
    else:
        print("encode(aspirin) returned None")
    print()

    # 3. Test get_neighbors().
    neighbors = get_neighbors(chrom)
    print(f"Number of neighbors (delta=1)  : {len(neighbors)}")
    valid_neighbors = [n for n in neighbors if decode(n)[0] is not None]
    print(f"Valid neighbors (mol not None) : {len(valid_neighbors)}")
    print()

    # 4. Test random_population().
    pop = random_population(size=10, seed=0)
    valid_pop = [c for c in pop if decode(c)[0] is not None]
    print(f"Population of 10: {len(valid_pop)}/10 decoded to valid molecules")
    print()

    # 5. Confirm 100% validity across a larger sample.
    N_SAMPLE = 200
    pop_large = random_population(size=N_SAMPLE, seed=1)
    n_valid = sum(1 for c in pop_large if decode(c)[0] is not None)
    print(f"Validity check ({N_SAMPLE} random chromosomes): {n_valid}/{N_SAMPLE} valid")
    assert n_valid == N_SAMPLE, "SELFIES validity guarantee violated — check vocabulary!"
    print("All assertions passed. representation.py is ready.")