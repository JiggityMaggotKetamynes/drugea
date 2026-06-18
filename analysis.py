"""
analysis.py
===========
Answers RQ1 and RQ2 from the project spec.

RQ1: Do self-adapted σ_i values recover the scaffold-decoration pattern?
     (core fragments frozen → low σ; peripheral R-groups active → high σ)

RQ2: Do evolved w_j values reveal which pharmacological constraints are
     hardest to satisfy in this chemical space?

Runs the EA for 100 generations, then analyses the final population's
strategy parameters (σ arrays and w arrays) across all survivors.

Usage
-----
    python analysis.py
    python analysis.py --seed 7 --gens 100
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from main import run_ea, EAConfig
from representation import FRAGMENT_VOCABULARY, N_CONSTRAINTS

# ---------------------------------------------------------------------------
# Human-readable fragment names (parallel to FRAGMENT_VOCABULARY)
# ---------------------------------------------------------------------------

FRAGMENT_NAMES = [
    # aromatic carbocycles
    "benzene",
    "naphthalene",
    # N-containing aromatics
    "pyridine",
    "pyrimidine",
    "pyrazine",
    "imidazole",
    "pyrazole",
    "indole",
    "benzimidazole",
    "quinoline",
    # O/S aromatics
    "furan",
    "thiophene",
    "benzofuran",
    "benzothiophene",
    # saturated N-heterocycles
    "piperidine",
    "piperazine",
    "morpholine",
    "pyrrolidine",
    "oxazoline",
    # functional linkers
    "amide",
    "sulfonamide",
    "urea",
    "ether",
    "ester",
    "nitrile",
    "fluoromethyl",
    "chloromethyl",
    # R-group decorators
    "methyl",
    "ethyl",
    "propyl",
    "isopropyl",
    "butyl",
    "methoxy",
    "ethoxy",
    "amine",
    "methylamine",
    "carboxylic acid",
    "alcohol",
    "ethanol",
]

# Truncate to actual validated vocabulary size (a few tokens may have been dropped)
FRAGMENT_NAMES = FRAGMENT_NAMES[: len(FRAGMENT_VOCABULARY)]

# Broad category for each fragment (used for grouping in RQ1 plot)
def _category(name: str) -> str:
    aromatic_cores = {
        "benzene", "naphthalene", "pyridine", "pyrimidine", "pyrazine",
        "indole", "benzimidazole", "quinoline", "benzofuran", "benzothiophene",
    }
    small_heterocycles = {
        "imidazole", "pyrazole", "furan", "thiophene",
        "piperidine", "piperazine", "morpholine", "pyrrolidine", "oxazoline",
    }
    linkers = {
        "amide", "sulfonamide", "urea", "ether", "ester",
        "nitrile", "fluoromethyl", "chloromethyl",
    }
    if name in aromatic_cores:
        return "aromatic core"
    if name in small_heterocycles:
        return "ring system"
    if name in linkers:
        return "linker / pharmacophore"
    return "R-group / decorator"


CONSTRAINT_NAMES = ["MW", "logP", "HBD", "HBA", "SA", "PAINS"]

# ---------------------------------------------------------------------------
# RQ1: per-fragment σ analysis
# ---------------------------------------------------------------------------

def analyse_sigma(population: list) -> None:
    """
    RQ1: Collect σ_i for every (chromosome, position) pair in the final
    population. Group by fragment identity, compute mean and std.
    """
    sigma_by_frag: defaultdict[int, List[float]] = defaultdict(list)

    for chrom in population:
        for frag_idx, sigma_val in zip(chrom.b, chrom.sigma):
            sigma_by_frag[frag_idx].append(float(sigma_val))

    # Only report fragments that actually appear in the final population
    rows = []
    for frag_idx, sigmas in sigma_by_frag.items():
        name = (
            FRAGMENT_NAMES[frag_idx]
            if frag_idx < len(FRAGMENT_NAMES)
            else f"frag_{frag_idx}"
        )
        rows.append({
            "idx":   frag_idx,
            "name":  name,
            "cat":   _category(name),
            "n":     len(sigmas),
            "mean":  float(np.mean(sigmas)),
            "std":   float(np.std(sigmas)),
        })

    rows.sort(key=lambda r: r["mean"])

    print("\nRQ1 — Per-fragment adapted σ_i  (final population)")
    print("─" * 68)
    print(f"  {'Fragment':>16}   {'Category':>22}   "
          f"{'n':>4}   {'Mean σ':>7}   {'Std σ':>6}")
    print("  " + "─" * 64)
    for r in rows:
        frozen_label = " ← frozen" if r["mean"] < 0.25 else (
                       " ← active" if r["mean"] > 0.55 else "")
        print(f"  {r['name']:>16}   {r['cat']:>22}   "
              f"{r['n']:>4}   {r['mean']:>7.4f}   {r['std']:>6.4f}"
              + frozen_label)

    # Plot: mean σ per fragment, coloured by category
    _plot_sigma(rows)


def _plot_sigma(rows: list) -> None:
    cat_colors = {
        "aromatic core":       "steelblue",
        "ring system":         "mediumseagreen",
        "linker / pharmacophore": "goldenrod",
        "R-group / decorator": "tomato",
    }

    names  = [r["name"]  for r in rows]
    means  = [r["mean"]  for r in rows]
    stds   = [r["std"]   for r in rows]
    colors = [cat_colors[r["cat"]] for r in rows]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(names))
    bars = ax.bar(x, means, yerr=stds, color=colors, alpha=0.8,
                  capsize=3, error_kw={"elinewidth": 1})

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1,
               label="σ init = 0.5")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean adapted σ_i  (lower = more frozen)")
    ax.set_title(
        "RQ1 — Self-adapted σ_i by fragment\n"
        "(core aromatics should be frozen ↓; R-groups should stay active ↑)"
    )
    ax.set_ylim(0, 1.0)

    # Legend patches
    import matplotlib.patches as mpatches
    patches = [
        mpatches.Patch(color=c, label=k)
        for k, c in cat_colors.items()
    ]
    patches.append(
        plt.Line2D([0], [0], color="gray", linestyle="--", label="initial σ=0.5")
    )
    ax.legend(handles=patches, fontsize=8, loc="upper left")

    fig.tight_layout()
    out = "surrogate/rq1_sigma_adaptation.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\n  [RQ1 plot saved → {out}]")


# ---------------------------------------------------------------------------
# RQ2: per-constraint w_j analysis
# ---------------------------------------------------------------------------

def analyse_weights(population: list) -> None:
    """
    RQ2: Collect all six w_j values from the final population.
    Constraints with high evolved w_j are those the EA learned to take
    seriously — i.e., the ones that most penalise binding quality when violated.
    """
    w_matrix = np.array([chrom.w for chrom in population])  # (N, 6)

    means = w_matrix.mean(axis=0)
    stds  = w_matrix.std(axis=0)
    mins  = w_matrix.min(axis=0)
    maxs  = w_matrix.max(axis=0)

    print("\nRQ2 — Evolved penalty weights w_j  (final population)")
    print("─" * 66)
    print(f"  {'Constraint':>12}   {'Mean w':>8}   {'Std w':>7}   "
          f"{'Min':>7}   {'Max':>7}   Interpretation")
    print("  " + "─" * 62)

    for j, name in enumerate(CONSTRAINT_NAMES):
        interp = ""
        if means[j] > 1.5:
            interp = "HIGH → hardest to satisfy"
        elif means[j] < 0.7:
            interp = "low → rarely violated"
        print(f"  {name:>12}   {means[j]:>8.4f}   {stds[j]:>7.4f}   "
              f"{mins[j]:>7.4f}   {maxs[j]:>7.4f}   {interp}")

    _plot_weights(means, stds)

    # Interpretation note
    print()
    ranked = sorted(zip(CONSTRAINT_NAMES, means), key=lambda x: -x[1])
    print("  Ranked by difficulty (highest w_j = hardest to satisfy):")
    for i, (nm, m) in enumerate(ranked, 1):
        print(f"    {i}. {nm:>6}  (mean w = {m:.4f})")


def _plot_weights(means: np.ndarray, stds: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(CONSTRAINT_NAMES))

    ax.bar(x, means, yerr=stds, color="steelblue", alpha=0.8,
           capsize=4, error_kw={"elinewidth": 1.2})
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1,
               label="w init = 1.0")
    ax.set_xticks(x)
    ax.set_xticklabels(CONSTRAINT_NAMES)
    ax.set_ylabel("Mean evolved w_j  (higher = harder constraint)")
    ax.set_title(
        "RQ2 — Self-adapted penalty weights by constraint\n"
        "(above 1.0 = EA learned this constraint needs more pressure)"
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    out = "surrogate/rq2_weight_adaptation.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\n  [RQ2 plot saved → {out}]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(seed: int = 42, n_gens: int = 100) -> None:
    print("=" * 65)
    print("DrugEA — Strategy Parameter Analysis  (RQ1 & RQ2)")
    print("=" * 65)
    print(f"Running EA for {n_gens} generations (seed={seed}) ...")
    print()

    cfg = EAConfig(
        mu=30,
        lambda_=30,
        n_gens=n_gens,
        seed=seed,
        verbose=True,
    )
    results = run_ea(cfg)
    population = results["population"]

    print(f"\nFinal population: {len(population)} individuals  "
          f"({sum(c.feasible for c in population)} feasible)")

    analyse_sigma(population)
    analyse_weights(population)

    print()
    print("=" * 65)
    print("Analysis complete.")
    print("  RQ1 plot → surrogate/rq1_sigma_adaptation.png")
    print("  RQ2 plot → surrogate/rq2_weight_adaptation.png")
    print("=" * 65)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="RQ1 & RQ2 strategy parameter analysis")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gens", type=int, default=100)
    args = p.parse_args()
    main(seed=args.seed, n_gens=args.gens)
