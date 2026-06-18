"""
surrogate/train.py
==================
Full training pipeline for AffinityMLP.

Pipeline
--------
1.  Download EGFR (CHEMBL203) IC50 data via chembl_webresource_client
    (~15 k raw records; cached to data/egfr_ic50.csv after first run)
2.  Preprocess: keep nM + exact '=' relations, IC50 range filter,
    RDKit SMILES validation, dedup by canonical SMILES (median IC50)
3.  Convert IC50 [nM] → ΔG_bind [kcal/mol]   via  ΔG = RT·ln(IC50_M)  at T=310K
4.  Compute 2048-bit Morgan fingerprints (radius=2)
5.  Stratified 70 / 15 / 15 train / val / test split by ΔG decile
6.  Train AffinityMLP  (Adam, ReduceLROnPlateau, early stopping patience=15)
7.  Calibration report:
      – Test RMSE + Pearson R + R²
      – Predicted-vs-actual scatter plot  → surrogate/calibration.png
      – MC Dropout uncertainty flagging   (std > 1.0 kcal/mol)
      – RMSE threshold check              (target < 1.5 kcal/mol)
8.  EA sensitivity analysis:
      Spearman ρ between clean and noise-perturbed rankings
      at σ = 0, 0.5, 1.0, 1.5 kcal/mol
9.  Save best checkpoint → surrogate/checkpoints/best.pt

Run
---
    cd ~/drugea
    python surrogate/train.py
"""

from __future__ import annotations

import math
import os
import sys
import warnings
from pathlib import Path

# Ensure project root on sys.path when running as a script directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")   # non-interactive — safe for headless / script runs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.DataStructs import ConvertToNumpyArray
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from surrogate.model import AffinityMLP, mol_to_fp

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ROOT      = Path(__file__).resolve().parent.parent   # ~/drugea
DATA_CACHE = _ROOT / "data" / "egfr_ic50.csv"
CKPT_DIR   = _ROOT / "surrogate" / "checkpoints"
CKPT_PATH  = CKPT_DIR / "best.pt"
CALIB_PNG  = _ROOT / "surrogate" / "calibration.png"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_CHEMBL_ID = "CHEMBL203"         # EGFR kinase

R_KCAL   = 0.001987                    # kcal / (mol·K)
T_KELVIN = 310.0                       # physiological temperature
RT       = R_KCAL * T_KELVIN           # ≈ 0.6160 kcal/mol
LN10     = math.log(10.0)

IC50_MIN_NM = 0.1                      # discard implausibly potent values
IC50_MAX_NM = 100_000.0                # discard very weak / inactive

BATCH_SIZE   = 64
MAX_EPOCHS   = 150
PATIENCE     = 15
LR           = 1e-3
MC_SAMPLES   = 50                      # forward passes for uncertainty
UNC_THRESH   = 1.0                     # kcal/mol — flag threshold
RMSE_TARGET  = 1.5                     # kcal/mol — minimum trusted RMSE

_MORGAN_GEN = GetMorganGenerator(radius=2, fpSize=2048)


# ---------------------------------------------------------------------------
# 1. Download
# ---------------------------------------------------------------------------

def download_egfr_ic50() -> pd.DataFrame:
    """
    Fetch EGFR IC50 data from ChEMBL via the webresource client.

    Returns a raw DataFrame with columns:
        chembl_id, smiles, ic50_value, units, relation
    """
    from chembl_webresource_client.new_client import new_client

    print(f"Downloading IC50 data for {TARGET_CHEMBL_ID} from ChEMBL ...")
    api = new_client.activity
    records = api.filter(
        target_chembl_id=TARGET_CHEMBL_ID,
        standard_type="IC50",
    ).only([
        "molecule_chembl_id",
        "canonical_smiles",
        "standard_value",
        "standard_units",
        "standard_relation",
    ])

    rows = []
    for rec in records:
        rows.append({
            "chembl_id": rec.get("molecule_chembl_id", ""),
            "smiles":    rec.get("canonical_smiles", "") or "",
            "ic50_value": rec.get("standard_value"),
            "units":     rec.get("standard_units", ""),
            "relation":  rec.get("standard_relation", "="),
        })

    df = pd.DataFrame(rows)
    print(f"  Raw records downloaded : {len(df):,}")
    return df


# ---------------------------------------------------------------------------
# 2+3. Preprocess + IC50 → ΔG conversion
# ---------------------------------------------------------------------------

def preprocess(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw ChEMBL IC50 data and convert to ΔG.

    Steps
    -----
    1. Keep only '=' relations (drop '>' / '<' censored data)
    2. Keep only nM units
    3. Drop rows with missing SMILES or IC50
    4. Convert IC50 to float; filter to [IC50_MIN_NM, IC50_MAX_NM]
    5. Validate SMILES with RDKit, canonicalize
    6. Deduplicate by canonical SMILES (median IC50)
    7. Convert IC50 [nM] → ΔG [kcal/mol]:
           ΔG = RT · ln(IC50_M) = RT · (ln(IC50_nM) − 9·ln10)

    Returns DataFrame with columns: smiles, ic50_nm, dg_bind
    """
    df = df_raw.copy()

    # 1. Exact measurements only
    df = df[df["relation"].isin(["=", ""])].copy()

    # 2. nM only
    df = df[df["units"] == "nM"].copy()

    # 3. Drop missing
    df = df.dropna(subset=["smiles", "ic50_value"])
    df = df[df["smiles"].str.len() > 0]

    # 4. Numeric IC50 + range filter
    df["ic50_nm"] = pd.to_numeric(df["ic50_value"], errors="coerce")
    df = df.dropna(subset=["ic50_nm"])
    df = df[(df["ic50_nm"] >= IC50_MIN_NM) & (df["ic50_nm"] <= IC50_MAX_NM)].copy()

    # 5. RDKit validation + canonicalisation
    valid: list[dict] = []
    for _, row in df.iterrows():
        mol = Chem.MolFromSmiles(row["smiles"])
        if mol is not None:
            valid.append({
                "smiles":   Chem.MolToSmiles(mol),
                "ic50_nm":  row["ic50_nm"],
            })

    df_clean = pd.DataFrame(valid)
    print(f"  After filtering / validation : {len(df_clean):,} records")

    # 6. Dedup: median IC50 per canonical SMILES
    df_dedup = (
        df_clean
        .groupby("smiles", as_index=False)["ic50_nm"]
        .median()
    )
    print(f"  After deduplication          : {len(df_dedup):,} unique molecules")

    # 7. IC50 [nM] → ΔG [kcal/mol]
    df_dedup["dg_bind"] = RT * (np.log(df_dedup["ic50_nm"]) - 9.0 * LN10)

    print(f"  ΔG range : [{df_dedup['dg_bind'].min():.2f}, "
          f"{df_dedup['dg_bind'].max():.2f}] kcal/mol")
    print(f"  ΔG mean  : {df_dedup['dg_bind'].mean():.2f} kcal/mol")

    return df_dedup[["smiles", "ic50_nm", "dg_bind"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4. Fingerprints
# ---------------------------------------------------------------------------

def compute_fingerprints(df: pd.DataFrame) -> np.ndarray:
    """
    Compute 2048-bit Morgan fingerprints for all molecules.

    Returns float32 array of shape (N, 2048).
    Any row whose SMILES fails RDKit is left as an all-zero vector
    (shouldn't occur after preprocess() validation).
    """
    print(f"Computing Morgan fingerprints for {len(df):,} molecules ...")
    fps = np.zeros((len(df), 2048), dtype=np.float32)
    for i, smiles in enumerate(df["smiles"]):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fps[i] = mol_to_fp(mol)
    print("  Done.")
    return fps


# ---------------------------------------------------------------------------
# 5. Stratified split
# ---------------------------------------------------------------------------

def stratified_split(
    fps: np.ndarray,
    dg: np.ndarray,
    train_frac: float = 0.70,
    seed: int = 42,
) -> dict:
    """
    Stratified 70 / 15 / 15 train / val / test split by ΔG decile.

    Falls back to random split if stratification fails (too few samples
    per bin or duplicate bin edges).

    Returns dict: X_train, X_val, X_test, y_train, y_val, y_test.
    """
    dg32 = dg.astype(np.float32)

    # Build decile labels for stratification
    try:
        strata = pd.qcut(dg, q=10, labels=False, duplicates="drop").values
    except Exception:
        strata = np.zeros(len(dg), dtype=int)

    # First split: train vs temp (val+test)
    try:
        X_tr, X_tmp, y_tr, y_tmp, s_tr, s_tmp = train_test_split(
            fps, dg32, strata,
            test_size=1.0 - train_frac,
            stratify=strata,
            random_state=seed,
        )
    except Exception:
        warnings.warn("Stratified split failed; using random split.")
        X_tr, X_tmp, y_tr, y_tmp = train_test_split(
            fps, dg32, test_size=1.0 - train_frac, random_state=seed
        )
        s_tmp = np.zeros(len(y_tmp), dtype=int)

    # Second split: val vs test (50/50 of temp)
    try:
        X_val, X_te, y_val, y_te = train_test_split(
            X_tmp, y_tmp,
            test_size=0.5,
            stratify=s_tmp,
            random_state=seed,
        )
    except Exception:
        X_val, X_te, y_val, y_te = train_test_split(
            X_tmp, y_tmp, test_size=0.5, random_state=seed
        )

    print(f"  Train: {len(X_tr):,}   Val: {len(X_val):,}   Test: {len(X_te):,}")

    return {
        "X_train": X_tr,  "y_train": y_tr,
        "X_val":   X_val, "y_val":   y_val,
        "X_test":  X_te,  "y_test":  y_te,
    }


# ---------------------------------------------------------------------------
# 6. Training
# ---------------------------------------------------------------------------

def _make_loader(X: np.ndarray, y: np.ndarray,
                 batch_size: int = BATCH_SIZE, shuffle: bool = True) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


class _EarlyStopping:
    def __init__(self, patience: int = PATIENCE):
        self.patience   = patience
        self.best_loss  = float("inf")
        self.counter    = 0
        self.best_state: dict | None = None

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - 1e-6:
            self.best_loss  = val_loss
            self.counter    = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            return False
        self.counter += 1
        return self.counter >= self.patience


def train_model(splits: dict) -> tuple[AffinityMLP, list[dict]]:
    """
    Train AffinityMLP with Adam + ReduceLROnPlateau + early stopping.

    Returns (best model, training history list of dicts).
    """
    train_loader = _make_loader(splits["X_train"], splits["y_train"])
    val_loader   = _make_loader(splits["X_val"],   splits["y_val"],   shuffle=False)

    model     = AffinityMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    criterion = nn.MSELoss()
    stopper   = _EarlyStopping(patience=PATIENCE)
    history: list[dict] = []

    print(f"\nTraining AffinityMLP  "
          f"(max {MAX_EPOCHS} epochs, early-stop patience={PATIENCE}) ...")

    for epoch in range(1, MAX_EPOCHS + 1):
        # ---- train ----
        model.train()
        train_loss = 0.0
        for Xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(Xb)
        train_loss /= len(splits["X_train"])

        # ---- validate ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                val_loss += criterion(model(Xb), yb).item() * len(Xb)
        val_loss /= len(splits["X_val"])

        scheduler.step(val_loss)
        history.append({"epoch": epoch,
                         "train_rmse": math.sqrt(train_loss),
                         "val_rmse":   math.sqrt(val_loss)})

        if epoch % 10 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch:>3d}  "
                  f"train_RMSE={math.sqrt(train_loss):.4f}  "
                  f"val_RMSE={math.sqrt(val_loss):.4f}  "
                  f"lr={lr_now:.2e}")

        if stopper(val_loss, model):
            print(f"  Early stopping at epoch {epoch} "
                  f"(best val_RMSE={math.sqrt(stopper.best_loss):.4f})")
            break

    # Restore best weights
    if stopper.best_state is not None:
        model.load_state_dict(stopper.best_state)
    model.eval()
    print(f"  Final best val RMSE : {math.sqrt(stopper.best_loss):.4f} kcal/mol\n")

    return model, history


# ---------------------------------------------------------------------------
# 7. Calibration
# ---------------------------------------------------------------------------

def calibrate(model: AffinityMLP, X_test: np.ndarray, y_test: np.ndarray) -> float:
    """
    Full calibration protocol on the held-out test set.

    1. Test RMSE, MAE, Pearson R, R²
    2. Predicted-vs-actual scatter plot → CALIB_PNG
    3. MC Dropout uncertainty flagging
    4. RMSE threshold assertion (target < RMSE_TARGET)

    Returns test RMSE.
    """
    print("=" * 55)
    print("Calibration report")
    print("=" * 55)

    fps_tensor = torch.from_numpy(X_test)
    model.eval()
    with torch.no_grad():
        y_pred = model(fps_tensor).numpy().astype(np.float64)

    y_true = y_test.astype(np.float64)

    rmse = math.sqrt(float(np.mean((y_true - y_pred) ** 2)))
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    r, _ = pearsonr(y_true, y_pred)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    print(f"Test set size : {len(y_true)}")
    print(f"RMSE          : {rmse:.4f} kcal/mol  (target < {RMSE_TARGET})")
    print(f"MAE           : {mae:.4f} kcal/mol")
    print(f"Pearson R     : {r:.4f}")
    print(f"R²            : {r2:.4f}")

    # --- scatter plot ---
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, s=6, alpha=0.4, color="steelblue",
               label="test molecules")
    lo = min(y_true.min(), y_pred.min()) - 0.5
    hi = max(y_true.max(), y_pred.max()) + 0.5
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="y = x  (perfect)")
    ax.set_xlabel("Actual ΔG_bind (kcal/mol)")
    ax.set_ylabel("Predicted ΔG_bind (kcal/mol)")
    ax.set_title(
        f"AffinityMLP Calibration — EGFR (CHEMBL203)\n"
        f"RMSE={rmse:.3f}  Pearson R={r:.3f}  n={len(y_true)}"
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(CALIB_PNG, dpi=150)
    plt.close(fig)
    print(f"Scatter plot  → {CALIB_PNG}")

    # --- MC Dropout uncertainty (batch) ---
    print(f"\nMC Dropout uncertainty flagging (mc_samples={MC_SAMPLES}) ...")
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

    mc_preds: list[np.ndarray] = []
    with torch.no_grad():
        for _ in range(MC_SAMPLES):
            mc_preds.append(model(fps_tensor).numpy())

    model.eval()
    stds      = np.stack(mc_preds).std(axis=0)    # (N_test,)
    high_unc  = int((stds > UNC_THRESH).sum())
    pct       = 100.0 * high_unc / len(y_test)
    print(f"  High-uncertainty predictions (std > {UNC_THRESH} kcal/mol): "
          f"{high_unc}/{len(y_test)} ({pct:.1f}%)")

    # --- RMSE threshold ---
    print()
    if rmse < RMSE_TARGET:
        print(f"  RMSE={rmse:.4f} < {RMSE_TARGET}  →  surrogate TRUSTED  ✓")
    else:
        print(f"  WARNING: RMSE={rmse:.4f} >= {RMSE_TARGET}  "
              f"→  surrogate may be unreliable")
        print("  Consider: more data, more epochs, or architecture adjustments.")

    return rmse


# ---------------------------------------------------------------------------
# 8. EA sensitivity analysis
# ---------------------------------------------------------------------------

def ea_sensitivity(model: AffinityMLP, X_test: np.ndarray, y_test: np.ndarray) -> None:
    """
    Measure how surrogate noise would degrade EA selection.

    For each noise level σ, Gaussian noise N(0,σ²) is added to model
    predictions on the test set.  Spearman ρ between the clean ranking
    and the noisy ranking quantifies how much selection pressure is
    preserved as surrogate error grows.

    ρ ≈ 1.0 → ranking is stable → EA selection pressure is preserved.
    ρ drops toward 0 → ranking randomises → EA loses directional signal.
    """
    print("\n" + "=" * 55)
    print("EA sensitivity analysis")
    print("=" * 55)
    print("Spearman ρ(clean ranking, noisy ranking) vs noise σ\n")

    model.eval()
    with torch.no_grad():
        y_clean = model(torch.from_numpy(X_test)).numpy().astype(np.float64)

    rng    = np.random.default_rng(0)
    sigmas = [0.0, 0.5, 1.0, 1.5]

    print(f"  {'σ (kcal/mol)':>14}   {'Spearman ρ':>10}   {'p-value':>12}")
    print("  " + "-" * 42)
    for sigma in sigmas:
        noisy      = y_clean + rng.normal(0.0, sigma, size=len(y_clean))
        rho, pval  = spearmanr(y_clean, noisy)
        print(f"  {sigma:>14.1f}   {rho:>10.4f}   {pval:>12.2e}")

    print()
    print("ρ close to 1.0 → EA molecular ranking preserved despite surrogate error.")
    print("A model with RMSE ≈ 1 kcal/mol typically gives ρ > 0.95 on drug-like sets.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 65)
    print("DrugEA — AffinityMLP Surrogate Training Pipeline")
    print("=" * 65)
    print()

    # 1+2+3. Data
    if DATA_CACHE.exists():
        print(f"Loading cached data from {DATA_CACHE}")
        df = pd.read_csv(DATA_CACHE)
        print(f"  {len(df):,} molecules loaded.")
        print(f"  ΔG range : [{df['dg_bind'].min():.2f}, "
              f"{df['dg_bind'].max():.2f}] kcal/mol")
    else:
        df_raw = download_egfr_ic50()
        df     = preprocess(df_raw)
        DATA_CACHE.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(DATA_CACHE, index=False)
        print(f"\nSaved processed data to {DATA_CACHE}")

    print()

    # 4. Fingerprints
    fps = compute_fingerprints(df)
    dg  = df["dg_bind"].values.astype(np.float32)
    print()

    # 5. Split
    print("Splitting dataset (70 / 15 / 15) ...")
    splits = stratified_split(fps, dg)
    print()

    # 6. Train
    model, history = train_model(splits)

    # 7. Calibrate
    rmse = calibrate(model, splits["X_test"], splits["y_test"])

    # 8. EA sensitivity
    ea_sensitivity(model, splits["X_test"], splits["y_test"])

    # 9. Save checkpoint
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    model.save(str(CKPT_PATH))
    print(f"\nCheckpoint saved → {CKPT_PATH}")

    print()
    print("=" * 65)
    print("Next: swap surrogate in fitness.py")
    print()
    print("  # Replace these two lines:")
    print("  SURROGATE = MockSurrogate()")
    print()
    print("  # With:")
    print("  from surrogate.model import AffinityMLP")
    print("  SURROGATE = AffinityMLP.load('surrogate/checkpoints/best.pt')")
    print("=" * 65)


if __name__ == "__main__":
    main()
