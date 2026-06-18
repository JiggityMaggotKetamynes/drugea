"""
surrogate/model.py
==================
AffinityMLP — PyTorch MLP surrogate for EGFR binding affinity prediction.

Architecture
------------
    Linear(2048 → 512) → BatchNorm1d → ReLU → Dropout(0.2)
    Linear( 512 → 128) → BatchNorm1d → ReLU → Dropout(0.2)
    Linear( 128 →   1)

Interface
---------
Matches MockSurrogate in fitness.py — swapping requires only one line change:
    SURROGATE = AffinityMLP.load('surrogate/checkpoints/best.pt')

    dg_bind, uncertainty = SURROGATE.predict(mol)   # identical call signature

Uncertainty estimation
----------------------
MC Dropout: Dropout layers stay active at inference; BatchNorm stays in eval
mode to use running statistics (not batch stats for single samples).
    mc_samples=1  → deterministic inference, uncertainty=0.0  (use in EA loop)
    mc_samples=50 → stochastic; returns (mean, std)  (use for calibration)

Checkpoint I/O
--------------
    model.save("surrogate/checkpoints/best.pt")
    model = AffinityMLP.load("surrogate/checkpoints/best.pt")
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.DataStructs import ConvertToNumpyArray
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

_MORGAN_GEN = GetMorganGenerator(radius=2, fpSize=2048)


def mol_to_fp(mol: Chem.Mol) -> np.ndarray:
    """Return a 2048-bit Morgan fingerprint as a float32 numpy array."""
    arr = np.zeros(2048, dtype=np.float32)
    ConvertToNumpyArray(_MORGAN_GEN.GetFingerprint(mol), arr)
    return arr


class AffinityMLP(nn.Module):
    """
    MLP surrogate for EGFR ΔG_bind prediction.

    Input  : 2048-dim Morgan fingerprint (float32 tensor).
    Output : scalar ΔG_bind in kcal/mol (negative = tighter binding = better).
    """

    def __init__(self, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )
        self._dropout_p = dropout
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def _enable_mc_dropout(self) -> None:
        """Eval mode for everything except Dropout — correct MC Dropout pattern."""
        self.eval()
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    def predict(
        self,
        mol: Chem.Mol,
        mc_samples: int = 1,
    ) -> Tuple[float, float]:
        """
        Predict ΔG_bind for a molecule.

        Parameters
        ----------
        mol        : valid RDKit Mol.
        mc_samples : 1 → fast deterministic (recommended for EA loop).
                     >1 → MC Dropout; returns (mean, std).

        Returns
        -------
        (dg_bind, uncertainty) — matches MockSurrogate interface in fitness.py.
        """
        fp = torch.from_numpy(mol_to_fp(mol)).unsqueeze(0)   # (1, 2048)

        if mc_samples == 1:
            self.eval()
            with torch.no_grad():
                return float(self(fp).item()), 0.0

        self._enable_mc_dropout()
        preds = []
        with torch.no_grad():
            for _ in range(mc_samples):
                preds.append(float(self(fp).item()))
        self.eval()

        arr = np.array(preds, dtype=np.float64)
        return float(arr.mean()), float(arr.std())

    def save(self, path: str) -> None:
        """Save model state dict and config to path (creates parent dirs)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.state_dict(), "dropout": self._dropout_p}, p)

    @classmethod
    def load(cls, path: str) -> "AffinityMLP":
        """Load AffinityMLP from a checkpoint saved by save()."""
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        model = cls(dropout=ckpt.get("dropout", 0.2))
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model


# ---------------------------------------------------------------------------
# Self-test  (run with: python surrogate/model.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    import os

    print("=" * 60)
    print("surrogate/model.py self-test")
    print("=" * 60)

    model = AffinityMLP()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters : {n_params:,}")
    print(f"Architecture:\n{model}\n")

    aspirin = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
    assert aspirin is not None

    # 1. Single deterministic inference
    dg, unc = model.predict(aspirin)
    print(f"Aspirin  ΔG={dg:.4f}  uncertainty={unc:.4f}  (random weights)")
    assert unc == 0.0, "mc_samples=1 must give uncertainty=0.0"
    print("  uncertainty=0.0 for mc_samples=1  ✓\n")

    # 2. MC Dropout inference
    dg_mc, unc_mc = model.predict(aspirin, mc_samples=50)
    print(f"Aspirin MC50  ΔG={dg_mc:.4f}  std={unc_mc:.4f}")
    assert unc_mc >= 0.0
    print("  MC Dropout std >= 0  ✓\n")

    # 3. Batch forward pass
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(16, 2048))
    assert out.shape == (16,), f"Expected shape (16,), got {out.shape}"
    print(f"Batch (16, 2048) → {tuple(out.shape)}  ✓\n")

    # 4. Save / load round-trip
    with tempfile.TemporaryDirectory() as td:
        ckpt_path = os.path.join(td, "ckpt", "test.pt")
        model.save(ckpt_path)
        assert Path(ckpt_path).exists(), "save() did not create file"
        model2 = AffinityMLP.load(ckpt_path)
        dg2, _ = model2.predict(aspirin)
    assert abs(dg - dg2) < 1e-5, f"Load/save mismatch: {dg} vs {dg2}"
    print("Save / load round-trip  ✓\n")

    # 5. MC Dropout is actually stochastic (std > 0 with random weights)
    dg_a, std_a = model.predict(aspirin, mc_samples=30)
    assert std_a > 0.0, "MC Dropout should produce variance with random weights"
    print("MC Dropout is stochastic (std > 0 on random weights)  ✓\n")

    print("All assertions passed. surrogate/model.py is ready.")
