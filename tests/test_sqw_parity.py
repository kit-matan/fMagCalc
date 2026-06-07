"""M2 parity test: the Fortran S(Q,w) pipeline (KKdMatrix + intensity) must
reproduce pyMagCalc's energies and intensities on the exported fixture.

Note on what is (not) asserted:
  * energies and intensities are physical observables and are checked tightly.
  * K / Kd are NOT asserted: they depend on the arbitrary phase LAPACK assigns
    to each eigenvector (especially within degenerate subspaces). The intensity
    is a gauge-invariant bilinear of K and Kd, so it is the meaningful check.
    K/Kd are still returned by run_sqw for debugging.

Regenerate the fixture with:
    python python/export_model.py --task sqw \\
        --model-dir ../pyMagCalc/examples/KFe3J --model-module spin_model \\
        --out python/fixtures/kfe3j_sqw.npz
"""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "python"))

FIXTURE = os.path.join(REPO, "python", "fixtures", "kfe3j_sqw.npz")
EXE = os.path.join(REPO, "build", "fmagcalc_sqw")


@pytest.mark.skipif(not os.path.exists(FIXTURE), reason="fixture not generated")
@pytest.mark.skipif(not os.path.exists(EXE), reason="fmagcalc_sqw not built")
def test_sqw_matches_pymagcalc():
    from fmagcalc import run_sqw

    d = np.load(FIXTURE)
    n, S = int(d["n"]), float(d["S"])
    r = run_sqw(d["H_plus"], d["H_minus"], d["Ud"], d["ff"], S, d["q_grid"])

    assert np.all(r["info"] == 0), f"KKd failures at q-indices {np.flatnonzero(r['info'])}"

    # Eigenvalues (Bogoliubov spectrum) — sanity check on the diagonalization.
    e_max = np.nanmax(np.abs(r["eigvals"] - d["eigvals"]))
    assert e_max < 1e-8, f"max |eigvals_f - eigvals_py| = {e_max:.3e}"

    # Energies (positive branch).
    en_max = np.nanmax(np.abs(np.sort(r["energies"], 1) - np.sort(d["energies"], 1)))
    assert en_max < 1e-8, f"max |E_f - E_py| = {en_max:.3e}"

    # Intensities — the gauge-invariant observable.
    inten_abs = np.nanmax(np.abs(r["intensities"] - d["intensities"]))
    assert inten_abs < 1e-7, f"max |I_f - I_py| = {inten_abs:.3e}"

    sig = d["intensities"] > 1e-3
    if sig.any():
        rel = np.nanmax(np.abs(r["intensities"][sig] - d["intensities"][sig]) / d["intensities"][sig])
        assert rel < 1e-6, f"max relative intensity error = {rel:.3e}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
