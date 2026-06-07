"""Phase-2 parity test: the Fortran model path (build H(q) from the bond
decomposition, then KKd + intensity) must reproduce pyMagCalc's energies and
intensities — without any per-q Hamiltonian built in Python.

Tolerances are a touch looser than the H-stack path (test_sqw_parity.py) because
the bond decomposition goes through a symbolic rewrite/expand, which introduces
a ~1e-10 reconstruction error in H(q) — still far inside physical tolerance.

Regenerate the fixture with:
    python python/export_model.py --task model \\
        --model-dir ../pyMagCalc/examples/KFe3J --model-module spin_model \\
        --out python/fixtures/kfe3j_model.npz
"""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "python"))

FIXTURE = os.path.join(REPO, "python", "fixtures", "kfe3j_model.npz")
LIB = os.path.join(REPO, "build", "libfmagcalc.dylib")


@pytest.mark.skipif(not os.path.exists(FIXTURE), reason="fixture not generated")
@pytest.mark.skipif(not os.path.exists(LIB), reason="libfmagcalc not built")
def test_sqw_model_matches_pymagcalc():
    from fmagcalc import run_sqw_model

    d = np.load(FIXTURE)
    n, S = int(d["n"]), float(d["S"])
    r = run_sqw_model(d["dvec"], d["Mb"], d["Ud"], d["ff"], S, d["q_grid"])

    assert np.all(r["info"] == 0), f"KKd failures at {np.flatnonzero(r['info'])}"

    en = np.nanmax(np.abs(np.sort(r["energies"], 1) - np.sort(d["energies"], 1)))
    assert en < 1e-8, f"max |E_f - E_py| = {en:.3e}"

    inten = np.nanmax(np.abs(r["intensities"] - d["intensities"]))
    assert inten < 1e-7, f"max |I_f - I_py| = {inten:.3e}"

    sig = d["intensities"] > 1e-3
    if sig.any():
        rel = np.nanmax(np.abs(r["intensities"][sig] - d["intensities"][sig]) / d["intensities"][sig])
        assert rel < 1e-6, f"max relative intensity error = {rel:.3e}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
