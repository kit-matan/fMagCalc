"""M1 parity test: the Fortran dispersion driver must reproduce pyMagCalc's
energies (the oracle) on the exported fixture.

Regenerate the fixture with:
    python python/export_model.py --model-dir ../pyMagCalc/examples/KFe3J \\
        --model-module spin_model --out python/fixtures/kfe3j_disp.npz
"""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "python"))

FIXTURE = os.path.join(REPO, "python", "fixtures", "kfe3j_disp.npz")
EXE = os.path.join(REPO, "build", "fmagcalc_disp")


@pytest.mark.skipif(not os.path.exists(FIXTURE), reason="fixture not generated")
@pytest.mark.skipif(not os.path.exists(EXE), reason="fmagcalc_disp not built")
def test_dispersion_matches_pymagcalc():
    from fmagcalc import run_dispersion

    data = np.load(FIXTURE)
    h_plus = data["H_plus"]
    oracle = data["energies"]  # (Nq, N) from pyMagCalc

    energies, info = run_dispersion(h_plus)

    assert np.all(info == 0), f"zgeev failures at q-indices {np.flatnonzero(info)}"
    assert energies.shape == oracle.shape

    # pyMagCalc sorts ascending and keeps the upper-N (magnon) branch; we do the
    # same. Sort each row to be robust to any residual ordering convention.
    e_sorted = np.sort(energies, axis=1)
    o_sorted = np.sort(oracle, axis=1)
    max_abs = np.nanmax(np.abs(e_sorted - o_sorted))
    assert max_abs < 1e-8, f"max |E_fortran - E_py| = {max_abs:.3e}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
