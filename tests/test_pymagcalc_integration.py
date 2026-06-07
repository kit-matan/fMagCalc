"""M7 integration test: drive pyMagCalc with backend="fortran" and confirm it
reproduces pyMagCalc's own NumPy result (the saved S(Q,w) oracle).

This exercises the opt-in path added to pyMagCalc.core.calculate_sqw
(_calculate_sqw_fortran -> fmagcalc.extract_bond_model + run_sqw_model).

Requires pyMagCalc importable (with the fortran-backend branch checked out) and
the compiled fMagCalc library. Skips cleanly otherwise.
"""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
PYMAGCALC = os.path.abspath(os.path.join(REPO, "..", "pyMagCalc"))
KFE3J = os.path.join(PYMAGCALC, "examples", "KFe3J")

sys.path.insert(0, os.path.join(REPO, "python"))
sys.path.insert(0, PYMAGCALC)
sys.path.insert(0, KFE3J)

FIXTURE = os.path.join(REPO, "python", "fixtures", "kfe3j_sqw.npz")
LIB = os.path.join(REPO, "build", "libfmagcalc.dylib")

pytestmark = [
    pytest.mark.skipif(not os.path.exists(FIXTURE), reason="fixture not generated"),
    pytest.mark.skipif(not os.path.exists(LIB), reason="libfmagcalc not built"),
    pytest.mark.skipif(not os.path.exists(KFE3J), reason="pyMagCalc KFe3J example not found"),
]


def _has_backend():
    try:
        import fmagcalc
        return getattr(fmagcalc, "backend", None) == "ctypes"
    except Exception:
        return False


@pytest.mark.skipif(not _has_backend(), reason="fMagCalc ctypes backend unavailable")
def test_pymagcalc_fortran_backend_matches_numpy_oracle():
    import logging
    logging.disable(logging.WARNING)
    import magcalc as mc
    import spin_model as kfe

    d = np.load(FIXTURE)
    q_grid = d["q_grid"]

    calc = mc.MagCalc(
        spin_magnitude=float(d["S"]),
        hamiltonian_params=[3.23, 0.11, 0.218, -0.195, [0, 0, 1], 0.0, 0.0],
        cache_file_base="KFe3J_m7",
        cache_mode="auto",
        spin_model_module=kfe,
    )

    res = calc.calculate_sqw(q_grid, backend="fortran")
    assert res is not None

    en = np.nanmax(np.abs(np.sort(res.energies, 1) - np.sort(d["energies"], 1)))
    assert en < 1e-8, f"max |E - E_oracle| = {en:.3e}"

    inten = np.nanmax(np.abs(res.intensities - d["intensities"]))
    assert inten < 1e-7, f"max |I - I_oracle| = {inten:.3e}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
