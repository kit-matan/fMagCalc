"""Regression test for the vertical-streak artifact.

The 9-spin GUI model has a Goldstone (zero-energy) degenerate subspace at
Gamma where pyMagCalc's KKdMatrix is numerically unstable: a machine-epsilon
change in H(q) — exactly what the bond-model reconstruction introduces — flips
the intensity from 0 to a spurious 3S, producing a bright vertical line at that
q in S(Q,w). The NumPy path (exact lambdified H) gives 0; the Fortran model path
used to give the spurious value.

The fix suppresses ill-conditioned zero-energy modes (magcalc_sqw.f90). This
test pins the q=0 intensities to 0 and checks full-grid parity with the saved
NumPy oracle. Without the fix, q=0 would be ~3S (=7.5 here) and this fails.

Regenerate with the helper in the commit message / docs (uses the
.config_gui_run.yaml 9-spin model).
"""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "python"))

FIXTURE = os.path.join(REPO, "python", "fixtures", "gui9_model.npz")
LIB = os.path.join(REPO, "build", "libfmagcalc.dylib")


@pytest.mark.skipif(not os.path.exists(FIXTURE), reason="fixture not generated")
@pytest.mark.skipif(not os.path.exists(LIB), reason="libfmagcalc not built")
def test_no_zero_mode_streak():
    from fmagcalc import run_sqw_model

    d = np.load(FIXTURE)
    S = float(d["S"])
    r = run_sqw_model(d["dvec"], d["Mb"], d["Ud"], np.ones_like(d["energies"]), S, d["q_grid"])

    # q=0 (first row) must be ~0, not the spurious ~3S Goldstone streak.
    i0 = int(np.argmin(np.sum(np.asarray(d["q_grid"]) ** 2, axis=1)))
    assert np.nanmax(r["intensities"][i0]) < 1e-6, \
        f"spurious intensity at Gamma: {r['intensities'][i0]}"

    # Full-grid parity with the NumPy oracle.
    iN = d["intensities"]
    inten = np.nanmax(np.abs(r["intensities"] - iN))
    assert inten < 1e-6, f"max |I_fortran - I_numpy| = {inten:.3e}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
