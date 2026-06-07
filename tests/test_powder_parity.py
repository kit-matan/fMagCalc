"""M6 parity test: fMagCalc powder average vs pyMagCalc.calculate_powder_average.

The q-sampling (Fibonacci sphere) is shared with the oracle, so the comparison
is sample-for-sample; only the per-q S(q,w) engine differs. Tolerances match the
model path (~1e-10 floor from the symbolic bond rewrite).

Regenerate the fixture with:
    python python/export_model.py --task powder \\
        --model-dir ../pyMagCalc/examples/KFe3J --model-module spin_model \\
        --out python/fixtures/kfe3j_powder.npz
"""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "python"))

FIXTURE = os.path.join(REPO, "python", "fixtures", "kfe3j_powder.npz")
LIB = os.path.join(REPO, "build", "libfmagcalc.dylib")


@pytest.mark.skipif(not os.path.exists(FIXTURE), reason="fixture not generated")
@pytest.mark.skipif(not os.path.exists(LIB), reason="libfmagcalc not built")
def test_powder_matches_pymagcalc():
    import fmagcalc

    d = np.load(FIXTURE)
    S = float(d["S"])
    r = fmagcalc.powder_average(d["dvec"], d["Mb"], d["Ud"], S,
                                d["q_magnitudes"], num_samples=int(d["num_samples"]))

    en = np.nanmax(np.abs(r["energies"] - d["energies"]))
    assert en < 1e-8, f"max |E_f - E_py| = {en:.3e}"

    inten = np.nanmax(np.abs(r["intensities"] - d["intensities"]))
    assert inten < 1e-7, f"max |I_f - I_py| = {inten:.3e}"

    sig = d["intensities"] > 1e-3
    if sig.any():
        rel = np.nanmax(np.abs(r["intensities"][sig] - d["intensities"][sig]) / d["intensities"][sig])
        assert rel < 1e-6, f"max relative intensity error = {rel:.3e}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
