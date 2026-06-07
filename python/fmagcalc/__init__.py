"""fMagCalc — Python wrapper around the Fortran LSWT core.

Once the f2py extension (`fmagcalc._core`) is built, this package exposes
``run_dispersion`` and ``run_sqw`` returning the same result shapes pyMagCalc
uses (energies (Nq, N), intensities (Nq, N)), so pyMagCalc can opt in to the
Fortran backend without changing call sites.

STATUS: scaffold. The extension is not built yet; see PLAN.md milestone M4.
"""
from __future__ import annotations

__version__ = "0.0.0"
__all__ = ["run_dispersion", "run_sqw"]


def _not_built(*_args, **_kwargs):
    raise NotImplementedError(
        "fMagCalc Fortran extension not built yet. Build with CMake/f2py first "
        "(see PLAN.md milestones M1–M4)."
    )


run_dispersion = _not_built
run_sqw = _not_built
