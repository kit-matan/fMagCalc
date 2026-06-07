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

# M1: dispersion is served by the standalone Fortran exe over a binary file
# (see _bin.py). M4 will swap this for an in-process f2py extension behind the
# same signature.
from ._bin import run_dispersion, run_sqw  # noqa: F401
