"""fMagCalc — Python wrapper around the Fortran LSWT core.

Once the f2py extension (`fmagcalc._core`) is built, this package exposes
``run_dispersion`` and ``run_sqw`` returning the same result shapes pyMagCalc
uses (energies (Nq, N), intensities (Nq, N)), so pyMagCalc can opt in to the
Fortran backend without changing call sites.

STATUS: scaffold. The extension is not built yet; see PLAN.md milestone M4.
"""
from __future__ import annotations

__version__ = "0.0.0"
__all__ = ["run_dispersion", "run_sqw", "backend"]

# M4: prefer the in-process ctypes path (libfmagcalc shared lib, zero file I/O).
# Fall back to the M1 subprocess/binary-file transport if the shared lib is not
# built. Both expose identical signatures.
import os as _os

from ._capi import _lib_path as _capi_lib_path

if _os.path.exists(_capi_lib_path()):
    from ._capi import run_dispersion, run_sqw  # noqa: F401
    backend = "ctypes"
else:  # pragma: no cover - exercised only without the shared lib
    from ._bin import run_dispersion, run_sqw  # noqa: F401
    backend = "subprocess"
