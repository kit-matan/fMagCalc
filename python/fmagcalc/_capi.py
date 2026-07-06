"""In-process transport via ctypes + the bind(C) shared library
(libfmagcalc.{dylib,so}). Zero file I/O: NumPy array pointers are passed
straight into the Fortran OpenMP kernels.

Same signatures as the subprocess transport in _bin.py, so the package can use
this as the default backend and fall back to _bin if the shared lib is absent.

Arrays are marshalled to Fortran (column-major) layout to match the bind(C)
declarations in src/magcalc_capi.f90.
"""
from __future__ import annotations

import ctypes
import os
import sys

import numpy as np

_LIBNAMES = {
    "darwin": "libfmagcalc.dylib",
    "win32": "fmagcalc.dll",
}


def _lib_path() -> str:
    """Locate libfmagcalc. Search order: (1) FMAGCALC_LIB env var, (2) inside
    this package (where a pip/scikit-build-core install puts it), (3) the
    source-tree ../../build directory (dev CMake builds)."""
    name = _LIBNAMES.get(sys.platform, "libfmagcalc.so")
    env = os.environ.get("FMAGCALC_LIB")
    if env:
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    packaged = os.path.join(here, name)
    if os.path.exists(packaged):
        return packaged
    build = os.path.join(os.path.dirname(os.path.dirname(here)), "build")
    return os.path.join(build, name)


_LIB = None


def _lib():
    global _LIB
    if _LIB is None:
        path = _lib_path()
        if not os.path.exists(path):
            raise FileNotFoundError(f"shared library not built: {path} (cmake --build build)")
        lib = ctypes.CDLL(path)
        cvp = ctypes.c_void_p
        ci = ctypes.c_int
        lib.magcalc_disp_c.argtypes = [ci, ci, cvp, cvp, cvp]
        lib.magcalc_disp_c.restype = None
        lib.magcalc_sqw_c.argtypes = [ci, ci, ctypes.c_double,
                                      cvp, cvp, cvp, cvp, cvp, cvp, cvp, cvp, cvp]
        lib.magcalc_sqw_c.restype = None
        lib.magcalc_sqw_model_c.argtypes = [ci, ci, ci, ctypes.c_double,
                                            cvp, cvp, cvp, cvp, cvp, cvp, cvp, cvp, cvp]
        lib.magcalc_sqw_model_c.restype = None
        _LIB = lib
    return _LIB


def _ptr(a):
    return a.ctypes.data_as(ctypes.c_void_p)


def run_dispersion(h_plus, **_):
    """In-process dispersion. h_plus: (Nq, 2N, 2N) complex. Returns (energies
    (Nq, N), info (Nq,))."""
    h = np.ascontiguousarray(h_plus, dtype=np.complex128)
    nq, two_n, _ = h.shape
    n = two_n // 2
    hf = np.asfortranarray(np.moveaxis(h, 0, -1))           # (2N,2N,Nq)
    energies = np.empty((n, nq), dtype=np.float64, order="F")
    info = np.empty(nq, dtype=np.intc)
    _lib().magcalc_disp_c(n, nq, _ptr(hf), _ptr(energies), _ptr(info))
    return energies.T.copy(), info.copy()


def run_sqw(h_plus, h_minus, ud, ff, S, q_grid, **_):
    """In-process S(Q,w). Returns dict with energies, intensities (Nq, N),
    info (Nq,), eigvals (Nq, 2N). (K/Kd are gauge-dependent and not returned by
    this path; use _bin.run_sqw for them.)"""
    hp = np.asfortranarray(np.moveaxis(np.ascontiguousarray(h_plus, np.complex128), 0, -1))
    hm = np.asfortranarray(np.moveaxis(np.ascontiguousarray(h_minus, np.complex128), 0, -1))
    nq = hp.shape[2]
    n = hp.shape[0] // 2
    udf = np.asfortranarray(np.ascontiguousarray(ud, np.complex128))
    qf = np.asfortranarray(np.ascontiguousarray(q_grid, np.float64).T)   # (3,Nq)
    fff = np.asfortranarray(np.ascontiguousarray(ff, np.float64).T)      # (N,Nq)

    energies = np.empty((n, nq), np.float64, order="F")
    intensities = np.empty((n, nq), np.float64, order="F")
    info = np.empty(nq, np.intc)
    evals = np.empty((2 * n, nq), np.complex128, order="F")

    _lib().magcalc_sqw_c(n, nq, float(S), _ptr(qf), _ptr(hp), _ptr(hm), _ptr(udf),
                         _ptr(fff), _ptr(energies), _ptr(intensities), _ptr(info), _ptr(evals))

    return dict(energies=energies.T.copy(), intensities=intensities.T.copy(),
                info=info.copy(), eigvals=evals.T.copy())


def run_sqw_model(dvec, Mb, ud, ff, S, q_grid, **_):
    """Phase-2 in-process S(Q,w): Fortran builds H(q) from the bond model, so
    no per-q Hamiltonian stack is marshalled — only Mb (built once).

    Args:
        dvec : (nb, 3) float — bond vectors d_b.
        Mb   : (nb, 2N, 2N) complex — coefficient matrices (S/params baked in).
        ud   : (3N, 3N) complex.  ff: (Nq, N) float.  q_grid: (Nq, 3) float.
    Returns dict: energies, intensities (Nq, N), info (Nq,), eigvals (Nq, 2N).
    """
    Mbf = np.asfortranarray(np.moveaxis(np.ascontiguousarray(Mb, np.complex128), 0, -1))  # (2N,2N,nb)
    nb = Mbf.shape[2]
    n = Mbf.shape[0] // 2
    nq = np.asarray(q_grid).shape[0]
    dvf = np.asfortranarray(np.ascontiguousarray(dvec, np.float64).T)   # (3,nb)
    udf = np.asfortranarray(np.ascontiguousarray(ud, np.complex128))
    qf = np.asfortranarray(np.ascontiguousarray(q_grid, np.float64).T)  # (3,Nq)
    fff = np.asfortranarray(np.ascontiguousarray(ff, np.float64).T)     # (N,Nq)

    energies = np.empty((n, nq), np.float64, order="F")
    intensities = np.empty((n, nq), np.float64, order="F")
    info = np.empty(nq, np.intc)
    evals = np.empty((2 * n, nq), np.complex128, order="F")

    _lib().magcalc_sqw_model_c(n, nq, nb, float(S), _ptr(qf), _ptr(dvf), _ptr(Mbf),
                               _ptr(udf), _ptr(fff), _ptr(energies), _ptr(intensities),
                               _ptr(info), _ptr(evals))
    return dict(energies=energies.T.copy(), intensities=intensities.T.copy(),
                info=info.copy(), eigvals=evals.T.copy())
