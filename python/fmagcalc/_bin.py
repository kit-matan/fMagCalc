"""Subprocess transport for the Phase-1 binary-file driver (fmagcalc_disp).

This is the M1 mechanism: marshal the (Nq, 2N, 2N) Hamiltonian stack to the
Fortran executable via a raw little-endian stream file, run it, read results
back. The f2py in-process path (M4) will later replace this with zero file I/O,
behind the same Python signature.

Byte layout must match src/magcalc_io.f90 (see docs/INTERFACE.md).
"""
from __future__ import annotations

import os
import struct
import subprocess
import tempfile

import numpy as np


def _build_dir() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(os.path.dirname(here)), "build")


def _default_exe() -> str:
    return os.path.join(_build_dir(), "fmagcalc_disp")


def run_dispersion(h_plus: np.ndarray, exe: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Run the Fortran dispersion driver on a stack of Hamiltonians.

    Args:
        h_plus: complex array, shape (Nq, 2N, 2N) — dynamical matrix per q.
        exe: path to fmagcalc_disp (defaults to ../../build/fmagcalc_disp).

    Returns:
        energies: float64 (Nq, N) — magnon branch (upper-half eigenvalues).
        info:     int32  (Nq,)   — 0 ok, nonzero = zgeev failure at that q.
    """
    exe = exe or _default_exe()
    if not os.path.exists(exe):
        raise FileNotFoundError(f"fmagcalc_disp not built: {exe} (run cmake --build build)")

    h = np.ascontiguousarray(h_plus, dtype=np.complex128)
    nq, two_n, two_n2 = h.shape
    if two_n != two_n2 or two_n % 2 != 0:
        raise ValueError(f"each H slab must be square 2N x 2N, got {h.shape}")
    n = two_n // 2

    # Fortran wants H(2N, 2N, Nq) in column-major. Our array is (Nq, 2N, 2N)
    # C-order; moving the q axis last and writing Fortran-order bytes yields the
    # exact element sequence Fortran reads (first index fastest).
    h_fort = np.moveaxis(h, 0, -1)  # (2N, 2N, Nq)

    with tempfile.TemporaryDirectory() as td:
        in_path = os.path.join(td, "disp_in.bin")
        out_path = os.path.join(td, "disp_out.bin")
        with open(in_path, "wb") as f:
            f.write(struct.pack("<i", n))
            f.write(struct.pack("<i", nq))
            f.write(np.asfortranarray(h_fort).tobytes(order="F"))

        subprocess.run([exe, in_path, out_path], check=True, capture_output=True, text=True)

        with open(out_path, "rb") as f:
            out_nq = struct.unpack("<i", f.read(4))[0]
            out_n = struct.unpack("<i", f.read(4))[0]
            energies = np.frombuffer(f.read(8 * out_n * out_nq), dtype="<f8")
            # Fortran wrote energies(n, nq) column-major => flat is mode-fastest
            # then q, i.e. exactly C-order (nq, n).
            energies = energies.reshape(out_nq, out_n, order="C").copy()
            info = np.frombuffer(f.read(4 * out_nq), dtype="<i4").copy()

    return energies, info


def run_sqw(h_plus, h_minus, ud, ff, S, q_grid, exe: str | None = None):
    """Run the Fortran S(Q,w) driver on a q-grid.

    Args mirror docs/INTERFACE.md (all NumPy, q-first stacking):
        h_plus, h_minus : complex (Nq, 2N, 2N)
        ud              : complex (3N, 3N)
        ff              : float   (Nq, N)
        S               : float
        q_grid          : float   (Nq, 3)

    Returns dict with energies (Nq,N), intensities (Nq,N), info (Nq,) and the
    intermediates K, Kd (Nq, 3N, 2N) and eigvals (Nq, 2N).
    """
    exe = exe or os.path.join(_build_dir(), "fmagcalc_sqw")
    if not os.path.exists(exe):
        raise FileNotFoundError(f"fmagcalc_sqw not built: {exe}")

    hp = np.ascontiguousarray(h_plus, dtype=np.complex128)
    hm = np.ascontiguousarray(h_minus, dtype=np.complex128)
    nq, two_n, _ = hp.shape
    n = two_n // 2

    def fbytes(a):  # logical Fortran-order bytes
        return np.asfortranarray(a).tobytes(order="F")

    with tempfile.TemporaryDirectory() as td:
        in_path = os.path.join(td, "sqw_in.bin")
        out_path = os.path.join(td, "sqw_out.bin")
        with open(in_path, "wb") as f:
            f.write(struct.pack("<i", n))
            f.write(struct.pack("<i", nq))
            f.write(struct.pack("<d", float(S)))
            f.write(fbytes(np.ascontiguousarray(ud, dtype=np.complex128)))      # (3N,3N)
            f.write(fbytes(np.moveaxis(hp, 0, -1)))                              # (2N,2N,Nq)
            f.write(fbytes(np.moveaxis(hm, 0, -1)))                             # (2N,2N,Nq)
            f.write(fbytes(np.ascontiguousarray(q_grid, dtype=np.float64).T))   # (3,Nq)
            f.write(fbytes(np.ascontiguousarray(ff, dtype=np.float64).T))       # (N,Nq)

        proc = subprocess.run([exe, in_path, out_path], check=True, capture_output=True, text=True)
        compute_seconds = float("nan")
        for tok in proc.stdout.split():
            if tok.startswith("compute_seconds="):
                compute_seconds = float(tok.split("=", 1)[1])

        with open(out_path, "rb") as f:
            out_nq = struct.unpack("<i", f.read(4))[0]
            out_n = struct.unpack("<i", f.read(4))[0]
            ne = out_n * out_nq
            energies = np.frombuffer(f.read(8 * ne), dtype="<f8").reshape(out_nq, out_n, order="C").copy()
            intensities = np.frombuffer(f.read(8 * ne), dtype="<f8").reshape(out_nq, out_n, order="C").copy()
            info = np.frombuffer(f.read(4 * out_nq), dtype="<i4").copy()
            nK = 3 * out_n * 2 * out_n * out_nq
            K = np.frombuffer(f.read(16 * nK), dtype="<c16").reshape(3 * out_n, 2 * out_n, out_nq, order="F")
            K = np.moveaxis(K, 2, 0).copy()
            Kd = np.frombuffer(f.read(16 * nK), dtype="<c16").reshape(3 * out_n, 2 * out_n, out_nq, order="F")
            Kd = np.moveaxis(Kd, 2, 0).copy()
            nev = 2 * out_n * out_nq
            eigvals = np.frombuffer(f.read(16 * nev), dtype="<c16").reshape(2 * out_n, out_nq, order="F").T.copy()

    return dict(energies=energies, intensities=intensities, info=info, K=K, Kd=Kd,
                eigvals=eigvals, compute_seconds=compute_seconds)
