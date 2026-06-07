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


def _default_exe() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(os.path.dirname(here))
    return os.path.join(repo, "build", "fmagcalc_disp")


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
