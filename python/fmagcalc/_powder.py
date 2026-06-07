"""Powder averaging (M6).

S(|q|, w) is S(q, w) averaged over a sphere of radius |q|. The heavy per-q work
runs in the Fortran model kernel (run_sqw_model); this module only generates the
sampling directions and averages — both cheap. The q-sampling reproduces
pyMagCalc.calculate_powder_average exactly (Fibonacci sphere) so results match
sample-for-sample.
"""
from __future__ import annotations

import numpy as np

Q_ZERO_THRESHOLD = 1e-10


def generate_powder_qvectors(q_magnitudes, num_samples):
    """Fibonacci-sphere q-vectors for each |q|. Returns (all_q (Npts,3),
    segment_sizes (len q_magnitudes)). Must match pyMagCalc exactly."""
    all_q = []
    segs = []
    for q_mag in q_magnitudes:
        if q_mag < Q_ZERO_THRESHOLD:
            vectors = np.array([[0.0, 0.0, 0.0]])
        else:
            idx = np.arange(0, num_samples, dtype=float) + 0.5
            phi = np.arccos(1 - 2 * idx / num_samples)
            theta = np.pi * (1 + 5 ** 0.5) * idx
            qx = q_mag * np.sin(phi) * np.cos(theta)
            qy = q_mag * np.sin(phi) * np.sin(theta)
            qz = q_mag * np.cos(phi)
            vectors = np.column_stack((qx, qy, qz))
        all_q.extend(vectors)
        segs.append(len(vectors))
    return np.asarray(all_q, dtype=np.float64), segs


def _average_segments(per_point, segment_sizes, n):
    out = []
    i = 0
    for c in segment_sizes:
        chunk = per_point[i:i + c]
        out.append(np.nanmean(chunk, axis=0) if chunk.size else np.full(n, np.nan))
        i += c
    return np.asarray(out)


def powder_average(dvec, Mb, ud, S, q_magnitudes, num_samples=100, ff_func=None):
    """Powder-averaged S(|q|, w) via the Fortran model path.

    Args:
        dvec, Mb, ud, S : bond model (see run_sqw_model).
        q_magnitudes    : 1D array of |q| values.
        num_samples     : sphere samples per |q|.
        ff_func         : optional callable |q| -> (N,) form factors. Default: 1.

    Returns dict: q_magnitudes, energies (Nmag, N), intensities (Nmag, N).
    """
    from ._capi import run_sqw_model

    q_mags = np.asarray(q_magnitudes, dtype=np.float64)
    n = Mb.shape[1] // 2
    all_q, segs = generate_powder_qvectors(q_mags, num_samples)

    if ff_func is None:
        ff = np.ones((all_q.shape[0], n))
    else:
        ff = np.array([ff_func(float(np.linalg.norm(q))) for q in all_q], dtype=np.float64)

    r = run_sqw_model(dvec, Mb, ud, ff, S, all_q)
    return dict(
        q_magnitudes=q_mags,
        energies=_average_segments(r["energies"], segs, n),
        intensities=_average_segments(r["intensities"], segs, n),
    )
