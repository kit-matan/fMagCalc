"""Degenerate magnons and near-zero energies on the exact-H path (2026-10-28).

Two defects, both found against exact answers and both fixed in `run_sqw`:

* `match_reorder` paired each +q eigenvector with a single best-overlapping -q column.
  Inside a degenerate pair (every q of a collinear antiferromagnet) no such column need
  exist, so the mode went unmatched and lost all its intensity. It now projects onto
  the degenerate subspace, as pyMagCalc's `linalg._match_and_reorder_minus_q` does.
* `sqw_intensity` zeroed every mode below 1e-3 meV. On the exact-H path that dropped
  well-conditioned modes; the cut now applies to `run_sqw_model` only.

pyMagCalc is the read-only oracle for the first (degenerate-group totals equal its
NumPy path, which pyMagCalc's tests/test_gapless_backends.py pins to Sunny 0.8.1 at
these same points), and the ferromagnetic chain's closed form for the second.
"""
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
PYMAGCALC = os.path.abspath(os.path.join(REPO, "..", "pyMagCalc"))
SUNNY_POINTS = os.path.join(PYMAGCALC, "tests", "data_afm_chain_sunny.json")
sys.path.insert(0, os.path.join(REPO, "python"))

mc = pytest.importorskip("magcalc", reason="pyMagCalc not installed (the oracle)")
fmagcalc = pytest.importorskip("fmagcalc")
pytestmark = pytest.mark.skipif(getattr(fmagcalc, "backend", None) != "ctypes",
                                reason="fMagCalc compiled library unavailable")

A = 3.0


def _calc(atoms, a, magstruct, J):
    import logging

    from magcalc.generic_model import GenericSpinModel

    logging.disable(logging.CRITICAL)
    cfg = {"crystal_structure": {"lattice_vectors": [[a, 0, 0], [0, 10, 0], [0, 0, 10]],
                                 "atoms_uc": atoms},
           "interactions": {"symmetry_rules": [
               {"type": "heisenberg", "distance": A, "value": J}]},
           "parameters": {}, "parameter_order": [], "magnetic_structure": magstruct}
    m = GenericSpinModel(cfg)
    th, ph = m.generate_magnetic_structure()
    m.set_magnetic_structure(th, ph)
    return mc.MagCalc(spin_model_module=m, spin_magnitude=1.0, cache_mode="none",
                      cache_file_base="fdeg", hamiltonian_params=[])


@pytest.mark.skipif(not os.path.exists(SUNNY_POINTS), reason="pyMagCalc fixture not found")
def test_degenerate_pair_totals_equal_the_numpy_oracle():
    calc = _calc([{"label": "A", "pos": [0, 0, 0], "spin_S": 1.0},
                  {"label": "B", "pos": [0.5, 0, 0], "spin_S": 1.0}], 2 * A,
                 {"type": "pattern", "pattern_type": "generic",
                  "directions": [[0, 0, 1], [0, 0, -1]]}, 1.0)
    with open(SUNNY_POINTS) as f:
        qs = [np.array(q) for q in json.load(f)["q_cart"]]
    rn = calc.calculate_sqw(qs, backend="numpy")
    rf = calc.calculate_sqw(qs, backend="fortran")
    tn = np.real(rn.intensities).sum(axis=1)
    tf = np.real(rf.intensities).sum(axis=1)
    assert np.sum(tf == 0) == 0, "degenerate modes lost all intensity"
    np.testing.assert_allclose(tf, tn, rtol=1e-8, atol=1e-10)


def test_small_but_nonzero_energies_keep_their_closed_form_intensity():
    calc = _calc([{"label": "A", "pos": [0, 0, 0], "spin_S": 1.0}], A,
                 {"type": "pattern", "pattern_type": "ferromagnetic", "direction": [0, 0, 1]},
                 -1.0)
    hs = [1e-6, 1e-4, 3e-3, 5e-3]                      # 4e-11 .. 9.9e-4 meV
    qs = [np.array([2 * np.pi * h / A, 0.4, 0.3]) for h in hs]
    r = calc.calculate_sqw(qs, backend="fortran")
    for q, I in zip(qs, r.intensities):
        qz = q[2] / np.linalg.norm(q)
        assert np.real(I[0]) == pytest.approx(0.5 * (1 + qz ** 2), rel=1e-9)


# --------------------------------------------------------------------------
# Accidental degeneracy: the block must be orthogonal in the bosonic metric
# --------------------------------------------------------------------------
@pytest.mark.parametrize("theta,phi", [(0.0, 0.0), (0.4, 0.0), (0.7, 0.0), (0.7, 0.9),
                                       (1.1, 0.3)])
def test_accidentally_degenerate_pair_total_equals_colpa(theta, phi):
    """Two modes at one energy with DIFFERENT u/v ratios, mixed by a site rotation.

    `apply_gs` orthonormalized a degenerate block with a Euclidean QR, which is
    orthogonal in the metric g = diag(1, -1) only for a symmetry-related pair (one
    u/v ratio) -- every degenerate case above. Here it is not, and the pair's total
    was 2-12 % off pyMagCalc's Colpa path (paraunitary by construction) until
    `g_orthogonalize` (2026-11-21), the port of pyMagCalc's
    `linalg._orthonormalize_degenerate_block`. theta = 0 is the unmixed control."""
    from magcalc.numerical import colpa_amplitudes_from_h

    A_ = np.diag([2.5, np.sqrt(13.0)])
    B_ = np.diag([1.5, 3.0])
    c, s = np.cos(theta), np.sin(theta)
    U = np.array([[c, -s * np.exp(1j * phi)], [s * np.exp(-1j * phi), c]])
    A2 = U.conj().T @ A_ @ U
    B2 = U.conj().T @ B_ @ U.conj()
    H2 = np.block([[A2, B2], [B2.conj().T, A2.conj()]])
    g = np.diag([1, 1, -1, -1.0])
    P = np.block([[np.zeros((2, 2)), np.eye(2)], [np.eye(2), np.zeros((2, 2))]])
    Hp, Hm = g @ H2, g @ (P @ H2.conj() @ P)
    ud = np.linalg.qr(np.random.default_rng(0).normal(size=(6, 6)))[0].astype(complex)
    q = np.array([[0.3, 0.1, 0.2]])

    out = fmagcalc.run_sqw(Hp[None], Hm[None], ud, np.ones((1, 2)), 0.5, q)
    assert np.allclose(out["energies"][0], 2.0, atol=1e-9)
    amps = colpa_amplitudes_from_h(Hp[None], q, ud, np.full(2, 0.5))
    ref = float(np.sum(amps.contract(q, "perp")[0]))
    assert float(out["intensities"][0].sum()) == pytest.approx(ref, rel=1e-10)
