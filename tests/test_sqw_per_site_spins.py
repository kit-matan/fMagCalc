"""K6 (2026-10-27): the S(Q,w) kernels take a per-site spin vector.

The intensity prefactor is sqrt(S_i/2) PER SITE, as in pyMagCalc's
linalg._spin_prefactor_vector; before K6 the kernel applied one global sqrt(S/2),
so pyMagCalc fell back to NumPy for every mixed-spin model. Pinned here:

  * a uniform vector reproduces the old scalar kernel BIT FOR BIT (run_sqw,
    run_sqw_model, powder_average);
  * the vector acts as an exact per-site row scale of K/Kd -- the same thing as
    folding sqrt(S_i/S) into the per-site form factor;
  * on a mixed-spin model the kernel equals pyMagCalc's NumPy path (the oracle),
    on the exact-H path and on the bond-model path;
  * the subprocess transport, whose file format carries one S, refuses a mixed
    vector instead of silently using S.
"""
import copy
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
PYMAGCALC = os.path.abspath(os.path.join(REPO, "..", "pyMagCalc"))
sys.path.insert(0, os.path.join(REPO, "python"))

FIX = os.path.join(REPO, "python", "fixtures")
LIB = os.path.join(REPO, "build", "libfmagcalc.dylib")

pytestmark = pytest.mark.skipif(not os.path.exists(LIB), reason="libfmagcalc not built")


def _load(name):
    return np.load(os.path.join(FIX, name))


def test_uniform_spin_vector_is_bit_identical_to_scalar():
    import fmagcalc

    d = _load("kfe3j_sqw.npz")
    n, S = int(d["n"]), float(d["S"])
    a = fmagcalc.run_sqw(d["H_plus"], d["H_minus"], d["Ud"], d["ff"], S, d["q_grid"])
    b = fmagcalc.run_sqw(d["H_plus"], d["H_minus"], d["Ud"], d["ff"], S, d["q_grid"],
                         spins=np.full(n, S))
    for k in ("energies", "intensities", "eigvals", "info"):
        assert np.array_equal(a[k], b[k], equal_nan=True), k

    m = _load("kfe3j_model.npz")
    n, S = int(m["n"]), float(m["S"])
    a = fmagcalc.run_sqw_model(m["dvec"], m["Mb"], m["Ud"], m["ff"], S, m["q_grid"])
    b = fmagcalc.run_sqw_model(m["dvec"], m["Mb"], m["Ud"], m["ff"], S, m["q_grid"],
                               spins=[S] * n)
    for k in ("energies", "intensities", "eigvals", "info"):
        assert np.array_equal(a[k], b[k], equal_nan=True), k

    p = _load("kfe3j_powder.npz")
    S = float(p["S"])
    kw = dict(num_samples=8)
    a = fmagcalc.powder_average(p["dvec"], p["Mb"], p["Ud"], S, p["q_magnitudes"][:6], **kw)
    b = fmagcalc.powder_average(p["dvec"], p["Mb"], p["Ud"], S, p["q_magnitudes"][:6],
                                spins=np.full(int(p["n"]), S), **kw)
    for k in ("energies", "intensities"):
        assert np.array_equal(a[k], b[k], equal_nan=True), k


def test_per_site_prefactor_is_a_row_scale():
    """sqrt(S_i/2) scales site i's rows of K and Kd, and the contraction already
    weights those rows by ff_i -- so spins=v must equal the scalar kernel run with
    ff_i * sqrt(v_i/S). Exact up to rounding (the 1e-6 zeroing of tiny K entries
    happens on the scaled K in the first and the unscaled in the second)."""
    import fmagcalc

    d = _load("kfe3j_sqw.npz")
    n, S = int(d["n"]), float(d["S"])
    v = S * np.array([1.0, 2.5, 0.7])
    assert v.shape == (n,)
    a = fmagcalc.run_sqw(d["H_plus"], d["H_minus"], d["Ud"], d["ff"], S, d["q_grid"], spins=v)
    b = fmagcalc.run_sqw(d["H_plus"], d["H_minus"], d["Ud"],
                         d["ff"] * np.sqrt(v / S)[None, :], S, d["q_grid"])
    c = fmagcalc.run_sqw(d["H_plus"], d["H_minus"], d["Ud"], d["ff"], S, d["q_grid"])
    assert np.array_equal(a["energies"], c["energies"])     # energies never see it
    np.testing.assert_allclose(a["intensities"], b["intensities"], rtol=1e-12, atol=1e-13)
    assert np.max(np.abs(a["intensities"] - c["intensities"])) > 1e-2   # and it matters


def test_bad_spin_vectors_raise():
    import fmagcalc

    d = _load("kfe3j_sqw.npz")
    args = (d["H_plus"], d["H_minus"], d["Ud"], d["ff"], float(d["S"]), d["q_grid"])
    with pytest.raises(ValueError, match="entries for 3 sites"):
        fmagcalc.run_sqw(*args, spins=[1.0, 2.0])
    with pytest.raises(ValueError, match="positive"):
        fmagcalc.run_sqw(*args, spins=[1.0, 0.0, 2.0])


def test_subprocess_transport_refuses_mixed_spin():
    from fmagcalc import _bin

    d = _load("kfe3j_sqw.npz")
    with pytest.raises(NotImplementedError, match="ctypes path"):
        _bin.run_sqw(d["H_plus"], d["H_minus"], d["Ud"], d["ff"], 1.0, d["q_grid"],
                     spins=[1.0, 2.0, 1.0])


# --- mixed spin against pyMagCalc's NumPy path (the oracle) -------------------

def _pymagcalc_ferri():
    """Alternating S = (1/2, 2) antiferromagnetic chain with form factors and an
    easy-axis anisotropy (gapped, so no Goldstone mode: fMagCalc zeroes |E| < 1e-3
    intensities by design and NumPy does not)."""
    if PYMAGCALC not in sys.path:
        sys.path.insert(0, PYMAGCALC)
    try:
        import magcalc as mc
        from magcalc.generic_model import GenericSpinModel
    except Exception as e:  # pragma: no cover
        pytest.skip(f"pyMagCalc not importable: {e}")
    lat = [[6.0, 0, 0], [0, 9.0, 0], [0, 0, 9.0]]
    cfg = {
        "crystal_structure": {"lattice_vectors": lat, "atoms_uc": [
            {"label": "A", "pos": [0.0, 0.0, 0.0], "spin_S": 0.5, "ion": "Cu2+"},
            {"label": "B", "pos": [0.5, 0.0, 0.0], "spin_S": 2.0, "ion": "Fe2+"}]},
        "interactions": {
            "symmetry_rules": [{"type": "heisenberg", "distance": 3.0, "value": "J"}],
            "single_ion_anisotropy": [{"value": -0.4, "axis": [0, 0, 1], "atoms": ["A", "B"]}]},
        "parameters": {"J": 1.3}, "parameter_order": ["J"],
        "magnetic_structure": {"type": "pattern", "pattern_type": "generic",
                               "directions": [[0, 0, 1], [0, 0, -1]]},
    }
    m = GenericSpinModel(copy.deepcopy(cfg))
    th, ph = m.generate_magnetic_structure()
    m.set_magnetic_structure(th, ph)
    calc = mc.MagCalc(spin_model_module=m, spin_magnitude=0.5, cache_mode="none",
                      cache_file_base="fmc_ferri", hamiltonian_params=[1.3])
    B = 2 * np.pi * np.linalg.inv(np.array(lat, float)).T
    h = np.linspace(0.02, 1.4, 29)
    qs = np.stack([h, 0.15 * h, 0.1 * np.sin(3 * h)], 1) @ B
    return calc, m, qs


def test_mixed_spin_matches_pymagcalc():
    import fmagcalc
    from magcalc.form_factors import get_form_factor

    calc, m, qs = _pymagcalc_ferri()
    oracle = calc.calculate_sqw(qs, backend="numpy")
    spins = np.array(m.spin_magnitudes(), float)
    ions = m.ion_list()
    ff = np.array([[get_form_factor(ions[i], float(np.linalg.norm(q))) for i in range(2)]
                   for q in qs])
    S = float(calc.spin_magnitude)
    Ud = np.asarray(calc.Ud_numeric, np.complex128)

    # The two bands never cross (their splitting is 2 J |S_A - S_B| + anisotropy),
    # so per-band comparison is well defined.
    r = fmagcalc.run_sqw(calc._build_h_stack(qs, S), calc._build_h_stack(qs, S, negate=True),
                         Ud, ff, S, qs, spins=spins)
    assert np.all(r["info"] == 0)
    np.testing.assert_allclose(r["energies"], oracle.energies, rtol=0, atol=1e-10)
    np.testing.assert_allclose(r["intensities"], oracle.intensities, rtol=0, atol=1e-10)
    # ...and the global-S kernel is what pyMagCalc used to refuse: badly wrong here.
    r_old = fmagcalc.run_sqw(calc._build_h_stack(qs, S), calc._build_h_stack(qs, S, negate=True),
                             Ud, ff, S, qs)
    assert np.max(np.abs(r_old["intensities"] - oracle.intensities)) > 0.1

    # Bond-model path (H(q) rebuilt in Fortran; ~1e-10 reconstruction floor, as in
    # test_sqw_model_parity.py).
    n, S_b, Ud_b, dvec, Mb = fmagcalc.extract_bond_model(calc)
    r = fmagcalc.run_sqw_model(dvec, Mb, Ud_b, ff, S_b, qs, spins=spins)
    assert np.all(r["info"] == 0)
    assert np.max(np.abs(r["energies"] - oracle.energies)) < 1e-8
    assert np.max(np.abs(r["intensities"] - oracle.intensities)) < 1e-7
