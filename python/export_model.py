"""Oracle bridge: drive pyMagCalc to produce (a) the Phase-1 inputs the Fortran
core needs and (b) pyMagCalc's own outputs, so fMagCalc can be validated for
numerical parity.

This module IMPORTS pyMagCalc read-only. It never modifies it.

Two model sources:
  * --model-dir/--model-module : the proven Python spin-model path (e.g. the
    KFe3J example). Robust, fast for small models. This is the default for M1.
  * --config                   : the YAML config-driven path.

Writes an .npz with: n, S, q_grid (Nq,3), H_plus (Nq,2N,2N) complex128,
energies (Nq,N) float64  [= pyMagCalc's own result, the oracle].

NOTE: top-level path setup + imports live at module scope so that pyMagCalc's
multiprocessing(spawn) workers can re-import cleanly; all real work runs under
the __main__ guard.

Example:
    python python/export_model.py \\
        --pymagcalc ../pyMagCalc \\
        --model-dir ../pyMagCalc/examples/KFe3J --model-module spin_model \\
        --params '[3.23, 0.11, 0.218, -0.195, [0,0,1], 0.0, 0.0]' --S 2.5 \\
        --out python/fixtures/kfe3j_disp.npz --nq 48
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys

import numpy as np


def build_q_grid(nq: int, seed: int) -> np.ndarray:
    """Generic (low-symmetry) q-points keep modes non-degenerate — the cleanest
    first parity check — plus a couple of explicit points."""
    rng = np.random.default_rng(seed)
    q = rng.uniform(0.0, 1.0, size=(nq, 3))
    q[:, 2] = 0.0          # KFe3J: kagome plane
    q[0] = [0.1, 0.0, 0.0]
    return q


def _make_calc(args):
    import magcalc as mc
    if args.model_dir:
        sys.path.insert(0, os.path.abspath(args.model_dir))
        model = importlib.import_module(args.model_module)
        params = json.loads(args.params)
        return mc.MagCalc(
            spin_magnitude=args.S,
            hamiltonian_params=params,
            cache_file_base=args.cache_base,
            cache_mode="w",
            spin_model_module=model,
        )
    return mc.MagCalc(config_filepath=os.path.abspath(args.config), cache_mode="auto")


def export_phase1_dispersion(args) -> dict:
    import sympy as sp
    from sympy import lambdify

    calc = _make_calc(args)
    n = int(calc.nspins)
    S = float(calc.spin_magnitude)

    lam = [s for s in calc.full_symbol_list if isinstance(s, sp.Symbol)]
    hfunc = lambdify(lam, calc.HMat_sym, modules=["numpy"], cse=True)

    q_grid = build_q_grid(args.nq, args.seed)
    h_plus = np.empty((args.nq, 2 * n, 2 * n), dtype=np.complex128)
    base = [S] + list(calc.hamiltonian_params)
    for i, q in enumerate(q_grid):
        h_plus[i] = np.asarray(hfunc(*(list(q) + base)), dtype=np.complex128)

    if np.abs(h_plus).max() == 0.0:
        raise RuntimeError("Exported Hamiltonian is identically zero — check the "
                           "model/params/cache; fixture would be trivial.")

    res = calc.calculate_dispersion(q_grid, serial=True)
    energies = np.asarray(res.energies, dtype=np.float64)
    return dict(n=n, S=S, q_grid=q_grid, H_plus=h_plus, energies=energies)


def extract_bond_model(calc):
    """Phase-2 model export: decompose the (numeric) dynamical matrix into
    H(q) = sum_b M_b * exp(i q.d_b), so a backend can build H(q) itself.

    S and the Hamiltonian parameters are baked into the M_b (the q-loop is then
    parameter-independent). Returns (n, S, Ud, dvec (3,nb), Mb (2n,2n,nb)).
    """
    import sympy as sp
    from sympy import exp, I, Add, Mul

    lam = [s for s in calc.full_symbol_list if isinstance(s, sp.Symbol)]
    kx, ky, kz = lam[0], lam[1], lam[2]
    subs = {lam[3]: calc.spin_magnitude}
    for i, v in enumerate(calc.hamiltonian_params):
        subs[lam[4 + i]] = v

    n = int(calc.nspins)
    n2 = 2 * n
    Hn = calc.HMat_sym.subs(subs).applyfunc(
        lambda e: sp.expand(sp.expand(e).rewrite(exp)))

    bonds: dict = {}
    for i in range(n2):
        for j in range(n2):
            for term in Add.make_args(sp.expand(Hn[i, j])):
                coeff = sp.Integer(1)
                arg = sp.Integer(0)
                for f in Mul.make_args(term):
                    if f.func == exp:
                        arg += f.args[0]
                    else:
                        coeff *= f
                # expand so coeff() distributes over compound args (kx & ky).
                phase = sp.expand(arg / I)  # kx*dx + ky*dy + kz*dz
                d = tuple(round(float(complex(phase.coeff(v)).real), 10)
                          for v in (kx, ky, kz))
                M = bonds.setdefault(d, np.zeros((n2, n2), complex))
                M[i, j] += complex(coeff)

    d_keys = sorted(bonds.keys())
    nb = len(d_keys)
    dvec = np.array(d_keys, dtype=np.float64)                 # (nb, 3)
    Mb = np.stack([bonds[d] for d in d_keys], axis=0)         # (nb, 2n, 2n)
    return n, float(calc.spin_magnitude), np.asarray(calc.Ud_numeric, np.complex128), dvec, Mb


def export_bond_model(args) -> dict:
    """Bond model + a q-grid + pyMagCalc oracle (energies, intensities), for the
    Phase-2 model-path parity test and benchmark."""
    from magcalc.numerical import process_calc_Sqw, _init_worker
    from magcalc.form_factors import get_form_factor
    import sympy as sp

    calc = _make_calc(args)
    n, S, Ud, dvec, Mb = extract_bond_model(calc)
    ion_list = calc.sm.ion_list() if hasattr(calc.sm, "ion_list") else None
    lam = [s for s in calc.full_symbol_list if isinstance(s, sp.Symbol)]
    _init_worker(calc.HMat_sym, lam)

    q_grid = build_q_grid(args.nq, args.seed)
    ff = np.ones((args.nq, n))
    energies = np.full((args.nq, n), np.nan)
    intensities = np.full((args.nq, n), np.nan)
    for i, q in enumerate(q_grid):
        if ion_list:
            qmag = float(np.linalg.norm(q))
            ff[i] = [get_form_factor(ion_list[j], qmag) for j in range(n)]
        _, en, inten = process_calc_Sqw((Ud, np.asarray(q, float), n, S,
                                         list(calc.hamiltonian_params), ion_list))
        energies[i], intensities[i] = en, inten

    return dict(n=n, S=S, Ud=Ud, dvec=dvec, Mb=Mb, q_grid=q_grid, ff=ff,
                energies=energies, intensities=intensities)


def export_phase1_sqw(args) -> dict:
    """S(Q,w) fixture: inputs (H_plus, H_minus, Ud, ff), pyMagCalc's oracle
    outputs (energies, intensities), AND the KKdMatrix intermediates
    (K, Kd, eigvals) so the Fortran Bogoliubov port can be validated step by
    step rather than only on the final intensity."""
    import sympy as sp
    from sympy import lambdify
    from magcalc.linalg import KKdMatrix
    from magcalc.numerical import process_calc_Sqw, _init_worker
    from magcalc.form_factors import get_form_factor

    calc = _make_calc(args)
    n = int(calc.nspins)
    S = float(calc.spin_magnitude)
    Ud = np.asarray(calc.Ud_numeric, dtype=np.complex128)

    ion_list = calc.sm.ion_list() if hasattr(calc.sm, "ion_list") else None

    lam = [s for s in calc.full_symbol_list if isinstance(s, sp.Symbol)]
    hfunc = lambdify(lam, calc.HMat_sym, modules=["numpy"], cse=True)
    # process_calc_Sqw uses the module-global lambdified function.
    _init_worker(calc.HMat_sym, lam)

    q_grid = build_q_grid(args.nq, args.seed)
    nq = args.nq
    base = [S] + list(calc.hamiltonian_params)

    h_plus = np.empty((nq, 2 * n, 2 * n), dtype=np.complex128)
    h_minus = np.empty_like(h_plus)
    ff = np.ones((nq, n), dtype=np.float64)
    K = np.empty((nq, 3 * n, 2 * n), dtype=np.complex128)
    Kd = np.empty_like(K)
    eigvals = np.empty((nq, 2 * n), dtype=np.complex128)
    energies = np.full((nq, n), np.nan)
    intensities = np.full((nq, n), np.nan)

    for i, q in enumerate(q_grid):
        h_plus[i] = np.asarray(hfunc(*(list(q) + base)), dtype=np.complex128)
        h_minus[i] = np.asarray(hfunc(*(list(-q) + base)), dtype=np.complex128)
        if ion_list:
            qmag = float(np.linalg.norm(q))
            ff[i] = [get_form_factor(ion_list[j], qmag) for j in range(n)]

        Kq, Kdq, ev = KKdMatrix(S, h_plus[i], h_minus[i], Ud, np.asarray(q, float), n)
        K[i], Kd[i], eigvals[i] = Kq, Kdq, ev

        _, en, inten = process_calc_Sqw((Ud, np.asarray(q, float), n, S, list(calc.hamiltonian_params), ion_list))
        energies[i], intensities[i] = en, inten

    if np.abs(h_plus).max() == 0.0:
        raise RuntimeError("Exported Hamiltonian is identically zero — trivial fixture.")

    return dict(n=n, S=S, q_grid=q_grid, H_plus=h_plus, H_minus=h_minus, Ud=Ud,
                ff=ff, energies=energies, intensities=intensities,
                K=K, Kd=Kd, eigvals=eigvals)


def main() -> None:
    ap = argparse.ArgumentParser(description="Export pyMagCalc oracle fixtures for fMagCalc")
    ap.add_argument("--pymagcalc", default="../pyMagCalc")
    ap.add_argument("--config")
    ap.add_argument("--model-dir")
    ap.add_argument("--model-module", default="spin_model")
    ap.add_argument("--params", default="[3.23, 0.11, 0.218, -0.195, [0,0,1], 0.0, 0.0]")
    ap.add_argument("--S", type=float, default=2.5)
    ap.add_argument("--cache-base", default="KFe3J_fmc")
    ap.add_argument("--out", required=True)
    ap.add_argument("--nq", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--task", choices=["disp", "sqw", "model"], default="disp")
    args = ap.parse_args()

    if not args.config and not args.model_dir:
        ap.error("provide either --config or --model-dir")

    import logging
    logging.disable(logging.WARNING)

    builder = {"disp": export_phase1_dispersion, "sqw": export_phase1_sqw,
               "model": export_bond_model}[args.task]
    data = builder(args)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, **data)
    if args.task == "model":
        print(f"[export_model] task=model n={data['n']} S={data['S']} Nq={len(data['q_grid'])} "
              f"n_bonds={data['dvec'].shape[0]} "
              f"E_range=[{np.nanmin(data['energies']):.3f},{np.nanmax(data['energies']):.3f}] "
              f"I_range=[{np.nanmin(data['intensities']):.3f},{np.nanmax(data['intensities']):.3f}] "
              f"-> {args.out}")
    else:
        extra = ""
        if args.task == "sqw":
            extra = f" I_range=[{np.nanmin(data['intensities']):.3f},{np.nanmax(data['intensities']):.3f}]"
        print(f"[export_model] task={args.task} n={data['n']} S={data['S']} Nq={len(data['q_grid'])} "
              f"H_absmax={np.abs(data['H_plus']).max():.3f} "
              f"E_range=[{np.nanmin(data['energies']):.3f},{np.nanmax(data['energies']):.3f}]{extra} "
              f"-> {args.out}")


# pyMagCalc path must be importable at module scope for spawn workers.
sys.path.insert(0, os.path.abspath(os.environ.get("FMAGCALC_PYMAGCALC", "../pyMagCalc")))

if __name__ == "__main__":
    main()
