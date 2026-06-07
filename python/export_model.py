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
    args = ap.parse_args()

    if not args.config and not args.model_dir:
        ap.error("provide either --config or --model-dir")

    import logging
    logging.disable(logging.WARNING)

    data = export_phase1_dispersion(args)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, **data)
    print(f"[export_model] n={data['n']} S={data['S']} Nq={len(data['q_grid'])} "
          f"H_absmax={np.abs(data['H_plus']).max():.3f} "
          f"E_range=[{np.nanmin(data['energies']):.3f},{np.nanmax(data['energies']):.3f}] "
          f"-> {args.out}")


# pyMagCalc path must be importable at module scope for spawn workers.
sys.path.insert(0, os.path.abspath(os.environ.get("FMAGCALC_PYMAGCALC", "../pyMagCalc")))

if __name__ == "__main__":
    main()
