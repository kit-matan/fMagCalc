"""Oracle bridge: drive pyMagCalc to produce (a) the Phase-1 inputs the Fortran
core needs and (b) pyMagCalc's own outputs, so fMagCalc can be validated for
numerical parity.

This module IMPORTS pyMagCalc read-only. It never modifies it. Make pyMagCalc
importable first, e.g.:

    pip install -e ../pyMagCalc          # or
    export PYTHONPATH=../pyMagCalc:$PYTHONPATH

Usage (sketch — fill in once the pyMagCalc entry point is wired):

    python export_model.py --config ../pyMagCalc/.config_gui_run.yaml \\
                           --out fixtures/cvo_disp.npz
"""
from __future__ import annotations

import argparse
import numpy as np


def export_phase1_dispersion(magcalc, q_grid: np.ndarray) -> dict:
    """Build the Phase-1 dispersion handoff from a pyMagCalc MagCalc instance.

    Returns a dict of arrays matching docs/INTERFACE.md:
      H_plus  (Nq, 2N, 2N) complex128
      energies(Nq, N)      float64   <- pyMagCalc's own result (the oracle)
    """
    # NOTE: this is a template. The exact pyMagCalc API for evaluating the
    # lambdified Hamiltonian over a q-grid lives in magcalc/numerical.py
    # (_init_worker / process_calc_disp) and magcalc/core.py (calculate_dispersion).
    # Wire these in at milestone M1.
    raise NotImplementedError("Wire to pyMagCalc.MagCalc at M1 — see PLAN.md §4")


def main() -> None:
    ap = argparse.ArgumentParser(description="Export pyMagCalc oracle fixtures for fMagCalc")
    ap.add_argument("--config", required=True, help="pyMagCalc YAML config")
    ap.add_argument("--out", required=True, help="output .npz path")
    args = ap.parse_args()
    print(f"[export_model] config={args.config} -> {args.out}")
    print("[export_model] STUB: implement at M1 (see PLAN.md).")


if __name__ == "__main__":
    main()
