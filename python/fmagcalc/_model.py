"""Bond-decomposition extraction: turn a pyMagCalc model's symbolic dynamical
matrix into H(q) = sum_b M_b exp(i q.d_b), so the Fortran model kernel can
build H(q) itself. Used both by the exporter and by pyMagCalc's opt-in
backend (M7).

Takes any object exposing the pyMagCalc MagCalc attributes:
full_symbol_list, HMat_sym, spin_magnitude, hamiltonian_params, nspins,
Ud_numeric. sympy is imported lazily so the rest of the package stays light.
"""
from __future__ import annotations

import numpy as np


def extract_bond_model(calc):
    """Returns (n, S, Ud (3n,3n), dvec (nb,3), Mb (nb,2n,2n)). S and the
    Hamiltonian parameters are baked into M_b, so the q-loop is
    parameter-independent."""
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
                phase = sp.expand(arg / I)
                d = tuple(round(float(complex(phase.coeff(v)).real), 10)
                          for v in (kx, ky, kz))
                M = bonds.setdefault(d, np.zeros((n2, n2), complex))
                M[i, j] += complex(coeff)

    d_keys = sorted(bonds.keys())
    dvec = np.array(d_keys, dtype=np.float64)
    Mb = np.stack([bonds[d] for d in d_keys], axis=0)
    return n, float(calc.spin_magnitude), np.asarray(calc.Ud_numeric, np.complex128), dvec, Mb
