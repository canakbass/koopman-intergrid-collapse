"""Mode-budget test for the quasi-periodic system (data_qp.py).

The observable of a frequency-modulated rotation has a pure point spectrum at the combination
frequencies m*omega + k*nu (Jacobi-Anger), so a linear Koopman generator CAN represent it -- given
enough modes. Expanding the render's harmonics against the Bessel amplitudes J_k(mA) puts 90% of
the oscillating energy in 13 modes, 95% in 22 and 99% in 57, while the models of run_qp.py carry
four oscillators. This runner sweeps that budget (--n_osc) to test whether the failure of the lift
there is a capacity limit rather than the absence of a linear generator.

Same injections as run_qp.py: data_qp replaces the observation model and diagnostics.track is
replaced because its true-phase reference assumes pure rotation.
"""
import argparse, sys
import torch
import data, data_qp

assert "diagnostics" not in sys.modules and "run" not in sys.modules
for _k in ("video", "batch"): setattr(data, _k, getattr(data_qp, _k))

import diagnostics, run
from models import KoopCT
from losses import LAM
diagnostics.track = data_qp.track     # ayni FM-farkindalikli track


def build(name, omega, seed, init="small", n_osc=4, n_static=4):
    if name == "oracle":
        g = torch.Generator().manual_seed(seed + 1000)
        w = (torch.randn(n_osc, generator=g) * 0.1).tolist(); w[0] = omega
        return KoopCT(n_osc=n_osc, n_static=n_static, omega_init=w, seed=seed)
    return KoopCT(n_osc=n_osc, n_static=n_static, seed=seed)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="koop"); p.add_argument("--variant", default="base")
    p.add_argument("--omega_mult", type=float, default=0.75); p.add_argument("--sampling", default="mr")
    p.add_argument("--seed", type=int, default=0); p.add_argument("--iters", type=int, default=3000)
    p.add_argument("--out", default="results_qp_budget.jsonl")
    p.add_argument("--lam", type=float, default=LAM); p.add_argument("--init", default="small"); p.add_argument("--warmup", type=int, default=0)
    p.add_argument("--dmd", default="none", choices=["none", "lock", "ft", "harm", "mr", "mrg"]); p.add_argument("--dmd_at", type=int, default=500)
    p.add_argument("--light", action="store_true"); p.add_argument("--tau", type=float, default=0.1)
    p.add_argument("--n_osc", type=int, default=4)
    a = p.parse_args()
    _n = a.n_osc
    run.build = lambda name, omega, seed, init="small": build(name, omega, seed, init, _n)
    main_rec = run.main(a)
