"""run.py for the quasi-periodic rotation (see data_qp.py).

Two injections, both before diagnostics binds anything: data_qp replaces the observation model, and
diagnostics.track is replaced because its true-phase reference is theta_0 + omega*t, which is the
mean rotation here and not the true phase. Everything else -- kappa, the inter-grid errors, the
autoencoder and chart diagnostics -- is the unmodified code that produced the other tables.
f_alias is left in place but is meaningless here: there is no single frequency to alias.
"""
import argparse, sys
import data, data_qp

assert "diagnostics" not in sys.modules and "run" not in sys.modules, "data_qp diagnostics'ten once takilmali"
for _k in ("video", "batch"): setattr(data, _k, getattr(data_qp, _k))

import diagnostics, run
from losses import LAM


diagnostics.track = data_qp.track


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="koop"); p.add_argument("--variant", default="base")
    p.add_argument("--omega_mult", type=float, default=0.75); p.add_argument("--sampling", default="regular")
    p.add_argument("--seed", type=int, default=0); p.add_argument("--iters", type=int, default=3000)
    p.add_argument("--out", default="results_qp.jsonl")
    p.add_argument("--lam", type=float, default=LAM); p.add_argument("--init", default="small"); p.add_argument("--warmup", type=int, default=0)
    p.add_argument("--dmd", default="none", choices=["none", "lock", "ft", "harm", "mr", "mrg"]); p.add_argument("--dmd_at", type=int, default=500)
    p.add_argument("--light", action="store_true"); p.add_argument("--tau", type=float, default=0.1)
    run.main(p.parse_args())
