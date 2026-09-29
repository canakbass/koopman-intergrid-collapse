"""run.py for the cylinder-flow benchmark (HANDOFF 5-B).

data_cyl is installed into the data module BEFORE diagnostics.py is imported, so diagnostics binds
the cylinder render/video/batch at its own import time and every metric -- kappa, loop_closure,
E_grid/E_mid, f_true -- is computed by the same code that produced the paper's tables. No core file
is modified and no diagnostic is reimplemented, which is what makes the two benchmarks comparable.
Import order is load-bearing, so it is asserted rather than assumed.
"""
import argparse, sys
import data, data_cyl

assert "diagnostics" not in sys.modules and "run" not in sys.modules, "data_cyl diagnostics'ten once takilmali"
for _k in ("render", "video", "batch", "test_set"): setattr(data, _k, getattr(data_cyl, _k))

import run                      # buradan sonra run.py -> diagnostics.py yamali data'yi baglar
from losses import LAM

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="koop"); p.add_argument("--variant", default="base")
    p.add_argument("--omega_mult", type=float, default=0.75); p.add_argument("--sampling", default="regular")
    p.add_argument("--seed", type=int, default=0); p.add_argument("--iters", type=int, default=3000)
    p.add_argument("--out", default="results_cyl.jsonl")
    p.add_argument("--lam", type=float, default=LAM); p.add_argument("--init", default="small"); p.add_argument("--warmup", type=int, default=0)
    p.add_argument("--dmd", default="none", choices=["none", "lock", "ft", "harm", "mr", "mrg"]); p.add_argument("--dmd_at", type=int, default=500)
    p.add_argument("--light", action="store_true"); p.add_argument("--tau", type=float, default=0.1)
    run.main(p.parse_args())
