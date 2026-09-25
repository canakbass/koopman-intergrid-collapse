"""Figure 1 right panel: decoded frames at grid time (correct) vs half-step (wrong) for one
learned model vs the oracle. Must run from the clock_diag/ directory (relative imports)."""
import math
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import data, diagnostics
from models import KoopCT

OUT = "../paper/figures/fig1_right.pdf"
OMEGA = 0.75 * math.pi  # rho = 3/8
SEQ = 0  # which of the 16 held-out rollout sequences to show

def load(tag):
    m = KoopCT().to(data.DEV)
    m.load_state_dict(torch.load(f"ckpt/{tag}.pt", map_location=data.DEV))
    m.eval()
    return m

learned = load("koop-small-wu0_base_w0.75_regular_s0")
oracle = load("oracle-small-wu0_base_w0.75_regular_s0")

with torch.no_grad():
    Rl = diagnostics.rollout(learned, OMEGA)
    Ro = diagnostics.rollout(oracle, OMEGA)

def frames(R, k_list):
    # R["xh"]: (n, T, 1, H, H), t step = 1/16 -> grid index k*16, half-step index k*16+8
    xh = R["xh"][SEQ, :, 0].clamp(0, 1).cpu().numpy()
    return [xh[k] for k in k_list]

grid_ks = [16, 32, 48]          # t = 1, 2, 3 (correct, on-grid)
half_ks = [24, 40, 56]          # t = 1.5, 2.5, 3.5 (wrong, mid-grid)

rows = [
    ("oracle\n$t{=}k\\Delta$", frames(Ro, grid_ks)),
    ("oracle\n$t{=}(k{+}\\frac{1}{2})\\Delta$", frames(Ro, half_ks)),
    ("learned\n$t{=}k\\Delta$", frames(Rl, grid_ks)),
    ("learned\n$t{=}(k{+}\\frac{1}{2})\\Delta$", frames(Rl, half_ks)),
]

plt.rcParams.update({"font.family": "serif", "font.size": 8, "pdf.fonttype": 42, "mathtext.fontset": "cm"})
fig, axes = plt.subplots(4, 3, figsize=(3.1, 4.3))
for r, (label, ims) in enumerate(rows):
    for c, im in enumerate(ims):
        ax = axes[r, c]
        ax.imshow(im, cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#888888"); spine.set_linewidth(0.5)
        if c == 0:
            ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=7)

fig.tight_layout(pad=0.3, h_pad=0.4, w_pad=0.2)
fig.savefig(OUT, dpi=300)
print("wrote", OUT)
