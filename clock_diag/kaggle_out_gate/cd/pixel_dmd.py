# Encoder'dan bağımsız teşhis: piksel dizilerinde (PCA-24) Δt1/Δt2 DMD -> Δt2-tutarlı temel mod var mı?
import json, math, numpy as np, torch
import data, data_pend as DP, dmd_init as D

def pca_seqs(x1, x2, r=24):
    n, T = x1.shape[:2]
    A1 = x1.reshape(n * T, -1).double().cpu().numpy(); A2 = x2.reshape(n * T, -1).double().cpu().numpy()
    mu = A1.mean(0); _, _, Vt = np.linalg.svd(A1 - mu, full_matrices=False)
    proj = lambda A: ((A - mu) @ Vt[:r].T).reshape(n, T, r)
    return proj(A1), proj(A2)

def report(name, x1, x2):
    z1, z2 = pca_seqs(x1, x2)
    P, om, _ = D.dmd_modal(z1, 6, 24 - 12)
    zp2 = np.einsum("ij,ntj->nti", np.linalg.inv(P), z2)
    lifted, mi = D.multirate_lift(om, zp2)
    (j, w, rbest), _ = D.partial_choose(om, zp2)
    out = dict(case=name, principal=[round(float(x), 3) for x in om], lifted=[round(float(x), 3) for x in lifted],
               residual=[round(float(x), 3) for x in mi["mr_residual"]], band_best_omega=round(w, 3), band_best_residual=round(rbest, 3))
    print(json.dumps(out)); return out

g = torch.Generator().manual_seed(7); n = 512
with torch.no_grad():
    x1, _ = DP.batch(n, 8, "regular", g, noise=0.0); x2, _ = DP.batch(n, 8, "d2", g, noise=0.0)
    res = [report("sarkaç (ω0=2.4, genlik 0.3–1.2)", x1, x2)]
    w = 0.763932 * math.pi
    x1, _ = data.batch(n, 8, "regular", w, g, noise=0.0); x2, _ = data.batch(n, 8, "d2", w, g, noise=0.0)
    res.append(report("sprite (ω=0.764π=2.40 rad/s)", x1, x2))
json.dump(res, open("pixel_dmd.json", "w"))
