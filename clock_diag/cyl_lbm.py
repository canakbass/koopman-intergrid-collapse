"""2D lattice-Boltzmann (D2Q9, BGK) flow past a cylinder at Re~100, run once on the CPU to build
the data source for the cylinder-flow benchmark (HANDOFF 5-B).

Why a stored period instead of a stored trajectory: the whole experiment measures the model
between grid points, so the ground truth has to be exact at arbitrary t, not only at simulation
steps. On the limit cycle the wake is periodic, so one period is enough -- render(phase) does a
Fourier interpolation in phase, which is exact for a band-limited periodic field. That mirrors
data.py's analytic render(sid, ang) and keeps the harmonics of the shedding frequency, which is
the mechanism the paper is about.

Output: cyl_period.npz with frames (N,H,H) vorticity over exactly one shedding period, the
measured period in lattice steps, and the Strouhal number.
"""
import argparse, numpy as np

C = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=np.float32)
W = np.array([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4, dtype=np.float32)
OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])


def equilibrium(rho, u):
    cu = 3.0 * np.tensordot(C, u, axes=([1], [0]))
    usq = 1.5 * (u[0] ** 2 + u[1] ** 2)
    return rho[None] * W[:, None, None] * (1 + cu + 0.5 * cu ** 2 - usq[None])


class Cylinder:
    def __init__(s, nx=360, ny=180, d=18.0, u0=0.1, re=100.0):
        s.nx, s.ny, s.u0, s.d = nx, ny, u0, d
        s.CX = nx // 5                                  # silindir merkezi
        s.X0, s.X1 = s.CX - 8, s.CX - 8 + 256           # iz bolgesi: 256x128 -> 8x4 blok ortalamasi -> 32x32
        s.Y0, s.Y1 = ny // 2 - 64, ny // 2 + 64
        s.nu = u0 * d / re
        s.tau = 3 * s.nu + 0.5
        x, y = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
        s.wall = ((x - nx / 5.0) ** 2 + (y - (ny / 2.0 - 0.5)) ** 2) < (d / 2.0) ** 2
        s.rho = np.ones((nx, ny), dtype=np.float32); s.u = np.zeros((2, nx, ny), dtype=np.float32); s.u[0] = u0
        s.u[1] += (0.05 * u0 * np.sin(4 * np.pi * y / ny)).astype(np.float32)   # asimetri: dokulmeyi tetikler (buyuk tutuldu, lineer buyume evresi kisalsin)
        s.f = equilibrium(s.rho, s.u)

    def step(s):
        s.rho = s.f.sum(0)
        s.u = np.tensordot(C, s.f, axes=([0], [0])) / s.rho[None]
        s.u[:, s.wall] = 0.0
        s.u[0, 0, :] = s.u0; s.u[1, 0, :] = 0.0                 # giris: sabit hiz
        s.rho[0, :] = 1.0
        feq = equilibrium(s.rho, s.u)
        fout = s.f - (s.f - feq) / s.tau
        for q in range(9): fout[q, s.wall] = s.f[OPP[q], s.wall]        # silindirde bounce-back
        fout[:, 0, :] = feq[:, 0, :]
        for q in range(9):                                             # akis
            s.f[q] = np.roll(np.roll(fout[q], C[q, 0], 0), C[q, 1], 1)
        s.f[:, -1, :] = s.f[:, -2, :]                                  # cikis: sifir egim
        return s

    def vorticity(s):
        dvdx = np.gradient(s.u[1], axis=0); dudy = np.gradient(s.u[0], axis=1)
        w = dvdx - dudy
        w[s.wall] = 0.0
        return w

    def probe(s):     # dokulme sinyali: silindirin 2 cap arkasinda tek nokta, capraz hiz
        return s.u[1, s.CX + int(2 * s.d), s.ny // 2]

    def frame(s, H=32):
        w = s.vorticity()[s.X0:s.X1, s.Y0:s.Y1]
        return w.reshape(H, (s.X1 - s.X0) // H, H, (s.Y1 - s.Y0) // H).mean((1, 3))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--warmup", type=int, default=30000)          # asgari; sonrasi genlik durağanlasana kadar
    p.add_argument("--max_warmup", type=int, default=300000)
    p.add_argument("--tol", type=float, default=0.002)           # ardisik bloklar arasi bagil genlik degisimi
    p.add_argument("--record", type=int, default=12000)
    p.add_argument("--harmonics", type=int, default=8)
    p.add_argument("--out", default="cyl_period.npz")
    a = p.parse_args()
    sim = Cylinder()
    print(f"tau={sim.tau:.4f} nu={sim.nu:.5f}", flush=True)
    for i in range(a.warmup): sim.step()
    A_prev, done = None, a.warmup
    while done < a.max_warmup:        # limit cevrimine oturma: genlik iki ardisik blokta ayni kalana kadar
        blk = np.array([sim.step().probe() for _ in range(4000)])
        done += 4000
        A = float(blk.std())
        rel = abs(A - A_prev) / A if (A_prev and A > 0) else float("nan")
        print(f"warmup {done} genlik={A:.5f} degisim={rel:.4%}", flush=True)
        if A > 0.01 and A_prev is not None and rel < a.tol: break
        A_prev = A
    else:
        print("UYARI: genlik max_warmup icinde duragan hale gelmedi", flush=True)
    sig, frames = [], []
    for i in range(a.record):
        sim.step(); sig.append(sim.probe()); frames.append(sim.frame())
    sig = np.array(sig); frames = np.array(frames, dtype=np.float32)
    x = sig - sig.mean()
    sp = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    k = int(np.argmax(sp[1:]) + 1)
    lo, mid, hi = sp[k - 1], sp[k], sp[k + 1]                       # parabolik tepe duzeltmesi
    kk = k + 0.5 * (lo - hi) / (lo - 2 * mid + hi)
    w0 = 2 * np.pi * kk / len(x)
    t = np.arange(len(frames))
    F = frames.reshape(len(t), -1)

    def design(w):
        return np.concatenate([np.ones((len(t), 1))] + [np.stack([np.cos(m * w * t), np.sin(m * w * t)], 1)
                                                        for m in range(1, a.harmonics + 1)], 1)

    def resid(w, cols):          # verilen w'de alan uyumunun bagil artigi
        B = design(w); G = F[:, cols]
        c, *_ = np.linalg.lstsq(B, G, rcond=None)
        return float(np.sqrt(((G - B @ c) ** 2).mean()) / G.std())

    cols = np.random.default_rng(0).choice(F.shape[1], 64, replace=False)   # tarama icin 64 piksel yeter
    for span, n in [(0.02, 81), (0.0008, 81)]:                             # iki asamali ince tarama
        cand = w0 * (1 + np.linspace(-span, span, n))
        w0 = float(cand[int(np.argmin([resid(w, cols) for w in cand]))])
    T = 2 * np.pi / w0
    print(f"periyot T={T:.3f} adim | St={sim.d / (T * sim.u0):.4f} | dokulme genligi={x.std():.2e} | tikanma={sim.d / sim.ny:.1%}", flush=True)
    B = design(w0)
    coef, *_ = np.linalg.lstsq(B, F, rcond=None)                    # periyodik model: A0 + sum_m harmonik
    res = F - B @ coef
    print(f"periyodik model artigi: {res.std() / F.std():.4%} (kucuk olmali)", flush=True)
    h = len(t) // 2              # duraganlik kontrolu: temel harmonigin genligi kaydin iki yarisinda ayni mi
    amp = []
    for sl in (slice(0, h), slice(h, 2 * h)):
        Bh = B[sl]; c, *_ = np.linalg.lstsq(Bh, F[sl], rcond=None)
        amp.append(float(np.sqrt((c[1] ** 2 + c[2] ** 2).sum())))
    print(f"temel genlik yarim-yarim: {amp[0]:.4f} -> {amp[1]:.4f} ({(amp[1]/amp[0]-1):+.3%})", flush=True)
    E = [float((coef[0] ** 2).sum())] + [float((coef[2 * m - 1] ** 2 + coef[2 * m] ** 2).sum())
                                         for m in range(1, a.harmonics + 1)]
    print("harmonik enerji payi:", [f"{e / sum(E[1:]):.4f}" for e in E[1:]], flush=True)
    np.savez(a.out, coef=coef.astype(np.float32), T=T, w0=w0, H=32,
             harmonics=a.harmonics, resid=res.std() / F.std(), probe=x.astype(np.float32))
    print("kaydedildi:", a.out, flush=True)
