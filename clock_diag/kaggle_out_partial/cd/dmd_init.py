# Spektral yeniden başlatma: ızgara latentlerine lineer DMD -> modal taban P -> L'yi DMD frekanslarına eşitle.
import math, numpy as np, torch, torch.nn as nn
import data

def harmonic_lift(phis, amps, M=6):
    """Izgara açıları φ_i (ana dal) -> sürekli frekanslar m_i·ω1. ω1 adayları: modların kendisi (±, alt harmonik bölümleri)."""
    phis = np.asarray(phis); amps = np.asarray(amps)
    cd = lambda a: np.abs(np.remainder(a + np.pi, 2 * np.pi) - np.pi)
    cands = set()
    for p in phis:
        for k in range(-3, 4):          # φ = m·ω1 − 2πk  ⇒ ω1 = (φ + 2πk)/m
            for m in range(1, M + 1):
                w = (p + 2 * np.pi * k) / m
                if 0.02 < abs(w) <= np.pi: cands.add(round(float(w), 6))
    best = None
    for w1 in cands:
        ms = np.arange(-M, M + 1); ms = ms[ms != 0]
        res = [cd(ms * w1 - p).min() for p in phis]; mm = [ms[np.argmin(cd(ms * w1 - p))] for p in phis]
        score = float(np.sum(amps * np.array(res))) + 1e-3 * float(np.sum(amps * np.abs(mm)))   # küçük harmonik tercih
        if best is None or score < best[0]: best = (score, w1, mm)
    _, w1, mm = best
    return np.array([m * w1 for m in mm]), float(w1), [int(m) for m in mm]

def dmd_modal(ze, n_osc, n_static, dt=1.0, harmonic=False):
    """ze (N,T,d) numpy, düzenli ızgara. Dönüş: P (d×d; z = P·c), ω (n_osc,), bilgi.
    Çift koordinatları SOL özvektörlerden (c_j = u_jᵀ z) -> katlı λ≈1 statik modlardan bağımsız, birebir modal.
    Kalan koordinatlar seçilen SAĞ özvektörleri yok eder -> seçilen modlar statik yuvalara sızmaz. P her zaman tersinir."""
    N, T, d = ze.shape
    X = ze[:, :-1].reshape(-1, d).T; Y = ze[:, 1:].reshape(-1, d).T
    A = Y @ np.linalg.pinv(X, rcond=1e-8)
    lam, V = np.linalg.eig(A); lamL, U = np.linalg.eig(A.T)
    U = U / np.linalg.norm(U, axis=0)
    left = [int(np.argmin(np.abs(lamL - l))) for l in lam]
    amp = np.array([np.abs(U[:, left[i]] @ X).mean() for i in range(d)])
    cplx = [i for i in np.argsort(-amp) if lam[i].imag > 1e-6]
    chosen = cplx[:n_osc]
    rows, om, vs = [], [], []
    for i in chosen:
        u = U[:, left[i]]; rows += [u.real, u.imag]; om.append(float(np.angle(lam[i])) / dt)   # c' = λc ⇒ (a,b) +angle(λ) döner
        vs += [V[:, i].real, V[:, i].imag]
    if vs:
        Uc, _, _ = np.linalg.svd(np.stack(vs, 1), full_matrices=True); comp = [Uc[:, k] for k in range(len(vs), d)]
    else:
        comp = [np.eye(d)[k] for k in range(d)]
    while len(om) < n_osc: rows += [comp.pop(0), comp.pop(0)]; om.append(0.0)
    rows += comp[:n_static]
    Pinv = np.stack(rows, 0); P = np.linalg.inv(Pinv)
    lift = {}
    if harmonic:
        k = len(chosen)
        lifted, w1, mm = harmonic_lift(np.array(om[:k]) * dt, amp[chosen])
        om[:k] = list(lifted / dt); lift = dict(w1=w1 / dt, harmonics=mm)
    info = dict(dmd_omegas=[float(w) for w in om], dmd_all=[float(np.angle(l)) for l in lam], dmd_mod=[float(abs(l)) for l in lam],
                cond_P=float(np.linalg.cond(P)), **{f"lift_{k}": v for k, v in lift.items()})
    return P, np.array(om), info

def multirate_lift(om1, zp2, dt1=1.0, dt2=2 ** 0.5 / 2, kmax=4):
    """zp2: Δt2 dizileri Δt1-modal koordinatlarda (N,T,d). Her çift j için kendi 2×2 haritasından işaretli Δt2 dönme açısı θ_j;
    ω = ω0 + 2πk/Δt1 adaylarından ωΔt2 ≡ θ_j olanı seç (modlar arası ve işaret kaynaklı hayalet eşleşme yok)."""
    cd = lambda a: np.abs(np.remainder(a + np.pi, 2 * np.pi) - np.pi)
    out, res, margin, psis = [], [], [], []
    for j, w0 in enumerate(om1):
        X = zp2[:, :-1, 2*j:2*j+2].reshape(-1, 2).T; Y = zp2[:, 1:, 2*j:2*j+2].reshape(-1, 2).T
        A = Y @ np.linalg.pinv(X, rcond=1e-8)
        th2 = np.arctan2(A[1, 0] - A[0, 1], A[0, 0] + A[1, 1]); psis.append(float(th2))   # işaretli Δt2 dönme açısı (modal tabanda +ω yönü)
        if abs(w0) < 1e-9: out.append(0.0); res.append(float("nan")); margin.append(float("nan")); continue
        ks = np.arange(-kmax, kmax + 1); cands = w0 + 2 * np.pi * ks / dt1
        dist = cd(cands * dt2 - th2) + 1e-3 * np.abs(ks)
        o = np.argsort(dist); out.append(float(cands[o[0]])); res.append(float(dist[o[0]])); margin.append(float(dist[o[1]] - dist[o[0]]))
    return np.array(out), dict(mr_residual=res, mr_margin=margin, dmd2_pair_phase=psis)

@torch.no_grad()
def multirate_reinit(model, batch_fn, g, n_seq=256, K=8):
    """batch_fn(n, K, g, mode) ; mode ∈ {regular, d2}"""
    x1, _ = batch_fn(n_seq, K, g, "regular"); x2, _ = batch_fn(n_seq, K, g, "d2")
    enc = lambda x: model.enc(x.reshape(-1, 1, data.H, data.H)).reshape(n_seq, K + 1, -1).double().cpu().numpy()
    ze1, ze2 = enc(x1), enc(x2)
    P, om, info = dmd_modal(ze1, model.n, model.d - 2 * model.n)
    Pinv = np.linalg.inv(P)
    lifted, mi = multirate_lift(om, np.einsum("ij,ntj->nti", Pinv, ze2))
    info.update(mi); info["dmd_principal"] = [float(w) for w in om]; info["dmd_omegas"] = [float(w) for w in lifted]
    _install(model, P, lifted)
    return info

def partial_choose(om1, zp2, band=(1.5, 4.0), dt1=1.0, dt2=2 ** 0.5 / 2, kmax=6):
    """Kısmi kilitleme: her çift için yalnızca |ω| ∈ band adaylarına bak, Δt2 işaretli açısıyla en tutarlı TEK çifti seç."""
    cd = lambda a: np.abs(np.remainder(a + np.pi, 2 * np.pi) - np.pi)
    best, table = None, []
    for j, w0 in enumerate(om1):
        if abs(w0) < 1e-9: continue
        X = zp2[:, :-1, 2*j:2*j+2].reshape(-1, 2).T; Y = zp2[:, 1:, 2*j:2*j+2].reshape(-1, 2).T
        A = Y @ np.linalg.pinv(X, rcond=1e-8); th2 = np.arctan2(A[1, 0] - A[0, 1], A[0, 0] + A[1, 1])
        c = w0 + 2 * np.pi * np.arange(-kmax, kmax + 1) / dt1
        c = c[(np.abs(c) >= band[0]) & (np.abs(c) <= band[1])]
        for w in c:
            r = float(cd(w * dt2 - th2)); table.append((j, float(w), r))
            if best is None or r < best[2]: best = (j, float(w), r)
    return best, table

def partial_install_info(model, batch_fn, g, band=(1.5, 4.0), n_seq=256, K=8):
    with torch.no_grad():
        x1, _ = batch_fn(n_seq, K, g, "regular"); x2, _ = batch_fn(n_seq, K, g, "d2")
        enc = lambda x: model.enc(x.reshape(-1, 1, data.H, data.H)).reshape(n_seq, K + 1, -1).double().cpu().numpy()
        ze1, ze2 = enc(x1), enc(x2)
        P, om, info = dmd_modal(ze1, model.n, model.d - 2 * model.n)
        (j, w, r), table = partial_choose(om, np.einsum("ij,ntj->nti", np.linalg.inv(P), ze2), band)
        om = om.copy(); om[j] = w
        _install(model, P, om)
    mask = torch.ones_like(model.omega); mask[j] = 0.0                # yalnızca seçilen çift dondurulur
    model.omega.register_hook(lambda gr: gr * mask)
    info.update(dict(partial_pair=int(j), partial_omega=w, partial_residual=r, partial_candidates=table, dmd_omegas=[float(x) for x in om]))
    return info

def _install(model, P, om):
    d = model.d; Pt = torch.tensor(P, dtype=torch.float32, device=data.DEV)
    li, lo = nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False)
    li.weight.copy_(torch.linalg.inv(Pt)); lo.weight.copy_(Pt)
    model.enc = nn.Sequential(model.enc, li.to(data.DEV)); model.dec = nn.Sequential(lo.to(data.DEV), model.dec)
    model.omega.copy_(torch.tensor(np.asarray(om), dtype=torch.float32))

@torch.no_grad()
def reinit(model, batch_fn, g, n_seq=256, K=8, harmonic=False):
    """batch_fn(n, K, g) -> (x (n,K+1,1,H,H), t) düzenli ızgara; veri kaynağından bağımsız"""
    x, _ = batch_fn(n_seq, K, g)
    ze = model.enc(x.reshape(-1, 1, data.H, data.H)).reshape(n_seq, K + 1, -1).double().cpu().numpy()
    P, om, info = dmd_modal(ze, model.n, model.d - 2 * model.n, harmonic=harmonic)
    d = model.d; Pt = torch.tensor(P, dtype=torch.float32, device=data.DEV)
    li, lo = nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False)
    li.weight.copy_(torch.linalg.inv(Pt)); lo.weight.copy_(Pt)
    model.enc = nn.Sequential(model.enc, li.to(data.DEV)); model.dec = nn.Sequential(lo.to(data.DEV), model.dec)
    model.omega.copy_(torch.tensor(om, dtype=torch.float32))
    return info

def selftest():
    rng = np.random.default_rng(0); d, n_osc = 12, 4
    Q = rng.normal(size=(d, d)); w_true = [0.75 * math.pi, 0.3, 1.9, 0.05]
    k = np.arange(9)
    ze = []
    for _ in range(64):
        u = rng.normal(size=d); z = []
        for kk in k:
            v = u.copy()
            for j, w in enumerate(w_true):
                c, s = math.cos(w * kk), math.sin(w * kk); a, b = u[2*j], u[2*j+1]
                v[2*j], v[2*j+1] = a * c - b * s, a * s + b * c
            z.append(Q @ v)
        ze.append(z)
    ze = np.array(ze)
    P, om, info = dmd_modal(ze, n_osc, d - 2 * n_osc)
    zp = np.einsum("ij,ntj->nti", np.linalg.inv(P), ze)
    err = 0.0
    for j, w in enumerate(om):
        a, b = zp[:, :-1, 2*j], zp[:, :-1, 2*j+1]; c, s = math.cos(w), math.sin(w)
        err = max(err, np.abs(np.stack([a * c - b * s, a * s + b * c], -1) - zp[:, 1:, 2*j:2*j+2]).max())
    ok = sorted(np.round(np.abs(om), 3)) == sorted(np.round(np.abs(w_true), 3)) and err < 1e-6
    print(f"selftest DMD modal: |ω| bulunan={sorted(np.round(np.abs(om),4))} gerçek={sorted(np.round(np.abs(w_true),4))} "
          f"modal tabanda tek adım hatası={err:.1e} cond(P)={info['cond_P']:.1f} {'OK' if ok else 'FAIL'}")
    return ok

def selftest_multirate():
    rng = np.random.default_rng(1); d = 10; Q = rng.normal(size=(d, d)); dt2 = 2 ** 0.5 / 2
    for w1 in [0.75 * math.pi, 0.763932 * math.pi]:
        w_true = [w1, 2 * w1, 3 * w1, 0.3]
        def seqs(dt):
            Z = []
            for _ in range(64):
                u = rng.normal(size=d); z = []
                for kk in range(9):
                    v = u.copy()
                    for j, w in enumerate(w_true):
                        c, s_ = math.cos(w * kk * dt), math.sin(w * kk * dt); a, b = u[2*j], u[2*j+1]
                        v[2*j], v[2*j+1] = a * c - b * s_, a * s_ + b * c
                    z.append(Q @ v)
                Z.append(z)
            return np.array(Z)
        z1, z2 = seqs(1.0), seqs(dt2)
        P, om, _ = dmd_modal(z1, 4, d - 8)
        lifted, info = multirate_lift(om, np.einsum("ij,ntj->nti", np.linalg.inv(P), z2))
        ok = sorted(np.round(np.abs(lifted), 4)) == sorted(np.round(np.abs(w_true), 4))
        print(f"selftest çok hızlı: ana dal |ω|/π={np.round(np.sort(np.abs(om))/np.pi,3)} -> kaldırılmış {np.round(np.sort(np.abs(lifted))/np.pi,3)} "
              f"(gerçek {np.round(np.sort(np.abs(w_true))/np.pi,3)}) min marj={np.nanmin(info['mr_margin']):.2f} {'OK' if ok else 'FAIL'}")
        if not ok: return False
    return True

def selftest_partial():
    rng = np.random.default_rng(3); d = 10; Q = rng.normal(size=(d, d)); dt2 = 2 ** 0.5 / 2
    w_true = [2.3, 4.6, 6.9, 0.3]          # sarkaç benzeri: temel + harmonikler (Δt1=1'de 4.6 ve 6.9 katlanır)
    def seqs(dt):
        Z = []
        for _ in range(64):
            u = rng.normal(size=d); z = []
            for kk in range(9):
                v = u.copy()
                for j, w in enumerate(w_true):
                    c, s_ = math.cos(w * kk * dt), math.sin(w * kk * dt); a, b = u[2*j], u[2*j+1]
                    v[2*j], v[2*j+1] = a * c - b * s_, a * s_ + b * c
                z.append(Q @ v)
            Z.append(z)
        return np.array(Z)
    z1, z2 = seqs(1.0), seqs(dt2)
    P, om, _ = dmd_modal(z1, 4, d - 8)
    (j, w, r), table = partial_choose(om, np.einsum("ij,ntj->nti", np.linalg.inv(P), z2))
    runner = sorted(t[2] for t in table)[1]
    ok = abs(abs(w) - 2.3) < 1e-6
    print(f"selftest kısmi: seçilen |ω|={abs(w):.4f} rad/s (gerçek temel 2.3), artık={r:.1e}, ikinci en iyi aday artığı={runner:.2f} {'OK' if ok else 'FAIL'}")
    return ok

def selftest_lift():
    w1 = 0.75 * math.pi; true = np.array([1, 2, 3, -1]) * w1
    phis = np.angle(np.exp(1j * true)); lifted, w, mm = harmonic_lift(phis, np.array([1.0, 0.8, 0.5, 0.3]))
    ok = np.allclose(np.abs(lifted), np.abs(true), atol=1e-6)
    print(f"selftest harmonik: ana dal φ/π={np.round(phis/np.pi,3)} -> kaldırılmış |ω|/π={np.round(np.abs(lifted)/np.pi,3)} (gerçek {np.round(np.abs(true)/np.pi,3)}) {'OK' if ok else 'FAIL'}")
    return ok

if __name__ == "__main__":
    import sys; sys.exit(0 if selftest() and selftest_lift() and selftest_multirate() and selftest_partial() else 1)
