# Adım 1–2 teşhisleri. Hepsi modelden bağımsız arayüzle (enc/dec/latent/vel) çalışır.
import math, numpy as np, torch, torch.nn.functional as F
from torch.func import jvp
from data import DEV, H, render, video, test_set, batch

_ANG = torch.arange(360, device=DEV) * (2 * math.pi / 360)
_B = torch.stack([render(torch.full((360,), s, device=DEV), _ANG)[:, 0].reshape(360, -1) for s in range(3)])
BANK = (_B - _B.mean(-1, keepdim=True)) / (_B - _B.mean(-1, keepdim=True)).norm(dim=-1, keepdim=True)

def probe(x, sid):
    B, T = x.shape[:2]
    v = x.reshape(B, T, -1); v = v - v.mean(-1, keepdim=True); v = v / (v.norm(dim=-1, keepdim=True) + 1e-8)
    pk, idx = torch.einsum("btp,bap->bta", v, BANK[sid]).max(-1)
    return _ANG[idx], pk

def slope(a, t):  # unwrap edilmiş açının eğimi, (B,T) numpy
    return np.array([np.polyfit(tt, np.unwrap(aa), 1)[0] for aa, tt in zip(a, t)])

def alias_of(omega): return omega - 2 * math.pi * round(omega / (2 * math.pi))

@torch.no_grad()
def rollout(model, omega, n=16, horizon=8.0, dt=1 / 16, seed=321):
    sid, th0 = test_set(n, seed)
    t = torch.arange(0, horizon + 1e-9, dt, device=DEV)[None].expand(n, -1).contiguous()
    x = video(sid, th0, omega, t)
    z, _ = model.latent(model.enc(x[:, 0]), t)
    xh = model.dec(z.reshape(-1, z.shape[-1])).reshape(x.shape)
    return dict(sid=sid, th0=th0, t=t, x=x, xa=video(sid, th0, alias_of(omega), t), z=z, xh=xh, dt=dt, omega=omega)

# ---------- Adım 1: Jacobian / sızıntı ----------
def jacobian_leak(model, R):
    t, z, xh = R["t"], R["z"], R["xh"]; d = z.shape[-1]; eps = 1e-2
    with torch.no_grad():   # ‖∂x̂/∂t‖: modeli t±ε'da ayrıca çalıştırıp merkezi fark (bağımsız yol); z0 hep t=0'da
        z0 = model.enc(R["x"][:, 0]); tt = t[:, 1:]
        zp_, _ = model.latent(z0, torch.cat([torch.zeros_like(tt[:, :1]), tt + eps], 1))
        zm_, _ = model.latent(z0, torch.cat([torch.zeros_like(tt[:, :1]), tt - eps], 1))
        xp_ = model.dec(zp_[:, 1:].reshape(-1, d)); xm_ = model.dec(zm_[:, 1:].reshape(-1, d))
    dxdt = ((xp_ - xm_) / (2 * eps)).view(tt.shape[0], tt.shape[1], -1).norm(dim=-1)
    zf = z[:, 1:].reshape(-1, d)
    with torch.no_grad(): zd = model.vel(zf)
    _, Jzd = jvp(model.dec, (zf,), (zd,))
    Jzd = Jzd.detach().flatten(1).norm(dim=-1).view_as(dxdt)              # ‖J_dec(z)·ż‖ (zincir kuralı yolu)
    ang, _ = probe(xh, R["sid"]); w_img = slope(ang.cpu().numpy(), t.cpu().numpy())
    # latent açısal hız: yörünge merkezine göre ilk 2 PCA düzleminde açı eğimi
    w_lat = []
    for zb, tb in zip(z.cpu().numpy(), t.cpu().numpy()):
        zc = zb - zb.mean(0); _, _, Vt = np.linalg.svd(zc, full_matrices=False)
        p = zc @ Vt[:2].T; w_lat.append(np.polyfit(tb, np.unwrap(np.arctan2(p[:, 1], p[:, 0])), 1)[0])
    w_lat = np.array(w_lat)
    # döngü kapanması: bir görüntü periyodu sonra latent aynı yere dönüyor mu?
    C = []
    for b, (zb, wi) in enumerate(zip(z.cpu().numpy(), w_img)):
        m = int(round(2 * math.pi / max(abs(wi), 1e-3) / R["dt"]))
        if 0 < m < zb.shape[0] - 8:
            C.append(np.linalg.norm(zb[m:] - zb[:-m], axis=1).mean() / np.sqrt(((zb - zb.mean(0)) ** 2).sum(1).mean()))
    with torch.no_grad():
        rms = (z - z.mean(1, keepdim=True)).pow(2).sum(-1).mean().sqrt()
        e_self = (model.enc(xh.reshape(-1, 1, H, H)).view_as(z) - z).norm(dim=-1).mean() / rms
        e_true = (model.enc(R["x"].reshape(-1, 1, H, H)).view_as(z) - z).norm(dim=-1).mean() / rms
    return dict(chain_ratio=float(Jzd.mean() / dxdt.mean()), w_img=float(np.median(w_img)), w_lat=float(np.median(np.abs(w_lat))),
                kappa=float(np.median(np.abs(w_img) / np.maximum(np.abs(w_lat), 1e-3))),
                loop_closure=float(np.mean(C)) if C else float("nan"), enc_self=float(e_self), enc_true=float(e_true))

# ---------- Adım 2: spektrum ve ara zaman ayrıştırması ----------
def dmd_dominant(Z, dt):
    """Z (n,T,d) numpy -> baskın salınım frekansı (genlik ağırlıklı), tüm frekanslar"""
    X = np.concatenate([zb[:-1] for zb in Z]).T; Y = np.concatenate([zb[1:] for zb in Z]).T
    A = Y @ np.linalg.pinv(X, rcond=1e-6)
    lam, V = np.linalg.eig(A)
    w = np.angle(lam) / dt
    amp = np.abs(np.linalg.pinv(V) @ Z[:, 0].T).mean(1) * np.linalg.norm(V, axis=0)
    osc = np.abs(w) > 0.05
    if not osc.any(): return 0.0, w
    return float(np.abs(w[osc])[np.argmax(amp[osc])]), w

def spectrum(model, R):
    out = {}
    with torch.no_grad():
        ze_true = model.enc(R["x"].reshape(-1, 1, H, H)).view_as(R["z"])
    out["dmd_pred"], _ = dmd_dominant(R["z"].cpu().numpy(), R["dt"])
    out["dmd_enc_true"], _ = dmd_dominant(ze_true.cpu().numpy(), R["dt"])
    if hasattr(model, "omega"):   # Koopman: hangi osilatör resmi sürüklüyor? (birim latent faz başına görüntü değişimi)
        z0 = R["z"][:, 0]; n = model.n; use = []
        for j in range(n):
            v = torch.zeros_like(z0); v[:, 2 * j] = -z0[:, 2 * j + 1]; v[:, 2 * j + 1] = z0[:, 2 * j]
            _, Jv = jvp(model.dec, (z0,), (v,)); use.append(Jv.detach().flatten(1).norm(dim=-1).mean().item())
        use = np.array(use); w = model.omega.detach().cpu().numpy()
        out["koop_omegas"] = w.tolist(); out["koop_usage"] = (use / use.sum()).tolist()
        out["koop_dominant"] = float(abs(w[use.argmax()]))
    return out

def intergrid(model, R):
    xh, x, xa, sid = R["xh"], R["x"], R["xa"], R["sid"]; T = x.shape[1]; s = int(round(1 / R["dt"]))
    gi = torch.arange(0, T, s, device=DEV); mi = torch.arange(s // 2, T, s, device=DEV)
    mse = lambda a, b: (a - b).pow(2).flatten(2).mean(-1)                      # (B,len)
    E_grid = mse(xh[:, gi], x[:, gi]).mean().item()
    E_mid = mse(xh[:, mi], x[:, mi]).mean().item()
    E_mid_alias = mse(xh[:, mi], xa[:, mi]).mean().item()
    best, best_a = [], []
    for k, m in enumerate(mi.tolist()):                                        # yol üzerinde en iyi kare (zamanlama serbest)
        cand = xh[:, k * s:(k + 1) * s + 1]
        best.append(mse(cand, x[:, m:m + 1].expand_as(cand)).min(1).values)
        best_a.append(mse(cand, xa[:, m:m + 1].expand_as(cand)).min(1).values)
    E_best = torch.stack(best).mean().item(); E_best_a = torch.stack(best_a).mean().item()
    with torch.no_grad():
        xm = x[:, mi].reshape(-1, 1, H, H)
        E_ae = (model.dec(model.enc(xm)) - xm).pow(2).mean().item()
    _, pk = probe(xh, sid)
    # latent koordinat kırılması mı, decoder interpolasyonu mu?
    z = R["z"]
    with torch.no_grad():
        ze = model.enc(x.reshape(-1, 1, H, H)).view_as(z)                      # encoder'ın gerçek kareye verdiği koordinat
        rms = (z - z.mean(1, keepdim=True)).pow(2).sum(-1).mean().sqrt()
        enc_err_grid = ((ze[:, gi] - z[:, gi]).norm(dim=-1).mean() / rms).item()
        enc_err_mid = ((ze[:, mi] - z[:, mi]).norm(dim=-1).mean() / rms).item()
        # ızgara dışı latent, eğitimdeki (ızgara üstü) latent dağılımının desteğinde mi? bank: ayrı 64 dizinin ızgara latentleri
        sid_b, th_b = test_set(64, 777)
        tb = torch.arange(0, 9.0, device=DEV)[None].expand(64, -1).contiguous()
        zb, _ = model.latent(model.enc(video(sid_b, th_b, R["omega"], tb)[:, 0]), tb)
        zb = zb.reshape(-1, zb.shape[-1]); sd = zb.std(0) + 1e-6
        nn_d = lambda q: torch.cdist(q.reshape(-1, q.shape[-1]) / sd, zb / sd).min(1).values.median().item()
        ood_mid = nn_d(z[:, mi]) / max(nn_d(z[:, gi[1:]]), 1e-9)
    return dict(enc_err_grid=enc_err_grid, enc_err_mid=enc_err_mid, ood_mid=ood_mid, E_grid=E_grid, E_mid=E_mid, mid_over_grid=E_mid / max(E_grid, 1e-9), E_mid_alias=E_mid_alias,
                E_best_on_path=E_best, E_best_on_path_alias=E_best_a, E_ae_mid=E_ae,
                timing_share=1 - E_best / max(E_mid, 1e-9), sharp_grid=pk[:, gi].mean().item(), sharp_mid=pk[:, mi].mean().item())

def track(R, omega, tol_deg=15):
    ang, _ = probe(R["xh"], R["sid"]); t = R["t"]
    cd = lambda a, b: torch.remainder(a - b + math.pi, 2 * math.pi).sub(math.pi).abs()
    fr = torch.remainder(t, 1.0); mid = (fr >= 0.25) & (fr <= 0.75); tol = math.radians(tol_deg)
    th0 = R["th0"][:, None]
    return dict(f_true=(cd(ang, th0 + omega * t) < tol)[mid].float().mean().item(),
                f_alias=(cd(ang, th0 + alias_of(omega) * t) < tol)[mid].float().mean().item())

@torch.no_grad()
def mse_grids(model, omega, sampling, K=8, seed=999):
    g = torch.Generator().manual_seed(seed); out = {}
    x, t = batch(128, K, sampling, omega, g, noise=0.0)
    out["train_mse"] = F.mse_loss(model.predict(x[:, 0], t), x).item()
    sid, th0 = test_set(128, seed + 1)
    for name, sp in [("half", 0.5), ("3_7", 3 / 7)]:
        tt = torch.arange(0, K + 1e-6, sp, device=DEV)[None].expand(128, -1).contiguous()
        xt = video(sid, th0, omega, tt)
        out[f"mse_{name}"] = F.mse_loss(model.predict(xt[:, 0], tt), xt).item()
    return out

def full(model, omega, sampling):
    model.eval(); R = rollout(model, omega)
    d = {}
    d.update(mse_grids(model, omega, sampling)); d.update(jacobian_leak(model, R))
    d.update(spectrum(model, R)); d.update(intergrid(model, R)); d.update(track(R, omega))
    # faz portresi için 4 dizinin latent'i (ilk 2 PCA, tüm diziler ortak)
    Z = R["z"][:4].cpu().numpy(); Zc = Z.reshape(-1, Z.shape[-1]); mu = Zc.mean(0)
    _, _, Vt = np.linalg.svd(Zc - mu, full_matrices=False)
    return d, dict(pca=((Z - mu) @ Vt[:2].T), t=R["t"][0].cpu().numpy())

def selftest_chain():
    import models
    for M in (models.KoopCT(), models.NODE()):
        M = M.to(DEV)
        with torch.no_grad():
            if hasattr(M, "omega"): M.omega.copy_(torch.tensor([2.0, 5.0, 0.7, 7.8]))
        r = jacobian_leak(M, rollout(M, 0.5 * math.pi, n=4))["chain_ratio"]
        print(f"selftest zincir kuralı {type(M).__name__}: ‖J·ż‖/‖∂x̂/∂t‖ = {r:.4f} (1 beklenir)")
        if abs(r - 1) > 0.02: return False
    return True

def selftest():
    """Ölçüm araçlarının kendi testi: bilinen doğrusal osilatörde DMD ω'yu bulmalı; ayrıştırma sanal modelde doğru olmalı."""
    ok = True
    for w in [0.5 * math.pi, 2.5 * math.pi]:
        t = np.arange(0, 8, 1 / 16); Z = np.stack([np.stack([np.cos(w * t + p), np.sin(w * t + p), np.full_like(t, 0.3)], 1) for p in (0, 1, 2)])
        f, _ = dmd_dominant(Z, 1 / 16); good = abs(f - w) < 1e-3; ok &= good
        print(f"selftest DMD ω={w/math.pi:.1f}π -> {f/math.pi:.4f}π {'OK' if good else 'FAIL'}")
    return ok
