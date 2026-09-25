# Ortak arayüz: enc, dec, latent(z0, t, offgrid) -> (z (B,T,d), z_offgrid (B,M,d)), vel(z) = ż
import math, torch, torch.nn as nn
from data import H

class Enc(nn.Module):
    def __init__(s, d):
        super().__init__()
        s.c = nn.Sequential(nn.Conv2d(1, 32, 4, 2, 1), nn.GELU(), nn.Conv2d(32, 64, 4, 2, 1), nn.GELU(),
                            nn.Conv2d(64, 64, 4, 2, 1), nn.GELU(), nn.Flatten(), nn.Linear(1024, 256), nn.GELU(), nn.Linear(256, d))
    def forward(s, x): return s.c(x)

class Dec(nn.Module):
    def __init__(s, d):
        super().__init__()
        s.l = nn.Sequential(nn.Linear(d, 256), nn.GELU(), nn.Linear(256, 1024), nn.GELU())
        s.c = nn.Sequential(nn.ConvTranspose2d(64, 64, 4, 2, 1), nn.GELU(), nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.GELU(),
                            nn.ConvTranspose2d(32, 1, 4, 2, 1))
    def forward(s, z): return s.c(s.l(z).view(-1, 64, 4, 4))

class KoopCT(nn.Module):
    """ż = Lz, L = ⊕ ω_j J  ⊕ 0 (statik)."""
    def __init__(s, n_osc=4, n_static=4, omega_init=None, seed=0):
        super().__init__()
        s.n, s.d = n_osc, 2 * n_osc + n_static
        s.enc, s.dec = Enc(s.d), Dec(s.d)
        g = torch.Generator().manual_seed(seed + 1000)
        w0 = torch.randn(n_osc, generator=g) * 0.1 if omega_init is None else torch.tensor(omega_init, dtype=torch.float)
        s.omega = nn.Parameter(w0)
    def omega_eff(s, a, b):          # (B,n); doğrusal Koopman'da sabit
        return s.omega.expand(a.shape[0], -1)
    def _flow(s, z0, t):
        n = s.n; a, b, st = z0[:, 0:2*n:2], z0[:, 1:2*n:2], z0[:, 2*n:]
        ph = t[:, :, None] * s.omega_eff(a, b)[:, None, :]
        c, sn = torch.cos(ph), torch.sin(ph)
        osc = torch.stack([a[:, None] * c - b[:, None] * sn, a[:, None] * sn + b[:, None] * c], -1).flatten(-2)
        return torch.cat([osc, st[:, None].expand(-1, t.shape[1], -1)], -1)
    def latent(s, z0, t, offgrid=False):
        z = s._flow(z0, t)
        if not offgrid: return z, None
        u = torch.rand(t.shape[0], t.shape[1] - 1, device=t.device)
        return z, s._flow(z0, t[:, :-1] + u * (t[:, 1:] - t[:, :-1]))   # her aralıkta rastgele ara zaman
    def vel(s, z):
        n = s.n; a, b = z[:, 0:2*n:2], z[:, 1:2*n:2]; w = s.omega_eff(a, b)
        v = torch.stack([-w * b, w * a], -1).flatten(-2)
        return torch.cat([v, torch.zeros_like(z[:, 2*n:])], -1)
    def predict(s, x0, t):
        z, _ = s.latent(s.enc(x0), t); B, T, d = z.shape
        return s.dec(z.reshape(B * T, d)).reshape(B, T, 1, H, H)

class KoopAmp(KoopCT):
    """ż_j = ω_j(r_j) J z_j,  ω_j(r) = ω_j·(1 + g_j(r²)); r dönmede korunur ⇒ akış kapalı formda kalır."""
    def __init__(s, **kw):
        super().__init__(**kw)
        s.g = nn.ModuleList([nn.Sequential(nn.Linear(1, 16), nn.Tanh(), nn.Linear(16, 1)) for _ in range(s.n)])
        for m in s.g: nn.init.zeros_(m[2].weight); nn.init.zeros_(m[2].bias)     # başlangıçta doğrusal modelle aynı
    def omega_eff(s, a, b):
        r2 = a ** 2 + b ** 2
        mod = torch.cat([s.g[j](r2[:, j:j + 1]) for j in range(s.n)], 1)
        return s.omega * (1 + mod)

class NODE(nn.Module):
    """ż = f_θ(z), RK4, adım <= hmax."""
    def __init__(s, d=12, hmax=0.125):
        super().__init__()
        s.d, s.hmax = d, hmax
        s.enc, s.dec = Enc(d), Dec(d)
        s.f = nn.Sequential(nn.Linear(d, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, d))
    def vel(s, z): return s.f(z)
    def _step(s, z, h):
        k1 = s.f(z); k2 = s.f(z + h / 2 * k1); k3 = s.f(z + h / 2 * k2); k4 = s.f(z + h * k3)
        return z + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    def latent(s, z0, t, offgrid=False):
        zs, subs, z = [z0], [], z0
        for k in range(1, t.shape[1]):
            dt = t[:, k] - t[:, k - 1]
            n = max(1, int(math.ceil(dt.max().item() / s.hmax))); h = (dt / n)[:, None]
            pick = int(torch.randint(0, max(1, n - 1), (1,))) if n > 1 else -1
            for j in range(n):
                z = s._step(z, h)
                if j == pick: subs.append(z)
            zs.append(z)
        return torch.stack(zs, 1), (torch.stack(subs, 1) if offgrid and subs else None)
    def predict(s, x0, t):
        z, _ = s.latent(s.enc(x0), t); B, T, d = z.shape
        return s.dec(z.reshape(B * T, d)).reshape(B, T, 1, H, H)
