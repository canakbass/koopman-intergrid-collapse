# Piksel sarkaç: θ'' = −ω0² sin θ, durgun bırakılır (θ̇0=0), θ0 = ±U(0.3,1.2). Görüntü: çubuk + top, 32×32.
import math, torch
from data import DEV, H, GX, GY
W0 = 2.4                      # küçük genlik frekansı: ρ = W0·Δ/2π ≈ 0.382 (genlikle hafif düşer)
PIV, LR, RB, RW = (0.0, -0.7), 0.9, 0.15, 0.05

def render(th):
    """th (N,) -> (N,1,H,H)"""
    bx = PIV[0] + LR * torch.sin(th); by = PIV[1] + LR * torch.cos(th)
    X, Y = GX[None], GY[None]; bx, by = bx[:, None, None], by[:, None, None]
    bob = torch.sigmoid(-(torch.sqrt((X - bx) ** 2 + (Y - by) ** 2 + 1e-9) - RB) / 0.04)
    px, py = X - PIV[0], Y - PIV[1]; dx, dy = bx - PIV[0], by - PIV[1]
    u = ((px * dx + py * dy) / (dx ** 2 + dy ** 2)).clamp(0, 1)
    rod = torch.sigmoid(-(torch.sqrt((px - u * dx) ** 2 + (py - u * dy) ** 2 + 1e-9) - RW) / 0.03)
    return torch.maximum(bob, rod)[:, None]

def theta_at(th0, t, dt=1 / 64):
    """RK4 ince ızgara + doğrusal interpolasyon. th0 (B,), t (B,T) -> (B,T)"""
    n = int(math.ceil(t.max().item() / dt)) + 2
    th, om = th0.clone(), torch.zeros_like(th0); traj = [th]
    f = lambda a, b: (b, -W0 ** 2 * torch.sin(a))
    for _ in range(n):
        k1 = f(th, om); k2 = f(th + dt / 2 * k1[0], om + dt / 2 * k1[1])
        k3 = f(th + dt / 2 * k2[0], om + dt / 2 * k2[1]); k4 = f(th + dt * k3[0], om + dt * k3[1])
        th = th + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]); om = om + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        traj.append(th)
    Tr = torch.stack(traj, 1)                                  # (B,n+1)
    idx = t / dt; i0 = idx.floor().long().clamp(max=Tr.shape[1] - 2); fr = idx - i0
    return Tr.gather(1, i0) * (1 - fr) + Tr.gather(1, i0 + 1) * fr

def sample_th0(B, g):
    a = 0.3 + 0.9 * torch.rand(B, generator=g); s = torch.where(torch.rand(B, generator=g) < 0.5, -1.0, 1.0)
    return (a * s).to(DEV)

def video(th0, t):
    th = theta_at(th0, t); B, T = th.shape
    return render(th.reshape(-1)).reshape(B, T, 1, H, H), th

def batch(B, K, mode, g, noise=0.01):
    th0 = sample_th0(B, g)
    from data import sample_times
    t = sample_times(B, K, mode, g)                                             # regular | d2 | mr
    x, _ = video(th0, t)
    return x + noise * torch.randn(x.shape, generator=g).to(DEV), t

_A = torch.linspace(-math.pi, math.pi, 721, device=DEV)[:-1]
_B = render(_A).reshape(720, -1); _B = _B - _B.mean(1, keepdim=True); BANK = _B / _B.norm(dim=1, keepdim=True)

def probe(x):
    B, T = x.shape[:2]; v = x.reshape(B, T, -1); v = v - v.mean(-1, keepdim=True); v = v / (v.norm(dim=-1, keepdim=True) + 1e-8)
    return _A[torch.einsum("btp,ap->bta", v, BANK).argmax(-1)]
