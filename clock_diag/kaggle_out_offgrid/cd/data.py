# Dönen asimetrik sprite videoları (32×32), gerçek ω biliniyor.
import math, torch
DEV = "cuda" if torch.cuda.is_available() else "cpu"
# TF32 kapalı: sonlu-fark Jacobian kontrolü ~1e-3 bağıl hassasiyette gürültüye boğuluyordu
torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
H = 32
_lin = torch.linspace(-1, 1, H, device=DEV)
GY, GX = torch.meshgrid(_lin, _lin, indexing="ij")
TAU = 0.05

def _box(u, v, cx, cy, hx, hy):
    return torch.sigmoid(-torch.maximum((u - cx).abs() - hx, (v - cy).abs() - hy) / TAU)

def _shapes(u, v):
    L = torch.maximum(_box(u, v, -0.30, 0.0, 0.16, 0.65), _box(u, v, 0.05, 0.50, 0.50, 0.16))
    A = torch.maximum(_box(u, v, 0.0, 0.1, 0.13, 0.60), _box(u, v, 0.28, -0.40, 0.30, 0.13))
    disk = torch.sigmoid(-(torch.sqrt((u + 0.4) ** 2 + v ** 2 + 1e-9) - 0.30) / TAU)
    K = torch.maximum(torch.maximum(disk, _box(u, v, 0.25, 0.0, 0.45, 0.09)), _box(u, v, 0.55, 0.20, 0.08, 0.16))
    return torch.stack([L, A, K])

def render(sid, ang):
    """sid (N,), ang (N,) -> (N,1,H,H)"""
    c, s = torch.cos(ang)[:, None, None], torch.sin(ang)[:, None, None]
    S = _shapes(c * GX + s * GY, -s * GX + c * GY)
    return S.gather(0, sid[None, :, None, None].expand(1, -1, H, H))[0][:, None]

def video(sid, th0, omega, t):
    """sid (B,), th0 (B,), t (B,T) -> (B,T,1,H,H)"""
    B, T = t.shape
    return render(sid.repeat_interleave(T), (th0[:, None] + omega * t).reshape(-1)).reshape(B, T, 1, H, H)

DT2 = 2 ** 0.5 / 2          # Δt1=1 ile ölçülemez ikinci hız
MR_FRAC = 0.125              # karışık modda Δt2 ızgarasındaki dizi oranı

def sample_times(B, K, mode, g):
    if mode in ("d2", "mr"):
        st = torch.ones(B, K)
        rows = torch.ones(B, dtype=torch.bool) if mode == "d2" else (torch.rand(B, generator=g) < MR_FRAC)
        st[rows] = DT2
        return torch.cat([torch.zeros(B, 1), torch.cumsum(st, 1)], 1).to(DEV)
    if mode == "exp": st = (-torch.log(torch.rand(B, K, generator=g))).clamp(0.05, 3.0)
    else: st = torch.ones(B, K)
    t = torch.cat([torch.zeros(B, 1), torch.cumsum(st, 1)], 1)
    if mode in ("j5", "j20"):
        t[:, 1:] += (torch.rand(B, K, generator=g) * 2 - 1) * (0.05 if mode == "j5" else 0.20)
    return t.to(DEV)

def batch(B, K, mode, omega, g, noise=0.01):
    sid = torch.randint(0, 3, (B,), generator=g).to(DEV)
    th0 = (torch.rand(B, generator=g) * 2 * math.pi).to(DEV)
    t = sample_times(B, K, mode, g)
    x = video(sid, th0, omega, t)
    return x + noise * torch.randn(x.shape, generator=g).to(DEV), t

def test_set(n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 3, (n,), generator=g).to(DEV), (torch.rand(n, generator=g) * 2 * math.pi).to(DEV)
