# Ikinci senaryo: ayni 32x32 karede, iki bagimsiz donen asimetrik sprite (kendi omega/sid/th0'i ile).
# Kanit: silent inter-grid failure ve multi-rate lifting tek nesneye ozgu bir renderer kazasi degil.
import math, torch
from data import DEV, H, GX, GY, _shapes, sample_times

OFFSETS = [(-0.48, 0.0), (0.48, 0.0)]
SCALE = 0.42

def _obj(sid, ang, ox, oy):
    c, s = torch.cos(ang)[:, None, None], torch.sin(ang)[:, None, None]
    u, v = (GX - ox) / SCALE, (GY - oy) / SCALE
    S = _shapes(c * u + s * v, -s * u + c * v)
    return S.gather(0, sid[None, :, None, None].expand(1, -1, H, H))[0][:, None]

def render2(sid1, ang1, sid2, ang2):
    """all args (N,) -> (N,1,H,H) composite frame, two objects, no occlusion by construction."""
    return torch.clamp(_obj(sid1, ang1, *OFFSETS[0]) + _obj(sid2, ang2, *OFFSETS[1]), 0, 1)

def video2(sid1, th01, omega1, sid2, th02, omega2, t):
    """sid* (B,), th0* (B,), t (B,T) -> (B,T,1,H,H)"""
    B, T = t.shape
    a1 = (th01[:, None] + omega1 * t).reshape(-1)
    a2 = (th02[:, None] + omega2 * t).reshape(-1)
    s1, s2 = sid1.repeat_interleave(T), sid2.repeat_interleave(T)
    return render2(s1, a1, s2, a2).reshape(B, T, 1, H, H)

def batch2(B, K, mode, omega1, omega2, g, noise=0.01):
    sid1 = torch.randint(0, 3, (B,), generator=g).to(DEV)
    sid2 = torch.randint(0, 3, (B,), generator=g).to(DEV)
    th01 = (torch.rand(B, generator=g) * 2 * math.pi).to(DEV)
    th02 = (torch.rand(B, generator=g) * 2 * math.pi).to(DEV)
    t = sample_times(B, K, mode, g)
    x = video2(sid1, th01, omega1, sid2, th02, omega2, t)
    return x + noise * torch.randn(x.shape, generator=g).to(DEV), t

def test_set2(n, seed):
    g = torch.Generator().manual_seed(seed)
    sid1 = torch.randint(0, 3, (n,), generator=g).to(DEV)
    sid2 = torch.randint(0, 3, (n,), generator=g).to(DEV)
    th01 = (torch.rand(n, generator=g) * 2 * math.pi).to(DEV)
    th02 = (torch.rand(n, generator=g) * 2 * math.pi).to(DEV)
    return sid1, th01, sid2, th02
