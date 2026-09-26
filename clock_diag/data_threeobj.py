# Bonus (Gorev A): uc bagimsiz donen nesne, data2.py'nin AYNI _obj/_shapes altyapisiyla.
# Uc ayrik bolge (ust-orta, sol-alt, sag-alt), cakisma yok. data.py/data2.py'yi DEGISTIRMEZ, sadece kullanir.
import math, torch
from data import DEV, H, sample_times
from data2 import _obj

OFFSETS3 = [(0.0, 0.55), (-0.5, -0.35), (0.5, -0.35)]
SCALE3 = 0.34

def _obj3(sid, ang, ox, oy):
    from data import GX, GY, _shapes
    c, s = torch.cos(ang)[:, None, None], torch.sin(ang)[:, None, None]
    u, v = (GX - ox) / SCALE3, (GY - oy) / SCALE3
    S = _shapes(c * u + s * v, -s * u + c * v)
    return S.gather(0, sid[None, :, None, None].expand(1, -1, H, H))[0][:, None]

def render3(sids, angs):
    """sids, angs: lists of 3 tensors (N,) -> (N,1,H,H) composite frame."""
    out = 0
    for k in range(3):
        out = out + _obj3(sids[k], angs[k], *OFFSETS3[k])
    return torch.clamp(out, 0, 1)

def video3(sids, th0s, omegas, t):
    """sids/th0s/omegas: lists of 3 (B,) tensors/scalars; t (B,T) -> (B,T,1,H,H)"""
    B, T = t.shape
    angs = [(th0s[k][:, None] + omegas[k] * t).reshape(-1) for k in range(3)]
    sids_r = [sids[k].repeat_interleave(T) for k in range(3)]
    return render3(sids_r, angs).reshape(B, T, 1, H, H)

def batch3(B, K, mode, omegas, g, noise=0.01):
    sids = [torch.randint(0, 3, (B,), generator=g).to(DEV) for _ in range(3)]
    th0s = [(torch.rand(B, generator=g) * 2 * math.pi).to(DEV) for _ in range(3)]
    t = sample_times(B, K, mode, g)
    x = video3(sids, th0s, omegas, t)
    return x + noise * torch.randn(x.shape, generator=g).to(DEV), t

def test_set3(n, seed):
    g = torch.Generator().manual_seed(seed)
    sids = [torch.randint(0, 3, (n,), generator=g).to(DEV) for _ in range(3)]
    th0s = [(torch.rand(n, generator=g) * 2 * math.pi).to(DEV) for _ in range(3)]
    return sids, th0s
