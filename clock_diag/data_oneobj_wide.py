# Tek-nesne kontrolu, ama data2.py'nin render olcegi/konumuyla (SCALE, OFFSETS[0]):
# iki-nesne sahnesiyle ayni goruntu karmasikligi (nesne boyutu/yerlesimi), sadece tek hareketli
# nesne. Amac: "genisletilmis latent kapasite (n_osc=6,n_static=4) + iki nesne" ile "ayni
# kapasite + tek nesne" arasinda ayrim yapabilmek icin sahne render'ini data2 ile sabit tutmak.
# Sadece data.py / data2.py'den okur, hicbirini degistirmez.
import math, torch
from data import DEV, H, sample_times
from data2 import OFFSETS, _obj

def render1(sid, ang):
    """sid (N,), ang (N,) -> (N,1,H,H); data2'nin nesne-1 konumu/olcegi."""
    return _obj(sid, ang, *OFFSETS[0])

def video1(sid, th0, omega, t):
    """sid (B,), th0 (B,), t (B,T) -> (B,T,1,H,H)"""
    B, T = t.shape
    a = (th0[:, None] + omega * t).reshape(-1)
    s = sid.repeat_interleave(T)
    return render1(s, a).reshape(B, T, 1, H, H)

def batch1(B, K, mode, omega, g, noise=0.01):
    sid = torch.randint(0, 3, (B,), generator=g).to(DEV)
    th0 = (torch.rand(B, generator=g) * 2 * math.pi).to(DEV)
    t = sample_times(B, K, mode, g)
    x = video1(sid, th0, omega, t)
    return x + noise * torch.randn(x.shape, generator=g).to(DEV), t

def test_set1(n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 3, (n,), generator=g).to(DEV), (torch.rand(n, generator=g) * 2 * math.pi).to(DEV)
