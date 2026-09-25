# FAZ 3: kisitli-genlik-araligi orneklemesi. data_pend.py DEGISTIRILMEDI, sadece import
# edilip render/theta_at/video/DEV/W0 kullanildi -- ayni sample_th0'in lo/hi parametreli
# hali (bkz. NOTES_nonlinear.md Bolum 7, PROPOSED_addition.md Secenek B adim 1).
import torch
import data_pend as DP

W0 = DP.W0


def sample_th0_range(B, g, lo, hi):
    """data_pend.sample_th0 ile AYNI (isaretli, |th0| yarisi negatif), ama genlik [lo,hi)
    araligindan (tam [0.3,1.2] yerine)."""
    a = lo + (hi - lo) * torch.rand(B, generator=g)
    s = torch.where(torch.rand(B, generator=g) < 0.5, -1.0, 1.0)
    return (a * s).to(DP.DEV)


def batch_range(B, K, mode, g, lo, hi, noise=0.01):
    """data_pend.batch ile AYNI, sample_th0 yerine sample_th0_range(lo,hi) kullanir."""
    th0 = sample_th0_range(B, g, lo, hi)
    from data import sample_times
    t = sample_times(B, K, mode, g)
    x, _ = DP.video(th0, t)
    return x + noise * torch.randn(x.shape, generator=g).to(DP.DEV), t
