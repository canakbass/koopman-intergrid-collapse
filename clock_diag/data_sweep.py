# YENİ dosya (Görev D, bkz. NOTES_robustness.md). data.py'ye DOKUNULMADI, sadece import edilip
# sample_times/batch ŞABLON alınarak MR_FRAC (α, karışım oranı) ve DT2 (β=Δt2/Δt1, ikinci hız)
# PARAMETRE olarak açılıyor -- mevcut data.py'de bu ikisi sabit (MR_FRAC=0.125, DT2=√2/2).
import math, torch
import data
from data import DEV


def sample_times_sweep(B, K, mode, g, mr_frac, dt2):
    """data.sample_times ile AYNI (mode='regular'/'exp'/'j5'/'j20' için DOĞRUDAN data.sample_times'a
    devrediyor, DEĞİŞTİRİLMEDEN); mode in {'d2','mr'} için mr_frac/dt2 PARAMETRE."""
    if mode in ("d2", "mr"):
        st = torch.ones(B, K)
        rows = torch.ones(B, dtype=torch.bool) if mode == "d2" else (torch.rand(B, generator=g) < mr_frac)
        st[rows] = dt2
        return torch.cat([torch.zeros(B, 1), torch.cumsum(st, 1)], 1).to(DEV)
    return data.sample_times(B, K, mode, g)


def batch_sweep(B, K, mode, omega, g, mr_frac, dt2, noise=0.01):
    """data.batch ile AYNI arayüz, tek fark sample_times_sweep (render data.video, DEĞİŞTİRİLMEDEN)."""
    sid = torch.randint(0, 3, (B,), generator=g).to(DEV)
    th0 = (torch.rand(B, generator=g) * 2 * math.pi).to(DEV)
    t = sample_times_sweep(B, K, mode, g, mr_frac, dt2)
    x = data.video(sid, th0, omega, t)
    return x + noise * torch.randn(x.shape, generator=g).to(DEV), t
