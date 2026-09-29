"""Cylinder-flow data source for the multi-rate experiments (HANDOFF 5-B).

The field comes from cyl_period.npz: a least-squares Fourier fit, in shedding phase, of the
vorticity field of a lattice-Boltzmann simulation at Re=100 on its limit cycle (cyl_lbm.py).
render() evaluates that series, so the reference is exact at any t and not only at simulation
steps -- which is what an inter-grid measurement needs. sid is accepted and ignored: there is one
flow and a sequence is identified by its initial phase alone, as in the single-object setting
where the method works.

Time convention is data.py's: t is in units of the training grid (Delta=1) and run.py sets
omega = omega_mult*pi, so rho = omega_mult/2 exactly as in Table 1. The simulation's period only
fixes the SHAPE of the field as a function of phase; the rate at which the phase advances is ours
to choose. The harmonics of the shedding frequency live in the field itself -- that is the
mechanism the paper is about, now coming from a PDE rather than from a rotating sprite.

Drop-in for data.py's render/video/batch/test_set, so run_cyl.py can hand it to the unmodified
diagnostics.py and the numbers stay comparable with the paper's tables.
"""
import math, os, numpy as np, torch
from data import DEV, H, sample_times

_P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cyl_period.npz")
if not os.path.exists(_P):
    raise FileNotFoundError(f"{_P} yok: once 'python cyl_lbm.py' calistir")
_d = np.load(_P)
M = int(_d["harmonics"])
# A0 (zaman-ortalamasi alan) atiliyor: dalgalanma alani uzerinde calisiliyor. Gerekce olculdu:
# ham alanda oracle 800 iterasyonda train_mse=0.00582'ye iniyor, bu da faz-ortalamasi alani
# tahmin etmenin MSE'siyle uc anlamli haneye kadar ayni -- yani model yalnizca zaman-ortalamasini
# ogrenmis oluyor. Ortalama atilinca ayni konfigurasyon 1e-4'e (kendi temel cizgisi 0.01359'un
# 136 kati altina) iniyor ve f_true=1.00 oluyor. Ortalamasi cikarilmis anlik goruntu ayrica
# silindir akisina DMD/Koopman uygulayan literaturun standardi.
_C = torch.tensor(_d["coef"][1:], dtype=torch.float32, device=DEV)      # (2M, H*H), sabit terim yok
T_SIM, RESID = float(_d["T"]), float(_d["resid"])


def _field(ang):                    # ang (N,) -> (N,H,H), faz serisinin degerlendirilmesi
    b = []
    for m in range(1, M + 1): b += [torch.cos(m * ang), torch.sin(m * ang)]
    return (torch.stack(b, 1) @ _C).reshape(-1, H, H)


with torch.no_grad():               # olcek: bir periyot boyunca robust aralik -> [0,1], sprite verisiyle ayni olcek
    _s = _field(torch.arange(256, dtype=torch.float32, device=DEV) * (2 * math.pi / 256)).flatten()
    LO, HI = [float(v) for v in torch.quantile(_s, torch.tensor([0.005, 0.995], device=DEV))]


def render(sid, ang):
    """sid (N,) yok sayilir, ang (N,) -> (N,1,H,H)"""
    return ((_field(ang) - LO) / (HI - LO)).clamp(0, 1)[:, None]


def video(sid, th0, omega, t):
    """th0 (B,), t (B,T) -> (B,T,1,H,H)"""
    B, T = t.shape
    return render(sid, (th0[:, None] + omega * t).reshape(-1)).reshape(B, T, 1, H, H)


def batch(B, K, mode, omega, g, noise=0.01):
    sid = torch.zeros(B, dtype=torch.long, device=DEV)
    th0 = (torch.rand(B, generator=g) * 2 * math.pi).to(DEV)
    t = sample_times(B, K, mode, g)
    x = video(sid, th0, omega, t)
    return x + noise * torch.randn(x.shape, generator=g).to(DEV), t


def test_set(n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.zeros(n, dtype=torch.long, device=DEV), (torch.rand(n, generator=g) * 2 * math.pi).to(DEV)
