"""Quasi-periodic (frequency-modulated) rotation: the non-periodic test of the phenomenon.

Every system in the paper so far is a periodic oscillator, so a fair objection is that the silent
inter-grid failure needs a line spectrum with harmonics. Here the glyph still rotates, but its phase
is

    theta(t) = theta_0 + omega*t + A*sin(nu*t),    nu = G*omega,  G = (sqrt(5)-1)/2 irrational,

so the instantaneous angular velocity omega + A*nu*cos(nu*t) varies by +-31% of omega and never
repeats: the trajectory is quasi-periodic, not periodic, and the observable's spectrum is a set of
Bessel sidebands at omega +- k*nu rather than a harmonic ladder. There is therefore no fixed-
frequency Koopman mode, which also makes this a second, independent test of the phase-residual gate
-- it should abstain here, as it does on the pendulum and unlike on the cylinder wake.

The phase is a closed-form function of t, so the reference stays exact at arbitrary query times,
which is what the inter-grid measurement requires. Drop-in for data.py's interface; run_qp.py
installs it and also replaces diagnostics.track, whose true-phase formula assumes pure rotation.
"""
import math, torch
from data import DEV, H, sample_times, render, test_set   # render/test_set unchanged: same glyphs

G = (5 ** 0.5 - 1) / 2          # irrasyonel: modulasyon frekansi omega ile kiyaslanamaz
A = 0.5                         # modulasyon derinligi (rad)


def phase(th0, omega, t):
    """th0 (B,), t (B,T) -> (B,T) gercek faz, her t icin kapali formda."""
    return th0[:, None] + omega * t + A * torch.sin(G * omega * t)


def video(sid, th0, omega, t):
    B, T = t.shape
    return render(sid.repeat_interleave(T), phase(th0, omega, t).reshape(-1)).reshape(B, T, 1, H, H)


def batch(B, K, mode, omega, g, noise=0.01):
    sid = torch.randint(0, 3, (B,), generator=g).to(DEV)
    th0 = (torch.rand(B, generator=g) * 2 * math.pi).to(DEV)
    t = sample_times(B, K, mode, g)
    x = video(sid, th0, omega, t)
    return x + noise * torch.randn(x.shape, generator=g).to(DEV), t


def track(R, omega, tol_deg=15):
    """diagnostics.track'in FM karsiligi: gercek faz theta_0 + omega t + A sin(nu t).
    Cekirdek surumu saf donme varsayiyor ve f_true'yu yanlis olcerdi. diagnostics cagri aninda
    import ediliyor, boylece bu modul onun oncesinde yuklenebiliyor."""
    import diagnostics
    ang, _ = diagnostics.probe(R["xh"], R["sid"])
    t, th0 = R["t"], R["th0"]
    tol = tol_deg * math.pi / 180
    mid = (torch.arange(t.shape[1], device=t.device) % int(round(1 / R["dt"]))) != 0
    cd = lambda a, b: (a - b + math.pi) % (2 * math.pi) - math.pi
    return dict(f_true=(cd(ang, phase(th0, omega, t)).abs() < tol)[:, mid].float().mean().item(),
                f_alias=float("nan"))     # tek bir alias frekansi yok
