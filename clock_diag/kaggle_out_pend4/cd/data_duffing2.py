# FAZ 4: ikinci genlik-bagimli-frekans nonlineer sistem -- sertlesen (hardening) Duffing-tipi
# acisal osilator: theta'' = -w0^2*theta - eps*theta^3 (undamped, konservatif, kuartik
# potansiyel). Piksel sarkacin (theta'' = -W0^2*sin(theta), YUMUSAYAN/softening -- genlik
# arttikca frekans DUSER) TAM TERSI yonde davranir: genlik arttikca frekans ARTAR.
#
# data_pend.py DEGISTIRILMEDI, sadece import edilip render/probe/BANK/sample_th0 AYNEN
# yeniden kullanildi (genlik araligi [0.3,1.2] rad hala [-pi,pi] icinde -- mevcut 721-noktali
# aci-bankasi (DP.BANK/DP.probe) fiziksel olarak gecerli, cunku o banka sadece render(th)'in
# th'ye gore tersine cevrilmesi, ODE'den bagimsiz). SADECE theta_at() icindeki ODE'nin sag
# tarafi degisiyor (bkz. theta_at_duffing altinda). data_pend2.py da DEGISTIRILMEDI; onun
# batch_range'i DP.video'yu (pendulum ODE) sabit cagirdigi icin, kisitli-araligi (restricted
# arm) egitimi icin AYNI mantigin Duffing karsiligi burada (sample_th0_range/batch_range)
# yeniden yazildi (data_pend2.py'nin BIREBIR kopyasi, sadece video_duffing cagiriyor).
import math, torch
import data_pend as DP

DEV = DP.DEV
render = DP.render          # degismedi, ODE'den bagimsiz (piksel cizimi th'ye gore)
probe = DP.probe            # degismedi, aci-bankasi ODE'den bagimsiz
sample_th0 = DP.sample_th0  # AYNI, [0.3,1.2] isaretli aralik, Faz 3 ile bire bir karsilastirilabilir


def theta_at_duffing(th0, t, dt=1 / 64, w0=2.4, eps=1.0):
    """RK4 ince izgara + dogrusal interpolasyon, DP.theta_at ile AYNI iskelet, tek fark ODE
    sag tarafi: f(a,b) = (b, -w0^2*a - eps*a^3). th0 (B,), t (B,T) -> (B,T).
    NOT: data_pend.py::theta_at DOKUNULMADAN birebir kopyalandi, ama BURADA (bu YENI dosyada,
    duzenlemek serbest) t, th0 ile AYNI cihaza tasiniyor -- cagiranlarin (run_pend4.py, Faz
    3'teki t1/t2 cagri deseniyle AYNI, .to(DEV) YOK) CPU tensoru dogrudan GPU'daki th0 ile
    kullanmasi durumunda Tr.gather sirasinda 'cihaz uyusmazligi' hatasi olusuyordu (Kaggle'da
    ilk kosumda gozlendi, 6/6 is hata verdi) -- data_pend.py'nin orijinal theta_at'i de AYNI
    riski tasiyor ama o dosyaya DOKUNULMUYOR; risk burada, yeni dosyada, guvenli sekilde
    kapatiliyor."""
    t = t.to(th0.device)
    n = int(math.ceil(t.max().item() / dt)) + 2
    th, om = th0.clone(), torch.zeros_like(th0); traj = [th]
    f = lambda a, b: (b, -w0 ** 2 * a - eps * a ** 3)
    for _ in range(n):
        k1 = f(th, om); k2 = f(th + dt / 2 * k1[0], om + dt / 2 * k1[1])
        k3 = f(th + dt / 2 * k2[0], om + dt / 2 * k2[1]); k4 = f(th + dt * k3[0], om + dt * k3[1])
        th = th + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]); om = om + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        traj.append(th)
    Tr = torch.stack(traj, 1)                                  # (B,n+1)
    idx = t / dt; i0 = idx.floor().long().clamp(max=Tr.shape[1] - 2); fr = idx - i0
    return Tr.gather(1, i0) * (1 - fr) + Tr.gather(1, i0 + 1) * fr


def video_duffing(th0, t, w0=2.4, eps=1.0):
    """DP.video ile AYNI, theta_at yerine theta_at_duffing cagirir."""
    from data import H
    th = theta_at_duffing(th0, t, w0=w0, eps=eps); B, T = th.shape
    return render(th.reshape(-1)).reshape(B, T, 1, H, H), th


def sample_th0_range(B, g, lo, hi):
    """data_pend2.sample_th0_range ile AYNI mantik (isaretli, |th0| lo/hi araliginda)."""
    a = lo + (hi - lo) * torch.rand(B, generator=g); s = torch.where(torch.rand(B, generator=g) < 0.5, -1.0, 1.0)
    return (a * s).to(DEV)


def batch_range(B, K, mode, g, lo, hi, w0=2.4, eps=1.0, noise=0.01):
    """data_pend2.batch_range ile AYNI, DP.video yerine video_duffing kullanir."""
    th0 = sample_th0_range(B, g, lo, hi)
    from data import sample_times
    t = sample_times(B, K, mode, g)
    x, _ = video_duffing(th0, t, w0=w0, eps=eps)
    return x + noise * torch.randn(x.shape, generator=g).to(DEV), t
