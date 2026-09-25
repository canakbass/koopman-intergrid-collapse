# FAZ 2: piksel sarkacta (1) Lusch ve ark. zengin kaybi, (2) r-bagimli yerel sertifika.
# clock_diag/run_pend.py, models.py, data_pend.py DEGISTIRILMEDEN kullanilir (sadece import).
import argparse, json, math, time, torch, torch.nn.functional as F
import data, data_pend as DP, dmd_init_nonlinear as DI
from models import KoopCT, KoopAmp

def composite_loss(model, x, t, a1=1e-3, a2=1e-9, a3=1e-14):
    """synth_amp_test.py'deki composite_loss ile AYNI (Lusch,Kutz,Brunton 2018 Denklem 7a-7e,
    sarkac satiri katsayilariyla), sadece piksel tensor sekli (B,T,1,H,H) icin uyarlanmis."""
    B, T = x.shape[0], x.shape[1]
    x0 = x[:, 0]
    z0 = model.enc(x0)
    z, _ = model.latent(z0, t)
    ze = model.enc(x.reshape(B * T, 1, data.H, data.H)).reshape(B, T, -1)
    xr0 = model.dec(z0)
    Lrecon = F.mse_loss(xr0, x0)
    xhat = model.dec(z.reshape(B * T, -1)).reshape(B, T, 1, data.H, data.H)
    Lpred = F.mse_loss(xhat[:, 1:], x[:, 1:])
    Llin = F.mse_loss(ze[:, 1:], z[:, 1:])
    Linf = (xr0 - x0).abs().amax() + (xhat[:, 1] - x[:, 1]).abs().amax()
    l2 = sum((p ** 2).sum() for p in model.parameters())
    return a1 * (Lrecon + Lpred) + Llin + a2 * Linf + a3 * l2

def gen_pair(n, K, g, dt1=1.0, dt2=2 ** 0.5 / 2):
    """AYNI th0'dan iki izgarada piksel gozlem -- r-bagimli sertifika icin (bkz. synth_amp_test.gen_pair)."""
    th0 = DP.sample_th0(n, g)
    t1 = torch.arange(K + 1, dtype=torch.float32)[None].expand(n, -1) * dt1
    t2 = torch.arange(K + 1, dtype=torch.float32)[None].expand(n, -1) * dt2
    x1, _ = DP.video(th0, t1)
    x2, _ = DP.video(th0, t2)
    return x1, x2, th0.abs()

def main(a):
    torch.manual_seed(a.seed)
    Cls = KoopAmp if a.model.endswith("amp") else KoopCT
    model = Cls(seed=a.seed).to(data.DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    g = torch.Generator().manual_seed(a.seed)
    n_pre = int(a.iters * a.pretrain_frac) if a.loss == "rich" else 0
    t0 = time.time()
    for it in range(a.iters):
        x, t = DP.batch(64, 8, a.sampling, g)
        if a.loss == "rich" and it >= n_pre:
            loss = composite_loss(model, x, t)
        else:
            loss = F.mse_loss(model.predict(x[:, 0], t), x)
        opt.zero_grad(); loss.backward(); opt.step()
        if not torch.isfinite(loss):
            print(json.dumps(dict(task="pendulum2", model=a.model, loss=a.loss, seed=a.seed, diverged_at_it=it)))
            return
    model.eval(); ge = torch.Generator().manual_seed(999); out = {}
    with torch.no_grad():
        th0 = DP.sample_th0(128, ge)
        for name, sp in [("train", 1.0), ("half", 0.5), ("3_7", 3 / 7)]:
            tt = torch.arange(0, 8 + 1e-6, sp, device=data.DEV)[None].repeat(128, 1)
            xt, _ = DP.video(th0, tt); out[f"mse_{name}"] = F.mse_loss(model.predict(xt[:, 0], tt), xt).item()
        tt = torch.arange(0, 8 + 1e-6, 1 / 16, device=data.DEV)[None].repeat(64, 1)
        xt, th = DP.video(th0[:64], tt); ang = DP.probe(model.predict(xt[:, 0], tt))
        fr = torch.remainder(tt, 1.0); mid = (fr >= 0.25) & (fr <= 0.75)
        out["f_true"] = ((ang - th).abs() < math.radians(8))[mid].float().mean().item()
        # r-bagimli yerel sertifika (bkz. dmd_init_nonlinear.py). ONEMLI: TRUE |th0| ile
        # binleniyor (dogrulama/tani amacli) -- modelin KENDI kodladigi yaricap r, genel
        # olarak |th0| ile AYNI OLCEKTE DEGIL (bkz. NOTES_nonlinear.md Bolum 3/6.2, encoder
        # keyfi radyal yeniden-parametreleme yapabiliyor); residual'in KENDISI yine de sadece
        # modelin enc/omega_eff'ine bakiyor, th0'a hic erismiyor.
        gc = torch.Generator().manual_seed(1234 + a.seed)
        x1, x2, th0_abs = gen_pair(256, 8, gc)
        r, res, om_pred = DI.local_gate_residuals(model, x1, x2)
        edges = [0.3, 0.5, 0.7, 0.9, 1.1, 1.2]
        cert = []
        for j in range(model.n):
            rows = []
            for lo, hi in zip(edges[:-1], edges[1:]):
                m = (th0_abs >= lo) & (th0_abs < hi)
                if m.sum() == 0: continue
                rows.append(dict(amp_lo=lo, amp_hi=hi, n=int(m.sum()), mean_residual=float(res[m, j].mean()),
                                  mean_r_encoded=float(r[m, j].mean())))
            cert.append(rows)
        out["local_gate"] = cert
        out["local_gate_omega_pred_mean"] = om_pred.mean(0).tolist()
    rec = dict(task="pendulum2", model=a.model, loss=a.loss, sampling=a.sampling, seed=a.seed,
               learned_omegas=model.omega.detach().cpu().tolist(), W0=DP.W0, secs=round(time.time() - t0), **out)
    with open(a.out, "a") as f: f.write(json.dumps(rec) + "\n")
    print(json.dumps({k: v for k, v in rec.items() if k != "local_gate"}))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="koopamp")
    p.add_argument("--loss", default="rich", choices=["simple", "rich"])
    p.add_argument("--pretrain_frac", type=float, default=0.15)
    p.add_argument("--sampling", default="mr")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--iters", type=int, default=3000)
    p.add_argument("--out", default="pend2_results.jsonl")
    main(p.parse_args())
