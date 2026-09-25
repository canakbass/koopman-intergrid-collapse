# Öneri: Faz 1+2+3 nonlineer araştırmasının makaleye olası katkısı

Bu dosya `paper/` veya `paper_l4dc/`'ye eklenmedi — sadece bir taslak/öneri. Tam analiz:
`clock_diag/NOTES_nonlinear.md`.

## Durum özeti (Faz 3 sonrası, GÜNCEL)

- Makaledeki mevcut cümle ("An amplitude-dependent generator does not close the gap
  either", `paper/sections/04_experiments.tex`) hâlâ **doğru ve geçerli** — değişmedi.
- Faz 1+2 araştırması bu cümlenin **neden** doğru olduğuna dair literatür-destekli, çok
  daha net bir açıklama üretti, ve r-bağımlı yerel sertifika fikrini **sentetik ortamda
  güçlü şekilde kanıtladı** (full_range: r=0.808, restricted: r=0.996 korelasyon,
  16-125x ekstrapolasyon sıçraması).
- **Faz 3 (TAMAMLANDI, bkz. NOTES Bölüm 7): piksel veride, DOĞRU deney tasarımıyla
  (full_range vs restricted eğitim, en az 3 tohum, sabit-tohumlu 256 değerlendirme
  yörüngesi, gerçek |θ0| ile binleme) test edildi. SONUÇ: DÜRÜST NEGATİF.** Sertifika
  residual'i piksel veride ekstrapolasyon bölgesini işaretlemiyor (sıçrama 1.37x, aynı
  boyutta bir "sıçrama" full_range'in kendi gürültü tabanında da var — 1.06-1.75x) ve
  residual ile gerçek frekans hatası arasındaki korelasyon **6/6 tohumda tutarlı şekilde
  NEGATİF** (full: -0.145/-0.259/-0.217, restricted: -0.477/-0.434/-0.120; havuzlanmış:
  full r=-0.156, restricted r=-0.268) — sentetikteki güçlü pozitif korelasyonun (0.808,
  0.996) tam tersi. Model genelleme farkı GERÇEK (ham MSE'de restricted ~100-200x daha
  kötü genelliyor), ama sertifika bu farkı doğru yönde işaretlemiyor. Aşağıdaki Seçenek B
  artık **denendi ve olumsuz sonuçlandı** (Seçenek A da buna göre güncellendi).

## Seçenek A (GÜNCELLENDİ, Faz 3 sonucuyla): Limitations'a kısa bir ek paragraf

`paper/sections/05_limitations.tex`'teki "The method assumes a fixed-frequency generator"
paragrafının hemen ardına, önerilen ek (İngilizce, makalenin diliyle tutarlı; Faz 3'ün
gerçek, doğru-tasarımlı piksel sayılarıyla güncellendi):

```latex
\paragraph{A path toward amplitude-dependent certification.}
A literature check situates this gap: Lusch et al.~\citep{lusch2018deep} train an
auxiliary network conditioning frequency on latent radius, in the same spirit as our
$\omega_j(r)$ attempt, and succeed on the full nonlinear pendulum (test error $1.1\times10^{-7}$)
with a low-dimensional state input, a richer loss (reconstruction, latent-space linearity,
and $L_\infty$ terms), and no multi-rate locking.
Reproducing their loss in our short-window regime did not help: $11$--$30\times$ worse
than our single-term loss on a controlled synthetic system (3 seeds, with vs.\ without a
short pretraining phase), and $2.9$--$3.4\times$ worse on the pixel pendulum (3 seeds),
suggesting their weighting does not transfer directly to short training horizons regardless
of observation modality.
More promisingly, because $r_j$ is exactly conserved along a KoopAmp trajectory, the
multi-rate residual of \cref{eq:lift} can be evaluated \emph{per trajectory} rather than
globally, yielding a local certificate indexed by amplitude.
On a controlled synthetic system with a known polynomial frequency-amplitude law, this
local residual rises $16$--$125\times$ when queried outside the amplitude range seen in
training, correlating with the true frequency error at $r=0.996$ across held-out
trajectories -- to our knowledge, no existing method offers such a certificate for
amplitude-dependent Koopman modes.
This does not transfer to the pixel pendulum: training two matched arms (amplitude
$\in[0.3,1.2]$ rad vs.\ a restricted $[0.3,0.7]$ rad range, 3 seeds each) and evaluating
the same certificate on 256 held-out trajectories spanning the full range, the residual
shows no reliable extrapolation signal -- the restricted arm's residual rises only
$1.37\times$ outside its training range, statistically indistinguishable from the
$1.06$--$1.75\times$ bin-to-bin noise floor observed even for the arm trained on the full
range -- and correlates \emph{negatively} with the true frequency error in all 6 runs
($r=-0.12$ to $-0.48$, pooled $r=-0.16$ and $r=-0.27$), the opposite sign from the
synthetic result, even though the restricted arm's raw generalization gap is real and large
(held-out MSE $100$--$200\times$ worse than the full-range arm).
We attribute this to the pixel CNN encoder's amplitude-radius representation not
generalizing the way the synthetic MLP encoder's did (\cref{sec:synth-certificate});
extending this certificate to convolutional observation models is an open problem.
```

Not: Bu paragraf artik "henuz test edilmedi" DEGIL, "test edildi ve piksel veride bu
haliyle calismiyor" diyor -- makalenin "biz her seyi durustce raporluyoruz, abartmiyoruz"
tonuyla tam tutarli: sentetik kanit (guclu, pozitif) ile piksel sonucu (durust, negatif)
ayni agirlikta, cekismeden veriliyor.

`paper_l4dc/main.tex` için (yer sınırlı, tek cümlelik versiyon), mevcut "Extending
identification to amplitude- or energy-dependent spectra... is open" cümlesinin sonuna
eklenebilir:

```latex
A per-trajectory version of the residual, exploiting the fact that $r_j$ is conserved
along a trajectory, shows a $16$--$125\times$ extrapolation signal ($r=0.996$ with true
error) on a controlled synthetic system, but a matched restricted-range experiment on the
pixel pendulum (3 seeds, 256 held-out trajectories) found no such signal ($r=-0.27$ pooled),
so this certificate does not yet transfer to convolutional observation models.
```

## Seçenek B (TAMAMLANDI -- Faz 3, bkz. `NOTES_nonlinear.md` Bölüm 7)

~~Yapılması gereken~~ Yapıldı (yeni dosyalar `clock_diag/data_pend2.py`,
`clock_diag/run_pend3.py`; mevcut kod DEĞİŞTİRİLMEDİ):
1. ~~`data_pend.py`'ye kısıtlı-genlik-aralığı eğitim modu eklemek~~ → `data_pend2.py::
   sample_th0_range(B,g,lo,hi)` eklendi (yeni dosya, ayrı).
2. ~~En az 3 tohum, hem full_range hem restricted kolu~~ → 3 tohum × 2 kol = 6 koşu,
   TAMAMI tamamlandı (hiçbiri diverge etmedi/timeout'a takılmadı, 282-302sn/koşu).
3. ~~Sertifikayı her ikisine de uygulayıp karşılaştırmak~~ → yapıldı, 256 sabit-tohumlu
   değerlendirme yörüngesiyle, gerçek |θ0| ile binlenerek (Bölüm 6.3'teki kodlanmış-r
   hatası TEKRARLANMADI).
4. **Sonuç: sinyal ZAYIF/GÜRÜLTÜYE GÖMÜLÜ değil, tersine YANLIŞ YÖNDE (negatif korelasyon,
   6/6 tohumda tutarlı).** Bu "belirsiz" değil, açık bir negatif sonuç — Seçenek A'nın
   "denedik, çalışmadı" çerçevesi (yukarıda güncellenmiş haliyle) doğru çerçeve.
5. Paper'a Deneyler bölümüne yeni bir alt-bölüm olarak EKLENMESİ önerilmiyor (sonuç
   negatif) — Limitations'daki kısa paragraf (Seçenek A, güncellendi) yeterli ve dürüst.

Gerçekleşen süre: kod yazma + smoke test ~20dk, 6 koşu toplam ~30dk (paralel değil,
sıralı — CPU, 12 çekirdek, tek seferde 1 koşu; tahmin edilenden hızlı çünkü hiçbir tohum
Faz 1'deki gibi patolojik yavaşlamadı), analiz ~10dk. Toplam ~1 saat, tahmin edilen
aralıkla (1-2 saat) tutarlı.

## Kararı bekleyen sorular (GÜNCEL)

- Seçenek A'nın güncellenmiş (Faz 3 sonuçlu) paragrafı `paper/sections/05_limitations.tex`'e
  eklensin mi? (Düşük risk, dürüst; kullanıcının onayı bekleniyor — bu dosya hâlâ paper/'a
  YAZILMADI, sadece taslak.)
- `paper_l4dc/main.tex`'e de eklensin mi, yoksa sadece arXiv (uzun) sürüm mü?
- Alternatif: bu tüm nonlineer-sertifika hattı (Faz 1-3) hiç makaleye eklenmeden, ayrı bir
  gelecekteki tez/çalışma notunda bırakılabilir — mevcut paper'ın kapsamı dışında kalan,
  kendi başına tutarlı bir negatif-sonuç araştırması olarak `NOTES_nonlinear.md`'de duruyor.
