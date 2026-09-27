# Silent Inter-Grid Collapse in Linear Koopman Autoencoders

Code and result artifacts for the paper *"On the Silent Inter-Grid Collapse in Linear Koopman
Autoencoders: Empirical Diagnostics, Multi-Rate Spectral Lifting, and Its Nonlinear Limits."*

## What this is about

Linear Koopman autoencoders trained only on a regular time grid can reach a grid training loss
indistinguishable from an oracle that knows the true generator, while their predictions at
intermediate, off-grid times are 90–958&times; worse — and nothing computed on the training grid
detects it. This repository contains:

- the rotating-sprite / pixel-pendulum / two-object probes used to reproduce that failure,
- the model-agnostic diagnostics used to localize it to the latent generator rather than the
  decoder or classical aliasing,
- **multi-rate spectral lifting**, the fix: a small fraction of training windows sampled at a
  second, incommensurate rate disambiguates the harmonic fold, and a phase-residual gate locks
  only the modes that second rate certifies,
- the second scene (two independently rotating objects) and the low-period rotation-number sweep
  (q = 3, 4, 6, 8) used to check the finding and the method's reach,
- four additional robustness checks probing the design (cheaper alternatives, decoder Lipschitz
  constraints, data/rate-ratio sensitivity, non-marginally-stable generators), and
- an exploratory side investigation into extending the method to amplitude-dependent (nonlinear)
  oscillators, kept separate from the core paper's claims.

## Repository layout

```
clock_diag/
  Core method
    data.py, data_pend.py, data2.py     rendering of the rotating-sprite / pixel-pendulum / two-object probes
    models.py                           Koopman autoencoder (KoopCT, KoopAmp), Neural ODE baseline
    losses.py                           training objective
    diagnostics.py                      chain consistency, phase gain/loop closure, chart consistency,
                                         latent support ratio, timing share, f_true / f_alias readout
    dmd_init.py, pixel_dmd.py           single-rate / multi-rate DMD spectral initialization and the
                                         phase-residual gate (the core method)
    grid.py, grid_rho.py                experiment grids (training + evaluation sweeps)
    run.py, run_pend.py, run2.py        single-run entry points (sprite / pendulum / two-object)
    report*.py                          scripts that turn results.jsonl into the tables in the paper
    results.jsonl, results_merged.jsonl,
    results_merged2.jsonl               recorded outputs from the runs reported in the paper (6-seed
                                         diagnostic sweep, q = 3/4/6/8 rotation-number resolution)
    ckpt/*_portrait.npz                 latent trajectory data behind the phase-portrait figures

  Robustness checks (each isolates one design question; see table below for which
  paper table/figure each maps to)
    data_damped.py, run_damped.py       does the fix extend to decaying/growing (non-marginally-stable)
                                         generators? (it does, unmodified -- the lift depends only on
                                         each eigenvalue's argument, never its modulus)
    data_offgrid.py, run_offgrid.py     does scattered off-grid supervision substitute for a coherent
                                         second rate? (it does not -- most of the gap is closed by the
                                         spectral relock itself)
    run_lipschitz.py                    does constraining the decoder's Lipschitz constant (spectral
                                         normalization) repair inter-grid accuracy on its own? (no)
    data_sweep.py, run_sweep.py         sensitivity of the method to the mixing fraction alpha and the
                                         rate ratio beta = Delta_2/Delta_1; empirically confirms the
                                         separation bound's predicted degradation near low-denominator
                                         rational beta
    data_oneobj_wide.py, run_oneobj_wide.py, data_threeobj.py, run3.py
                                         isolate whether the two-object scene's milder control failure is
                                         driven by latent capacity, object count, or sampling regime, and
                                         test a third independent object
    run_noise.py, run_pend_noise.py     observation-noise sweep for the gate threshold (false-reject on
                                         rigid rotation, false-accept on the pendulum)
    models_robustness.py                model classes used only by the checks above (additive subclasses
                                         of models.py; the original classes are untouched)
    build_*_kernel.py                   Kaggle kernel packaging scripts used to run the corresponding
                                         experiments on GPU
    report_extended.py                  merges/reports the robustness-check results above

  Exploratory: toward an amplitude-dependent certificate (NOT part of the core paper's claims)
    synth_amp_test.py, synth_certificate_test.py
                                         synthetic testbed for a richer (Lusch et al.) training loss and
                                         for a per-trajectory, r-dependent local certificate
    dmd_init_nonlinear.py               the per-trajectory certificate itself (local_gate_residuals)
    data_pend2.py, run_pend2.py, run_pend3.py
                                         pixel-pendulum experiments testing whether the certificate,
                                         which works cleanly in the synthetic testbed, transfers to a
                                         convolutional observation model (it does not)
    data_duffing2.py, run_pend4.py      a second, independent nonlinear system (a hardening cubic
                                         oscillator, the opposite sign of the pendulum's softening law)
                                         used to check whether the certificate's failure is specific to
                                         the pendulum (it is not)
```

## Reproducing a run

```bash
cd clock_diag
python run.py --help        # single rotating-sprite training run
python run_pend.py --help   # single pixel-pendulum training run
python run2.py --help       # single two-independent-object training run
python grid.py              # sweep used for the main diagnostics table
```

The recorded raw outputs from every run reported in the paper are already checked in, so the
tables can be reproduced from data alone, without retraining. This is what each paper table
reads from and which script prints it:

| Paper table/result | Data file(s) | Command |
|---|---|---|
| Table 1 (6-seed diagnostic sweep) | `results_merged2.jsonl` | `python report.py results_merged2.jsonl` |
| ρ=1/4 anomaly + q=3/4/6/8 resolution | `results_merged2.jsonl` (same file; filter `omega_mult` ∈ {0.5, 0.75, 0.7639, 1.6667, 0.6667, 0.3333, 2.5, 2.7639} for ρ ∈ {1/4, 3/8, golden, 5/6, 1/3, 1/6, 5/4, 1+golden}) | `python report_rho.py results_merged2.jsonl` (covers 1/4, 3/8, golden, 5/4; the q=3/6 rows need the same manual filter on the raw file) |
| Table 2 (single-rate DMD init, 5 seeds) | `kaggle_out/results.jsonl` + `results_t2_moreseeds.jsonl` | `python report_extended.py t2` |
| Table 3 (multi-rate lifting, 5 seeds) | `kaggle_out_gpu/results.jsonl` + `results_t3_moreseeds.jsonl` | `python report_extended.py t3` |
| Table 4 (gate, rigid rotation) | `kaggle_out_gate/results.jsonl` | `python report_gate.py kaggle_out_gate` |
| Table 5 (pixel pendulum, 5 seeds, incl. the false accept) | `kaggle_out_partial/`, `kaggle_out_gate/`, `results_pend_moreseeds.jsonl` | `python report_extended.py t5` |
| Table 6 (two objects, 5 seeds) + three-object check | `kaggle_out_scenario2/results2.jsonl` + `results2_moreseeds.jsonl` / `results3.jsonl` | `python report_extended.py axis2` |
| Capacity-vs-object-count isolation | `results_axis1.jsonl`, `results_oneobj_wide.jsonl` | `python report_extended.py axis1` (then `axis2`) |
| Neural ODE, 5 seeds | `results_node5.jsonl` | `python report_extended.py node` |
| Gate noise sweep (rigid rotation / pendulum) | `results_noise.jsonl` / `results_pend_noise.jsonl` | `python report_extended.py noise` / `pendnoise` |
| α/β sensitivity | `kaggle_out_sweep/sweep_results.jsonl` | `python report_mr.py kaggle_out_sweep` |
| Non-marginal (damped/growing) generators | `kaggle_out_damped/damped_results.jsonl` | inspect directly (small file, no dedicated report script) |
| Decoder Lipschitz control | `kaggle_out_lipschitz/lipschitz_results.jsonl` | inspect directly |
| Off-grid-supervision ablation | `kaggle_out_offgrid/offgrid_results.jsonl` | inspect directly |
| Amplitude-dependent certificate (Appendix) | `synth_amp_results.json`, `synth_certificate_results.json`, `pend2_*.jsonl`, `pend3_results.jsonl`, `pend4_results.jsonl` | inspect directly, or rerun via `synth_amp_test.py` / `synth_certificate_test.py` / `run_pend2.py` / `run_pend3.py` / `run_pend4.py` |

Every `run*.py` / `data*.py` script is self-contained and documents in its header comment which
existing files it imports from (and never modifies) and which new ones it adds; `--help` lists
its own arguments. `kaggle_out_*/` directories hold the raw outputs (results files and logs) from
the GPU runs reported in the paper; `build_*_kernel.py` scripts show exactly how each was packaged
and launched, for anyone who wants to rerun training from scratch rather than just read the results.

Requires Python 3, PyTorch, and NumPy (see `requirements.txt`).

## Citation

If this code is useful, please cite the paper (details to be added once the preprint is posted).

## License

MIT (see `LICENSE`).
