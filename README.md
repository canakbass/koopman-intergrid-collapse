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

  Four robustness checks (see NOTES_robustness.md for full results)
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
    models_robustness.py                model classes used only by the four checks above (additive
                                         subclasses of models.py; the original classes are untouched)
    build_{damped,offgrid,lipschitz,sweep,q36,scenario2,moreseeds,pend4}_kernel.py
                                         Kaggle kernel packaging scripts used to run the corresponding
                                         experiments on GPU

  Exploratory: toward an amplitude-dependent certificate (NOT part of the core paper's claims;
  see NOTES_nonlinear.md and PROPOSED_addition.md for the full writeup)
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
python report.py            # prints the corresponding table from results.jsonl
```

The four robustness checks and the exploratory certificate experiments are single-file, self-contained
scripts with a `--help` flag; each is documented in `NOTES_robustness.md` / `NOTES_nonlinear.md`
alongside the raw, seed-by-seed results it produced. `kaggle_out_*/` directories hold the recorded
outputs (results files and logs) from the GPU runs reported in the paper.

Requires Python 3, PyTorch, and NumPy (see `requirements.txt`).

## Citation

If this code is useful, please cite the paper (details to be added once the preprint is posted).

## License

MIT (see `LICENSE`).
