# Silent Inter-Grid Collapse in Linear Koopman Autoencoders

Code and result artifacts for the paper *"On the Silent Inter-Grid Collapse in Linear Koopman
Autoencoders: Empirical Diagnostics, Multi-Rate Spectral Lifting, and Its Nonlinear Limits."*

Linear Koopman autoencoders trained only on a regular time grid can reach a grid training loss
indistinguishable from an oracle that knows the true generator, while their predictions at
intermediate, off-grid times are 90-958x worse. This repository contains the code that produces
that observation, the model-agnostic diagnostics used to localize the failure, and the multi-rate
spectral lifting method proposed to fix it on rigid rotation, plus a second scene (two independent
rotating objects) used to check the finding is not specific to the single-sprite renderer.

## Contents

```
clock_diag/
  data.py, data_pend.py, data2.py   rendering of the rotating-sprite / pixel-pendulum / two-object probes
  models.py                   Koopman autoencoder, NODE baseline
  losses.py                   training objective
  diagnostics.py              chain consistency, phase gain/loop closure, chart consistency,
                               latent support ratio, timing share, f_true / f_alias readout
  dmd_init.py, pixel_dmd.py   single-rate / multi-rate DMD spectral initialization and the
                               phase-residual gate
  grid.py, grid_rho.py        experiment grids (training + evaluation sweeps)
  run.py, run_pend.py, run2.py    single-run entry points (sprite / pendulum / two-object)
  report*.py                  scripts that turn results.jsonl into the tables in the paper
  results.jsonl, results2.jsonl, *.txt, *.log   recorded outputs from the runs reported in the paper
  ckpt/*_portrait.npz         latent trajectory data behind the phase-portrait figures
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

Requires Python 3, PyTorch, and NumPy (see `requirements.txt`).

## Citation

If this code is useful, please cite the paper (details to be added once the preprint is posted).

## Author

Hayrettin Can Akbaş, Manisa Celal Bayar Üniversitesi

## License

MIT (see `LICENSE`).
