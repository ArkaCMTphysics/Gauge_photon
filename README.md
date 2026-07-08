# Gauge_photon

Code and figures for **"A universal gauge structure in the photon content of light–matter systems"** (Arka Dutta and Marcus Kollar, University of Augsburg).

The paper shows that across the gauge family of a truncated light–matter Hamiltonian, the ground-state photon number is exactly a parabola in the gauge parameter,

```
⟨a†a⟩_λ = ⟨a†a⟩_0 + C λ + Q λ²,      Q = ⟨G²⟩,
```

with an invariant minimum and a curvature fixed by the fluctuation of the operator that couples to the field. This is established for the quantum Rabi model (exact solution via tunable coherent states) and confirmed for the Dicke model (exact diagonalization), where the curvature is shown to be a diagnostic of the diamagnetic (Thomas–Reiche–Kuhn) term that also forbids superradiance.

This repository contains the numerical code used to produce every figure in the paper, so the results can be independently reproduced.

## Contents

```
src/
  make_fig1.py   generates the schematic overview figure (00_gauge_orbit_overview):
                 the gauge orbit of the photon dressing, with phase-space insets
                 showing the field being squeezed along the orbit
  makefigs.py    generates Figs. 1-14 (quantum Rabi model: spectra, exact
                 diagonalization, tunable coherent-state coefficients and
                 photon-number cross-checks)
  dicke_gi.py    builds the gauge-invariant Dicke Hamiltonian and runs
                 exact diagonalization (Garziano et al. resummed form)
  dicke_figs.py  generates Figs. 15-19 (Dicke gauge parabolas, curvature,
                 discriminant, gauge of minimal photon content, and invariant
                 minimal photon number vs. coupling), using dicke_gi.py
figs/
  00_*.pdf, 01_*.pdf ... 19_*.pdf   the resulting figures, numbered as in the
                          paper (PNG copies included alongside each PDF)
```

## Reproducing the figures

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd src
python3 make_fig1.py    # -> ../figs/00_gauge_orbit_overview.*
python3 makefigs.py     # -> ../figs/01_*.pdf ... 14_*.pdf
python3 dicke_figs.py   # -> ../figs/15_*.pdf ... 19_*.pdf
```

Each run prints a diagnostic line per figure (and, for the Dicke script, an explicit `Ncut` convergence check) so the numbers behind each plot are visible, not just the plot itself.

## Notes on reproducibility

- The quantum Rabi model figures use exact diagonalization at a fixed photon cutoff (`Nf=160` by default, stated per-figure in the code) and, for the coefficient-decay figure (`07_cn_decay`), 80-digit arbitrary-precision arithmetic via `mpmath` — standard double precision cannot resolve the coefficient magnitudes shown there (they reach ~$10^{-25}$).
- The Dicke figures use an adaptive photon cutoff, $N_\mathrm{cutoff}(g) = \max(24,\, \lfloor 20 + 34\,g/\omega_c \rfloor)$, and `dicke_figs.py` prints a convergence check (cutoff vs. cutoff+14) confirming results are stable to machine precision.
- Figures 10 and 11 (computational-cost scaling) plot a pre-recorded set of wall-clock timings and their fit; wall-clock time is machine-specific, so these are reference data rather than something meant to be regenerated bit-for-bit on a different machine.

## Requirements

See `requirements.txt`. Tested with the versions listed there; the code has no other dependencies.

## Citation

The associated paper is currently in preparation for submission. This section will be updated with a DOI/arXiv reference once available.

## License

Code is released under the MIT License (see `LICENSE`). Figures are derived outputs of this code and may be reused with attribution to the paper above.
