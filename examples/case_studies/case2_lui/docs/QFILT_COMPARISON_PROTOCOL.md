# Qfilt comparison protocol

When the upstream `Qfilt` dataset arrives, run the same experiments
that `phase1p5_*` already ran on `Qraw`, then produce a direct
comparison. This file documents the exact protocol so the campaign
stays honest and reproducible.

## Prerequisites

1. `Qfilt` arrives as a dataset sibling to `data/baseflow/` — e.g.
   `examples/case_studies/case2_lui/data/baseflow_filt/` — with identical shape `(599, 679)`
   (pre-transpose), same grid, same `sponge.npy`.
2. Record the filter provenance inside `data/baseflow_filt/`:
   - source (upstream pipeline, filter name, α value, number of
     passes);
   - author and date;
   - the raw dataset it was produced from.

## Protocol — same six runs, parallel columns

### Track A — mean-flow diagnostic

Run these scripts, first on `Qraw` (done), then on `Qfilt`:

```bash
# Already done for Qraw:
.venv/bin/python examples/case_studies/case2_lui/scripts/phase1p5_baseflow_spectrum.py

# To run for Qfilt (after copy/point paths):
LINSTABPY_QFILT_DIR=examples/case_studies/case2_lui/data/baseflow_filt \
  .venv/bin/python examples/case_studies/case2_lui/scripts/phase1p5_baseflow_spectrum.py
#  (script would need a --case-dir flag added; currently hard-coded)
```

Artefacts to compare side-by-side:

- `artifacts/phase1p5/qraw_u_2d_fft.png` vs `qfilt_u_2d_fft.png` — expect
  the diagonal shock ridges and corners to be visibly darker under
  Qfilt.
- `artifacts/phase1p5/qraw_u_with_stations.png` vs `qfilt_u_with_stations.png`
  — expect thicker shocks (visible spatial smoothing) and preserved
  bulk structure.
- Gradient-peak table (values of `max|∂Ū/∂x|`, `max|∂Ū/∂y|`):
  document both directly.

### Track B — EVP contamination

Exactly the same three EVP configs, pointed at `data/baseflow_filt/`:

1. `evp_stride2x1_oscillatory_target_ncv500_tol1e4.jsonc` — the
   original "row 5" reproduction. 2 modes at `eps_tol=1e-4`.
2. `evp_stride2x1_stationary_target_ncv500_tol1e4.jsonc` — nev=2 at
   stationary target, same tolerance.
3. `evp_stride2x1_m21_screening_qraw.jsonc` (rename to
   `...screening_qfilt.jsonc`) — the PLAN Phase 2 screening pair,
   20 modes at `eps_tol=5e-3` per target.

For each run, score with:

```bash
.venv/bin/python examples/case_studies/case2_lui/scripts/phase1p5_score_run.py <npz>
.venv/bin/python examples/case_studies/case2_lui/scripts/phase1p5_plot_spectrum.py \
    "Qraw osc"=<qraw_osc_npz> "Qfilt osc"=<qfilt_osc_npz>
.venv/bin/python examples/case_studies/case2_lui/scripts/phase1p5_plot_modes.py <qfilt_osc_npz>
.venv/bin/python examples/case_studies/case2_lui/scripts/phase1p5_gradient_vs_modes.py <qfilt_osc_npz>
```

### Track C — direct comparison figure

Produce one figure that tells the Qfilt story in one glance:

- `artifacts/phase1p5/qraw_vs_qfilt_spectrum.png` — two-panel: Qraw spectrum
  (target + wall-local cluster) | Qfilt spectrum (target + whatever
  remains).
- `artifacts/phase1p5/qraw_vs_qfilt_mode_peak_shift.png` — mode 0 and mode 1
  peak locations on the `(x, y)` domain for both fields; expect the
  peaks to either shift away from the shock foot / wall or weaken
  substantially.

## Success / failure criteria

**Qfilt is a clear win if:**

- Screening run (nev=20) returns ≥ 1 mode with `|Re ω| ≥ 0.2` for the
  oscillatory target, passing the PLAN osc-branch gate.
- Returned modes have peak locations DIFFERENT from Qraw's shock-foot
  and recovery-wall locations.
- Mean-flow gradient peaks: `max|∂Ū/∂x|` reduced by ≥ 50%,
  `max|∂Ū/∂y|` either reduced or unchanged (if unchanged because of
  Dirichlet BC preservation upstream, note it and move on).

**Qfilt is a partial win if:**

- Oscillatory gate still fails (no `|Re ω|` mode) but mode peaks
  shift and Mode 0 family moves off the shock foot — indicates the
  local-gradient attractor is weakened but the physical oscillatory
  branch still isn't attracted enough.
- Action in that case: re-localise target (PLAN Phase 3), sweep
  target grid around plausible oscillatory seeds under γ=1.31.

**Qfilt is a neutral/negative if:**

- Returned modes look identical to Qraw's. Then the `Qfilt` upstream
  pipeline is not reducing the relevant mean-flow-gradient sources.
  Action: request filter provenance details, compare gradient peaks
  side-by-side with the Qraw diagnostic from Entry 6 of `worklog.md`.

## Reference numbers from Qraw (Entry 5 of worklog)

- Qraw M21 osc-target returned `ω = 0 - 0.683i` and `0 - 1.580i`.
- Postcheck residual median: 5.7e-5 (passed tol gate).
- Target distance median: 0.72 (failed 0.1·|ω_target|=0.125 gate).
- Max |Re ω|: 0.000 (failed osc-branch gate).
- Mode 0 peak at shock foot (x=0.818, y=-0.099). Coincides with
  `|∂Ū/∂x|_max` at (x=0.815, y=-0.092).
- Mode 1 peak at recovery wall (x=0.962, y=-0.150). Near
  `|∂Ū/∂y|_max` along the wall.

Anything from Qfilt must beat these to be worth promoting.
