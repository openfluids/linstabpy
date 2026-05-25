# Phase 1.5 artifacts

Diagnostic outputs produced by the `examples/case_studies/case2_lui/phase1p5_*.py` scripts.
Each script writes into this directory directly (they call
`(SCRIPT.parent / "phase1p5").mkdir(exist_ok=True)`), so the set is
append-only until pruned.

## File → generator map

### Baseflow diagnostics (`phase1p5_baseflow_spectrum.py`)

| file | purpose |
|---|---|
| `qraw_u_with_stations.png` | Qraw ū field with the 4 wall-normal stations marked |
| `qraw_u_wall_normal_spectrum.png` | wall-normal \|FFT\|² of ū at the 4 stations |
| `qraw_multi_variable_spectrum.png` | same, stacked for ρ,u,v,T |
| `qraw_u_2d_fft.png` | 2D FFT of ū — shock-ridge content visible |
| `qraw_T_2d_fft.png` | 2D FFT of T |
| `summary.txt` | numeric verdict: Qraw is not cleanly band-limited (energy above M2 Nyquist, plateau level) |

### Filter reference (`phase1p5_lele_filter_reference.py`)

Applies Lele (1992) F2/F4 compact filters to Qraw as a Qfilt stand-in.

| file | purpose |
|---|---|
| `qfilt_ref_{rho,u,v,T}_wall_normal.png` | filtered vs raw wall-normal profiles |
| `qfilt_ref_{rho,u,T}_2d_fft.png` | filtered 2D FFT spectra |
| `qfilt_ref_{rho,u,T}_shock_profile_yidx339.png` | filtered shock-normal profile at η=339 |

### Screening analysis (`phase1p5_analyze_screening.py`)

Output stem encodes the runs dir's parent (the campaign name).

| file | purpose |
|---|---|
| `screening_{campaign}_spectrum.png` | per-campaign spectrum overview |
| `screening_{campaign}_{run_label}_modes.png` | \|u\|(x,y) of the dominant modes per run |

Campaigns present:
- `evp_stride2_m2_screening_qraw` — symmetric stride (2,2) M2 (current)

### Mode classifier (`phase1p5_classify_modes.py`)

Chu energy localisation metrics + WALL_LOCAL / GLOBAL / INTERMEDIATE label.

| file | purpose |
|---|---|
| `{perf_run_stem}_chu_class.csv` | columns: `j,omega_re,omega_im,wall_frac,shock_foot_frac,global_extent,peak_x,peak_y,class` |

Present runs: `bosc_m2`, `bstat_m2`, `bstat_m1`, plus the four `probe_m2_*` sweeps.

### Classifier plots (`phase1p5_plot_classified_spectrum.py`)

| file | purpose |
|---|---|
| `{perf_run_stem}_classified_spectrum.png` | per-run spectrum colored by class |
| `combined_classified_spectrum.png` | all runs overlaid |

### Resolvent gain (`phase1p5_plot_resolvent_gain.py`)

| file | purpose |
|---|---|
| `resolvent_{campaign}_gain_curve.png` | σ₁(ω) curve |
| `resolvent_{campaign}_gain_table.txt` | tabular ω → σ₁, St |

Campaigns present:
- `resolvent_m2_qraw_scan` — coarse scan, ω ∈ [0.2, 3.0]
- `resolvent_m2_qraw_refined` — refined scan targeting GLOBAL ω-positions [0.10, 1.74]

### Resolvent mode shapes (`phase1p5_plot_resolvent_modes.py`)

| file | purpose |
|---|---|
| `case2_perf_omega_{0p40,1p60,3p00}_*_fr_modes.png` | forcing/response mode pair at sampled ω |

### Other

| file | generator | purpose |
|---|---|---|
| `qraw_gradient_vs_mode_peaks.png` | `phase1p5_gradient_vs_modes.py` | baseflow-gradient magnitude vs eigenmode peak locations |
| `global_oscillatory_candidates.png` | `phase1p5_plot_global_modes.py` | GLOBAL-classified oscillatory eigenmodes |
| `topology_probe_0p4_m6p2_modes.png` | `phase1p5_plot_modes.py` (arg-driven) | topology probe at ω=(0.4, -6.2i) |
| `qraw_osc_target_contaminated_modes.png` | one-off (worklog.md §task #11) | early M21 oscillatory-target modes showing wall-local contamination; kept as narrative evidence |

## Regenerating

All scripts are idempotent — re-running overwrites. Example:

```bash
.venv/bin/python examples/case_studies/case2_lui/phase1p5_baseflow_spectrum.py
.venv/bin/python examples/case_studies/case2_lui/phase1p5_classify_modes.py <path/to/modes.npz>
.venv/bin/python examples/case_studies/case2_lui/phase1p5_plot_resolvent_gain.py <campaign_name>
```
