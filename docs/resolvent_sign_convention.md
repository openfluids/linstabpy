# Resolvent sign convention

This page states which sign the resolvent uses for the forcing frequency,
why it changed, what the change does to results you already have, how to
reproduce an old result, and how to convert one.

Read the summary. Read the effect table before you reuse an old gain map or
an old mode.

## Summary

- The resolvent is now `H(omega) = [(gamma - i*omega) I - L0]^-1`.
  It is the response to a forcing `f * exp(-i*omega*t)`.
  This is the same time convention as the stability solver (`omega = eig(i*L0)`,
  `q ~ exp(-i*omega*t)`).
- Before the change the code inverted `L0 - (gamma + i*omega) I`.
  That is the LinStab2D (MATLAB) matrix. It equals `-H(-omega)`.
- Gains at one frequency are unchanged for a flow with `W = 0` (spanwise or
  azimuthal base velocity), including `beta != 0`. They change for a flow with
  `beta != 0` and `W != 0`: the gain map is mirrored in `omega`.
- Forcing and response modes change in every case. They are conjugated, and
  for `beta != 0` the spanwise velocity component also changes sign.
- The old behaviour is one setting away:
  `solver.resolvent_sign_convention: matlab_parity`.
- The change is listed under "Changed (breaking)" in `CHANGELOG.md`.

## The convention

Notation: `L0` is the assembled operator, with `dq/dt = L0 q`. `omega` is the
real forcing frequency. `gamma >= 0` is the discount. `B` and `C` are the input
and output masks and `F` is the optional filter. `Wc` is the Chu energy weight.
`W` alone always means the spanwise (azimuthal) component of the base flow
velocity.

| Convention | Matrix that is inverted | Setting |
|------------|-------------------------|---------|
| `physical` (default) | `(gamma - i*omega) I - L0` | none |
| `matlab_parity` | `L0 - (gamma + i*omega) I` | `solver.resolvent_sign_convention: matlab_parity` |

The leading gains come from the weighted normal problem `invWc H^H Wc H v = sigma^2 v`
for `H = C F inv(matrix) F B`, with `Wc` in place of `W`. The discount `gamma` shifts the real part of the
spectrum in the same stabilising direction in both conventions.

## Derivation

1. The stability solver uses `q ~ exp(-i*omega*t)`, so `dq/dt = -i*omega*q`.
2. Force the equation `dq/dt = L0 q + f` with `f = f_hat * exp(-i*omega*t)`.
   Then `-i*omega q_hat = L0 q_hat + f_hat`.
3. Solve for the response: `q_hat = (-i*omega I - L0)^-1 f_hat`.
4. With a discount, use `dq/dt = (L0 - gamma I) q + f`. The result is
   `q_hat = [(gamma - i*omega) I - L0]^-1 f_hat`.

Martini and Schmidt (2024, TCFD, eq. 22) and Towne et al. (2024, eq. 9 to 10 and
37 to 39) write the same resolvent for this time convention.

The old matrix has `+i*omega` and the opposite overall sign. Put `omega -> -omega`
in the new matrix and change the sign: `-[(gamma + i*omega) I - L0] = L0 - (gamma + i*omega) I`.
So, for every operator and every `omega`:

```
H_old(omega) = -H_new(-omega)
```

This one relation gives every row of the effect table.

## What the change does

Singular values do not change when a matrix changes sign, so the gain at `omega`
in the old code is the gain at `-omega` in the new code:

```
gain_new(omega) = gain_old(-omega)
forcing_new(omega) = forcing_old(-omega)
response_new(omega) = - response_old(-omega)      (relative sign between forcing and response)
```

When the spectrum of `L0` is symmetric under `omega -> -conj(omega)`, the gain
map is even in `omega` and `gain_old(-omega) = gain_old(omega)`. That is the
case for a real `L0` and for `beta != 0` with `W = 0`. It is not the case for
`beta != 0` with `W != 0`, nor for Floquet and swirl cases.

### Effect table

| Case | Gain at a frequency | Gain map over `omega` | Forcing and response modes |
|------|---------------------|-----------------------|----------------------------|
| `beta = 0`, real `L0` | unchanged | even, unchanged | new = complex conjugate of old; the response also changes sign relative to the forcing |
| `beta != 0`, `W = 0` | unchanged | even, unchanged | new = complex conjugate of old with the sign of the spanwise (azimuthal) velocity component reversed; the response also changes sign relative to the forcing |
| `beta != 0`, `W != 0`; Floquet; swirl | `gain_new(omega) = gain_old(-omega)` | mirrored about `omega = 0` | `forcing_new(omega) = forcing_old(-omega)`; `response_new(omega) = -response_old(-omega)` (the response changes sign relative to the forcing) |

Modes are defined up to a unit complex factor. The statements about modes hold
up to that factor. The relative sign between a forcing and its response is not
removed by that factor.

The case `beta != 0`, `W = 0` follows from this: complex conjugation of `L0(beta)`
gives `L0(-beta)`, and reversing the spanwise coordinate maps `L0(-beta)` back
to `L0(beta)` while it changes the sign of the spanwise velocity component.
The tests below confirm it.

### Wave direction

Rebuild a field as `Re[q_hat(x) exp(-i*omega*t)]`. Where the old modes were the
conjugates of the new ones (the first two rows of the table), a wave that travels
downstream (`d arg(q_hat)/dx > 0`) came out as an upstream wave. With the new modes the wave travels in the
correct direction. This is the reason to use the new default for any
visualisation or phase argument.

### Poles and the link to the stability solver

The resolvent matrix is singular at `omega = omega_eig`, where
`omega_eig = eig(i*L0)` is a stability-solver eigenvalue (with `gamma = 0`).
Before the change the pole was at `omega = -omega_eig`. Two consequences:

- A real eigenvalue part `Re(omega_eig)` now gives a gain peak at
  `omega = +Re(omega_eig)`. Before, the peak was at `-Re(omega_eig)`.
- `analysis.omega_target` now selects one frequency for both the shift-invert
  eigenvalue search and the resolvent. The two refer to the same frequency.

`solvers/router/band_marking.py` uses the pole at `+omega_eig`, and it centres its
eigenvalue search at the median of the requested frequencies. Under
`matlab_parity` it uses `-omega_eig` and the negated median, as before.

If your scripts negate the frequency window to follow an eigenvalue, remove that
step when you move to the default.

## Harmonic resolvent

The harmonic resolvent (`solvers/harmonic_resolvent.py`, through resolvent4py)
uses the same time convention as the classical resolvent, `exp(-i*omega*t)`.
The perturbation is `q(t) = sum_j q_j exp(-i*omega_j*t)` and the periodic
operator is `A(t) = sum_k Ahat_k exp(-i*k*omega0*t)`. With
`omega_j = omega + j*omega0` the ladder is

```
(gamma - i*omega_j) q_j - sum_k Ahat_k q_{j-k} = f_j
```

At `N_h = 0` this is the classical equation, so the harmonic gain at `omega_j`
equals the classical gain at the same `omega_j`. The `resolvent_sign_convention`
setting does not change the harmonic path.

resolvent4py fixes its own convention, `exp(+i*omega*t)`, with the ladder
`(gamma + i*w_p) q_p - sum_k X_k q_{p-k}`. The code maps one onto the other
(`_harmonic_generator`): it passes the frequencies `w_p = -omega_{-p}` and the
files `X_k = Ahat_{-k}`. Both are reversals of an ordered list. Block `p` of the
resolvent4py ladder is then harmonic `j = -p`. A reversal of the blocks does not
change singular values.

What carries the new convention:

- `PeriodicBaseFlow.Qhat[k]` is the coefficient of `exp(-i*k*omega0*t)`.
  `reconstruct(t)` returns `Qhat[0] + 2 Re sum_{k>=1} Qhat[k] exp(-i*k*omega0*t)`.
- `HarmonicBlocks.Ahat[k]` is the coefficient of `exp(-i*k*omega0*t)`. The
  files written by `write_resolvent4py_coo` and `write_resolvent4py_vecs` hold
  `k = -N_h..N_h` in this convention.
- `perts_freqs` are the frequencies `omega_j` of `exp(-i*omega_j*t)`, in ascending order.

Files written before this change use `exp(+i*k*omega0*t)`. To convert them,
replace each harmonic `k` by `-k`:

- a periodic base-flow `.npz`: take the complex conjugate of `Qhat[1:]`. `Qhat[0]`
  is real and stays;
- a list of `Ahat_k` files or arrays: reverse the order, so that the file for
  `k` becomes the file for `-k`;
- a list of `dQ_k` vectors: reverse the order in the same way.

Gains from the old code at `N_h = 0` are the classical gains at `-omega_j`. For
`N_h >= 1` the old ladder with `omega_j` solved the new ladder with `-omega_j`
and the harmonics reversed. A gain map computed with the old convention at `omega`
is therefore the new map at `-omega`.

Evidence (`tests/test_harmonic_exp_minus.py`): a complex 3 x 3 periodic operator with
`A(t) = A_0 + A_1 exp(-i*omega0*t) + A_{-1} exp(+i*omega0*t)`, `omega0 = 1.3`, with
Floquet multipliers below 4e-3. The late-time response of
`dq/dt = (A(t) - gamma) q + f exp(-i*omega*t)`, found with `solve_ivp` (tolerances
1e-12) and projected on `exp(-i*(omega + j*omega0)*t)` for `j = -1, 0, 1`, is compared
with the solve of the ladder (`N_p = 8`). The relative difference is 2e-12 at
`gamma = 0` and 9e-13 at `gamma = 0.2` with the mapping. The ladder as resolvent4py
reads the same files, without the mapping, differs by 0.95 and 0.86.

## Check which convention produced a file

- A resolvent map store (`output_npz` of `resolvent_map`) and a driver result
  file carry the key `resolvent_sign_convention` with the value `physical` or
  `matlab_parity`.
- A file without that key was written before the change. It used `matlab_parity`.
- `resolvent_map` refuses to resume a store whose convention differs from the
  current run. The error names both conventions and says what to do. A map never
  mixes conventions.

## Reproduce an old result

Add one line to the `solver` block of the YAML file:

```yaml
solver:
  resolvent_sign_convention: matlab_parity
```

The key sits beside `solver.resolvent`. It does not go inside that block,
because a `solver.resolvent` block replaces the `solver.linsolve` settings for the
resolvent solve.

In a script, the same setting is `Config.load({"solver": {"resolvent_sign_convention": "matlab_parity"}})`,
or a `solver_cfg` that contains the key. To build the old matrix directly:

```python
from linstabpy.linalg.shifted import shifted_operator

K_old = shifted_operator(L0, omega, gamma, matlab_parity=True)  # L0 - (gamma + i*omega) I
K_new = shifted_operator(L0, omega, gamma)  # (gamma - i*omega) I - L0
```

The old matrix is reproduced bit for bit. Use this setting when you compare
with LinStab2D MATLAB output. Examples 1, 2 and 5 in `examples/linstab2d/` set it
for that reason.

## Convert an old result without recomputing

Use `H_old(omega) = -H_new(-omega)` and the effect table.

Gain map from the old code, `beta != 0` and `W != 0` (also Floquet, swirl):
`gain_new(beta, f) = gain_old(beta, -f)`. Reverse the frequency axis. An old map
that covers `f >= 0` gives the new map for `f <= 0`. To get the new map for
`f >= 0`, you need old values at negative `f`, or a new run.

Gain map from the old code, `W = 0` (including `beta = 0`): nothing to convert.

Modes from the old code:

- `beta = 0`: take the complex conjugate of the forcing. Take the complex
  conjugate of the response and reverse its sign relative to the forcing.
- `beta != 0`, `W = 0`: do the same, and also reverse the sign of the spanwise
  (azimuthal) velocity component of both modes.
- `beta != 0`, `W != 0`: the forcing at `omega` is the old forcing at `-omega`. The response at `omega` is the old response at `-omega` with its sign reversed relative to the forcing.

If a conversion is not exact for your case, rerun with the default.

## What did not change

- The gains of every flow with `W = 0`.
- The eigenvalue convention: `omega = eig(i*L0)`, unstable if `Im(omega) > 0`.
- The meaning of `gamma`: it moves the pole off the real axis, in the same
  direction as before.
- The energy weight, the masks, the filter, and the solver settings.
- The files in `examples/savarino/output/compare/` still hold valid gains. The
  Savarino base flow has `W = 0` exactly, so its gain map is the same in both
  conventions. Their modes were computed with the old sign.

## Evidence

All of these run in the test suite.

| Test | What it shows |
|------|---------------|
| `tests/test_resolvent_sign_convention.py::test_default_shifted_operator_matches_time_integration` | A stable, non-normal complex 6 x 6 operator with an asymmetric spectrum. The late-time solution of `dq/dt = (L - gamma) q + f exp(-i*omega*t)`, found with a matrix exponential, equals `solve(shifted_operator(L, omega, gamma), f)`. The relative difference is between 3e-16 and 7e-16 for `omega` in `-3, 2, 3.1` and `gamma` in `0, 0.4`. The pass limit is 1e-10. |
| `tests/test_resolvent_sign_convention.py::test_matlab_parity_does_not_match_time_integration` | The old matrix gives relative differences of 1.2 to 1.5 against the same reference (120 % to 150 %). |
| `tests/test_resolvent_sign_oracle.py` | A diagonal operator with the eigenvalue `-0.2 + 3i`. The leading gain peaks at `omega = -3` (value 5.000; at `+3` it is 0.167). Under `matlab_parity` it peaks at `+3`. |
| `tests/test_resolvent_sign_effect.py` | An axisymmetric jet with azimuthal wavenumber 3. With `W = 0` the gains agree to 1e-8 and the modes match `conj(old)` with the spanwise component reversed (overlap 1.000000; with the plain conjugate the overlap is 0.78 for the forcing). With swirl the new gain at `omega` equals the old gain at `-omega`, and the map is not even. |
| `tests/test_resolvent_sign_oracle.py::test_real_operator_gain_equal_and_modes_conjugate_under_both_conventions` | A real operator: equal gains, conjugate modes. |
| `tests/test_resolvent_sign_convention.py::test_adjoint_solve_follows_the_assembled_matrix` | The adjoint solve inverts the conjugate transpose of the assembled matrix, in both conventions. |
| `tests/test_band_marking.py`, `tests/test_mark_unstable_band.py` | The pole location in both conventions. |
| `tests/test_router_entry.py` | A map store records the convention and refuses a mixed resume. |

Example 1 of `examples/linstab2d/` (compressible Blasius boundary layer) against
the LinStab2D MATLAB reference:

| Setting | Gain error (relative) | Mode error (relative L2) | Result |
|---------|-----------------------|--------------------------|--------|
| `matlab_parity` | 3.8e-11 | 7.9e-11 | PASS |
| `physical` | 3.8e-11 | 1.0 (the modes are conjugates of the reference) | SOFT_FAIL |

The gains agree in both settings. The modes agree with the MATLAB reference only
under `matlab_parity`.
