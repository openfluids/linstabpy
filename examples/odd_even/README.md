# Odd-even decoupling on a wavy-wall mesh

This case reproduces the figure and the numbers of the section "Odd-even
decoupling on curvilinear meshes" in the linstabpy paper. It computes the
leading resolvent response of a model boundary layer on a Cartesian mesh and
on a curvilinear mesh with a wavy lower wall, for two resolutions and six
values of the operator-dissipation coefficient epsilon.

The case uses the SciPy operator path only. It needs NumPy, SciPy and
matplotlib, not PETSc.

```
PYTHONPATH=src .venv/bin/python -m examples.odd_even.odd_even
PYTHONPATH=src .venv/bin/python -m examples.odd_even.odd_even_fig
```

The first command solves the 24 cases and writes `odd_even.csv` (tracked)
and `output/odd_even.npz` (not tracked). The second command reads both and
writes `odd_even.png`. The 161 x 81 cases take about one minute each with
serial `splu`.

## Setup

| Item | Value |
| --- | --- |
| Box | 0 <= x <= 4, 0 <= y <= 1 |
| Wall | y_w = 0.15 sin(2 pi x / 4), displacement decays as (1 - eta)^2 |
| Base flow | U = 1 - exp(-(eta / 0.15)^2), V = W = 0, uniform density and temperature |
| Flow | Ma = 0.5, Re = 10^4, omega = 1, beta = 1 |
| Scheme | SBP order 4 |
| Boundary conditions | Dirichlet on all variables at inlet and top, on u, v, w, T at the wall, Neumann at the outlet |
| Sponge | quadratic from 0 to 2 over the last 15 % of the box length |
| Meshes | 81 x 41 and 161 x 81 |
| epsilon | 0, 0.005, 0.01, 0.02, 0.05, 0.1 |

## Columns of `odd_even.csv`

| Column | Meaning |
| --- | --- |
| `leak_into_dx` | `‖Dx s‖ / ‖Dy s‖` for the field `s = (-1)^j` that alternates along eta |
| `gain_l2` | leading singular value of `(-i omega I - L)^-1`, unweighted l2 norm |
| `r_n` | energy fraction of the streamwise response removed by one (1/4, 1/2, 1/4) pass along each direction |
| `nnz` | nonzero entries of `L` after the boundary conditions |
| `wall_factor_s`, `wall_total_s` | wall time of the LU factorization and of the whole case, in seconds |
