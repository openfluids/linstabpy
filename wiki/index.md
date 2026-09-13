---
title: LinStabPy — Compressible Hydrodynamic Stability and Resolvent Analysis
description: Sparse linear stability and resolvent analysis of compressible viscous flows using summation-by-parts finite differences and PETSc/SLEPc.
---

# LinStabPy

**LinStabPy** builds the discrete linearized compressible [Navier–Stokes](https://grokipedia.com/page/Navier–Stokes_equations) operator about a given base flow and solves the eigenvalue and [resolvent](https://grokipedia.com/page/Resolvent_formalism) problems on it at HPC scale.

Intuition first. You bring a steady flow field computed elsewhere. LinStabPy answers two questions. **Stability**: does a small disturbance grow on its own, at what frequency, and with what spatial shape? **Resolvent**: at a given frequency, which steady harmonic forcing produces the largest response, and how large is that amplification?

The discrete state is the primitive perturbation vector \( \mathbf{q}' \) at each grid point, giving 5 unknowns per point before boundary elimination (\( \rho', u', v', w', T' \)). The code assembles the [sparse](https://grokipedia.com/page/Sparse_matrix) matrix \( \mathcal{L}_0 \) such that \( \partial_t \mathbf{q}' = \mathcal{L}_0 \mathbf{q}' \) in the undisturbed interior, then applies sponge, dissipation, and boundary recipes in a fixed order. Everything downstream consumes that one operator family.

---

## Governing Equations & Mathematical Formulation

### 1. Perturbation Decomposition

A two-dimensional steady [compressible](https://grokipedia.com/page/Compressible_flow) base flow \( \bar{\mathbf{q}}(x,y) \) is perturbed by small three-dimensional unsteady fluctuations:

\[
\mathbf{q}(x, y, z, t) = \bar{\mathbf{q}}(x, y) + \mathbf{q}'(x, y, z, t)
\]

Assuming periodicity in the spanwise direction \( z \) and harmonic time dependence with spanwise [wavenumber](https://grokipedia.com/page/Wavenumber) \( \beta \) and frequency \( \omega \):

\[
\mathbf{q}'(x, y, z, t) = \hat{\mathbf{q}}(x, y) \exp\left( i (\beta z - \omega t) \right)
\]

where the state vector of primitive perturbation variables is:

\[
\hat{\mathbf{q}} = \begin{bmatrix} \hat{\rho} & \hat{u} & \hat{v} & \hat{w} & \hat{T} \end{bmatrix}^T
\]

### 2. Linearized Dynamical System (LNSE)

Linearizing the compressible Navier–Stokes equations around \( \bar{\mathbf{q}} \) yields the semi-discrete linear dynamical system:

\[
\mathbf{M} \frac{\partial \mathbf{q}'}{\partial t} = \mathcal{L}_0(\beta) \mathbf{q}' + \mathbf{f}'
\]

where \( \mathcal{L}_0 \) incorporates [summation-by-parts](https://grokipedia.com/page/Summation_by_parts) (SBP) [finite differences](https://grokipedia.com/page/Finite_difference_method) (orders 2, 4, 6, 8), coordinate metrics, characteristic [sponge layers](https://grokipedia.com/page/Perfectly_matched_layer), and boundary conditions.

### 3. Modal Stability: Generalized Eigenvalue Problem

Under harmonic time dependence, the homogeneous system reduces to:

\[
\mathcal{A}(\beta) \hat{\mathbf{q}} = \omega \mathcal{B} \hat{\mathbf{q}}
\]

* **Angular frequency**: \( \operatorname{Re}(\omega) \)
* **[Temporal growth rate](core-concepts.md)**: \( \operatorname{Im}(\omega) > 0 \) denotes exponential instability (\( \propto \exp(\operatorname{Im}(\omega) t) \))
* Solved via [SLEPc](https://en.wikipedia.org/wiki/SLEPc) `EPS` using [Krylov–Schur](solver-glossary.md) with spectral [shift-invert](solver-glossary.md) transformation.

### 4. Input–Output Resolvent Analysis

Under external harmonic forcing \( \hat{\mathbf{f}} \), the resolvent operator \( \mathcal{R}(\beta, \omega) \) maps inputs to linear responses:

\[
\hat{\mathbf{q}} = \mathcal{R}(\beta, \omega) \hat{\mathbf{f}}, \qquad \mathcal{R}(\beta, \omega) = \mathbf{C} \left( -i\omega \mathbf{B} - \mathbf{A}(\beta) \right)^{-1} \mathbf{B}_f
\]

The optimal energy [gain](core-concepts.md) \( \sigma_1^2 \) is defined as the supremum over all forcing shapes under the [Chu](core-concepts.md) compressible energy norm:

\[
\sigma_1^2(\beta, \omega) = \max_{\hat{\mathbf{f}} \neq 0} \frac{\langle \hat{\mathbf{q}}, \mathbf{W}_q \hat{\mathbf{q}} \rangle}{\langle \hat{\mathbf{f}}, \mathbf{W}_f \hat{\mathbf{f}} \rangle} = \left\| \mathbf{W}_q^{1/2} \mathcal{R}(\beta, \omega) \mathbf{W}_f^{-1/2} \right\|_2^2
\]

Large gains with \( \operatorname{Im}(\omega) < 0 \) quantify [non-normal](https://en.wikipedia.org/wiki/Defective_matrix) convective amplification, receptivity to environmental disturbances, and flow-control sensitivity.

---

!!! tip "How to read these docs as a mathematician or engineer"
    We keep solver language exact and explain the fluids side clearly:

    * **Base flow**: A steady solution of the compressible Navier–Stokes equations, loaded from `.npz` files. We linearize about it; the code never computes it.
    * **Temporal stability**: [Eigenvalues](https://grokipedia.com/page/Eigenvalues_and_eigenvectors) \( \omega = \operatorname{eig}(i \mathcal{L}_0) \). Growth in time is \( \operatorname{Im}(\omega) > 0 \). The shift-invert target you pass in YAML lives in \( \omega \) space.
    * **Spatial stability**: Same physics, swapped role of axes. Frequency is prescribed real, streamwise wavenumber is the eigenvalue.
    * **Resolvent**: For fixed \( (\beta, \omega) \), the gain is the leading weighted [singular value](https://grokipedia.com/page/Singular_value_decomposition) of \( \mathcal{R} \). High gain with all \( \operatorname{Im}(\omega) < 0 \) indicates transient or convective amplification.
    * **Chu norm**: The compressible energy inner product. Diagonal in our discretization, built from quadrature weights, base density and temperature, \( \gamma \), and [Mach number](https://grokipedia.com/page/Mach_number).
    * **Solver objects**: `KSP`, `PC`, `EPS`, `ST`, and `MatShell` are defined from zero in the [solver glossary](solver-glossary.md), with links to the official [PETSc](https://en.wikipedia.org/wiki/PETSc) and SLEPc docs.

---

<div class="grid cards" markdown>

- :material-chart-bell-curve-cumulative:{ .lg .middle } __Spectra & Sensitivity__

    ---

    Growth rates, frequencies, [adjoint](https://grokipedia.com/page/Adjoint) modes, and where the flow is most sensitive to a structural perturbation.

    [:octicons-arrow-right-24: Read Overview](overview.md)

- :material-layers-triple-outline:{ .lg .middle } __LNSE Discretization__

    ---

    Grids, SBP difference stencils (orders 2, 4, 6, 8), metric invariants, and energy-stable boundary condition recipes.

    [:octicons-arrow-right-24: Explore Architecture](architecture.md)

- :material-console-line:{ .lg .middle } __Orchestrated Workflow__

    ---

    End-to-end staged pipeline from mean-flow ingest and YAML config to [MPI](https://grokipedia.com/page/Message_Passing_Interface) execution and contract-stable NPZ payloads.

    [:octicons-arrow-right-24: Study Workflow](workflow.md)

- :material-tune-vertical:{ .lg .middle } __Resolvent & Solvers__

    ---

    Cost-aware parametric sweeps across \( (\beta, \omega) \) with anchor factorizations, [preconditioner](https://grokipedia.com/page/Preconditioner) reuse, and residual gates.

    [:octicons-arrow-right-24: Module Guide](modules/index.md)

</div>

---

## Quick Example

You can run a case from the terminal with MPI, or import the package in Python. Import stays lightweight and does not start MPI or PETSc until a solver stage is invoked.

```bash
# Run one case on 4 MPI ranks. Config and output travel by env vars (see Getting started).
export LINSTABPY_CASE_CONFIG=case.yaml LINSTABPY_CASE_OUTPUT_NPZ=results/out.npz
mpirun -n 4 linstabpy
```

```python
# Load and validate a case configuration without pulling PETSc
import linstabpy
from linstabpy.config.typed import load_case_config

config = load_case_config("case.yaml")
print(f"Case: {config.meta.case_name}, Mode: {config.solver.mode}")
```

The YAML file is the case definition. It names the base flow files, the [Reynolds](https://grokipedia.com/page/Reynolds_number) and Mach numbers, the grid and boundary choices, and what to compute. Keep it next to the results for reproducible scientific provenance.

Results land in contract-stable `.npz` files. Load them with `numpy.load()`:

```python
import numpy as np

data = np.load("results/case_temporal.npz")
omega = data["omega"]          # Complex eigenvalues
modes = data["modes"]          # Eigenvectors [n_points, 5, n_modes]
print(f"Computed {len(omega)} eigenvalues. Leading growth rate: {np.max(np.imag(omega)):.4e}")
```

---

## Documentation Directory

| Section | Core Focus | Target Audience |
|---|---|---|
| [**Getting started**](getting-started/index.md) | Install, first runs, examples, your first case | Start here. Do this today |
| [**Overview**](overview.md) | What LinStabPy computes and where it stops | First read, scope, vocabulary |
| [**Architecture**](architecture.md) | Component relationships & C4 models | Contributors, system developers |
| [**Workflow**](workflow.md) | Steps of a run in order | Running and orchestrating cases |
| [**Module guide**](modules/index.md) | Per-package derivations and code notes | Advanced customization, algorithms |
| [**Boundaries**](boundaries.md) | YAML schema, CLI flags, env vars, NPZ contract | Reference when building cases |
| [**Solver glossary**](solver-glossary.md) | Every KSP, PC, EPS, ST, and MatShell term defined and linked | Open beside any solver page |
| [**Core concepts**](core-concepts.md) | Base flow, growth rate, gain, Chu norm, wavemaker, sponge, and friends | Open beside any science page |

---

## Cite linstabpy

If this software contributed to published work, please cite the linstabpy paper together with the LinStab2D methodology paper:

> Hamada, G. Y. R. and Frantz, R. A. S. linstabpy: open-source compressible stability and resolvent analysis. *Theor. Comput. Fluid Dyn.* (in preparation, 2026).

> Martini, E. and Schmidt, O. T. (2024). LinStab2D: stability and resolvent analysis of compressible viscous flows in MATLAB. *Theor. Comput. Fluid Dyn.* 38, 665–685. doi:10.1007/s00162-024-00706-0.

The full bibliography lives on the [references](references.md) page, and `CITATION.cff` carries the machine-readable entry (DOI and version land with the first tagged Zenodo release).
