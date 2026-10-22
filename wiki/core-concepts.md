# Core concepts

Short, exact definitions of the physics and math vocabulary used across this wiki. Each entry states what the thing is, the convention this repo uses, and where to read more. Solver objects (`KSP`, `PC`, `EPS`, `ST`, `MatShell`) live in the [solver glossary](solver-glossary.md). Code structure lives in the [module guide](modules/index.md).

Shared conventions: the perturbation state is $q' = [\rho', u', v', w', T']^T$ per grid point; the assembled operator satisfies $\partial_t q' = L_0 q'$ in the interior; temporal eigenvalues are $\omega = \mathrm{eig}(i L_0)$ with $\exp(-i\omega t)$ time dependence.


## Stability vocabulary

**Base flow.** A steady solution $\bar q(x, y)$ of the compressible [Navier-Stokes](https://grokipedia.com/page/Navier–Stokes_equations) equations, loaded from files and never computed here. Starting point of every analysis. Nondimensional, convective scaling: freestream density, velocity, and temperature equal 1. See [baseflow management](modules/baseflow-management.md).

**Perturbation.** The small disturbance $q'(x, y, z, t)$ about the base flow. LinStabPy solves linear equations for $q'$ only. Nonlinear saturation is out of scope by construction.

**Discrete temporal operator.** The assembled sparse matrix $iL_0$ whose [eigenvalues](https://grokipedia.com/page/Eigenvalues_and_eigenvectors) are the reported $\omega$. "Discrete" marks it as the matrix acting on grid vectors, as opposed to the continuous linearized operator. Every temporal eigensolve targets this matrix through shift-invert; the $i$ puts frequencies on the real axis and growth rates on the imaginary axis under the $\exp(-i\omega t)$ convention.

**Temporal growth rate.** The imaginary part $\mathrm{Im}(\omega)$ of a temporal [eigenvalue](https://grokipedia.com/page/Eigenvalues_and_eigenvectors). Mode amplitude evolves as $\exp(\mathrm{Im}(\omega)\, t)$, so positive means exponential growth in time at a fixed station. Reported for every accepted mode with its postcheck residual.

**Angular frequency.** The real part $\mathrm{Re}(\omega)$. Oscillations run as $\exp(-i\,\mathrm{Re}(\omega)\, t)$. [Shift-invert](solver-glossary.md) targets in YAML live in full complex $\omega$ space, so you aim at a frequency and growth rate together.

**Direct mode.** A right eigenvector of the stability problem: the spatial shape $v(x, y)$ of the disturbance. Modes are normalized in the Chu norm (or $L^2$ where stated) before any comparison or plot.

**Spatial amplification.** At prescribed real spanwise [wavenumber](https://grokipedia.com/page/Wavenumber) $\beta$ and real frequency $\omega$, the spatial problem returns a complex streamwise wavenumber $\alpha$. Amplification along $x$ is read from $\mathrm{Im}(\alpha)$ while following one branch; the sign interpretation depends on the branch, so never judge convective or absolute behavior from a single point. Build the spectrum first, then track the branch. See the tutorial and [stability analysis](modules/stability-analysis.md).

**Generalized eigenvalue problem.** An eigenproblem of the form $A x = \lambda B x$ with $B \neq I$ (see [generalized eigenvectors](https://grokipedia.com/page/Generalized_eigenvector)). Ours is the spatial pencil $(L_c + \omega L_w)\, v = \alpha\, R\, v$. Boundary rows make $R$ singular, so $\lambda = \infty$ eigenvalues exist for the constrained subspace. Tiny or near-full problems go to dense [QZ](https://en.wikipedia.org/wiki/QZ_algorithm), which handles singular $B$ natively; production sizes use shift-invert [EPS](solver-glossary.md).

**Boundary layer.** The thin region near a solid wall where viscosity brings the flow from freestream speed to rest at the wall. Blasius flow, the base flow of example 1, is the flat-plate similarity solution of the [boundary-layer](https://grokipedia.com/page/Boundary_layer) equations. Background: [Scholarpedia: Boundary Layer Theory](http://www.scholarpedia.org/article/Boundary_Layer_Theory).


## Resolvent vocabulary

**Modal vs non-modal.** Modal stability asks about $t \to \infty$: do all eigenvalues sit on the stable side. Non-modal stability asks about finite time: how far can a perturbation wander before (possibly) decaying. For normal operators the answers agree; for non-normal ones they can differ wildly, and the finite-time answer is an optimization problem, not an eigenvalue. See Frantz, Loiseau and Robinet (2023), *Krylov methods for large-scale dynamical systems*, *Appl. Mech. Rev.* 75(3), 030802, modal and non-modal stability.

**Resolvent.** The operator $H = C F ((\gamma - i\omega) I - L_0)^{-1} F B$ mapping harmonic forcing to harmonic response at frequency $\omega$. Mathematically an instance of the [resolvent formalism](https://grokipedia.com/page/Resolvent_formalism). $B$ and $C$ restrict inputs and outputs (regions, variables), $F$ is an optional spatial filter inside the masks, $\gamma \ge 0$ is the discount below. Never formed as entries; applied through one factorization. The resolvent is the frequency-domain analogue of optimal-perturbation analysis below: same question (worst input, largest output), asked per frequency instead of per time horizon. The sign of $\omega$ follows $\exp(-i\omega t)$, as for the eigenvalues; see the [sign convention](../docs/resolvent_sign_convention.md). See [resolvent analysis](modules/resolvent-analysis.md).

**Optimal energy gain.** The largest achievable energy ratio $\sigma_1^2 = \max_{f \neq 0} \|Hf\|_E^2 / \|f\|_F^2$: response energy in the Chu norm over forcing energy in the forcing norm. Computed as $\sigma = \sqrt{\mathrm{Re}(\lambda)}$ from the weighted normal eigenproblem, never from an explicit [SVD](https://grokipedia.com/page/Singular_value_decomposition) of $H$. "Optimal" means optimal over all forcing shapes at that $(\beta, \omega)$, not over frequencies.

**Forcing mode.** The optimal input shape $v$: the unit-norm vector attaining the leading gain. Normalized as $v^H W v = 1$ under the `chu` forcing norm or $v^H v = 1$ under `l2`. What to inject for maximum effect.

**Response mode.** The optimal output shape $u = Hv/\sigma$: what grows when the optimal forcing is applied. Always recovered by applying $H$, which is why the forward residual sits at roundoff by construction.

**Gain map.** Leading gains sampled over a $(\beta, \omega)$ grid: spanwise [wavenumber](https://grokipedia.com/page/Wavenumber) versus frequency. Peaks mark the flow's preferred amplifier settings. Production maps reuse anchor factorizations across nearby $\omega$ under a residual gate instead of refactoring per point. See [solver routing](modules/solver-routing.md).

**Discount ($\gamma$).** A real shift added as $(\gamma - i\omega) I - L_0$ with $\gamma > 0$ pushing poles off the imaginary axis. Keeps the shifted solve well posed near lightly damped eigenvalues. $\gamma = 0$ is the classical resolvent. Gains of a discounted resolvent answer the discounted question, so report $\gamma$ with the number.

**Nonnormality.** $L_0$ does not commute with its [adjoint](https://grokipedia.com/page/Adjoint) in the energy inner product ($H^H W \neq W H$ in the weighted sense). Consequence: all modes stable does not mean quiet. Quantitatively, with $\lambda_r$ the leading real part and $\kappa(V) = \|V\|\|V^{-1}\|$ the eigenvector conditioning, $\exp(2\tau\lambda_r) \le \|\exp(\tau L)\|^2 \le \kappa(V)\,\exp(2\tau\lambda_r)$: the upper bound is where transient growth lives, and it grows with nonnormality. (In repo convention $\lambda_r = \mathrm{Im}(\omega)$.) Mechanically, picture almost-colinear eigenvectors interfering constructively before decaying; the extreme is a non-diagonalizable Jordan block with algebraic growth. See Frantz, Loiseau and Robinet (2023), *Krylov methods for large-scale dynamical systems*, *Appl. Mech. Rev.* 75(3), 030802, non-modal stability. Nonnormal operators support large transient amplification, which is exactly what the gain map measures. The related spectral picture is the [pseudospectrum](https://en.wikipedia.org/wiki/Pseudospectrum): $\{z : \|(zI - A)^{-1}\| > 1/\varepsilon\}$. Scholarpedia's [Stability](http://www.scholarpedia.org/article/Stability) article works the classic two-by-two transient-growth example.

**Optimal perturbation and transient growth.** The finite-time gain $\mathcal{G}(\tau) = \max_{x_0} \|\exp(\tau L)x_0\|^2/\|x_0\|^2$ equals $\sigma_1^2$, the squared leading singular value of the propagator; the maximizer is the first right singular vector and the response is $\sigma_1$ times the first left one. With the adjoint available this recasts as an eigenproblem rather than an SVD. Textbook illustration: plane Couette flow at $Re = 300$, linearly stable yet with optimal gain near 100, streamwise vortices growing into streaks. See Frantz, Loiseau and Robinet (2023), *Krylov methods for large-scale dynamical systems*, *Appl. Mech. Rev.* 75(3), 030802, non-modal stability and illustration.

**Receptivity.** The sensitivity of the flow to external disturbances: incoming noise, roughness, turbulence. Resolvent gains and [adjoint](https://grokipedia.com/page/Adjoint) modes quantify it. High gain at some $(\beta, \omega)$ means the environment can drive a large response there even when every eigenvalue is stable. Downstream of receptivity lies [transition to turbulence](http://www.scholarpedia.org/article/Transition_to_turbulence).


## Energy and adjoint vocabulary

**Chu energy norm.** The compressible disturbance-energy inner product. Diagonal in our discretization: $W = \tfrac{1}{2}\,\mathrm{diag}(W_\rho, W_u, W_v, W_w, W_T)$ built from quadrature weights, base density and temperature, $\kappa$, and Mach number (times radius in axisymmetric mode). Defines "large" for gains, mode normalization, and adjoints. A zero diagonal entry is a hard error at shell construction, not a warning.

**Adjoint mode.** An eigenvector of the energy-weighted adjoint $W^{-1} L_0^H W$. Where the direct mode shows what grows, the adjoint shows what the eigenvalue is sensitive to: the receptive shape. Returned in the temporal $\omega$ convention as conjugates of direct eigenvalues for real-valued base flows.

**Structural sensitivity (wavemaker).** The pointwise product of direct and adjoint mode magnitudes (Giannetti-Luchini). Marks where a localized structural tweak moves the eigenvalue most: the wavemaker region. The control-oriented output of running direct plus adjoint analysis together.


## Numerics vocabulary

**SBP operator.** A [summation-by-parts](https://grokipedia.com/page/Summation_by_parts) [finite-difference](https://grokipedia.com/page/Finite_difference_method) matrix: discrete derivative plus diagonal norm satisfying a discrete analogue of integration by parts, which bounds the discrete energy. Even interior orders 2, 4, 6, 8 only, each with validated boundary closures. Order 4 is the usual start.

**Sponge layer.** An artificial damping region near outflow and farfield boundaries that absorbs outgoing waves before they reflect. Implemented as a damping field added to the operator during assembly. Related idea, different mechanism: the [perfectly matched layer](https://grokipedia.com/page/Perfectly_matched_layer). A zero sponge array means no damping, which is valid for some cases and wrong for others; the baseflow plots show it.

**Boundary recipe.** The declarative list of [Dirichlet](https://grokipedia.com/page/Dirichlet_boundary_condition) and [Neumann](https://grokipedia.com/page/Neumann_boundary_condition) rules applied last in assembly: which borders, which variables, corner handling, curvilinear normals. Publishes the free degree-of-freedom sets every downstream solve must respect.

**BC penalty.** The number behind `bc: penalty:`. For Dirichlet rows, assembly zeroes the row and column, then writes `penalty` on the diagonal, so the constrained equation reads `penalty` times $q_i = 0$: the DOF is pinned to zero and decoupled from interior equations. In the temporal EVP (on $iL_0$) these pinned DOFs sit near $i \times$ `penalty`, far from physical targets, and the free-DOF sets plus acceptance checks keep them out of results. For Neumann rows, the normal-derivative stencil row is multiplied by `penalty`: the homogeneous constraint is unchanged, but the row is scaled to balance against interior rows in the factorization.

How to set it: keep the path default unless diagnostics complain. The driver and Song default to `1e8` (`bc.penalty` in YAML); assembly, the router, and the Martini examples default to `1e4` (nested `bc: penalty:` in their YAMLs, top-level `bc_penalty` or `penalty` in router dicts, `bc_penalty` kwarg in the direct API). Raise toward `1e6` (a validation case does exactly this) if constrained-Dirichlet leakage diagnostics show boundary modes leaking into results. Lower back down if you see conditioning trouble: a `1e8` diagonal next to order-1 interior entries is exactly the kind of scaling that can inflate residuals, and direct LU tolerates only so much.

**Artificial dissipation.** The term $-\varepsilon Q_{\mathrm{diss}}$ subtracted from $L_0$, with $Q_{\mathrm{diss}}$ the Mattsson-Svard-Nordstrom operator. Damps grid-scale sawtooth modes the centered SBP stencils admit. Two halves must both be set (build the operator, set the strength) or nothing happens silently. Off on rectilinear meshes, on for meshes curved in both directions. See the Workflow page.

**Implicit filter ($\alpha$).** The Mattsson-style spatial filter applied inside the resolvent masks and on the mesh. Strength parameter `alpha_filter`: $0.0$ is maximum filtering in the MATLAB convention, `"none"` disables. The adjoint path uses the adjoint-compliant variant so discrete adjointness holds.

**Spanwise wavenumber ($\beta$).** The real [wavenumber](https://grokipedia.com/page/Wavenumber) of the $e^{i\beta z}$ spanwise dependence. Enters the linearized physics, so each $\beta$ needs its own assembled $L_0$. $\beta = 0$ is the two-dimensional problem. Gain maps sweep $\beta$ on the outer loop.

**Floquet exponent.** The detuning parameter for spanwise-periodic [Floquet](https://grokipedia.com/page/Floquet_theory) analysis on streaky or modulated base flows. `continuum` is the default form; `matlab_parity` reproduces the LinStab2D algebra exactly for validation. Set per case, never guessed.

**Matrix dual path.** Operators assemble locally in SciPy CSC (from CSR blocks) and convert to PETSc AIJ for distributed solves, or assemble AIJ directly where supported. Layouts, backends, and the conversion bridge are defined in the [solver glossary](solver-glossary.md).


## Workflow vocabulary

**Shift target ($\sigma$).** The complex $\omega$-space point the eigensolver searches around. Lives in the YAML solver block, set explicitly on the spectral transformation. Near a branch it isolates modes cleanly; on a crowded real axis it returns mixtures. Moving slightly off-axis is a numerical search choice, not a physics change. Solver mechanics live in the [solver glossary](solver-glossary.md).

**Acceptance.** The postchecks a computed mode must pass before it is trusted: full-operator residual, boundary-condition leakage, target distance, spectral boxes, conjugate completeness. Converged Ritz pairs fail these regularly on interior non-Hermitian problems, which is why acceptance exists as a separate stage from convergence.

**Provenance.** The reproducibility record attached to every result: config hash, git SHA, backend and precision, mesh and norm settings. Stored inside the NPZ beside the numbers so a figure always traces back to the run that produced it.

## Beyond linear scope

Linear analysis predicts onset. What happens after onset is nonlinear and outside this repo, but two landmarks orient the map.

**Hopf bifurcation.** What happens after a mode goes unstable: an equilibrium sheds a limit cycle when a complex-conjugate eigenvalue pair crosses the imaginary axis. Linear analysis predicts the crossing ($\mathrm{Im}(\omega) = 0$ at criticality); the cycle itself needs the nonlinear terms. See [Scholarpedia: Andronov-Hopf bifurcation](http://www.scholarpedia.org/article/Andronov-Hopf_bifurcation) and [Bifurcation](http://www.scholarpedia.org/article/Bifurcation).

**Transition to turbulence.** The full path from laminar flow to turbulence: receptivity, linear amplification (modal or transient), secondary instability, breakdown. This repo covers the linear amplification step and the sensitivity fields around it. See [Scholarpedia: Transition to turbulence](http://www.scholarpedia.org/article/Transition_to_turbulence).

If a term appears elsewhere in this wiki without a definition or a link, that is a bug in that page. Link it here.
