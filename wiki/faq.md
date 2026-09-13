# Frequently asked questions

Short answers to the questions new users actually ask, each pointing at the page with the full story. Conventions are fixed repo-wide: $\omega = \mathrm{eig}(i L_0)$, unstable exactly when $\mathrm{Im}(\omega) > 0$.

## Which analysis do I need?

**Temporal stability** if you ask "does the flow sustain self-excited disturbances" — growth rates and frequencies of global modes. **Spatial stability** if you prescribe a real frequency and ask how a disturbance grows downstream — the unknown is the complex streamwise wavenumber $\alpha$ from a quadratic-turned-generalized problem of twice the size. **Adjoint modes** if you ask "where is the flow receptive" — sensitivity and wavemaker fields from $W^{-1}L_0^H W$. **Resolvent** if you ask "which external forcing is amplified most" — optimal gains with forcing and response shapes at real $\omega$. Temporal, adjoint, and resolvent act on the same assembled $L_0$; spatial builds its own pencil from the same coefficients. Details in [core concepts](core-concepts.md), solver mechanics in the [solver glossary](solver-glossary.md).

## My spectrum moved when I changed the shift. Is that normal?

Yes, if you changed which branch the solver locked onto. Shift-invert finds eigenvalues near the target; on a crowded real axis, neighboring targets can return mixtures from different branches. Fix the branch first (small `nev`, a few nearby targets, compare postcheck residuals and target distances), then scale up. Moving the target slightly off the real axis often isolates a branch more cleanly. This is a numerical search choice, not a physics change. See [getting started](getting-started/index.md#your-first-own-case) and the paper's note that slow convergence near branches means widening the subspace or moving the shift.

## Do I turn the dissipation on?

Run the grid-scale leak test first: feed the operator a field alternating along one grid direction and constant along the other; a transverse derivative must return zero, so $\|D_x q\|/\|D_y q\|$ measures pure metric mixing. At round-off (the Song and Savarino meshes read below $10^{-11}$), the mesh is rectilinear: leave $\varepsilon = 0$ and confirm insensitivity under refinement. At order one (0.26 on one LES curved-wall mesh, 0.10–0.12 on a wavy-wall test), the mesh is bent: switch dissipation on and tune $\varepsilon$ on that mesh until the gain or eigenvalue stops moving. A value tuned elsewhere does not transfer. And remember the two halves: building the operator and setting the strength are separate flags, and setting only one is a silent no-op. Costs and the full procedure come from the paper's sawtooth section: $\varepsilon = 10^{-2}$ damps grid content at rate $\approx 0.3$ while resolved scales see $\varepsilon \times 10^{-2}$, for 15% more nonzeros and 3–4× wall time.

## My resolvent response shows point-to-point wiggles. What is that?

The odd–even (sawtooth) mode: the $q_i = (-1)^i$ family that every centered first-derivative stencil annihilates exactly, at any order. Only viscous terms damp it, at rate $\sim 1/(\mathrm{Re}\,h^2)$, so at high $\mathrm{Re}\,h^2$ it sits nearly neutral and whatever projects onto it stays. On rectilinear meshes it decouples and refinement removes it; on bent meshes the cross-metrics couple it into the resolved field and you need the dissipation from the previous answer. See the sawtooth discussion in [core concepts](core-concepts.md#numerics-vocabulary).

## What does `bc: penalty:` do, and what value?

Dirichlet rows are zeroed and the diagonal is set to `penalty`, pinning the DOF (`penalty`·$q_i = 0$); pinned modes land near $i \times$ `penalty`, far from physical targets. Neumann rows are scaled by it for row balance. Driver default `1e8`, library and router default `1e4`. Keep the default unless constrained-leakage diagnostics complain, then raise toward `1e6` first. Full mechanics in [core concepts](core-concepts.md#numerics-vocabulary).

## Why does `linstabpy run case.yaml` run the wrong case?

Because that command never existed. The CLI takes no case arguments — only `--version`. Case and output travel by `LINSTABPY_CASE_CONFIG` and `LINSTABPY_CASE_OUTPUT_NPZ`, and unknown words are silently ignored by design of the parser. See [getting started](getting-started/index.md#the-cli-as-it-really-is).

## Why did my run print `Cache valid` and finish instantly?

Rank 0 found an output NPZ whose stored config hash and git SHA match the current ones, so it skipped the compute. Point each run at its own output path, or export `LINSTABPY_FORCE_RECOMPUTE=1` deliberately. See [getting started](getting-started/index.md#the-cli-as-it-really-is).

## Which SBP order, and why only even ones?

Order 4 to start. Only even orders 2, 4, 6, 8 ship because only they have validated diagonal-norm boundary closures in this family. Higher order buys accuracy per point but stiffens the operator and grows stencils. See [core concepts](core-concepts.md#numerics-vocabulary).

## How big should `ncv` be?

Bigger than `nev`, then bigger again near branches. Measured guidance from the paper's campaign: on a 450k-unknown resolvent point, cutting the Krylov space from 20 vectors to 5 cut solves per point from 40 to 10 and moved the gain by less than $10^{-10}$ — subspace size sets cost, not precision, once past the threshold. If accepted-mode quality plateaus while the solve stays active, the subspace is too small for the neighborhood, not the tolerance too loose. Defaults per path live in the [solver glossary](solver-glossary.md#slepc-eps-eigensolvers-and-st-spectral-transformations).

## `nconv > 0`. Am I done?

No. Converged Ritz pairs can still fail full-operator residuals, boundary leakage, or target distance on interior non-Hermitian problems. Trust the acceptance postchecks, not the convergence count: requested `nev` vs converged `nconv`, explicit postcheck residuals, leakage diagnostics, and target distances for accepted and rejected modes alike. See [getting started](getting-started/index.md#your-first-own-case).

## Can I compare my gain with another paper's number?

Only if both weights match: response weight $W$ and forcing weight $W_f$. Kinetic-energy-only norms misrank compressible modes because thermodynamic fluctuations carry real energy share, and a Chu gain is simply not the same number as an $L^2$ gain. Always report forcing norm, response norm, $\beta$, $\omega$, $\gamma$, grid, and SBP order with the value. See [core concepts](core-concepts.md#resolvent-vocabulary).

## Direct or iterative linear solves?

Direct (`KSPPREONLY` + `PCLU` + MUMPS) unless memory says otherwise — it is the accuracy reference, and one factorization serves every iteration and every $H$/$H^H$ apply at that shift. Iterative paths (`fgmres_lu`, `strumpack_fgmres`) are explicit opt-ins for cases that do not fit direct factorization, always with `true_residual` on because transformed residuals can lie. The decision procedure with measured costs is in the [solver glossary](solver-glossary.md#choosing-a-configuration).

## One rank or many?

Factorization dominates (about four fifths of a resolvent point: 60 s factor vs 0.44 s solves and 5.8 s eigensolve at 450k unknowns), and it scales poorly: measured 190.5 s at 4 ranks, 154.6 s at 8, 148.2 s at 16, while peak memory doubled from 18.6 to 40.6 GiB. Start at 4 ranks with one thread each, compare on your machine, and scale out across cases (scheduler arrays) before scaling up ranks. Numbers from `examples/savarino/*.csv`; protocol in [getting started](getting-started/index.md#the-example-family-map).

## Where do I put a question this page does not answer?

Open an issue with the section that failed you, the command, the error, and the `--diagnose` output. Install problems especially — they are the most useful reports this project gets.
