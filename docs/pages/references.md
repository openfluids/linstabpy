# References

The papers and manuals behind this wiki, grouped by what they justify. One line each on why it is cited. To cite this software, see [how to cite](index.md#cite-linstabpy) on the home page.

## The package and its lineage

* Hamada, G. Y. R., Wolf, W. R., Junqueira-Junior, C., Robinet, J.-C. and Frantz, R. A. S. LinStabPy: an efficient Python framework for compressible stability and resolvent analysis. *Theor. Comput. Fluid Dyn.* (under revision, 2026). The paper for this codebase. Prefer it when citing results.
* Martini, E. and Schmidt, O. T. (2024). LinStab2D: stability and resolvent analysis of compressible viscous flows in MATLAB. *Theor. Comput. Fluid Dyn.* 38, 665–685. doi:10.1007/s00162-024-00706-0. The methodology ancestor: linearized equations, SBP discretization, metrics, boundary conditions, filtering. Cite alongside the linstabpy paper.
* Padovan, A., Anantharaman, V., Rowley, C. W., Vollmer, B., Colonius, T. and Bodony, D. J. (2025). resolvent4py: a parallel Python package for analysis, model reduction and control of large-scale linear systems. *SoftwareX* 31, 102286. doi:10.1016/j.softx.2025.102286. Operator-handling design influence.

## Stability and transition theory

* Schmid, P. J. and Henningson, D. S. (2001). *Stability and Transition in Shear Flows*. Springer. The textbook companion for modal spectra, branches, and transition physics.
* Schmid, P. J. (2007). Nonmodal stability theory. *Annu. Rev. Fluid Mech.* 39, 129–162. Transient growth, optimal perturbations, and the finite-time viewpoint behind gain maps.
* Trefethen, L. N., Trefethen, A. E., Reddy, S. C. and Driscoll, T. A. (1993). Hydrodynamic stability without eigenvalues. *Science* 261, 578–584. Why spectra alone mislead for nonnormal operators; the founding case for pseudospectra thinking.
* Frantz, R. A. S., Loiseau, J.-Ch. and Robinet, J.-Ch. (2023). Krylov methods for large-scale dynamical systems: application in fluid dynamics. *Appl. Mech. Rev.* 75(3), 030802. doi:10.1115/1.4056808. Arnoldi factorization, Ritz convergence, modal vs non-modal bounds, Krylov practice. The review behind the [solver glossary](solver-glossary.md).
* Hanifi, A., Schmid, P. J. and Henningson, D. S. (1996). Transient growth in compressible boundary layer flow. *Phys. Fluids* 8, 826–837. doi:10.1063/1.868864. Compressible transient growth; the wall-normal mapping parameter in the Blasius example follows this work.

## Resolvent and adjoint methods

* McKeon, B. J. and Sharma, A. S. (2010). A critical-layer framework for turbulent pipe flow. *J. Fluid Mech.* 658, 336–382. Resolvent framing for amplifier flows.
* Luchini, P. and Bottaro, A. (2014). Adjoint equations in stability analysis. *Annu. Rev. Fluid Mech.* 46, 493–. The adjoint machinery behind receptivity and wavemaker fields.
* Chu, B.-T. (1965). On the energy transfer to small disturbances in fluid flow (Part I). *Acta Mech.* 1(3), 215–234. The compressible energy inner product used for every gain and norm here.

## Discretization

* Strand, B. (1994). Summation by parts for finite difference approximations for d/dx. *J. Comput. Phys.* 110(1), 47–67. First-derivative SBP operators.
* Mattsson, K. and Nordstrom, J. (2004). Summation by parts operators for finite difference approximations of second derivatives. *J. Comput. Phys.* 199(2), 503–540. Second-derivative SBP operators.
* Mattsson, K., Svard, M. and Nordstrom, J. (2004). Stable and accurate artificial dissipation. *J. Sci. Comput.* 21, 57–79. doi:10.1023/B:JOMP.0000027955.75872.3f. The dissipation operator behind the epsilon knob.
* Linders, V., Lundquist, T. and Nordstrom, J. (2018). On the order of accuracy of finite difference operators on diagonal norm based summation-by-parts form. *SIAM J. Numer. Anal.* 56, 1048–1063. doi:10.1137/17M1139333. Why diagonal-norm boundary closures lose order — and why only even orders 2, 4, 6, 8 ship here.
* Lele, S. K. (1992). Compact finite difference schemes with spectral-like resolution. *J. Comput. Phys.* 103, 16–42. Centered stencils annihilate the sawtooth mode at any order; the reason grid-scale content needs watching.
* Fornberg, B. (1988). Generation of finite difference formulas on arbitrarily spaced grids. *Math. Comp.* 51, 699–706. Weight generation behind the mesh operators.

## Solver stack

* Balay, S. et al. (2025). *PETSc/TAO Users Manual* (ANL-21/39, Rev. 3.24). Argonne National Laboratory. doi:10.2172/2998643. Distributed matrices, KSP/PC, the runtime under every solve.
* Hernandez, V., Roman, J. E. and Vidal, V. (2005). SLEPc: a scalable and flexible toolkit for the solution of eigenvalue problems. *ACM Trans. Math. Software* 31(3), 351–362. doi:10.1145/1089014.1089019. EPS eigensolvers and spectral transformations.
* Amestoy, P. R., Duff, I. S., Koster, J. and L'Excellent, J.-Y. (2001). A fully asynchronous multifrontal solver using distributed dynamic scheduling. *SIAM J. Matrix Anal. Appl.* 23(1), 15–41. MUMPS, the default sparse-direct backend.
