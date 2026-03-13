# Resolvent4py: a parallel Python package for analysis, model reduction and control of large-scale linear systems

Alberto Padovan<sup>a</sup> , Vishal Anantharaman<sup>b</sup> , Clarence W. Rowley<sup>c</sup> , Blaine Vollmer<sup>a</sup> , Tim Colonius<sup>b</sup> , Daniel J. Bodony<sup>a</sup>

## Abstract

In this paper, we present resolvent4py, a parallel Python package for the analysis, model reduction and control of large-scale linear systems with millions or billions of degrees of freedom. This package provides the user with a friendly Python-like experience (akin to that of well-established libraries such as numpy and scipy), while enabling MPI-based parallelism through mpi4py, petsc4py and slepc4py. In turn, this allows for the development of streamlined and efficient Python code that can be used to solve several problems in fluid mechanics, solid mechanics, graph theory, molecular dynamics and several other fields.

Keywords: Python, Parallel computing, Model reduction, Resolvent analysis, Stability analysis, Harmonic resolvent analysis

## Metadata

## 1. Motivation and significance

The development of this package is motivated by recent (and not-so-recent) advances in the field of fluid mechanics, where linear analysis has been shown to provide great insights into flow physics. For example, linear systems theory can be used to study the stability of fluid flows around equilibria, develop feedback control strategies to modify the flow behavior and compute reducedorder models to accelerate physical simulations. (We refer the reader to the foundational work of [\[1\]](#page-11-0) and to the review papers in [\[2,](#page-11-1) [3\]](#page-11-2) for a more in-depth

<sup>a</sup>Department of Aerospace Engineering, University of Illinois Urbana-Champaign, 104 S. Wright St., Urbana, IL 61802

<sup>b</sup>Department of Mechanical and Civil Engineering, California Institute of Technology, 1200 E. California Boulevard, Pasadena, CA 91125

<sup>c</sup>Department of Mechanical and Aerospace Engineering, Princeton University, Olden St., Princeton, NJ 08540

Table 1: Code metadata

| Nr. | Code metadata description                                       | Metadata                                       |
|-----|-----------------------------------------------------------------|------------------------------------------------|
| C1  | Current code version                                            | v1.0.1                                         |
| C2  | Permanent link to code/repository used for this code version    | https://github.com/albertopadovan/resolvent4py |
| C3  | Permanent link to Reproducible Capsule                          | –                                              |
| C4  | Legal Code License                                              | MIT                                            |
| C5  | Code versioning system used                                     | Git (GitHub)                                   |
| C6  | Software code languages, tools, and services used               | Python, Petsc4py, Slepc4py, Mpi4py             |
| C7  | Compilation requirements, operating environments & dependencies | See README file in package distribution        |
| C8  | If available Link to developer documentation/manual             | https://albertopadovan.github.io/resolvent4py/ |
| C9  | Support email for questions                                     | alberto.padovan.94@gmail.com                   |

overview of the importance and impact of linear systems theory in the field of fluid mechanics.) For systems of moderate dimension (e.g., fewer than 10,000 states), all these tasks can be performed straightforwardly with a few lines of Python code thanks to scientific computing libraries like scipy. For larger systems, the need for distributed-memory parallelism significantly increases the implementation complexity of these algorithms, and it often requires the development of application-specific software written in compiled languages such as C, C++ and Fortran. In fact, until the recent development of Python libraries like mpi4py and petsc4py, compiled languages represented the only way to write MPI-based[1](#page-1-0) code. The objective of this package is to address these issues and provide a user-friendly pythonic environment similar to that of scipy and numpy, while exploiting the large-scale parallelism offered by the mpi4py, petsc4py and slepc4py libraries. In turn, this enables the development of user-friendly, streamlined and efficient code to perform dayto-day engineering tasks and physical analyses of large-scale systems.

Remark 1. Although the development of this package is motivated by problems in the field of fluid mechanics, resolvent4py is application agnostic,

<span id="page-1-0"></span><sup>1</sup>MPI stands for Message Passing Interface, and it refers to the communication protocol used to exchange data across different processors in distributed-memory systems.

and its functionalities can be leveraged for any linear system of equations regardless of their origin.

## 2. Software description

resolvent4py is a Python library for the analysis, model reduction and control of large-scale linear systems. It relies on mpi4py [\[4\]](#page-11-3) for distributedmemory parallelism, and it leverages the data structures and functionalities provided by the petsc4py [\[5\]](#page-11-4) and slepc4py [\[6\]](#page-11-5) libraries.

### 2.1. Software architecture

At the core of this package is an abstract class, called LinearOperator, which serves as a blueprint for user-defined child classes that can be used to define a linear operator L. All child classes must implement at least three methods: apply(), which defines the action of L on a vector, apply\_mat(), which defines the action of L on a matrix and destroy() to free the memory. The implementation of the second method may seem redundant to the reader who is familiar with native Python, where the dot() method can be used to apply the action of a linear operator to vectors and matrices alike. However, the need for this method is dictated by the fact that matrix-vector products and matrix-matrix products are performed with different functions and data structures in petsc4py.

If desired, additional methods can be implemented. For example, we often define the action of the hermitian transpose of a linear operator on vectors and matrices (apply\_hermitian\_transpose() and apply\_hermitian\_trans pose\_mat(), respectively), the action of the inverse L <sup>−</sup><sup>1</sup> on vectors, solve(), etc. Below is a list of concrete subclasses of LinearOperator that are provided by resolvent4py:

- 1. MatrixLinearOperator: the linear operator L is defined by a sparse, MPI-distributed PETSc matrix.
- 2. LowRankLinearOperator: the linear operator L is defined as the product of low-rank factors,

$$L = U\Sigma V^*, \tag{1}$$

where U and V are "tall and skinny" matrices (i.e., matrices with much fewer columns than rows) stored as MPI-distributed SLEPc BV structures[2](#page-2-0) , while Σ is a sequential (i.e., non-distributed) two-dimensional numpy array of appropriate size.

<span id="page-2-0"></span><sup>2</sup>The SLEPc BV structure (where BV stands for "Basis Vector") holds a tall and skinny dense matrix whose rows are distributed across different MPI processors. The columns,

3. LowRankUpdatedLinearOperator: the linear operator L is defined as L = A + M, where A is itself a resolvent4py linear operator and M is a low-rank linear operator, as described in the previous bullet point. An operator of this form arises in several applications (e.g., in linear feedback control theory [\[7\]](#page-11-6)) and, if A and L are full-rank, then the inverse L −1 can be computed using the Woodbury matrix inversion lemma [\[8\]](#page-11-7),

$$L^{-1} = A^{-1} - A^{-1}U\Sigma \left(I + V^*A^{-1}U\Sigma\right)^{-1}V^*A^{-1}.$$
 (2)

- 4. ProjectionLinearOperator: the linear operator L is defined either as L = Φ (Ψ<sup>∗</sup>Φ)<sup>−</sup><sup>1</sup> Ψ<sup>∗</sup> or L = I − Φ (Ψ<sup>∗</sup>Φ)<sup>−</sup><sup>1</sup> Ψ<sup>∗</sup> , where Φ and Ψ are tall and skinny matrices of size N ×r stored as SLEPc BVs. In both cases, L defines a projection (i.e., L <sup>2</sup> = L).
- 5. ProductLinearOperator: the linear operator L is defined as the product of other resolvent4py linear operators L<sup>i</sup> of conformal dimensions,

$$L = L_m L_{m-1} \dots L_2 L_1. \tag{3}$$

### 2.2. Software functionalities

Once a linear operator is properly defined, resolvent4py currently allows for several types of analyses.

- 1. Eigendecomposition: the user can compute the left and right eigendecomposition of the linear operator L using the Arnoldi algorithm [\[9\]](#page-11-8). The shift-and-invert technique [\[10\]](#page-11-9) can be used to compute the eigenvalues that are closest to a target value s ∈ C, where C is the set of complex numbers.
- 2. Singular value decomposition (SVD): the user can compute an SVD of the linear operator using randomized linear algebra [\[11,](#page-12-0) [12\]](#page-12-1). This is useful for resolvent analysis [\[1,](#page-11-0) [13,](#page-12-2) [14\]](#page-12-3) and harmonic resolvent analysis [\[15,](#page-12-4) [16,](#page-12-5) [17,](#page-12-6) [18\]](#page-12-7). For the specific case of resolvent analysis, the package also offers the possibility of computing the SVD using time-stepping techniques [\[19,](#page-12-8) [20\]](#page-12-9).
- 3. Linear time-invariant balanced truncation: given the linear operator L, and appropriately-defined input and output matrices B and C, the user may compute and balance the associated controllability and observability Gramians for model reduction purposes [\[21,](#page-12-10) [22,](#page-13-0) [23,](#page-13-1) [24\]](#page-13-2). Specifically,

however, are not distributed, and this allows for significantly reduced parallel overhead when the number of MPI processors is much larger than the number of columns, as is usually the case in practice.

we implement the algorithm outlined in [\[25\]](#page-13-3). We remark that there exist additional model reduction formulations for linear system, (see, e.g., [\[26,](#page-13-4) [27,](#page-13-5) [28,](#page-13-6) [29\]](#page-13-7) for H<sup>2</sup> and H<sup>∞</sup> model reduction and variations thereof), but these are not yet implemented in the package.

Additionally, resolvent4py ships with several functions—available under the resolvent4py/utils directory and directly accessible to the user via the resolvent4py namespace—that further facilitate the use of our package and allow for streamlined application-specific code development. These include support for:

- 1. parallel I/O through petsc4py,
- 2. MPI-based communications using mpi4py,
- 3. manipulation of PETSc matrices/vectors and SLEPc BVs.

All these features are thoroughly documented using sphinx, and demonstrated with several examples that ship with the package.

<span id="page-4-0"></span>

### 2.3. Sample code snippet

In this subsection, we show a code snippet similar to the one used in section [3.2](#page-6-0) to perform resolvent analysis on the hypersonic flow over a cone. The first thing performed by this piece of code is to read in the sparse complex-valued matrix A of size N ×N. This matrix is stored on disk in sparse COO format by means of PETSc-compatible binary files "rows.dat", "columns.dat" and "values.dat". After assembling A as a sparse PETSc matrix, we create a corresponding MatrixLinearOperator object and we perform resolvent analysis using the RSVD-∆t algorithm presented in [\[19,](#page-12-8) [20\]](#page-12-9). The desired singular values and vectors of the resolvent operator are then saved to file using the parallel I/O routines available under the resolvent4py namespace. Running this piece of code with, e.g., 20 processors can be done with the command mpiexec -n 20 python code\_snippet.py.

Listing 1: Resolvent analysis via time stepping

```
import resolvent4py as res4py
# Define problem size and read jacobian from file
N = 121404 # Number of global rows (and columns) of A
Nloc = res4py.compute_local_size(N) # Number of local rows
file_names = ["rows.dat", "columns.dat", "values.dat"]
A = res4py.read_coo_matrix(file_names, ((Nloc, N), (Nloc, N)))
# Create resolvent4py linear operator
L = res4py.linear_operators.MatrixLinearOperator(A)
```

```
# Define the inputs to the rsvd-dt routine
dt = 1e-5 # Time step size
omega = 1.0 # Fundamental frequency (sets period T = 2pi/omega)
n_omegas = 5 # Number of harmonics of the fundamental to resolve
n_periods = 100 # Number of periods to integrate through
n_loop = 1 # Number of power iterations
n_rand = 30 # Number of random vectors
n_svals = 3 # Number of singular values/vectors to resolve
tol = 1e-3 # Tolerance to declare that transients have decayed
verbose = 1 # Verbosity level
Ulst, Slst, Vlst = res4py.linalg.resolvent_analysis_rsvd_dt(L, dt,
   omega, n_omegas, n_periods, n_loops, n_rand, n_svals, tol,
   verbose)
# Save to file
np.save("S.npy", Slst[1])
res4py.write_to_file("U.dat", Ulst[1])
res4py.write_to_file("V.dat", Vlst[1])
```

Remark 2. We would like to stress how this snippet exposes the applicationagnostic nature of our package. The governing equations (e.g., the linearized fluid dynamics equations, the linearized molecular dynamics equations, or any other linear equation from other branches of physics) are embedded in the sparse matrix A that is passed as an input to the script. Once A is loaded from file, resolvent4py performs the desired analysis without additional knowledge of the underlying physics.

## 3. Illustrative examples

<span id="page-5-0"></span>

### 3.1. Hypersonic flow over a compression ramp

The first example is a Mach–7.7 flow over a 15 deg compression ramp of length L = 0.1 m. In the freestream, the Mach number is M<sup>∞</sup> = 7.7, the pressure is p<sup>∞</sup> = 760 P a, the temperature is T<sup>∞</sup> = 125 K, the streamwise velocity is u<sup>∞</sup> = 1726 m/s and the Reynolds number based on L is Re<sup>∞</sup>,L = 4.2 × 10<sup>5</sup> . This flow configuration is studied extensively in [\[30\]](#page-13-8), and we refer to the latter for details on boundary conditions, spatial discretization, mesh and nondimensionalization. As discussed in [\[30\]](#page-13-8), this flow admits a two-dimensional steady state solution (see figure [1\)](#page-6-1) that is unstable with respect to threedimensional perturbations. These instabilities can be detected by linearizing

<span id="page-6-1"></span>![](_page_6_Figure_0.jpeg)

Figure 1: Mach number contours of two-dimensional steady state solution for the Mach– 7.7 flow over a 15 deg compression ramp.

the governing equations about the aforementioned steady state and studying the dynamics of three-dimensional, spanwise-periodic perturbations with wavenumber β. Upon linearization, the spatially-discretized compressible Navier-Stokes equations can be written compactly as

$$\frac{d}{dt}q_{\beta} = A_{\beta}q_{\beta}, \quad q_{\beta} \in \mathbb{C}^{N}, \tag{4}$$

where the state vector q<sup>β</sup> denotes the wavenumber-β Fourier coefficients of the spatially-discretized flow variables. Here, given a computational grid of size n<sup>y</sup> × n<sup>x</sup> = 236 × 1076 and 5 flow variables, the size of the state vector is N = 5 n<sup>y</sup> n<sup>x</sup> ≈ 1.27 × 10<sup>6</sup> . The eigenvalues of A<sup>β</sup> and corresponding eigenvectors for (non-dimensional) wavenumber β = 2π/0.066 are computed using resolvent4py with 240 processors on Stampede3 at Texas Advanced Computing Center. The eigenvalues are shown in figure [2,](#page-7-0) where we see good agreement with the results in Cao et al. [\[30\]](#page-13-8). Indeed, we find that the flow exhibits multiple instabilities corresponding to the five eigenvalues λ with real part greater than zero. The flow structures associated with these instability modes are shown in figure [3,](#page-7-1) where we plot contours of the real part of the spanwise velocity component from the eigenvectors corresponding to three unstable eigenvalues. Here, we find that the eigenvector corresponding to λ<sup>1</sup> exhibits very good qualitative agreement with its counterpart in figure 14a in [\[30\]](#page-13-8). The eigenvectors corresponding to λ<sup>2</sup> and λ<sup>3</sup> are very similar in nature to those in figures 14b and 14c in [\[30\]](#page-13-8), but it is important to observe that a direct comparison cannot be made since the modes shown in figures 14b and 14c in [\[30\]](#page-13-8) correspond to wavenumbers β = 0.070 and β = 0.079, respectively.

<span id="page-6-0"></span>

### 3.2. Hypersonic flow over a cone

Another illustrative example is a Mach-5.9 flow over a 7 deg half-angle blunt cone. The freestream quantities are M<sup>∞</sup> = 5.9, p<sup>∞</sup> = 3396.3 P a,

<span id="page-7-0"></span>![](_page_7_Figure_0.jpeg)

Figure 2: Eigenvalues of the linearized Navier-Stokes equations at non-dimensional spanwise wavenumber β = 2π/0.066 computed using resolvent4py (black dots). The red crosses denote the eigenvalues in [\[30\]](#page-13-8) for comparison. The eigenvectors associated with λ1, λ<sup>2</sup> and λ<sup>3</sup> are shown in figure [3.](#page-7-1)

<span id="page-7-1"></span>![](_page_7_Figure_2.jpeg)

Figure 3: Real part of the spanwise velocity component from the eigenvectors corresponding to eigenvalues λ1, λ<sup>2</sup> and λ<sup>3</sup> in figure [2.](#page-7-0)

<span id="page-8-0"></span>![](_page_8_Figure_0.jpeg)

Figure 4: Mach number contours of two-dimensional steady state solution for the Mach– 5.9 flow over a 7 deg blunt cone, truncated to solely the post-shock regime and normalized by domain length.

T<sup>∞</sup> = 76.74 K, u<sup>∞</sup> = 1036.1 m/s, and Re<sup>∞</sup> = 18 × 10<sup>6</sup> and the domain length is L = 0.4893 m. Bluntness effects produce unique instability phenomena in comparison with sharp cones, and are extensively studied using various DNS and operator theoretic methods in receptivity studies [\[31\]](#page-13-9). At high Mach numbers, a discrete higher-order mode termed the second Mack mode becomes destabilized and dominates laminar-to-turbulent transition. This mechanism manifests itself as acoustic waves trapped within the boundary layer. Likewise, response modes are modulated by the entropy layer observed near the front of the cone, which can contribute to distinct turbulent transition mechanisms.

The spatially-discretized Navier-Stokes equations are linearized about the base flow shown in figure [4,](#page-8-0) and the singular value decomposition of the resolvent operator is computed using the RSVD-∆t algorithm using a script almost identical to the one shown in section [2.3.](#page-4-0) The temperature perturbations extracted from the leading right and left singular vectors of the resolvent are seen in figure [5.](#page-9-0) The response mode corresponding to maximal amplification appears trapped in the boundary layer, while the forcing mode follows the decaying entropy layer's contour within the boundary layer, indicative of the importance of the high entropy gradients of the flow due to bluntness.

## 4. Impact

In the fluid mechanics community alone, linear systems tools like stability analysis [\[1,](#page-11-0) [30,](#page-13-8) [32\]](#page-13-10), resolvent analysis [\[13,](#page-12-2) [14,](#page-12-3) [33\]](#page-14-0), open-loop and feedback control design [\[34,](#page-14-1) [35,](#page-14-2) [36\]](#page-14-3), model reduction of linear time-invariant and time-

<span id="page-9-0"></span>![](_page_9_Figure_0.jpeg)

Figure 5: Temperature forcing and response modes computed using RSVD-∆t plotted in streamwise and wall-normal coordinates normalized by the streamwise length of the domain.

periodic systems via balanced truncation [\[21,](#page-12-10) [23,](#page-13-1) [25,](#page-13-3) [37\]](#page-14-4), etc., are featured in hundreds of peer-reviewed publications and industry applications. However, despite the popularity of these methods and algorithms, there is no unified platform that helps with their streamlined implementation. resolvent4py aims to bridge this gap by offering a Python package that can significantly reduce the upfront cost associated with developing parallel code to answer pressing research questions in physics and engineering. Specifically, not only does resolvent4py ship with fully-parallel state-of-the-art algorithms [\[9,](#page-11-8) [11,](#page-12-0) [12,](#page-12-1) [25\]](#page-13-3) that are used on a daily basis by researchers and engineers in several branches of physics, but it also offers a user-friendly, pythonic environment that allows for rapid code development, testing and deployment. In other words, our package can help scientists and engineers solve very large-scale problems of practical interest at a fraction of the human cost that would otherwise be necessary to develop application-specific software. The simple and self-contained code snippet in section [2.3](#page-4-0) is a testament to the friendly environment enabled by our package.

Additionally, we believe that the current version of resolvent4py can serve as a stepping stone for additional software development—both by the original developers and the scientific community at large—that will be featured in future releases. For example, the infrastructure of resolvent4py enables the straightforward implementation of time-periodic balanced truncation using "Frequential Gramians" [\[37\]](#page-14-4), harmonic resolvent analysis via time stepping [\[38\]](#page-14-5), wavelet-based resolvent analysis [\[39\]](#page-14-6) and One-Way Navier-Stokes (OWNS) for slowly-developing flows [\[40\]](#page-14-7). Furthermore, it should be possible

to implement the algorithms discussed in [\[41,](#page-14-8) [42\]](#page-14-9) for the efficient computation of low-rank solutions to Lyapunov and Riccati equations of large-scale, sparse linear systems such as those that arise from the spatial discretization of partial differential equations. These solutions could then be leveraged for model reduction and closed-loop control design.

## 5. Conclusions

In this manuscript, we introduced resolvent4py, a parallel Python package for the analysis, model reduction and control of large-scale linear systems. Large-scale linear systems with millions or billions of degrees of freedom are ubiquitous in the sciences and engineering, and this package is built to provide a friendly pythonic environment for rapid development and deployment of MPI-based parallel software to solve problems of practical interest. Our library leverages mpi4py for distributed-memory parallelism, and it uses the functionalities and data structures provided by petsc4py and slepc4py to enable stability analysis, input/output analysis, model reduction, control and optimization of linear time-invariant and time-periodic systems. Additionally, the package ships with several functions and features that are available to the user through the resolvent4py namespace and that facilitate user-specific and application-specific software development within the larger resolvent4py infrastructure. Finally, although the primary focus of this package is on linear systems, we wish to remark that the data structures and functionalities of resolvent4py can be extended straightforwardly to accommodate the analysis, model reduction and control of large-scale nonlinear systems.

## Acknowledgements

This material is based upon work supported by the National Science Foundation under Grant No. 2139536, issued to the University of Illinois at Urbana-Champaign by the Texas Advanced Computing Center under subaward UTAUS-SUB00000545 with Dr. Daniel Stanzione as the PI. DB gratefully acknowledges support from the Office of Naval Research (N00014-21- 1-2256), and CR gratefully acknowledges support from the Air Force Office of Scientific Research (FA9550-19-1-0005). TC and VA gratefully acknowledge support from The Boeing Company (CT-BA-GTA-1) and the Office of Naval Research (N00014-25-1-2072). BV was supported by the LDRD Program at Sandia National Laboratories. Sandia is managed and operated by NTESS under DOE NNSA contract DE-NA0003525. The computations in section [3.1](#page-5-0) were performed on TACC's Stampede3 under ACCESS allocation CTS090004.

## References

- <span id="page-11-0"></span>[1] P. J. Schmid, D. S. Henningson, Stability and Transition in Shear Flows, Vol. 142, Springer-Verlag New York, 2001.
- <span id="page-11-1"></span>[2] C. W. Rowley, S. T. M. Dawson, Model reduction for flow analysis and control, Annual Review of Fluid Mechanics 49 (2017) 387–417.
- <span id="page-11-2"></span>[3] M. R. Jovanović, From bypass transition to flow control and data-driven turbulence modeling: An input–output viewpoint, Annual Review of Fluid Mechanics 53 (Volume 53, 2021) (2021) 311–345. [doi:https:](https://doi.org/https://doi.org/10.1146/annurev-fluid-010719-060244) [//doi.org/10.1146/annurev-fluid-010719-060244](https://doi.org/https://doi.org/10.1146/annurev-fluid-010719-060244).
- <span id="page-11-3"></span>[4] L. Dalcin, P. Kler, R. Paz, A. Cosimo, Parallel distributed computing using python, Advances in Water Resources 34 (9) (2011) 1124–1139. [doi:10.1016/j.advwatres.2011.04.013](https://doi.org/10.1016/j.advwatres.2011.04.013).
- <span id="page-11-4"></span>[5] S. Balay, S. Abhyankar, M. Adams, S. Benson, J. Brown, P. Brune, K. Buschelman, E. Constantinescu, L. Dalcin, A. Dener, V. Eijkhout, J. Faibussowitsch, W. Gropp, V. Hapla, T. Isaac, P. Jolivet, D. Karpeyev, D. Kaushik, M. Knepley, F. Kong, S. Kruger, D. May, L. C. McInnes, R. Mills, L. Mitchell, T. Munson, J. Roman, K. Rupp, P. Sanan, J. Sarich, B. Smith, H. Zhang, S. Zampini, H. Zhang, J. Zhang, PETSc/TAO Users Manual, ANL-21/39 - Revision 3.23 (2025). [doi:](https://doi.org/10.2172/2476320) [10.2172/2476320](https://doi.org/10.2172/2476320).
- <span id="page-11-5"></span>[6] V. Hernandez, J. E. Roman, V. Vidal, SLEPc: A scalable and flexible toolkit for the solution of eigenvalue problems, ACM Trans. Math. Software 31 (3) (2005) 351–362.
- <span id="page-11-6"></span>[7] M. Natarajan, J. B. Freund, D. J. Bodony, Actuator selection and placement for localized feedback flow control, Journal of Fluid Mechanics 809 (2016) 775–792. [doi:10.1017/jfm.2016.700](https://doi.org/10.1017/jfm.2016.700).
- <span id="page-11-7"></span>[8] M. A. Woodbury, Inverting modified matrices, in: Memorandum Rept. 42, Statistical Research Group, Princeton Univ., 1950, p. 4.
- <span id="page-11-8"></span>[9] W. E. Arnoldi, The principle of minimized iterations in the solution of the matrix eigenvalue problem, Quart. Appl. Math. 9 (1951) 17–29. [doi:10.1090/qam/42792](https://doi.org/10.1090/qam/42792).
- <span id="page-11-9"></span>[10] Y. Saad, Numerical Methods for Large Eigenvalue Problems, SIAM, 2011. [doi:10.1137/1.9781611970739](https://doi.org/10.1137/1.9781611970739).

- <span id="page-12-0"></span>[11] N. Halko, P. G. Martinsson, J. A. Tropp, Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions, SIAM Review 53 (2011) 217–288.
- <span id="page-12-1"></span>[12] J. H. M. Ribeiro, C.-A. Yeh, K. Taira, Randomized resolvent analysis, Phys. Rev. Fluids 5 (2020) 033902. [doi:10.1103/PhysRevFluids.5.](https://doi.org/10.1103/PhysRevFluids.5.033902) [033902](https://doi.org/10.1103/PhysRevFluids.5.033902).
- <span id="page-12-2"></span>[13] M. R. Jovanović, B. Bamieh, Componentwise energy amplification in channel flows, Journal of Fluid Mechanics 534 (2005) 145–183.
- <span id="page-12-3"></span>[14] B. J. McKeon, A. S. Sharma, A critical-layer framework for turbulent pipe flow, Journal of Fluid Mechanics 658 (2010) 336–382.
- <span id="page-12-4"></span>[15] A. Padovan, S. E. Otto, C. W. Rowley, Analysis of amplification mechanisms and cross-frequency interactions in nonlinear flows via the harmonic resolvent, Journal of Fluid Mechanics 900 (2020). [doi:doi:](https://doi.org/doi:10.1017/jfm.2020.497) [10.1017/jfm.2020.497](https://doi.org/doi:10.1017/jfm.2020.497).
- <span id="page-12-5"></span>[16] A. Padovan, C. W. Rowley, Analysis of the dynamics of subharmonic flow structures via the harmonic resolvent: Application to vortex pairing in an axisymmetric jet, Phys. Rev. Fluids 7 (2022) 073903. [doi:10.](https://doi.org/10.1103/PhysRevFluids.7.073903) [1103/PhysRevFluids.7.073903](https://doi.org/10.1103/PhysRevFluids.7.073903).
- <span id="page-12-6"></span>[17] W. Wu, C. Meneveau, R. Mittal, A. Padovan, C. W. Rowley, L. Cattafesta, Response of a turbulent separation bubble to zero-netmass-flux jet perturbations, Phys. Rev. Fluids 7 (2022) 084601. [doi:](https://doi.org/10.1103/PhysRevFluids.7.084601) [10.1103/PhysRevFluids.7.084601](https://doi.org/10.1103/PhysRevFluids.7.084601).
- <span id="page-12-7"></span>[18] M. R. Islam, Y. Sun, Identification of cross-frequency interactions in compressible cavity flow using harmonic resolvent analysis, Journal of Fluid Mechanics 1000 (2024) A13. [doi:10.1017/jfm.2024.949](https://doi.org/10.1017/jfm.2024.949).
- <span id="page-12-8"></span>[19] E. Martini, D. Rodríguez, A. Towne, A. V. Cavalieri, Efficient computation of global resolvent modes, Journal of Fluid Mechanics 919 (2021) A3. [doi:10.1017/jfm.2021.364](https://doi.org/10.1017/jfm.2021.364).
- <span id="page-12-9"></span>[20] A. Farghadan, E. Martini, A. Towne, Scalable resolvent analysis for three-dimensional flows, Journal of Computational Physics 524 (2025) 113695. [doi:https://doi.org/10.1016/j.jcp.2024.113695](https://doi.org/https://doi.org/10.1016/j.jcp.2024.113695).
- <span id="page-12-10"></span>[21] B. Moore, Principal component analysis in linear systems: Controllability, observability, and model reduction, IEEE Transactions on Automatic Control 26 (1) (1981) 17–32.

- <span id="page-13-0"></span>[22] G. E. Dullerud, F. Paganini, A Course in Robust Control Theory: a Convex Approach, Springer New York, 2000.
- <span id="page-13-1"></span>[23] C. W. Rowley, Model reduction for fluids using balanced proper orthogonal decomposition, Int. J. of Bifurcation and Chaos 15 (03) (2005) 997–1013.
- <span id="page-13-2"></span>[24] U. Baur, P. Benner, L. Feng, Model order reduction for linear and nonlinear systems: a system-theoretic perspective 21 (4) (2014) 331–358.
- <span id="page-13-3"></span>[25] G. Dergham, D. Sipp, J.-C. Robinet, A. Barbagallo, Model reduction for fluids using frequential snapshots, Physics of Fluids 23 (2011) 064101. [doi:10.1063/1.3590732](https://doi.org/10.1063/1.3590732).
- <span id="page-13-4"></span>[26] S. Gugercin, A. C. Antoulas, C. Beattie, H<sup>2</sup> model reduction for largescale linear dynamical systems 30 (2) (2008) 609–638.
- <span id="page-13-5"></span>[27] G. Flagg, C. A. Beattie, S. Gugercin, Interpolatory H<sup>∞</sup> model reduction, Systems & Control Letters 62 (7) (2013).
- <span id="page-13-6"></span>[28] P. Benner, S. Gugercin, K. Willcox, A survey of projection-based model reduction methods for parametric dynamical systems 57 (4) (2015) 483– 531.
- <span id="page-13-7"></span>[29] P. Benner, V. Mehrmann, D. C. Sorensen, Dimension reduction of largescale systems, in: Dimension Reduction of Large-Scale Systems, Vol. 45 of Lecture Notes in Computational Science and Engineering, Springer, 2005, pp. 3–32. [doi:10.1007/3-540-27909-1\\_1](https://doi.org/10.1007/3-540-27909-1_1).
- <span id="page-13-8"></span>[30] S. Cao, J. Hao, I. Klioutchnikov, H. Olivier, C.-Y. Wen, Unsteady effects in a hypersonic compression ramp flow with laminar separation, Journal of Fluid Mechanics 912 (2021) A3. [doi:10.1017/jfm.2020.1093](https://doi.org/10.1017/jfm.2020.1093).
- <span id="page-13-9"></span>[31] D. A. Cook, J. W. Nichols, Three-dimensional receptivity of hypersonic sharp and blunt cones to free-stream planar waves using hierarchical input-output analysis, Phys. Rev. Fluids 9 (2024) 063901. [doi:10.1103/](https://doi.org/10.1103/PhysRevFluids.9.063901) [PhysRevFluids.9.063901](https://doi.org/10.1103/PhysRevFluids.9.063901).
- <span id="page-13-10"></span>[32] B. Vollmer, A. Padovan, D. J. Bodony, [Surface instabilities in laminar](https://arxiv.org/abs/2503.11631) [compressible boundary layers with sublimation](https://arxiv.org/abs/2503.11631) (2025). [arXiv:2503.](http://arxiv.org/abs/2503.11631) [11631](http://arxiv.org/abs/2503.11631).
  - URL <https://arxiv.org/abs/2503.11631>

- <span id="page-14-0"></span>[33] B. Herrmann, P. J. Baddoo, R. Semaan, S. L. Brunton, B. J. McKeon, Data-driven resolvent analysis, Journal of Fluid Mechanics 918 (2021) A10. [doi:10.1017/jfm.2021.337](https://doi.org/10.1017/jfm.2021.337).
- <span id="page-14-1"></span>[34] O. Semeraro, J. O. Pralits, C. W. Rowley, D. S. Henningson, Riccatiless approach for optimal control and estimation: an application to twodimensional boundary layers, Journal of Fluid Mechanics 731 (2013) 394–417.
- <span id="page-14-2"></span>[35] C.-A. Yeh, K. Taira, Resolvent-analysis-based design of airfoil separation control, Journal of Fluid Mechanics 867 (2019) 572–610. [doi:10.1017/](https://doi.org/10.1017/jfm.2019.163) [jfm.2019.163](https://doi.org/10.1017/jfm.2019.163).
- <span id="page-14-3"></span>[36] J. Woo, S. R. Murthy, D. J. Bodony, Resolvent-based framework for jet noise reduction of a low-bypass ratio coannular nozzle, 2024. [doi:](https://doi.org/10.2514/6.2024-2805) [10.2514/6.2024-2805](https://doi.org/10.2514/6.2024-2805).
- <span id="page-14-4"></span>[37] A. Padovan, C. W. Rowley, Continuous-time balanced truncation for time-periodic fluid flows using frequential Gramians, Journal of Computational Physics 496 (2024) 112597. [doi:https://doi.org/10.1016/](https://doi.org/https://doi.org/10.1016/j.jcp.2023.112597) [j.jcp.2023.112597](https://doi.org/https://doi.org/10.1016/j.jcp.2023.112597).
- <span id="page-14-5"></span>[38] A. Farghadan, J. Jung, R. Bhagwat, A. Towne, Efficient harmonic resolvent analysis via time stepping, Theoretical and Computational Fluid Dynamics 38 (3) (2024) 331. [doi:10.1007/s00162-024-00694-1](https://doi.org/10.1007/s00162-024-00694-1).
- <span id="page-14-6"></span>[39] E. Ballouz, B. Lopez-Doriga, S. T. Dawson, H. J. Bae, Wavelet-based resolvent analysis of non-stationary flows, Journal of Fluid Mechanics 999 (2024) A53. [doi:10.1017/jfm.2024.903](https://doi.org/10.1017/jfm.2024.903).
- <span id="page-14-7"></span>[40] A. Towne, G. Rigas, O. Kamal, E. Pickering, T. Colonius, Efficient global resolvent analysis via the one-way Navier–Stokes equations, Journal of Fluid Mechanics 948 (2022) A9. [doi:10.1017/jfm.2022.647](https://doi.org/10.1017/jfm.2022.647).
- <span id="page-14-8"></span>[41] P. Benner, J.-R. Li, T. Penzl, Numerical solution of large-scale lyapunov equations, riccati equations, and linear-quadratic optimal control problems, Numerical Linear Algebra with Applications 15 (9) (2008) 755–777. [doi:https://doi.org/10.1002/nla.622](https://doi.org/https://doi.org/10.1002/nla.622).
- <span id="page-14-9"></span>[42] P. Benner, J. Saak, Numerical solution of large and sparse continuous time algebraic matrix riccati and lyapunov equations: a state of the art survey, GAMM-Mitteilungen 36 (1) (2013) 32–52. [doi:https://doi.](https://doi.org/https://doi.org/10.1002/gamm.201310003) [org/10.1002/gamm.201310003](https://doi.org/https://doi.org/10.1002/gamm.201310003).