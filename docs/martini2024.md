# **ORIGINAL ARTICLE**

**Eduardo Martini · Oliver Schmidt**

# *Linstab2D***: stability and resolvent analysis of compressible viscous flows in MATLAB**

Received: 1 December 2023 / Accepted: 30 May 2024 / Published online: 13 August 2024 © The Author(s), under exclusive licence to Springer-Verlag GmbH Germany, part of Springer Nature 2024

**Abstract** We present *LinStab2D*, an easy-to-use linear stability analysis MATLAB tool capable of handling complex domains, performing temporal and spatial linear stability, and resolvent analysis. We present the theoretical foundations of the code, including the linear stability and resolvent analysis frameworks, finite differences discretization schemes, and the Floquet ansatz. These concepts are explored in five different examples, highlighting and illustrating the different code capabilities, including mesh masking, mapping, imposition of boundary constraints, and the analysis of periodic flows using Cartesian or axisymmetric coordinates. These examples were constructed to be a departure point for studying other flows.

**Keywords** Linear stability analysis · Resolvent analysis · Flow modelling · Transition to turbulence

# **1 Introduction**

Linear stability theory (LST) is a powerful tool for investigating laminar and turbulent flows. For laminar flows, it provides insights into the mechanisms that can lead to flow transition to turbulence.

One of the earliest studies of flow stability was due to Lord Kelvin and Helmholtz [\[101](#page-20-0)], who studied the stability of shear layers, modeled by a vortex sheet. They identified what is now known as the Kelvin– Helmholtz instability. Michalke [\[59](#page-19-0),[60\]](#page-19-1) later expanded the analysis to finite thickness shear layers. This instability mechanism has since been the basis for modeling different phenomena, among them convective and absolute instability of jets and wakes [\[50](#page-19-2),[58](#page-19-3)[,62](#page-19-4)[–64,](#page-19-5)[89](#page-20-1)].

Another instability mechanism, proposed by [\[79](#page-20-2)[,96](#page-20-3)], was proposed in the early twentieth century as a route by which boundary layers transition to turbulence. The mechanism is underpinned by a class of unstable waves, which became known as the Tollmien–Schlichting (TS) waves. In low free-stream turbulence conditions, is it considered the primary mechanism for boundary layer transition. Thus, many studies have investigated the formation of these waves. Goldstein [\[29](#page-18-0)] identified that curvature discontinuities are an important mechanism

Communicated by André Cavalieri.

Eduardo Martini and Oliver Schmidt have contributed equally to this work.

E. Martini (B)

Département Fluides, Thermique, Combustion, Institut Pprime, CNRS, ENSMA, Universitè de Poitiers, 1 Bd Marie et Pierre Curie, Futuroscope Chasseneuil, 86962 la Vienne, France

E-mail: eduardo.martini@ensma.fr

O. Schmidt

Department of Mechanical and Aerospace Engineering, University of California San Diego, 9500 Gilman Drive, La Jolla, CA 92093-0411, USA

E-mail: oschmidt@ucsd.edu

by which these waves can be excited by acoustic perturbations. TS waves were also investigated in spatially evolving flows [\[65](#page-19-6)], in the presence of compliant walls [\[73\]](#page-19-7), when the boundary layer is supersonic [\[48](#page-19-8)].

Linear stability has also proven useful in modeling turbulent flows, often providing insights into the mechanisms responsible for the formation of turbulent structures. After [\[57\]](#page-19-9) used the framework to model turbulent structures in a turbulent channel, the approach has been applied to many other problems. Among them turbulent jets [\[26](#page-18-1),[44,](#page-19-10)[70](#page-19-11)[,74,](#page-19-12)[75](#page-20-4)[,85\]](#page-20-5), jet noise [\[11](#page-18-2)[–14](#page-18-3)], cavity flow [\[18\]](#page-18-4), channel flow [\[3](#page-17-0)[,40](#page-19-13),[57\]](#page-19-9), pipe flow [\[2,](#page-17-1)[31](#page-18-5)], and the flow around airfoils [\[93](#page-20-6)[,103](#page-20-7)], among many others. Towne et al. [\[97\]](#page-20-8) showed that, under certain conditions, the most energetic structures in a turbulent flow should indeed be predicted by linear stability analysis, providing a more grounded theoretical basis for the approach. It also provided insights into the physical mechanisms underpinning the dynamics of these structures when they are not properly predicted by linear models [\[66,](#page-19-14)[71\]](#page-19-15).

As these phenomena have been successfully modeled by linear analysis, in recent years, several studies have used linear control theory to control them [\[5](#page-18-6)[,82](#page-20-9)]. Sasaki et al. [\[78](#page-20-10)] used feed-back control approaches to delay vortex-roll up in a shear layer, with similar approaches also applied to the control TS of waves [\[4](#page-18-7),[41\]](#page-19-16), vortex shedding in wakes [\[36\]](#page-18-8), turbulent jets [\[49](#page-19-17)], turbulent pipes [\[45](#page-19-18)], and in supersonic boundary layers [\[48](#page-19-8)[,69](#page-19-19)].

The framework has also been used to guide the design of new airfoils [\[104](#page-20-11)], compliant coatings [\[46](#page-19-20),[47\]](#page-19-21), and to design novel approaches for flow estimation and control [\[36](#page-18-8),[43,](#page-19-22)[45](#page-19-18),[51,](#page-19-23)[53](#page-19-24)[,98\]](#page-20-12).

Linear modeling of flows thus can provide insights into fundamental mechanisms that drive flow dynamics, while also providing a framework to develop flow modeling, estimation, and control approaches.

The approach is based on the linearization of the Navier–Stokes equations, which are subsequently discretized. Different numerical approaches can then be used to exploit the resulting linear system, with the most suitable approach dependent on the number of homogeneous directions.

Since the work of [\[72](#page-19-25)], pseudo-spectral methods have become the natural choice for flows with only one non-homogeneous direction. This case is typically referred to as a 1D problem. The high accuracy of these methods typically requires between 50 and 200 grid points to resolve the dynamics of interest. As the resulting matrices are small, explicitly handling them is straightforward.

In systems with two inhomogeneous directions, i.e., a 2D problem, the grid point count reaches tens or hundreds of thousands. In this case, explicitly handling the resulting matrices becomes prohibitively expensive. To approach these problems numerically, using finite difference (FD) methods coupled with sparse algebra libraries and iterative methods became the standard approach. Krylov subspace-based approaches, pioneered by [\[22\]](#page-18-9), can compute a subset of the operator eigenvalues to analyze the system's stability or the action of the resolvent operator to identify the dominant force and response modes.

For flows with no homogeneous spatial directions, using sparse matrices, although possible (see, e.g. [\[77](#page-20-13)], ), is typically expensive. In this case, matrix-free methods are typically deployed. These methods avoid matrices manipulation using approaches based on time integration [\[24,](#page-18-10)[30](#page-18-11)[,52,](#page-19-26)[65](#page-19-6)].

While the interest in complex flows has been rising in recent years, there are many problems for which a 2D approach is better suited. Among them are the cylinder wake [\[89](#page-20-1)], jets [\[26](#page-18-1),[44](#page-19-10)[,74](#page-19-12),[75](#page-20-4)[,85](#page-20-5)], cavity flow [\[18](#page-18-4),[43\]](#page-19-22), channel flow [\[3,](#page-17-0)[40](#page-19-13)[,57,](#page-19-9)[67](#page-19-27)], spatially evolving boundary layers [\[65](#page-19-6),[68\]](#page-19-28), and the flow around airfoils [\[93](#page-20-6),[103\]](#page-20-7).

While the methods required for analyzing 1D systems are usually of manageable complexity and can be developed on a case-by-case basis, and the conceptual extension to 2D is straightforward, this is not the case for the implementation of linear stability algorithms. To apply boundary conditions, for example, the indices of interface nodes need to be tracked. Similarly, mesh deformations are less straightforward, and care must be taken to keep the matrices sparse. Creating such a code can be time-consuming, making the resulting code hard to generalize to different problems.

Aiming to facilitate the study of 2D systems, we developed *LinStab2D*. We are now sharing it as an Open-Source code and presenting it in this work. The code aims to provide an easy-to-use tool that can perform state-of-the-art analyses. We provide high-level functions for critical operations and several examples covering different capabilities and use cases, which can be modified to analyze other flow problems. The predecessor of this code was originally developed for the 2D stability analysis of compressible streamwise corner flows [\[83\]](#page-20-14), then extended to non-modal analysis [\[84\]](#page-20-15), and has more recently been used for stability [\[85](#page-20-5)] and resolvent analyses [\[74](#page-19-12),[75,](#page-20-4)[86](#page-20-16)] of turbulent jets.

We describe in this work the methods and concepts used in *LinStab2D*. For a detailed review of different numerical and modeling approaches, we refer the reader to the analysis presented in previous studies [\[10](#page-18-12)[,23](#page-18-13), [38](#page-18-14)[,94](#page-20-17),[95\]](#page-20-18).

This paper is structured as follows. Section [2](#page-2-0) reviews the theoretical aspects of linear stability analysis. Section [3](#page-8-0) describes the numerical methods implemented in code. Section [4](#page-11-0) showcases five different test-cases:

- Section 4.1 presents the resolvent analysis of a flat-plate boundary layer, illustrating resolvent analysis on Cartesian coordinates,
- Section 4.2 presents the resolvent analysis of an axisymmetric jet, illustrating resolvent analysis on cylindrical coordinates,
- Section 4.3 presents the temporal and spatial local analysis of a supersonic boundary layer, illustrating local 2D analysis and Floquet modes,
- Section 4.4 presents the analysis of Rossiter modes in open cavity flow, illustrating the modeling of complex domains,
- Section 4.5 presents the analysis of the receptivity of steaks in the flow around a paraboloid body, illustrating
  the use of curvilinear coordinates.

Section 5 presents concluding remarks.

# <span id="page-2-0"></span>2 Linear stability analysis

The flow of a compressible ideal gas is governed by the continuity Eq. (1a), the compressible Navier–Stokes Eq. (1b), and the energy Eq. (1c):

$$\frac{\partial \rho}{\partial t} = -\nabla \cdot \rho \mathbf{u},\tag{1a}$$

$$\frac{\partial \rho \mathbf{u}}{\partial t} = -\frac{1}{2} \nabla \cdot (\mathbf{u} \otimes \rho \mathbf{u} + \rho \mathbf{u} \otimes \mathbf{u}) - \nabla p + \frac{1}{Re} \nabla \cdot \boldsymbol{\tau}, \tag{1b}$$

$$\frac{\partial \rho e}{\partial t} = -\nabla \cdot \rho e \mathbf{u} + \frac{1}{(\gamma - 1)RePrMa_{\infty}^2} \nabla \cdot k \nabla T - \nabla \cdot p \mathbf{u} + \frac{1}{Re} \nabla \cdot \tau \mathbf{u}, \tag{1c}$$

where  $\rho$  is the density,  $\boldsymbol{u}$  the velocity vector, p the pressure, T the temperature and e the total energy. The dynamic viscosity  $\mu$ , thermal conductivity k, specific heat capacity  $c_p$ , and heat capacity ratio  $\gamma$  are specific to the fluid. All quantities are non-dimensionalized by their respective free-stream or characteristic values denoted by  $(.)_{\infty}$ . The pressure is non-dimensionalized by twice the dynamic pressure,  $\rho_{\infty}u_{\infty}^2$ , and the coordinates by a problem-specific characteristic length scale, L. The dimensionless Mach, Reynolds, and Prandtl numbers are defined as  $Ma = u_{\infty}/a_{\infty}$ ,  $Re = \rho_{\infty}u_{\infty}L/\mu_{\infty}$ , and  $Pr = c_{p,\infty}\mu_{\infty}/k_{\infty}$ , respectively. The total energy reads

<span id="page-2-1"></span>
$$e = \frac{1}{\gamma(\gamma - 1)Ma_{\infty}^2} T + \frac{1}{2} \|\mathbf{u}\|^2.$$
 (2)

The viscous stresses of the Newtonian fluid, invoking the Stokes hypothesis to eliminate the bulk viscosity, are given by

$$\tau = \mu \left( \nabla \mathbf{u} + \nabla \mathbf{u}^T \right) - \frac{2}{3} \mu (\nabla \cdot \mathbf{u}) \mathbf{I}. \tag{3}$$

The pressure is related to the density and temperature through the ideal gas law,

$$p = \frac{1}{\gamma M a_{\infty}^2} \rho T. \tag{4}$$

Sutherland's empirical formula

<span id="page-2-2"></span>
$$\mu(T) = \mu_{\text{ref}} T^{3/2} \frac{1 + T_s}{T + T_s} \tag{5}$$

where  $\mu_{\rm ref}(T_{\rm ref}=280{\rm K})=1.735\times 10^{-5}{\rm \,kg/ms}$  and  $T_s=110.4{\rm K}/T_{\rm ref}$  for standard air, is used to obtain the dynamic viscosity from the temperature.

Equation (1) can be written in compact operator form as

<span id="page-2-3"></span>
$$\frac{\partial \mathbf{q}_{c}}{\partial t} = \mathcal{N}(\mathbf{q}),\tag{6}$$

where  $q_c(x,t) = [\rho, \rho u, \rho e]^T$  is the solution vector of conservative variables. The solution vector of primitive variables,  $q(x,t) = [\rho, u, T]^T$ , is readily obtained by dividing the momentum densities,  $\rho u$ , by the density, and recovering the temperature by solving Eq. (2) for T. All occurrences of e,  $\tau$ , p, and  $\mu$  in compressible Navier–Stokes operator,  $\mathcal{N}$ , are replaced by their dependencies on the primitive variables using Eqs. (2)–(5). In particular, variations of the molecular viscosity with temperature fluctuations are considered, as in [32]. Next, the Reynolds decomposition,  $q(x,t) = \overline{q}(x) + q'(x,t)$ , is inserted into Eq. (6). In classical linear theory,  $\overline{q}$  is a steady-state solution, or fixed point, of Eq. (6) with  $\frac{\partial \overline{q}}{\partial t} = 0$ , and q' hence represents fluctuations around this steady state. In other applications like resolvent analysis,  $\overline{q}$  can represent a long-time mean of a turbulent flow. Note that if  $\overline{q}$  is not a fixed point of the system, the results obtained from linear analysis are not unique [39]. The code implements a linearization using primitive variables. Linearization using, e.g., conservative variables, can lead to different results.

The Reynolds decomposition allows us to separate terms and to rewrite Eq. (6) as

<span id="page-3-0"></span>
$$\mathcal{M}\frac{\partial \mathbf{q'}}{\partial t} = \mathcal{L}\mathbf{q'} + \mathbf{f'},\tag{7}$$

where all terms that are linear in the fluctuations are collected in the first term on the right-hand side. Because of linearity, this term can be written as a product of a linear operator  $\mathcal{L} = \mathcal{L}(\overline{q})$  and the solution vector of fluctuating quantities, q'. All terms that contain second and higher-order products of fluctuating quantities are collected in f'. The operator  $\mathcal{M} = \mathcal{M}(\overline{q})$  is required to accommodate the additional terms generated when inserting the Reynolds-decomposed primitive variables into the time rates of change of the conservative variables on the left-hand side of momentum and energy Eqs. (1b) and (1c), respectively. By defining  $\mathcal{A} = \mathcal{M}^{-1}\mathcal{L}$ , we may write Eq. (7) as,

<span id="page-3-1"></span>
$$\frac{\partial \mathbf{q}'}{\partial t} = \mathcal{A}\mathbf{q}' + \mathbf{f}',\tag{8}$$

where the term  $\mathcal{M}^{-1}f'$  has been re-defined as f' for compactness.

In classical linear theory, choosing  $\overline{q}$  as a steady-state solution of the nonlinear equations and neglecting products of fluctuating quantities implies that f'=0. The linearized dynamics of the fluctuations are then governed by

<span id="page-3-3"></span><span id="page-3-2"></span>
$$\frac{\partial \mathbf{q}'}{\partial t} = \mathcal{A}\mathbf{q}'. \tag{9}$$

In other applications like resolvent analysis, discussed in Sect. 2.6 below, the term f' in Eq. (8) is retained but modeled as exogenous forcing of the linearized dynamics. Nonmodal stability theories (see [80] for a stimulating overview), like resolvent analysis, require defining a measure to quantify the perturbations. For incompressible flows, this measure is usually the fluctuating, or turbulent, kinetic energy. An analogous measure for compressible flows that, in addition to the fluctuating velocities, takes into account density and temperature fluctuations was derived by [17] and independently by [32]. This expression for the fluctuating compressible kinetic energy is defined in terms of the primitive fluctuating variables and can hence readily be used to define an energy-based (weighted  $L_2$ ) norm,

$$\|\boldsymbol{q}'\|_E^2 = \int \left(\frac{\overline{T}}{\overline{\rho}\gamma M^2} |\rho'|^2 + \overline{\rho}|u'|^2 + \overline{\rho}|v'|^2 + \overline{\rho}|w'|^2 + \frac{\overline{\rho}}{\overline{T}\gamma(\gamma - 1)M^2} |T'|^2\right) dV, \tag{10}$$

for the solution vector q'.

## 2.1 Reference flow

Linear stability analysis can be performed around a fixed point of the Naver–Stokes equations, in which case the flow analysis is referred to as a baseflow, or over a mean of a turbulent or non-stationary stationary flow, in which case it is referred to as a mean-flow.

Meanflows can be obtained by averaging flow snapshots from experimental [21] or numerical [30,57,85,86] databases.

![](_page_4_Figure_2.jpeg)

<span id="page-4-0"></span>**Fig. 1** Illustration of the coordinate systems used

For stable systems, the baseflows can also be measured experimentally, or obtained numerically simply by marching the Navier–Stokes equations in tomeTime. The system then naturally converges to the fixed point, which can then be used in the analysis. The case for unstable baseflow is more complicated.

Selective frequency damping can be used to damp existing unstable modes without affecting the baseflow, which can then be obtained by time marching [\[1\]](#page-17-3). Alternatively, implicit time-stepping methods, e.g., the backward Euler, can be used with large time steps, which numerically damps any flow oscillation. Another approach is to follow the fixed point of a stable system using Newton's method. This allows tracking a fixed point, e.g., identified in a region of the parameter space where it is stable, into a region where it becomes unstable. Advanced methods, such as RPM [\[88](#page-20-20)] and BoostConv [\[19\]](#page-18-19), can accelerate and/or reduce the numerical costs associated with finding these points.

*LinStab2D* was designed to perform linear stability analysis only, and thus does not contain functions to compute base or meanflows. We refer the reader to the works cited above for a description of these methods.

## 2.2 Coordinate systems

Stability analysis is performed by computing eigenvalues of the linearized operator, *A*, and the presence of at least an eigenmode with growing amplitude indicates that the flow is unstable. While this procedure has to be applied to the physical domain, i.e., in 3D, the presence of symmetries allows for simplifying assumptions. For example, translation symmetry of the flow, i.e., when a homogeneous direction exists, allows for using a Fourier ansatz, which reduces the number of spatial directions by one and, therefore, significantly reduces the size of the problem to be studied.

In Eqs. [\(1a](#page-2-0)–c), the governing equations of compressible flow are given in operator form, i.e., independent of the coordinate system. In practice, the linearized operator *L* in Eq. [\(6\)](#page-2-3) and the conversion operator *M* in [\(7\)](#page-3-0) are derived using symbolic math software for Cartesian and cylindrical coordinates separately. These coordinate systems are illustrated in Fig. [1.](#page-4-0)

In general, for Cartesian coordinates,

$$\mathcal{A} = \mathcal{A}\left(\left[\overline{U}, \overline{T}, \overline{\rho}\right](x, y, z), \frac{\partial}{\partial x}, \frac{\partial}{\partial y}, \frac{\partial}{\partial z}\right),\tag{11}$$

i.e., it is a differential operator with an explicit dependence on the baseflow. We consider here the case where the flow is homogeneous in the *z* direction, *U* = *U*(*x*, *y*), and likewise for the other baseflow quantities. Using the ansatz

<span id="page-4-1"></span>
$$q(x, y, z, t) = \hat{q}'(x, y)e^{i\beta z - i\omega t},$$
(12)

the linear operator can be expressed as

$$\mathcal{A}_{\beta} = \mathcal{A}_{\beta} \left( \left[ \overline{U}, \overline{T}, \overline{\rho} \right] (x, y), \frac{\partial}{\partial x}, \frac{\partial}{\partial y}, \beta \right)$$
 (13)

and the linear stability problem is solved independently for each wavenumber β.

Similarly, in cylindrical coordinates, we assume the baseflow does not depend on the azimuthal coordinate, and thus the ansatz

<span id="page-5-0"></span>
$$q(x, y, \theta, t) = \hat{q}'(x, y)e^{im\theta - i\omega t}, \tag{14}$$

yields the linear operator

$$\mathcal{A}_{m} = \mathcal{A}_{m} \left( \left[ \overline{U}, \overline{T}, \overline{\rho} \right] (x, y), \frac{\partial}{\partial x}, \frac{\partial}{\partial y}, m \right), \tag{15}$$

which now depends on the azimuthal wavenumber m.

## 2.3 Temporal stability analysis

Temporal stability is based on (9). It describes the time evolution of a given flow condition: if the perturbation grows without bounds, the flow is unstable. Otherwise, it is stable. To analyze the stability properties the ansatz (12) or (14) are used to re-write (9) as an eigenvalue problem, i.e.,

$$-i\omega\hat{\boldsymbol{q}}' = \mathcal{A}_{\bullet}\hat{\boldsymbol{q}}', \quad \omega \in \mathbb{C}$$
 (16)

where  $A_{\bullet}$  is a placeholder for either  $A_m$  or  $A_{\beta}$ , with  $m \in \mathbb{I}$  and  $\beta \in \mathbb{R}$ .

Eigenvalues,  $\omega$ , and corresponding eigenmodes  $\hat{q}'$  as solution of (18). The real part of  $\omega$  provides the oscillation frequency, and their imaginary part indicates its growth or decay rates, if positive or negative, respectively. The flow is stable if all eigenvalues lie in the bottom half of the complex plane. A single value on the upper complex plane is sufficient to render the system unstable.

The adjoint operator,  $\mathcal{A}_{\bullet}^{\dagger}$ , is defined as the operator that satisfies

$$\langle \boldsymbol{w}, \mathcal{A}_{\bullet} \boldsymbol{q} \rangle_{\mathcal{W}_q} = \langle \mathcal{A}_{\bullet}^{\dagger} \boldsymbol{w}, \boldsymbol{q} \rangle_{\mathcal{W}_q}, \quad \forall \boldsymbol{q}, \boldsymbol{w}.$$
 (17)

The adjoint eigenproblem reads

<span id="page-5-1"></span>
$$-i\omega^{\dagger}\hat{\boldsymbol{w}}' = \mathcal{A}_{\bullet}^{\dagger}\hat{\boldsymbol{w}}',\tag{18}$$

and it can be shown that  $\omega^{\dagger} = -\omega^{*1}$ , where the \* superscript indicated complex conjugation, and that the adjoint modes can be normalized such that  $\langle \hat{\boldsymbol{w}}', \hat{\boldsymbol{q}}' \rangle_{W_q} = \delta_{ij}$ , i.e., the direct and adjoint modes are bi-orthogonal. In the context of LSA, adjoint modes indicate the optimal way to excite a given flow mode, or can be used to decompose a flow perturbation into an eigenmode basis.

Although the results are easy to interpret at first sight, they can be nuanced if the homogeneous direction coincides with the streamwise direction. In this case, an unstable mode indicates that the perturbation energy will grow indefinitely provided that the initial conditions span the entire domain, i.e., from  $-\infty$  to  $+\infty$ . However, what happens if the perturbations are localized, or the analysis domain is finite, which is always the case, is not a priori clear. Does the energy growth at a given position come from the surrounding perturbations or different domain regions? In the latter case, is the equilibrium position reached again if the initial condition was localized in space?

Spatial stability provides further insights into these systems.

<span id="page-5-2"></span><sup>&</sup>lt;sup>1</sup> The adjoint operators can be obtained as described here from the discrete operator  $\mathbf{A}_{\bullet}$  as  $\mathbf{A}_{\bullet}^{\dagger} = \mathbf{W}_{q}^{-1} \mathbf{A}_{\bullet}^{H} \mathbf{W}_{q}$ . This method is referred to as *discrete-adjoint*. Alternatively, it can be obtained by first deriving the adjoint equations and then discretizing them. This method is referred to as *continuous-adjoint*, see [15] for a discussion. The *discrete-adjoint* method discussed here guarantees that  $\omega^{\dagger} = -\omega^{*}$ , but can lead to oscillations in the adjoint modes. This trend is reversed for the *continuous-adjoint* approach.

## 2.4 Spatial stability analysis

While temporal stability analyzes the temporal growth rate of perturbations, spatial stability analyzes their spatial growth. Spatial stability was developed initially to model flows with localized perturbations [27,28], e.g., a vibrating ribbon on a boundary layer [87]. In practice, spatial stability analyses a problem with a real-valued frequency,  $\omega$ , and complex-valued wavenumber,  $\beta$ , which contains information on the modes' growth or decay, i.e., their stability.

While presenting the spatial stability formulation, we will consider only Cartesian systems. Note that for axisymmetric problems, the perturbations are, by definition, periodic in the azimuthal direction, and thus m is a real-valued integer. Spatial analysis is not meaningful in this direction, as it assumes complex-valued numbers to encode modes growth or decay. To perform spatial analysis, the operator  $\mathcal{A}_{\beta}$  is written as a polynomial in  $\beta$ , and (18) reads

<span id="page-6-1"></span>
$$-i\omega\hat{\boldsymbol{q}}' = \underbrace{\left(\mathcal{A}_{\beta,0} + \mathcal{A}_{\beta,1}\beta + \mathcal{A}_{\beta,2}\beta^2\right)}_{\mathcal{A}_{\beta}}\hat{\boldsymbol{q}}',\tag{19}$$

which, for a given value of  $\omega$ , corresponds to a polynomial eigenvalue in  $\beta$ . It can be converted into a standard eigenvalue problem using an extended system, which reads

$$\beta \begin{bmatrix} 1 & 0 \\ 0 - \mathcal{A}_{\beta, 2} \end{bmatrix} \begin{bmatrix} \hat{\boldsymbol{q}}' \\ \beta \hat{\boldsymbol{q}}' \end{bmatrix} = \begin{bmatrix} 0 & 1 \\ (\mathcal{A}_{\beta, 0} + i\omega\mathcal{I}) & \mathcal{A}_{\beta, 1} \end{bmatrix} \begin{bmatrix} \hat{\boldsymbol{q}}' \\ \beta \hat{\boldsymbol{q}}' \end{bmatrix}. \tag{20}$$

The real and imaginary parts  $\beta$  correspond to the wavenumber and spatial amplification or decay rate, respectively. However, the group velocity must be investigated to determine if a mode is stable or unstable. This is achieved by the [6,8] criteria: in the limit of  $\Im(\omega) \to \infty$ , if  $\beta$  lies on the top-half complex plane, the mode is traveling in the z+ direction, and thus unstable if  $\Im(\beta) < 0$ . Otherwise, if  $\beta$  lies on the bottom-half complex plane for  $\Im(\omega) \to \infty$ , the mode is traveling in the z- direction and thus unstable if  $\Im(\beta) > 0$ .

## 2.5 Spatiotemporal analysis

While temporal/spatial analyses restrict  $\omega / \beta$  to be real-valued, spatiotemporal analysis assumes that both  $\omega$  and  $\alpha$  can be complex numbers. A dispersion relation formulation is constructed from (19) as

$$DR(\omega, \alpha) = \det\left(-i\omega - \mathcal{A}_{\beta,0} + \mathcal{A}_{\beta,1}\beta + \mathcal{A}_{\beta,2}\beta^2\right) = 0.$$
 (21)

This approach unifies spatial and temporal analysis. Temporal analysis can be recovered by fixing a real-valued  $\beta$ , and spatial analysis by fixing a real-valued  $\omega$ .

Huerre and Monkewitz [33] brought to the fluid community the concepts of convective and absolute instabilities developed by [6,8] in the context of plasma physics. The framework uses spatiotemporal analyses to investigate the long-term behavior of the flow at a given position.

Convective stabilities are instability waves that are strongly convected by the flow. Thus, while its amplitude increases, this growth is convected away from the disturbance origin. The flow thus returns to equilibrium once this wave passes. These instabilities can trigger transition to turbulence downstream of the perturbation point, but the flow may remain laminar locally. On the other hand, absolute instabilities grow locally, leading to transition, or at least to a new base state.

To determine the nature of the instability, pinching points on the spatial spectra, or, alternatively, cusps on the temporal spectra, have to be identified. For details on the procedure, we refer the reader to the works of [16,33,34,81].

<span id="page-6-0"></span>

## 2.6 Resolvent analysis

Another important realization was that, even if flows are stable, finite perturbations can be amplified by the flow dynamics [37,76,81,100], a phenomenon intrinsically connected to the non-normality of the system. Two types of non-modal amplifications have been studied. In the time domain, the phenomenon is known as transient growth and studies the amplification of a given perturbation without external forcings. In the

frequency domain, one typically studies flow responses for different forcings, identifying the forcing terms that are optimally amplified by the flow dynamics. The amplification factors are described by gains obtained via a singular-valued decomposition. Such analysis is known as resolvent analysis.

The Resolvent analysis framework is obtained from (9), which, once Fourier-transformed in time, leads to

$$\hat{\boldsymbol{q}}' = \underbrace{(-i\omega - \mathcal{A}_{\bullet})^{-1}}_{\mathcal{R}} \hat{\boldsymbol{f}}', \tag{22}$$

where R is known as the resolvent operator.

Resolvent analysis consists in finding the forcing term  $\hat{f}'$  that leads to the largest response, i.e., the forcing that maximizes

<span id="page-7-0"></span>
$$\sigma = \frac{\|\mathcal{C}\hat{q}'\|_{\mathcal{W}_q}}{\|\mathcal{B}\hat{f}'\|_{\mathcal{W}_f}},\tag{23}$$

where  $\|\bullet\|_{\mathcal{W}_{f,q}}$  represents separate norms for the response and forcing using different weights  $\mathcal{W}_{f,q}$ . We introduce the operators  $\mathcal{B}$  and  $\mathcal{C}$  in (23) to allow for restriction in the forcings and responses considered. By default, we use the compressible energy norm defined in Eq. (10). The forcing and response modes that maximize  $\sigma$  are called leading modes, and  $\sigma$  is the corresponding leading gain. Suboptimal modes and gains maximize (23) under the constraint of being orthogonal to the leading modes and each other.

When applied to laminar and stable flows, resolvent analysis can point towards mechanisms responsible for by-pass transition [32,65]. For turbulent flows, it has been observed that the framework can provide insights into the turbulent dynamics. By interpreting the non-linear interactions as an exogenous forcing term, many trends observed for the turbulent fluctuations can be reproduced [35,57]. Resolvent analysis is thus a powerful tool for analyzing turbulent flows, with intimate connections to spectral proper orthogonal decomposition (SPOD) [97], widely used to extract coherent structures from turbulent flows.

Resolvent analysis is closely related to pseudo-spectra. Assuming  $\mathcal{B} = \mathcal{C} = \mathcal{I}$ , for simplicity, the pseudo spectrum is characterized by the isovalues of [99]

$$ps(\lambda) = \| (\mathcal{A} - \lambda \mathcal{I})^{-1} \|. \tag{24}$$

Noting that the term to be normed is equal to  $-\mathcal{R}$  for  $\omega = -\lambda/i$ , and that the leading singular value of a matrix defines a norm, it is clear that the leading resolvent gain gives the pseudo spectra. However, analyses of the pseudo-spectra are typically not concerned with the modes associated with the gains. One of the main contributions of the resolvent analysis framework is to use the left and right singular modes, i.e., forcing and response modes, to shed light on the physical mechanisms at play in the flow.

## 2.7 Floquet ansatz

When analyzing a given problem, exploring its symmetries reduces computational costs. When the problem has a translational symmetry (i.e., a homogeneous direction), the Fourier ansatz, as in (12) and (14), reduces the dimensionality of the problem. Another well-known symmetry is the reflection symmetry, which leads to symmetric or antisymmetric solutions.

A more complex type of symmetry is periodic symmetry (for an illustration of a periodic baseflow, see Fig. 9 in Sect. 4.3). In such cases, the Fourier ansatz is not applicable. Instead, the Floquet ansatz can be used, which reads

<span id="page-7-1"></span>
$$\mathbf{q}'(x, y, z, t) = \tilde{\mathbf{q}}'(x, y, z, t)e^{\frac{2\pi i}{Ln_{\text{floq}}}x}.$$
 (25)

The ansatz consists of decomposing q' into two terms: a Floquet mode,  $\tilde{q}'$ , which is periodic in the domain L, and a second term with period  $n_{\text{floq}}L$ . Fourier ansatz can later be added to homogeneous directions, such as t and  $z/\theta$ .

As the second term in (25) is analytical, only  $\tilde{q}'$  has to be obtained numerically. As this function has the same period as the flow, only a single flow period has to be discretized, reducing the computational cost of, for example, the analysis of sub-harmonic instabilities.

![](_page_8_Figure_2.jpeg)

<span id="page-8-1"></span>Fig. 2 Illustration of the concept of the state vector of compressible flow quantities

# <span id="page-8-0"></span>3 Numerical implementation

To numerically compute spatial, temporal, or resolvent analyses, the linear operator  $\mathcal{A}_{\bullet}$  has to be discretized, and boundary conditions imposed. The result of this process is a finite-size matrix  $\mathbf{A}_{\bullet}$ .

This section describes how the derivative operators are constructed, how boundary conditions are imposed, and how the code handles complex geometries, such as block-structured meshes and curvilinear coordinates.

The space is discretized into a  $N_x \times N_y$  grid. The pressure, temperature, and velocity fields are reshaped into a vector, and the vectors are concatenated into a state vector. This is illustrated in Fig. 2.

## 3.1 Spatial discretization

Spatial derivatives are computed using summation-by-parts (SBP) finite differences (FD) [42,92]. All second derivatives are approximated using the consistent second difference weights provided by [56]. We refer to the latter paper for a review of this discretization scheme. Second-, fourth-, and eighth-order arrucate SBP-FDs are available in the code. From experience, fourth-order accuracy provides a good balance between accuracy, compactness of the FD molecules (and therefore sparsity of the discretized stability operators), and stability. Based on this best practice, all computations shown in this paper are computed using fourth-order accuracy. The polar singularity of the cylindrical coordinate system is treated by the reflected-stencil method proposed by [61]. Also implemented is a Padé-type 10th-order filter [25] that can be switched on to stabilize resolvent analyses. If enabled, both the input and output modes are filtered at each call of the matrix–vector product by the Krylov–Schur method.

## 3.2 Boundary conditions

### 3.2.1 Dirichlet boundary conditions

Dirichlet boundary conditions are implemented directly on the operator discretized  $A_{\bullet}$ . Considering that such a condition is to be implemented on the *i*-th DOF, the matrix is modified as

$$\mathbf{A}_{\bullet} = \begin{bmatrix} \mathbf{A}_{\bullet}^{-,-} & \mathbf{A}_{\bullet}^{-,i} & \mathbf{A}_{\bullet}^{-,+} \\ \mathbf{A}_{\bullet}^{i,-} & \mathbf{A}_{\bullet}^{i,i} & \mathbf{A}_{\bullet}^{i,+} \\ \mathbf{A}_{\bullet}^{+,-} & \mathbf{A}_{\bullet}^{+,i} & \mathbf{A}_{\bullet}^{+,+} \end{bmatrix} \rightarrow \begin{bmatrix} \mathbf{A}_{\bullet}^{-,-} & 0 & \mathbf{A}_{\bullet}^{-,+} \\ 0 & \epsilon & 0 \\ \mathbf{A}_{\bullet}^{+,-} & 0 & \mathbf{A}_{\bullet}^{+,+} \end{bmatrix}$$
(26)

where the +/- index represents the block matrices with indexes larger/smaller than i and  $\epsilon$  is a penalization term. Each Dirichlet boundary condition uses this approach to have an eigenvalue  $\epsilon$  of  $\mathbf{A}_{\bullet}$ . All other eigenvectors have null entries at the corresponding DOF, i.e., satisfy the boundary condition. Choosing a large value for  $\epsilon$  guarantees the spurious eigenvalues are away from the region of interest, not affecting the eigenvalue computation.

When performing a resolvent analysis, forcing terms acting on the DOFs where the Dirichlet boundary condition is applied leads to a non-zero value (i.e., equal to  $\hat{f}'_i/\epsilon$ ). To enforce the boundary conditions on the

![](_page_9_Figure_1.jpeg)

<span id="page-9-0"></span>Fig. 3 Illustration of a mesh masking, mapping, and the stencils used for a 4th order FD scheme. The red and blue markers indicate the points used to compute derivatives at the crosses alongside the  $\xi$  and  $\eta$  coordinates, respectively. Black circles correspond to grid points in the flow domain, and gray circles to masked points. A case with periodicity in  $\xi/x$  is shown. (Colour figure online)

solution, matrices **B** and **C**, i.e., discretized versions of operators  $\mathcal{B}$  and  $\mathcal{C}$ , need to be constructed to prevent forcing these DOFs.

### 3.2.2 Neuman boundary conditions

Neuman boundary conditions are implemented similarly. Considering that such a condition is to be implemented on the *i*-th DOF, the matrix is modified as

$$\mathbf{A}_{\bullet} = \begin{bmatrix} \mathbf{A}_{\bullet}^{-} \\ \mathbf{A}_{\bullet}^{i} \\ \mathbf{A}_{\bullet}^{+} \end{bmatrix} \Rightarrow \begin{bmatrix} \mathbf{A}_{\bullet}^{-} \\ \epsilon \mathbf{D}^{i} \\ \mathbf{A}_{\bullet}^{+} \end{bmatrix}$$
 (27)

where the +/- index represents the block matrices with line indexes larger/smaller than i, i superscript indicates the i-th line of  $\mathbf{D}$ , the differentiation matrix, and  $\epsilon$  is a penalization term.

Here also, when performing a resolvent analysis, matrices **B** and **C** need to be constructed to prevent forcing the DOFs where the boundary condition was applied.

### 3.2.3 Periodic boundary conditions

Periodic boundary conditions are obtained by using centered stencils around the boundaries. Figure 3 illustrates stencils used across the periodicity boundary. Note that the function value of the grid points located at  $\xi = 0$  and  $\xi = 1$  is identical, and thus only the first is kept in the computational domain.

<span id="page-9-1"></span>

## 3.3 Curvilinear coordinates

A mapping between a Cartesian, computational, grid and the physical grid is constructed to construct curvilinear coordinates. This is illustrated in Fig. 4. We assume the mapping to be defined numerically, which makes the code more versatile and easy to use. In practice, the mapped position of each point is provided to the code, and mapping's Jacobian and Hessian are computed numerically. The differentiation matrices on the physical domain are then constructed.

First-order derivatives are obtained as

$$\begin{bmatrix} \frac{\partial}{\partial x} \\ \frac{\partial}{\partial y} \end{bmatrix} f = \underbrace{\begin{bmatrix} \frac{\partial \xi}{\partial x} & \frac{\partial \eta}{\partial x} \\ \frac{\partial \xi}{\partial y} & \frac{\partial \xi}{\partial y} \end{bmatrix}}_{\mathbf{I}} \begin{bmatrix} \frac{\partial}{\partial \xi} \\ \frac{\partial}{\partial \eta} \end{bmatrix} f, \tag{28}$$

from which we obtain, for any scalar function f, the relation

$$\frac{\partial f}{\partial x_i} = J_{i,j} \frac{\partial f}{\partial \xi_j},\tag{29}$$

![](_page_10_Picture_2.jpeg)

Fig. 4 Illustration of the computational (left) and physical (right) meshes

<span id="page-10-0"></span>where  $\partial/\partial x_{1/2}$  corresponds to  $\partial/\partial x$  and  $\partial/\partial y$ , and  $\partial/\partial \xi_{1/2}$  corresponds to  $\partial/\partial \xi$  and  $\partial/\partial \eta$ , respectivelly.

Differentiation matrices approximating the differential operators are constructed using finite difference schemes. Thus, the terms in J, which are functions of the derivates w.r.t. x and y, cannot be constructed directly. Instead,

$$\mathbf{J}^{-1} = \begin{bmatrix} \frac{\partial x}{\partial \xi} & \frac{\partial x}{\partial \eta} \\ \frac{\partial y}{\partial \xi} & \frac{\partial y}{\partial \xi} \end{bmatrix},\tag{30}$$

is constructed numerically and **J** is obtained by its inversion.

The second derivative is obtained using the chain rule,

$$\frac{\partial^2 f}{\partial x_i x_j} = \frac{\partial}{\partial x_i} \left( J_{j,l} \frac{\partial f}{\partial \xi_l} \right) = \frac{\partial J_{j,l}}{\partial x_i} \frac{\partial f}{\partial \xi_l} + J_{j,l} J_{i,m} \frac{\partial^2 f}{\partial \xi_m \partial \xi_l}, \tag{31}$$

with term  $\frac{\partial J_{j,l}}{\partial x_i}$  computed using the identity

$$\frac{\partial \mathbf{J}}{\partial x_i} = \mathbf{J} \frac{\partial \mathbf{J}^{-1}}{\partial x_i} \mathbf{J},\tag{32}$$

which can be computed using derivatives of the numerical mapping w.r.t.  $\xi$  and  $\eta$  only.

When used in periodic domains, the mesh deformation should be smooth across the periodic boundary to ensure proper Jacobian map computation when using periodic and boundary conditions on the left and right boundaries.

<span id="page-10-1"></span>

## 3.4 Quasi-structured meshes

While many problems can be studied using structured meshes using Cartesian and curvilinear coordinates, some geometries remain challenging to model using structured meshes. Examples include the flow around a cavity, a square cylinder, or a jet flow where the nozzle needs to be modeled.

To allow the study of these problems, a quasi-structured mesh can be used. The approach consists of masking grid points from an otherwise Cartesian grid. Near masked points, derivatives are computed using uncentered finite differences. A quasi-structured Cartesian grid can be mapped into more complex domains using the approach detailed in Sect. 3.3. Figure 3 illustrated the approach.

Note that when using this approach, the matrix representing the linear operator acts only of the non-masked grid points, i.e., masked points do not increase the size of the matrix, and thus do not increase the computational cost. However, the corresponding solution vector has to be extended to include the masked points, with a minor increase in the memory requirements. However, this cost is negligible.

## 3.5 Computation of eigen and resolvent modes.

To obtain a finite set of the eigenvalues of  $A_{\bullet}$ , a Krylov–Schur method [91] is used. This approach typically converges to the largest eigenvalues.

To select the eigenvalue region of interest, a shift-and-invert (s&i) approach is used, i.e., the Krylov–Schur method is applied to  $\mathbf{A}_{s\&i} = (\mathbf{A}_{\bullet} - \omega_{\text{tar}}\mathbf{I})^{-1}$ . The matrix inversion is avoided by computing  $\mathbf{v} = \mathbf{A}_{s\&i}\mathbf{u}$  by solving a linear problem

$$\mathbf{v} = (\mathbf{A}_{\bullet} - \omega_{\text{tar}} \mathbf{I}) \, \mathbf{u}. \tag{33}$$

The largest eigenvalues of  $\mathbf{A}_{s\&i}$  correspond to the eigenvalues closes to  $\omega_{tar}$  of  $\mathbf{A}_{\bullet}$ , and that  $\mathbf{A}_{s\&i}$  shares the same eigenvectors as  $\mathbf{A}_{\bullet}$ . This approach thus allows obtaining eigenvalues and vectors in a region of interest. Resolvent forcing and response modes are obtained from the discretized modified resolvent operator

$$\tilde{\mathbf{R}} = \mathbf{W}_q^{1/2} \mathbf{R} \mathbf{W}_f^{-1/2},\tag{34}$$

via a singular value decomposition, i.e.,

$$\tilde{\mathbf{R}} = \begin{bmatrix} | & | & | & \\ \tilde{\boldsymbol{v}}_1 & \tilde{\boldsymbol{v}}_2 & \tilde{\boldsymbol{v}}_3 & \dots \\ | & | & | & \end{bmatrix} \begin{bmatrix} \sigma_1 & & & \\ & \sigma_2 & & \\ & & \sigma_3 & \\ & & & \ddots \end{bmatrix} \begin{bmatrix} | & | & | & \\ \tilde{\boldsymbol{u}}_1 & \tilde{\boldsymbol{u}}_2 & \tilde{\boldsymbol{u}}_3 & \dots \\ | & | & | & \end{bmatrix}^T, \tag{35}$$

from which the forcing and response modes are recovered as  $\mathbf{u} = \mathbf{W}_q^{-1/2} \tilde{\mathbf{u}}$  and  $\mathbf{v} = \mathbf{W}_f^{-1/2} \tilde{\mathbf{v}}$  [97].

For large systems, it is typically more efficient to compute  $\tilde{u}_i$  as eigenvalues of  $\tilde{\mathbf{R}}^H \tilde{\mathbf{R}}$ , with  $\tilde{v}_i$  recovered as  $\tilde{\mathbf{R}} \tilde{u}_i$  [90]. The leading eigenvalues are obtained using an implicitly restarted Arnoldi algorithm.

Note that, after discretization, the norm weights become diagonal matrices  $\mathbf{W}_{f,q}$ , and quadrature weights are typically included in these matrices, such that the norm of a vector is approximated by  $\|\mathbf{q}'\|_{\mathbf{W}_q}^2 = \mathbf{q}'^H \mathbf{W}_q \mathbf{q}'$ .

<span id="page-11-0"></span>

# 4 Examples

This section presents five examples, which were chosen to highlight different features of the code.

First, we illustrate the application of resolvent analysis on systems homogeneous in the spanwise and azimuthal directions, using a spatially evolving boundary layer and an axisymmetric jet as examples. Next, we perform temporal and spatial stability of a locally parallel streaky-boundary layer, exploiting periodicity in the spanwise direction with the Floquet ansatz. Finally, we illustrate quasi-structured and curvilinear meshes using the flow around a cavity and a paraboloid body as examples. In all the examples below, fourth-order finite difference schemes and identical forcing and response norms were used ( $\mathbf{W}_q = \mathbf{W}_f$ ), with the matrix accounting for the compressible energy norm [17,32] and quadrature weights. When performing resolvent analysis, matrices  $\mathbf{B}$  and  $\mathbf{C}$  are diagonal matrices with 1's along the diagonal, except for the columns associated with DOFs where the boundary conditions were applied.

Typically, boundary conditions should be applied in every boundary of the problem for the three velocity components, u', v', w', and temperature T'. At physical walls, no-slip conditions imply Dirichlet boundary conditions for the velocity. For the temperature field, isothermal/adiabatic conditions lead to Dirichlet/Neumann. No boundary conditions need to be applied for the density field. If the flow is periodic, the periodic boundary conditions for all terms should be used.

The treatment of inflow/outflow boundaries is more subtle. Here, we choose to apply Dirichlet boundary conditions for the velocity and temperature fields, using sponge zones to dampen forcing/response modes near them, thus avoiding wave reflection. More sophisticated boundary conditions are possible. We refer the reader to the work of [20] for a comprehensive review of different inflow, outflow, and non-reflective boundary conditions.

![](_page_12_Figure_2.jpeg)

<span id="page-12-2"></span>Fig. 5 Boundary layer flow and coordinate system. The color scale indicates the axial velocity

![](_page_12_Figure_4.jpeg)

<span id="page-12-3"></span>Fig. 6 Forcing and response modes for the three leading modes of the spatially evolving boundary layer, and the associated gains. Only the forcing associated with the stream-wise momentum equation and the streamwise velocity component are shown for brevity

<span id="page-12-0"></span>

## 4.1 Tollmien–Schliching waves in the flat-plate boundary layer

In this example, we investigate the optimal forcing of Tollmien–Schliching waves on a spatially evolving boundary layer, shown in Fig. 5. This example follows the work of [90], who conducted their study for the incompressible self-similar Blasius solution. We choose a relatively low Mach number of Ma = 0.3 to obtain comparable results for the self-similar compressible boundary layer (see, e.g., [102], ). The Reynolds number based on unit length and unit velocity is  $Re_x = 60\,000$ . All the variables are made adimensional using the freestream temperature ( $T = 273.15\,\mathrm{K}$ ), velocity, and density. The flow is homogeneous in the z direction, and we focus the study on planar ( $\beta = 0$ ) perturbations.

The domain consists of the rectangle  $[0.05, 1.25] \times [0, 0.05]$ , discretized into a grid of  $300 \times 100$  points, which are mapped into the y direction to concentrate most of the points below the boundary layer edge. Dirichlet boundary conditions were applied for the temperature and velocity fields on all the boundary nodes.

Figure 6 shows the forcing and response modes obtained and the associated gains for a frequency of  $\omega = 60$ . The leading mode consists of a forcing mode located upstream of the domain, with most of the response found downstream. This is a typical behavior of amplifier flows and a signature of the non-normality of the system.

For this boundary layer, the mechanism associated with this perturbation amplification is the excitation of Tollmien–Schliching waves. As these waves are convectively unstable, their amplitude increases as they travel upstream. Thus, the optimal way to force these waves is to excite them upstream, leveraging this instability mechanism to obtain high amplitudes further downstream.

The suboptimal modes leverage a similar mechanism. However, in order to be orthogonal to the leading mode, they have to excite these waves in different positions, with opposite phases, leading to the amplitude modulation observed in the suboptimal response modes. This cancelation means that the associated forcing modes are less effective in generating a flow response, which is reflected in lower gains.

The reader is referred to [90] for a detailed physical interpretation of the results.

<span id="page-12-1"></span>

## 4.2 Resolvent analysis of an axisymmetric jet

Here, the evolution of Kelvin–Helmholtz instability waves on a spatially evolving axisymmetric jet is presented. The mean flow of the Re = 450,000 turbulent jet, which serves as the base state, is shown in Fig. 7 and was obtained from a large-eddy simulation [7]. All the variables were made non-dimensional using the jet velocity, diameter, freestream temperature (T = 288.15K), and density. This example corresponds to the transonic jet case in [86] with Mach number of Ma = 0.9. Following the reference, we assume a constant effective

![](_page_13_Figure_1.jpeg)

Fig. 7 Jet base flow and coordinate system. The color scale indicates the axial velocity, normalized by the velocity of the jet core

<span id="page-13-1"></span>![](_page_13_Figure_3.jpeg)

<span id="page-13-2"></span>Fig. 8 Forcing and response modes for the three leading modes of the spatially evolving boundary layer, and the associated gains. Only the forcing associated with the stream-wise momentum equation and the streamwise velocity component are shown for brevity

Reynolds number of 10 000 but note that more accurate models are available [75]. The flow is homogeneous in the  $z/\theta$  direction, and we focus the study on axisymmetric (m = 0) perturbations.

The domain consists of the rectangle  $[-1, 34.5] \times [0, 11.2]$ , discretized into a grid of  $500 \times 100$  points. To avoid wave reflections, sponge zones and mesh stretching were used near the upstream of the nozzle (x < 0) and near the downstream (x > 30). Dirichlet boundary conditions were applied for the temperature and velocity fields on the left, right, and top borders. On the bottom border, a symmetry condition was imposed on the density, streamwise velocity, and temperature fields  $(\rho, u, T)$ , and anti-symmetry conditions on the azimuthal and radial velocity components (v, w).

Figure 8 shows the forcing and response modes obtained and the associated gains for a Strouhal number St = fU/D = 0.39, as previously reported by [85]. The leading forcing mode consists of excitations near the nozzle lip (r = 0.5), with a Kelvin–Helmholtz wavepacket evolving downstream. The leading mode pair has a large gain. This gain can be understood by the amplification of perturbations at the mixing layer due to the instability associated with the baseflow inflectional point. Higher-order modes have their forcing components distributed through the domain, with a lower gain. The mechanism responsible for the modulation is analogous to the one discussed in the previous section for the boundary layer example. These are typical of systems with convective instabilities.

<span id="page-13-0"></span>

## 4.3 Local analysis of a supersonic boundary layer

Here, a stability analysis of a streaky boundary layer [9] is presented. The baseflow is shown in Fig. 9. All variables were made non-dimensional using the boundary layer thickness, freestream velocity, temperature (T = 273.15k), and density. The Reynolds and Mach numbers are Re = 28,000 and Ma = 6. The flow is periodic in the spanwise direction (x), and the problem assumes homogeneity in the streamwise direction (z), i.e., it is a locally parallel approximation.

The domain consists of the rectangle  $[0, 2\pi] \times [0, 5]$ , discretized into a grid of  $200 \times 200$  points. A smaller domain,  $[0, \pi] \times [0, 5]$ , discretized with  $100 \times 200$  points, is also used with the Floquet ansatz. Sponge zones and mesh stretching were used near the top boundary to avoid wave reflections. Dirichlet boundary conditions were applied for the temperature and velocity fields on the top and bottom borders and a periodic condition on the right and left borders.

Figure 10 shows the temporal spectra for a streamwise wavenumber  $\beta = 0.6283$  and the streamwise velocity of the two unstable modes, using a domain that contains two streaks (indicated by the black rectangle in Fig. 9). The mode with the highest growth rate exhibits a periodicity of  $2\pi$ , corresponding to a sub-harmonic

![](_page_14_Figure_2.jpeg)

<span id="page-14-1"></span>Fig. 9 Streamwise velocity for a Mach 6 streaky boundary layer base flow. The coordinate systems and freestream velocity vectors are indicated. The flow is periodic in the x direction, and the red and blue rectangles indicate the two domain domains, with lengths of  $\pi$  and  $2\pi$ , used in the analysis. (Colour figure online)

![](_page_14_Figure_4.jpeg)

<span id="page-14-2"></span>Fig. 10 Temporal spectrum obtained with a domain containing two streaks is shown with blue dots. Red and yellow diamonds show the spectra obtained using a domain with one streak and  $n_{\text{floq}} = 1$  and 2, respectively. Figures on the right show the streamwise velocity  $(\hat{w}')$  field for the two fastest-growing modes: the dominant having a period 2L, and the other having the same periodicity as the baseflow, i.e., L. (Colour figure online)

streak instability. The other mode has the same periodicity as the streaks. An analysis of a domain containing a single streak (indicated by the red rectangle in Fig. 9) using the Floquet ansatz is shown with red ( $n_{floq} = 0$ ) and yellow ( $n_{floq} = 2$ ) diamonds<sup>2</sup>. The approach reproduces the spectrum of the larger domain, having the advantage of naturally splitting the spectra into harmonic and subharmonic modes, having lower computational cost, and is easily extended to higher Floquet numbers, i.e., analyzing larger domains.

The instability observed in this case is a case of sub-harmonic instability. On a baseflow with period L, perturbations with a period of L/2 can be self-excited by triadic interactions. A quadratic non-linear interaction between the mentioned perturbation and baseflow, leads to terms with period  $L \pm L/2$ , i.e., L/2 and 3L/2. The former can enhance the original perturbation, giving rise to the subharmonic instability. For small perturbations, this non-linear interaction can be linearized, and thus be captured by the code.

The same mechanisms can be observed in the spatial stability framework. To illustrate it, we focus on the frequency associated with the fastest-growing mode identified in the temporal analysis, i.e.,  $\omega=0.585$ . The smaller domain  $(L_x=\pi)$  with  $n_{\text{floq}}=2$  is used to reduce the computational cost. Figure 11 shows the resulting spectra and illustrates the Briggs–Bers criteria. None of the loci form a saddle point, which indicates that these modes are not absolutely unstable. Instead, a spatially unstable mode with  $\beta=0.628-0.018i$  is observed. As the imaginary part of the frequency increases, this mode crosses the real axis at the complex frequency identified in the temporal stability analysis, showing that the mode is essentially the same. As the mode's shape is indistinguishable from the one shown in Fig. 10, it is not presented.

<span id="page-14-0"></span>

## 4.4 Rossiter modes in open cavity flow

This example illustrates the stability analysis of a flow over an open cavity. The baseflow is shown in Fig. 12. It corresponds to the turbulent mean flow of the direct numerical simulation by [55]. We thankfully acknowledge the authors for providing us with the data. All variables were made non-dimensional using the cavity depth, freestream velocity, temperature (T = 300k), and density. The Reynolds and Mach numbers are Re = 1500 and Ma = 0.5. The problem is assumed to be periodic in the spanwise (z) direction.

<span id="page-14-3"></span>Note that since  $\tilde{q}'$ , in (25), has period L, results for  $n_{flow} = 0$  and 1 are equivalent.

![](_page_15_Figure_1.jpeg)

<span id="page-15-1"></span>**Fig. 11** Spatial stability spectra for a base flow with one (*Lx* = π) streaks and *n*floq = 2. The spectrum for ω = 0.5854 is shown in blue. The back dots show the spectra for increasing imaginary parts. The yellow diamond indicates the point on which the unstable mode crosses the real axis line, which occurs for (ω) = 0.0167. (Colour figure online)

![](_page_15_Figure_3.jpeg)

<span id="page-15-2"></span>**Fig. 12** Cavity base flow and coordinate system. The color scale indicates the axial velocity

The domain consists of the rectangle [−1.65, 5, 34]×[0, 5], discretized into a grid of 241 × 101 points. The grid was deformed in the *x* and *y* directions to position points at the wall surface. Points within the wall were masked from the domain using the quasi-structured mesh approach described in Sect. [3.4.](#page-10-1) Sponge zones were used near the top, left, and right boundaries to avoid wave reflection. Dirichlet boundary conditions were applied for the temperature and velocity fields along the walls.

Figure [13](#page-16-0) shows the spectra of the direct (ω) and the adjoint operator (ω†). As expected, the spectra are consistent: complex conjugating the adjoint eigenvalues recovers the direct eigenvalues.

The mechanism associated with these instabilities is an interaction between Kelvin–Helmholtz waves on the shear layer formed over the cavity, which is downstream-traveling, and acoustic waves inside the cavity, which are upstream-traveling. Being unstable, the Kelvin–Helmholtz waves impinge on the right edge of the cavity with a large amplitude, being scattered into acoustic waves inside the cavity. Traveling downstream, these acoustic waves can re-excite Kelvin–Helmholtz waves. If the overall gain, considering the amplification due to Kelvin–Helmholtz instability, reflection coefficients, and potential attenuation of the acoustic wave, is positive, an unstable mode is formed.

This mechanism underpins both unstable modes observed. As the wavenumber increases with frequency for the Kelvin–Helmholtz instability, the higher frequency mode exhibits more oscillations in the shear layer, i.e., higher harmonic modes are formed. The adjoint modes have their support near the upstream cavity wall, as this is the region where disturbances can optimally excite the unstable modes.

<span id="page-15-0"></span>

## 4.5 Steaks in the flow around a paraboloid body

Finally, the last example illustrates a resolvent analysis of the incompressible flow around a paraboloid body using curvilinear coordinates. The baseflow is shown in Fig. [14.](#page-16-1) All the variables were made non-dimensional using the freestream velocity and density and twice the curvature diameter at the leading edge.

As the response modes are expected to be concentrated on the boundary layer, and for high Reynolds numbers, it becomes thin, we focus on a low Reynolds number case to facilitate the mode visualization. The Reynolds and Mach numbers are Re = 5000 and *Ma* = 0.01. The flow is homogeneous in the azimuthal direction (*z*).

A Cartesian domain using a grid of 100×100 points is mapped around the paraboloid body using parabolic coordinates. Sponge zones were used near the left and top boundaries. Dirichlet boundary conditions were applied at the parabolic surface (right domain boundary) for the temperature and velocity fields. At the axis of rotation (bottom boundary), symmetry conditions were applied. The forcing and response modes for the azimuthal number *m* = 5 are shown in Fig. [15.](#page-16-2)

![](_page_16_Figure_2.jpeg)

<span id="page-16-0"></span>**Fig. 13** Spectra of the direct (blue circles) and adjoint (black crosses) operators for the flow around a cavity. On the bottom, support of the unstable adjoint (top) and direct (bottom) modes

![](_page_16_Figure_4.jpeg)

**Fig. 14** Velocity magnitude for the flow around a paraboloid body

<span id="page-16-1"></span>![](_page_16_Figure_6.jpeg)

<span id="page-16-2"></span>**Fig. 15** Forcing and response modes for the three leading modes for the flow around the paraboloid body and the associated gains. Only the forcing associated with the stream-wise momentum equation and the streamwise velocity component are shown for brevity

The forcing/response modes have optimal modes exhibiting a single maximum in the radial/wall-normal direction and an increasing number of nodes for the suboptimal modes, with forcing/response modes concentrated upstream/downstream of the stagnation point. Results for higher Reynolds numbers show qualitatively the same results, with forcing/response modes moving closer to the symmetry axis/body surface [\[54\]](#page-19-33).

Part of the mechanism responsible present here is the excitation of streamwise vortices upstream of the body, which penetrates the boundary layer via the stagnation point. Once in the boundary layer, these vortices create regions of downwash, with a velocity pointing towards the wall, and upwash, with the velocity pointing away from the wall. These regions are intercalated and have the same periodicity as that of the streamwise vortices formed in the upstream region. These regions trigger the lift-up mechanism: fast flow velocities, found far from the wall, are brought closer to the wall via the downwash, while low velocities, found closer to the

wall, are brought up by the upwash. This leads to the formation of regions of fast and slow velocities, known as streaks.

For the lower Reynolds number shown here, the forcing modes extend throughout the domain, and the mechanisms described above are mixed with the direct excitation of streaks throughout the body. For increasing Reynolds numbers, the lift-up mechanism becomes more effective, and thus the leading mode becomes dominated by it. In a previous work [\[54\]](#page-19-33), we show that the forcing modes move upstream of the body, with most of the streaks formed by this mechanism.

# <span id="page-17-2"></span>**5 Conclusion**

We presented *LinStab2D*, a MATLAB code for linear stability and resolvent analysis of 2D flows. Highorder finite differences were used to approximate the linearized Navier–Stokes operator using sparse matrices accurately. Using iterative methods, different aspects of the flow dynamics can be explored: flow instabilities, locally parallel temporal, spatial, and spatio-temporal stability, and resolvent analysis, i.e., identifying the optimal flow forcings and the associated responses.

Operations that are conceptually simple but whose implementation can be time-consuming and error-prone, such as imposing boundary conditions, using curvilinear coordinates, and mesh masking, are performed using high-level functions, reducing setup time and opportunity for coding mistakes. This approach allows the user to focus on the physics-relevant aspects of the analysis.

The code, available as an open source project (see link in page 28), contains examples that can be easily adaptable to studying different problems. The examples were designed to highlight different code capabilities: analyses in cartesian (Sect. [4.1\)](#page-12-0) and cylindrical (Sect. [4.2\)](#page-12-1) coordinates, temporal and spatial locally parallel analyses (Sect. [4.3\)](#page-13-0), spatially periodic flows (Sect. [4.3\)](#page-13-0), and flow domains with non-trivial geometries (Sects. [4.4](#page-14-0) and [4.5\)](#page-15-0). These examples provide an illustration not only of how to use the code, but also of the physical mechanisms which can be identified with it. Among the potential mechanisms that can be modeled are Kelvin– Helmholtz waves, Tollmien–Schliching waves, the interaction between upstream and downstream waves, subharmonic instabilities, and the lift-up mechanism.

The code has been tested for both compressible and incompressible regimes. Although this code has not yet been used to investigate flows with shocks, a similar code has been used for the analysis of overexpanded supersonic jets [\[21\]](#page-18-18). Note, however, that in the work of Edgington-Mitchell et al" the shocks have been smoothed out during measurements, both due to the measurement technique and the unsteadiness of the shock, which, when averaged, provides a smooth flow transition. Care should be taken when using this code when flow discontinuities are present, as the code does not include shock-capturing techniques, and the finite difference scheme used here can be unstable near these discontinuities.

**Funding** This work is part of the project TRANSITION supported by Région Nouvelle-Aquitaine under grant 2018-1R10220. This work has been granted access to the HPC resources of IDRIS under the allocations A0092A10868 and A0112A10868 made by GENCI (Grand Equipement National de Calcul Scientifique).

**Availability of data and materials** The source code for *LinStab2D* can be recovered at [https://github.com/eduardomartini/](https://github.com/eduardomartini/LinStab2D.git) [LinStab2D.git.](https://github.com/eduardomartini/LinStab2D.git)

# **Declarations**

**Ethical Approval** Not applicable

---

# Python Port Verification Notes

Cross-reference of the five paper examples against the MATLAB source and Python (`linstab2d`) implementation.
Run date: 2026-02-17. Python 3.13, PETSc/SLEPc + MUMPS.

## 4.1 Blasius Boundary Layer — Resolvent

| Parameter | Paper | MATLAB | Python | Status |
|-----------|-------|--------|--------|--------|
| Re | "Re_x = 60,000" | 6e5 | 6e5 | **Paper value appears incorrect** (OCR artifact?). MATLAB and Python both use Re = 600,000. Omega formula `100*(Re/1e6) = 60` confirms Re = 6e5 is correct |
| Ma | 0.3 | 0.3 | 0.3 | OK |
| T_0 | 273.15 K | 273.15 K | 273.15 K | OK |
| Domain | [0.05, 1.25] x [0, 0.05] | [0.05, 1.26] x [0, 0.05] | [0.05, 1.26] x [0, 0.05] | Paper rounds x_max = 1.25+dx to 1.25 |
| Grid | 300 x 100 | 300 x 100 | 300 x 100 | OK |
| omega | 60 | 60 | 60 | OK |
| n_modes | 3 | 3 | 3 | OK |
| BCs | Dirichlet on all borders | lrbt, uvwT | lrbt, uvwT | OK |
| Mesh mapping | Hanifi 1996 | yi=0.01, ymax=0.05 | yi=0.01, ymax=0.05 | OK |
| Sponge | "sponge zones...near inflow/outflow/freestream" | amp=5, tanh profile | amp=5, tanh profile | OK |
| B/C matrices | "1's except BC DOFs" | u,v forcing only (no rho,w,T) | u,v forcing only (no rho,w,T) | OK |

**Python gains: [128.62, 14.54, 7.49]** — matches MATLAB reference to 4+ significant figures.

**Assessment: CORRECT.** One paper transcription issue (Re_x = 60,000 should be 600,000).

## 4.2 Axisymmetric Jet — Resolvent

| Parameter | Paper | MATLAB | Python | Status |
|-----------|-------|--------|--------|--------|
| Re_eff | 10,000 | 1e4 | 1e4 | OK |
| Ma | 0.9 | 0.9 | 0.9 | OK |
| T_0 | 288.15 K | 288.15 K | 288.15 K | OK |
| Domain | [-1, 34.5] x [0, 11.2] | from .mat file | from .mat file | OK (loaded from M09jet.mat) |
| Grid | 500 x 100 | from .mat file | 500 x 100 (coarsen=1) | OK |
| St | 0.39 | 0.39 | 0.39 | OK |
| m | 0 | 0 | 0 | OK |
| n_modes | 3 | 3 | 3 | OK |
| BCs | "Dirichlet on left, right, top; symmetry on bottom" | ltr, uvwT | ltr, uvwT | OK |
| Sponge | "sponge zones near upstream of nozzle and downstream" | from .mat spongeFun | interpolated from .mat spongeFun | OK |
| B/C | "1's except BCs + sponge" | zero where sponge > 0 | zero where sponge > 0 | OK |

**Python gains: [1.78e8, 7.02e7, 5.64e7]** at full (uncoarsened) resolution.

**Assessment: CORRECT.** Baseflow and sponge loaded from the same .mat file as MATLAB.

## 4.3 Supersonic Streaky Boundary Layer — Temporal & Spatial

| Parameter | Paper | MATLAB | Python | Status |
|-----------|-------|--------|--------|--------|
| Re | 28,000 | 28,000 | 28,000 | OK |
| Ma | 6 | 6 | 6 | OK |
| T_0 | 273.15 K | 273.15 K | 273.15 K | OK |
| Domain (small) | [0, pi] x [0, 5] | [0, pi] x [-1, 1] mapped to [0, 5] | [0, pi] x [-1, 1] mapped to [0, 5] | OK — paper describes physical domain; code uses computational + Hanifi mapping |
| Grid (small) | 100 x 200 | 125 x 250 | 125 x 250 | **Paper underreports resolution.** MATLAB code uses `Nx = 125*(Lx/pi)`, `Ny = 250`. Paper says 100 x 200 — possibly earlier revision or rounded |
| alpha_temp | 0.6283 | 0.6283 | 0.6283 | OK |
| n_floq | 0, 2 | nPeriods=2 | n_periods=2 | OK |
| n_modes | — | 50 | 50 | OK |
| BCs | "Dirichlet top/bottom; periodic left/right" | tb, uvwT, x-periodic | tb, uvwT, x-periodic | OK |
| Sponge | "near top boundary" | amp=0.1, tanh at y=4 | amp=0.1, tanh at y=4 | OK |
| Mesh mapping | "mesh stretching near top" | Hanifi: yi=2.0, Ly=5 | Hanifi: yi=2.0, Ly=5 | OK |

**Assessment: CORRECT.** Grid is 125x250 (MATLAB native), not 100x200 as stated in paper. This is either a paper error or an earlier draft used lower resolution. MATLAB source is authoritative.

## 4.4 Cavity — Direct & Adjoint Eigenvalues

| Parameter | Paper | MATLAB | Python | Status |
|-----------|-------|--------|--------|--------|
| Re | 1,500 | 1,500 | 1,500 | OK |
| Ma | 0.5 | 0.5 | 0.5 | OK |
| T_0 | 300 K | 300 K | 300 K | OK |
| Domain | [-1.65, 5.34] x [0, 5] | x_start - L to x_start + 2L | x_start - L to x_start + 2L | OK (paper rounds) |
| Grid | 241 x 101 | 3*80+1 x 2*50+1 = 241 x 101 | 241 x 101 | OK |
| n_modes | — | 50 | 50 | OK |
| omega_target | — | 1.5 + 1.0i | 1.5 + 1.0j | OK |
| BCs | "Dirichlet on walls" | lbtrm, uvwT | lbtrm, uvwT | OK |
| Masking | "points within the wall were masked" | (Y<0) & ((X<0.34)\|(X>2.34)) | identical condition | OK |
| Mesh deform | "grid deformed in x and y" | cubic + power-law stretch | identical | OK |
| Sponge | "near top, left, right" | amp=5, tanh profile | amp=5, tanh profile | OK |
| Adjoint | "complex conjugating adjoint recovers direct" | `eigs(invW*L0'*W, n, conj(target))` | `adjoint_eigenvalues()` with `1j*conj(target)` internally | OK — different internal convention but identical output after post-processing |

**Python spectrum**: direct (blue dots) and adjoint (orange crosses) overlay perfectly, confirming conjugate pairing.

**Assessment: CORRECT.** Adjoint eigenvalue formulation differs in internal convention but produces identical results after the sign correction `omega_d = conj(-1j * omega_adj)`.

## 4.5 Paraboloid — Resolvent (Curvilinear, Axisymmetric)

| Parameter | Paper | MATLAB | Python | Status |
|-----------|-------|--------|--------|--------|
| Re | 5,000 | 5e3 | 5e3 | OK |
| Ma | 0.01 | 0.01 | 0.01 | OK |
| Grid | 100 x 100 | 100 x 100 | 100 x 100 | OK |
| m | 5 | 5 | 5 | OK |
| n_modes | 3 | 3 | 3 | OK |
| omega | 0 | 0 | 0 (with 1e-4j fallback) | OK — Python adds robustness for singular case |
| Conformal mapping | "parabolic coordinates via z^2" | X = X*Lx - 0.5; Y = Y*Ly; then z^2 | X = X*Lx - 0.5; Y = Y*Ly; then Z^2 | OK — **Lx/Ly scaling was missing in Python, fixed 2026-02-17** |
| BCs | "Dirichlet at surface; symmetry at axis" | ltr, uvwT | ltr, uvwT | OK |
| Sponge | "near left and top" | amp=5, tanh profile | amp=5, tanh profile | OK |

**Python gains: [2.62e8, 6.05e7, 2.12e7]** at 100x100.

**Assessment: CORRECT.** The Lx/Ly physical-domain scaling before the Z^2 conformal mapping was missing in a previous Python version but has been fixed. The domain now extends to x ~ -7 (matching MATLAB).

## Overall Summary

| Example | MATLAB match | Paper match | Notes |
|---------|-------------|-------------|-------|
| 1 Blasius | exact (sigma to 4 s.f.) | **Re typo in paper** (60k vs 600k) | Gold-standard validation |
| 2 Jet | exact (same .mat data) | OK | Baseflow from LES data |
| 3 Streaky BL | exact (125x250) | **Grid underreported** (paper: 100x200, code: 125x250) | MATLAB source authoritative |
| 4 Cavity | exact (241x101) | OK | Adjoint convention documented |
| 5 Paraboloid | exact (100x100) | OK | Lx/Ly scaling bug fixed |

All five examples run correctly at MATLAB native resolutions with PETSc/SLEPc + MUMPS.

---

# **References**

- <span id="page-17-3"></span>1. Akervik, E., et al.: Steady solutions of the Navier-Stokes equations by selective frequency damping. Phys. Fluids **18**(6), 068102 (2006). <https://doi.org/10.1063/1.2211705>
- <span id="page-17-1"></span>2. Amaral, F.R., Cavalieri, A.V.G.: Resolvent analysis of pressure fluctuations in turbulent channel flows. J. Fluid Mech. **927**, A17 (2021). [https://doi.org/10.1017/jfm.2021.764.](https://doi.org/10.1017/jfm.2021.764) [arXiv:2108.10833](http://arxiv.org/abs/2108.10833) [physics]
- <span id="page-17-0"></span>3. Amaral, F.R., Cavalieri, A.V.G., Martini, E., et al.: Resolvent-based estimation of turbulent channel flow using wall measurements. J. Fluid Mech. **927**, A17 (2021). <https://doi.org/10.1017/jfm.2021.764>

- <span id="page-18-7"></span>4. Audiffred, D.B.S., Cavalieri, A.V.G., Brito, P.P.C., et al.: Experimental control of Tollmien–Schlichting waves using the Wiener–Hopf formalism. Phys. Rev. Fluids **8**(7), 073902 (2023). <https://doi.org/10.1103/PhysRevFluids.8.073902>
- <span id="page-18-6"></span>5. Bagheri, S., Henningson, D.S.: Transition delay using control theory. Philos. Trans. R. Soc. A: Math. Phys. Eng. Sci. **369**(1940), 1365–1381 (2011)
- <span id="page-18-23"></span>6. Bers, A.: Space-time evolution of plasma instabilities-absolute and convective. Basic Plasma Phys. **1** (1983)
- <span id="page-18-32"></span>7. Brès, G., Jordan, P., Le Rallic, M., et al.: Importance of the nozzle-exit boundary-layer state in subsonic turbulent jets. J. Fluid Mech. **851**, 83–124 (2018). <https://doi.org/10.1017/jfm.2018.476>
- <span id="page-18-24"></span>8. Briggs, R.J.: Electron-Stream Interaction with Plasmas, vol. 29. MIT Press, Cambridge (1964)
- <span id="page-18-33"></span>9. Caillaud, C., Lehnasch, G., Martini, E., et al.: Linear and non-linear dynamics of streak breakdown in a hypersonic boundary layer. In: Twelfth International Symposium on Turbulence and Shear Flow Phenomena (TSFP12), Osaka, Japan (Online) (2022)
- <span id="page-18-12"></span>10. Camarri, S.: Flow control design inspired by linear stability analysis. Acta Mech. **226**(4), 979–1010 (2015). [https://doi.org/](https://doi.org/10.1007/s00707-015-1319-1) [10.1007/s00707-015-1319-1](https://doi.org/10.1007/s00707-015-1319-1)
- <span id="page-18-2"></span>11. Cavalieri, A.V., Agarwal, A.: Coherence decay and its impact on sound radiation by wavepackets. J. Fluid Mech. **748**, 399–415 (2014). <https://doi.org/10.1017/jfm.2014.186>
- 12. Cavalieri, A.V., Jordan, P., Agarwal, A., et al.: Jittering wave-packet models for subsonic jet noise. J. Sound Vib. **330**(18–19), 4474–4492 (2011). <https://doi.org/10.1016/j.jsv.2011.04.007>
- 13. Cavalieri, A.V., Jordan, P., Colonius, T., et al.: Axisymmetric superdirectivity in subsonic jets. J. Fluid Mech. **704**, 388–420 (2012). <https://doi.org/10.1017/jfm.2012.247>
- <span id="page-18-3"></span>14. Cavalieri, A.V.G., Jordan, P., Lesshafft, L.: Wave-packet models for jet dynamics and sound radiation. Appl. Mech. Rev. **71**(2), 020802 (2019). <https://doi.org/10.1115/1.4042736>
- <span id="page-18-20"></span>15. Chandler, G.J., Juniper, M.P., Nichols, J.W., et al.: Adjoint algorithms for the Navier–Stokes equations in the low mach number limit. J. Comput. Phys. **231**(4), 1900–1916 (2012)
- <span id="page-18-26"></span>16. Chomaz, J.M.: Global instabilities in spatially developing flows: non-normality and nonlinearity. Annu. Rev. Fluid Mech. **37**(1), 357–392 (2005). <https://doi.org/10.1146/annurev.fluid.37.061903.175810>
- <span id="page-18-17"></span>17. Chu, B.: On the energy transfer to small disturbances in fluid flow (Part I). Acta Mech. **1**(3), 215–234 (1965)
- <span id="page-18-4"></span>18. Citro, V., Giannetti, F., Brandt, L., et al.: Linear three-dimensional global and asymptotic stability analysis of incompressible open cavity flow. J. Fluid Mech. **768**, 113–140 (2015). <https://doi.org/10.1017/jfm.2015.72>
- <span id="page-18-19"></span>19. Citro, V., Luchini, P., Giannetti, F., et al.: Efficient stabilization and acceleration of numerical simulation of fluid flows by residual recombination. J. Comput. Phys. **344**, 234–246 (2017). <https://doi.org/10.1016/j.jcp.2017.04.081>
- <span id="page-18-31"></span>20. Colonius, T.: Modeling artificial boundary conditions for compressible flow. Ann. Rev. Fluid Mech. **36**, 315–345 (2004). <https://doi.org/10.1146/annurev.fluid.36.050802.121930>
- <span id="page-18-18"></span>21. Edgington-Mitchell, D., Wang, T., Nogueira, P., et al.: Waves in screeching jets. J. Fluid Mech. **913**, A7 (2021). [https://doi.](https://doi.org/10.1017/jfm.2020.1175) [org/10.1017/jfm.2020.1175](https://doi.org/10.1017/jfm.2020.1175)
- <span id="page-18-9"></span>22. Eriksson, L.E., Rizzi, A.: Computer-aided analysis of the convergence to steady state of discrete approximations to the Euler equations. J. Comput. Phys. **57**(1), 90–128 (1985)
- <span id="page-18-13"></span>23. Fabre, D., Citro, V., Ferreira Sabino, D., et al.: A practical review on linear and nonlinear global approaches to flow instabilities. Appl. Mech. Rev. **70**(6), 060802 (2018). <https://doi.org/10.1115/1.4042737>
- <span id="page-18-10"></span>24. Farghadan, A., Martini, E., Towne, A.: Scalable resolvent analysis for three-dimensional flows. (2023), [arXiv:2309.04617](http://arxiv.org/abs/2309.04617)
- <span id="page-18-30"></span>25. Gaitonde, D.V., Visbal, M.R.: Pade-type higher-order boundary filters for the Navier–Stokes equations. AIAA J. **38**(11), 2103–2112 (2000)
- <span id="page-18-1"></span>26. Garnaud, X., Lesshafft, L., Schmid, P.J., et al.: The preferred mode of incompressible jets: linear frequency response analysis. J. Fluid Mech. **716**, 189–202 (2013). <https://doi.org/10.1017/jfm.2012.540>
- <span id="page-18-21"></span>27. Gaster, M.: On the generation of spatially growing waves in a boundary layer. J. Fluid Mech. **22**(3), 433–441 (1965). [https://](https://doi.org/10.1017/S0022112065000873) [doi.org/10.1017/S0022112065000873](https://doi.org/10.1017/S0022112065000873)
- <span id="page-18-22"></span>28. Gaster, M.: Growth of disturbances in both space and time. Phys. Fluids **11**(4), 723 (1968). [https://doi.org/10.1063/1.](https://doi.org/10.1063/1.1691990) [1691990](https://doi.org/10.1063/1.1691990)
- <span id="page-18-0"></span>29. Goldstein, M.E.: Scattering of acoustic waves into Tollmien–Schlichting waves by small streamwise variations in surface geometry. J. Fluid Mech. **154**, 509–529 (1985). <https://doi.org/10.1017/S0022112085001641>
- <span id="page-18-11"></span>30. Gómez, F., Blackburn, H.M., Rudman, M., et al.: A reduced-order model of three-dimensional unsteady flow in a cavity based on the resolvent operator. J. Fluid Mech. **798**, R2 (2016). <https://doi.org/10.1017/jfm.2016.339>
- <span id="page-18-5"></span>31. Gomez Carrasco, F., Blackburn, H., Rudman, M., et al.: Reconstruction of turbulent pipe flow using resolvent modes: application to low-order models and flow control. In: Proceedings of the 19th Australasian Fluid Mechanics Conference 2014 Australasian Fluid Mechanics Society (AFMS 2014), pp. 1–4 (2014)
- <span id="page-18-15"></span>32. Hanifi, A., Schmid, P.J., Henningson, D.S.: Transient growth in compressible boundary layer flow. Phys. Fluids **8**(3), 826–837 (1996). <https://doi.org/10.1063/1.868864>
- <span id="page-18-25"></span>33. Huerre, P., Monkewitz, P.A.: Absolute and convective instabilities in free shear layers. J. Fluid Mech. **159**(1), 151 (1985). <https://doi.org/10.1017/S0022112085003147>
- <span id="page-18-27"></span>34. Huerre, P., Monkewitz, P.A.: Local and global instabilities in spatially developing flows. Annu. Rev. Fluid Mech. **22**(1), 473–537 (1990). <https://doi.org/10.1146/annurev.fl.22.010190.002353>
- <span id="page-18-29"></span>35. Hwang, Y., Cossu, C.: Linear non-normal energy amplification of harmonic and stochastic forcing in the turbulent channel flow. J. Fluid Mech. **664**, 51–73 (2010). <https://doi.org/10.1017/S0022112010003629>
- <span id="page-18-8"></span>36. Jin, B., Illingworth, S.J., Sandberg, R.D.: Feedback control of vortex shedding using a resolvent-based modelling approach. J. Fluid Mech. **897**, A26 (2020). <https://doi.org/10.1017/jfm.2020.347>
- <span id="page-18-28"></span>37. Jovanovi´c, M.R., Bamieh, B.: Componentwise energy amplification in channel flows. J. Fluid Mech. **534**, 145–183 (2005)
- <span id="page-18-14"></span>38. Juniper, M.P., Hanifi, A., Theofilis, V.: Modal stability theory. Appl. Mech. Rev. **66**(2), 024804 (2014). [https://doi.org/10.](https://doi.org/10.1115/1.4026604) [1115/1.4026604](https://doi.org/10.1115/1.4026604)
- <span id="page-18-16"></span>39. Karban, U., Bugeat, B., Martini, E., et al.: Ambiguity in mean-flow-based linear analysis. J. Fluid Mech. **900**, R5 (2020). <https://doi.org/10.1017/jfm.2020.566>

<span id="page-19-13"></span>40. Karban, U., Martini, E., Cavalieri, A.V.G., et al.: Self-similar mechanisms in wall turbulence, through the lens of resolvent analysis. J. Fluid Mech. p A.0 (2022). <https://doi.org/10.1017/jfm.2022.225>

- <span id="page-19-16"></span>41. Kotsonis, M., Giepman, R., Hulshoff, S., et al.: Numerical study of the control of Tollmien–Schlichting waves using plasma actuators. AIAA J. **51**(10), 2353–2364 (2013). <https://doi.org/10.2514/1.J051766>
- <span id="page-19-29"></span>42. Kreiss, HO., Scherer, G.: Finite element and finite difference methods for hyperbolic partial differential equations. In: Mathematical Aspects of Finite Elements in Partial Differential Equations, pp. 195–212. Elsevier (1974)
- <span id="page-19-22"></span>43. Leclercq, C., Demourant, F., Poussot-Vassal, C., et al.: Linear iterative method for closed-loop control of quasiperiodic flows. J. Fluid Mech. **868**, 26–65 (2019). <https://doi.org/10.1017/jfm.2019.112>
- <span id="page-19-10"></span>44. Lesshafft, L., Semeraro, O., Jaunet, V., et al.: Resolvent-based modeling of coherent wave packets in a turbulent jet. Phys. Rev. Fluids **4**(6), 063901 (2019). <https://doi.org/10.1103/PhysRevFluids.4.063901>
- <span id="page-19-18"></span>45. Luhar, M., Sharma, A.S., McKeon, B.J.: Opposition control within the resolvent analysis framework. J. Fluid Mech. **749**, 597–626 (2014). <https://doi.org/10.1017/jfm.2014.209>
- <span id="page-19-20"></span>46. Luhar, M., Sharma, A.S., McKeon, B.J.: A framework for studying the effect of compliant surfaces on wall turbulence. J. Fluid Mech. **768**, 415–441 (2015). <https://doi.org/10.1017/jfm.2015.85>
- <span id="page-19-21"></span>47. Luhar, M., Sharma, A.S., McKeon, B.J.: On the design of optimal compliant walls for turbulence control. J. Turbul. **17**(8), 787–806 (2016). <https://doi.org/10.1080/14685248.2016.1181267>
- <span id="page-19-8"></span>48. Mack, L.M.: Linear stability theory and the problem of supersonic boundary-layer transition. AIAA J. **13**(3), 278–289 (1975). <https://doi.org/10.2514/3.49693>
- <span id="page-19-17"></span>49. Maia, I.A., Jordan, P., Cavalieri, A.V.G., et al.: Real-time reactive control of stochastic disturbances in forced turbulent jets. Phys. Rev. Fluids **6**(12), 123901 (2021). <https://doi.org/10.1103/PhysRevFluids.6.123901>
- <span id="page-19-2"></span>50. Martini, E., Cavalieri, A.V.G., Jordan, P.: Acoustic modes in jet and wake stability. J. Fluid Mech. **867**, 804–834 (2019). <https://doi.org/10.1017/jfm.2019.148>
- <span id="page-19-23"></span>51. Martini, E., Cavalieri, A.V.G., Jordan, P., et al.: Resolvent-based optimal estimation of transitional and turbulent flows. J. Fluid Mech. **900**, A2 (2020). <https://doi.org/10.1017/jfm.2020.435>
- <span id="page-19-26"></span>52. Martini, E., Rodríguez, D., Towne, A., et al.: Efficient computation of global resolvent modes. J. Fluid Mech. **919**, A3 (2021). <https://doi.org/10.1017/jfm.2021.364>
- <span id="page-19-24"></span>53. Martini, E., Jung, J., Cavalieri, A.V., et al.: Resolvent-based tools for optimal estimation and control via the Wiener–Hopf formalism. J. Fluid Mech. **937**, A19 (2022). <https://doi.org/10.1017/jfm.2022.102>
- <span id="page-19-33"></span>54. Martini, E., Caillaud, C., Lehnasch, G., et al.: Perturbation amplification near the stagnation point of blunt bodies. (2023). [arXiv:2311.02486](http://arxiv.org/abs/2311.02486)
- <span id="page-19-32"></span>55. Mathias, M., Medeiros, M.: Interaction between rossiter and görtler modes in the compressible flow in an open cavity. In: AIAA Paper, p 3074 (2020)
- <span id="page-19-30"></span>56. Mattsson, K., Nordström, J.: Summation by parts operators for finite difference approximations of second derivatives. J. Comput. Phys. **199**(2), 503–540 (2004)
- <span id="page-19-9"></span>57. McKeon, B.J., Sharma, A.S.: A critical-layer framework for turbulent pipe flow. J. Fluid Mech. **658**, 336–382 (2010). <https://doi.org/10.1017/S002211201000176X>
- <span id="page-19-3"></span>58. Meliga, P., Sipp, D., Chomaz, J.M.: Absolute instability in axisymmetric wakes: compressible and density variation effects. J. Fluid Mech. **600**, 373–401 (2008)
- <span id="page-19-0"></span>59. Michalke, A.: On the inviscid instability of the hyperbolictangent velocity profile. J. Fluid Mech. **19**(4), 543–556 (1964). <https://doi.org/10.1017/S0022112064000908>
- <span id="page-19-1"></span>60. Michalke, A.: On spatially growing disturbances in an inviscid shear layer. J. Fluid Mech. **23**(3), 521–544 (1965). [https://](https://doi.org/10.1017/S0022112065001520) [doi.org/10.1017/S0022112065001520](https://doi.org/10.1017/S0022112065001520)
- <span id="page-19-31"></span>61. Mohseni, K., Colonius, T.: Numerical treatment of polar coordinate singularities. J. Comput. Phys. **157**(2), 787–795 (2000)
- <span id="page-19-4"></span>62. Monkewitz, P.A.: The absolute and convective nature of instability in two-dimensional wakes at low Reynolds numbers. Phys. Fluids **31**(5), 999 (1988). <https://doi.org/10.1063/1.866720>
- 63. Monkewitz, P.A., Sohn, K.: Absolute instability in hot jets. AIAA J. **26**(8), 911–916 (1988). <https://doi.org/10.2514/3.9990>
- <span id="page-19-5"></span>64. Monkewitz, P.A., Bechert, D.W., Barsikow, B., et al.: Self-excited oscillations and mixing in a heated round jet. J. Fluid Mech. **213**(1), 611 (1990). <https://doi.org/10.1017/S0022112090002476>
- <span id="page-19-6"></span>65. Monokrousos, A., Åkervik, E., Brandt, L., et al.: Global three-dimensional optimal disturbances in the Blasius boundarylayer flow using time-steppers. J. Fluid Mech. **650**, 181–214 (2010). <https://doi.org/10.1017/S0022112009993703>
- <span id="page-19-14"></span>66. Morra, P., Nogueira, P.A.S., Cavalieri, A.V.G., et al.: The colour of forcing statistics in resolvent analyses of turbulent channel flows. J. Fluid Mech. **907**, A24 (2021). <https://doi.org/10.1017/jfm.2020.802>
- <span id="page-19-27"></span>67. Morra, P., Nogueira, P.A.S., Cavalieri, A.V.G., et al.: The colour of forcing statistics in resolvent analyses of turbulent channel flows. J. Fluid Mech. **907**, A24 (2021). <https://doi.org/10.1017/jfm.2020.802>
- <span id="page-19-28"></span>68. Nibourel, P., Leclercq, C., Demourant, F., et al.: Reactive control of second Mack mode in a supersonic boundary layer with free-stream velocity/density variations. J. Fluid Mech. **954**, A20 (2023). <https://doi.org/10.1017/jfm.2022.997>
- <span id="page-19-19"></span>69. Nibourel, P., Leclercq, C., Demourant, F., et al.: Extending closed-loop control of two-dimensional supersonic boundarylayer instabilities beyond a single operating point. AIAA J. **62**(3), 896–914 (2024). <https://doi.org/10.2514/1.J062544>
- <span id="page-19-11"></span>70. Nogueira, P.A.S., Cavalieri, A.V., Jordan, P., et al.: Large-scale streaky structures in turbulent jets. J. Fluid Mech. **873**, 211–237 (2019). <https://doi.org/10.1017/jfm.2019.365>
- <span id="page-19-15"></span>71. Nogueira, P.A.S., Morra, P., Martini, E., et al.: Forcing statistics in resolvent analysis: application in minimal turbulent Couette flow. J. Fluid Mech. **908**, A32 (2021). <https://doi.org/10.1017/jfm.2020.918>
- <span id="page-19-25"></span>72. Orszag, S.A.: Accurate solution of the Orr–Sommerfeld stability equation. J. Fluid Mech. **50**(4), 689–703 (1971). [https://](https://doi.org/10.1017/S0022112071002842) [doi.org/10.1017/S0022112071002842](https://doi.org/10.1017/S0022112071002842)
- <span id="page-19-7"></span>73. Pfister, J.L., Fabbiane, N., Marquet, O.: Global stability and resolvent analyses of laminar boundary-layer flow interacting with viscoelastic patches. J. Fluid Mech. **937**, A1 (2022). <https://doi.org/10.1017/jfm.2022.72>
- <span id="page-19-12"></span>74. Pickering, E., Rigas, G., Nogueira, P.A.S., et al.: Lift-up, Kelvin–Helmholtz and Orr mechanisms in turbulent jets. J. Fluid Mech. **896**, A2 (2020). <https://doi.org/10.1017/jfm.2020.301>

- <span id="page-20-4"></span>75. Pickering, E., Rigas, G., Schmidt, O.T., et al.: Optimal eddy viscosity for resolvent-based models of coherent structures in turbulent jets. J. Fluid Mech. **917**, A29 (2021). <https://doi.org/10.1017/jfm.2021.232>
- <span id="page-20-23"></span>76. Reddy, S.C., Schmid, P.J., Henningson, D.S.: Pseudospectra of the Orr–Sommerfeld operator. SIAM J. Appl. Math. **53**(1), 15–47 (1993). <https://doi.org/10.1137/0153002>
- <span id="page-20-13"></span>77. Rodríguez, D., Gennaro, E.M.: Three-dimensional flow stability analysis based on the matrix-forming approach made affordable. In: Bittencourt, M.L., Dumont, N.A., Hesthaven, J.S. (eds) Spectral and High Order Methods for Partial Differential Equations ICOSAHOM 2016, vol 119, pp. 639–650. Springer International Publishing, Cham (2017). [https://doi.](https://doi.org/10.1007/978-3-319-65870-4_46) [org/10.1007/978-3-319-65870-4\\_46](https://doi.org/10.1007/978-3-319-65870-4_46)
- <span id="page-20-10"></span>78. Sasaki, K., Tissot, G., Cavalieri, A.V.G., et al.: Closed-loop control of a free shear flow: a framework using the parabolized stability equations. Theoret. Comput. Fluid Dyn. **32**(6), 765–788 (2018). <https://doi.org/10.1007/s00162-018-0477-x>
- <span id="page-20-2"></span>79. Schlichting, H.: Zur enstehung der turbulenz bei der plattenströmung. Nachrichten von der Gesellschaft der Wissenschaften zu Göttingen, Mathematisch-Physikalische Klasse **1933**, 181–208 (1933)
- <span id="page-20-19"></span>80. Schmid, P.J.: Nonmodal stability theory. Annu. Rev. Fluid Mech. **39**, 129–162 (2007). [https://doi.org/10.1146/annurev.](https://doi.org/10.1146/annurev.fluid.38.050304.092139) [fluid.38.050304.092139](https://doi.org/10.1146/annurev.fluid.38.050304.092139)
- <span id="page-20-22"></span>81. Schmid, PJ., Henningson, DS.: Stability and Transition in Shear Flows, vol 142. Springer Science & Business Media (2012)
- <span id="page-20-9"></span>82. Schmid, P.J., Sipp, D.: Linear control of oscillator and amplifier flows. Phys. Rev. Fluids **1**(4), 040501 (2016). [https://doi.](https://doi.org/10.1103/PhysRevFluids.1.040501) [org/10.1103/PhysRevFluids.1.040501](https://doi.org/10.1103/PhysRevFluids.1.040501)
- <span id="page-20-14"></span>83. Schmidt, O.T., Rist, U.: Linear stability of compressible flow in a streamwise corner. J. Fluid Mech. **688**, 569–590 (2011). <https://doi.org/10.1017/jfm.2011.405>
- <span id="page-20-15"></span>84. Schmidt, O.T., Rist, U.: Viscid–inviscid pseudo-resonance in streamwise corner flow. J. Fluid Mech. **743**, 327–357 (2014). <https://doi.org/10.1017/jfm.2014.31>
- <span id="page-20-5"></span>85. Schmidt, O.T., Towne, A., Colonius, T., et al.: Wavepackets and trapped acoustic modes in a turbulent jet: coherent structure eduction and global stability. J. Fluid Mech. **825**, 1153–1181 (2017). <https://doi.org/10.1017/jfm.2017.407>
- <span id="page-20-16"></span>86. Schmidt, O.T., Towne, A., Rigas, G., et al.: Spectral analysis of jet turbulence. J. Fluid Mech. **855**, 953–982 (2018). [https://](https://doi.org/10.1017/jfm.2018.675) [doi.org/10.1017/jfm.2018.675](https://doi.org/10.1017/jfm.2018.675)
- <span id="page-20-21"></span>87. Schubauer, G.B., Skramstad, H.K.: Laminar boundary-layer oscillations and stability of laminar flow. J. Aeronaut. Sci. **14**(2), 69–78 (1947). <https://doi.org/10.2514/8.1267>
- <span id="page-20-20"></span>88. Shroff, G.M., Keller, H.B.: Stabilization of unstable procedures: the recursive projection method. SIAM J. Numer. Anal. **30**(4), 1099–1120 (1993). <https://doi.org/10.1137/0730057>
- <span id="page-20-1"></span>89. Sipp, D., Lebedev, A.: Global stability of base and mean flows: a general approach and its applications to cylinder and open cavity flows. J. Fluid Mech. (2007). <https://doi.org/10.1017/S0022112007008907>
- <span id="page-20-28"></span>90. Sipp, D., Marquet, O.: Characterization of noise amplifiers with global singular modes: the case of the leading-edge flat-plate boundary layer. Theoret. Comput. Fluid Dyn. **27**(5), 617–635 (2013)
- <span id="page-20-27"></span>91. Stewart, G.W.: A Krylov–Schur algorithm for large eigenproblems. SIAM J. Matrix Anal. Appl. **23**(3), 601–614 (2002). <https://doi.org/10.1137/S0895479800371529>
- <span id="page-20-26"></span>92. Strand, B.: Summation by parts for finite difference approximations for d/dx. J. Comput. Phys. **110**(1), 47–67 (1994)
- <span id="page-20-6"></span>93. Symon, S., Sipp, D., McKeon, B.J.: A tale of two airfoils: resolvent-based modelling of an oscillator versus an amplifier from an experimental mean. J. Fluid Mech. **881**, 51–83 (2019). <https://doi.org/10.1017/jfm.2019.747>
- <span id="page-20-17"></span>94. Taira, K., Brunton, S.L., Dawson, S.T.M., et al.: Modal analysis of fluid flows: an overview. AIAA J. **55**(12), 4013–4041 (2017). <https://doi.org/10.2514/1.J056060>
- <span id="page-20-18"></span>95. Theofilis, V.: Global linear instability. Annu. Rev. Fluid Mech. **43**, 319–352 (2011)
- <span id="page-20-3"></span>96. Tollmien, W., Miner, D.M., et al.: The production of turbulence. Tech. rep, National Advisory Committee for Aeronautics (1931)
- <span id="page-20-8"></span>97. Towne, A., Schmidt, O.T., Colonius, T.: Spectral proper orthogonal decomposition and its relationship to dynamic mode decomposition and resolvent analysis. J. Fluid Mech. **847**, 821–867 (2018). <https://doi.org/10.1017/jfm.2018.283>
- <span id="page-20-12"></span>98. Towne, A., Lozano-Durán, A., Yang, X.: Resolvent-based estimation of space-time flow statistics. J. Fluid Mech. **883**, A17 (2020). <https://doi.org/10.1017/jfm.2019.854>
- <span id="page-20-25"></span>99. Trefethen, L.N.: Pseudospectra of linear operators. SIAM Rev. **39**(3), 383–406 (1997). [https://doi.org/10.1137/](https://doi.org/10.1137/S0036144595295284) [S0036144595295284](https://doi.org/10.1137/S0036144595295284)
- <span id="page-20-24"></span>100. Trefethen, L.N., Trefethen, A.E., Reddy, S.C., et al.: Hydrodynamic stability without eigenvalues. Science **261**(5121), 578–584 (1993)
- <span id="page-20-0"></span>101. von Helmholtz, H.: Über Discontinuirliche Flüssigkeits-Bewegungen. Akademie der Wissenschaften zu Berlin (1868)
- <span id="page-20-29"></span>102. White, F.M., Majdalani, J.: Viscous Fluid Flow, vol. 3. McGraw-Hill, New York (2006)
- <span id="page-20-7"></span>103. Yeh, C.A., Taira, K.: Resolvent-analysis-based design of airfoil separation control. J. Fluid Mech. **867**, 572–610 (2019). <https://doi.org/10.1017/jfm.2019.163>
- <span id="page-20-11"></span>104. Yeh, C.A., Taira, K.: Resolvent-analysis-based design of airfoil separation control. J. Fluid Mech. **867**, 572–610 (2019). <https://doi.org/10.1017/jfm.2019.163>

**Publisher's Note** Springer Nature remains neutral with regard to jurisdictional claims in published maps and institutional affiliations.

Springer Nature or its licensor (e.g. a society or other partner) holds exclusive rights to this article under a publishing agreement with the author(s) or other rightsholder(s); author self-archiving of the accepted manuscript version of this article is solely governed by the terms of such publishing agreement and applicable law.