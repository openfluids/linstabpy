# *Bifurcations in shock-wave/laminar-boundarylayer interaction: global instability approach*

# *J.-C*h*. ROBINET*

SINUMEF Laboratory, ENSAM CER de Paris, 151, Boulevard de l'Hopital, 75013 Paris, France ˆ

(Received 31 October 2005 and in revised form 1 November 2006)

The principal objective of this paper is to study some unsteady characteristics of an interaction between an incident oblique shock wave impinging on a laminar boundary layer developing on a flat plate. More precisely, this paper shows that some unsteadiness, in particular the low-frequency unsteadiness, originates in a supercritical Hopf bifurcation related to the dynamics of the separated boundary layer. Various direct numerical simulations were carried out of a shock-wave/laminar-boundarylayer interaction (SWBLI). Three-dimensional unsteady Navier–Stokes equations are numerically solved with an implicit dual time stepping for the temporal algorithm and high-order AUSMPW+ scheme for the spatial discretization. A parametric study on the oblique shock-wave angle has been performed to characterize the unsteady behaviour onset. These numerical simulations have shown that starting from the incident shock angle and the spanwise extension, the flow becomes three-dimensional and unsteady. A linearized global stability analysis is carried out in order to specify and to find some characteristics observed in the direct numerical simulation. This stability analysis permits us to show that the physical origin generating the threedimensional characters of the flow results from the existence of a three-dimensional stationary global instability.

# *1. Introduction*

Effective design of supersonic air vehicles requires accurate simulation methods for predicting aerothermodynamic loads (i.e. mean and fluctuating surface pressure, skin friction and heat transfer). Shock wave/turbulent-boundary-layer interaction (SWTBLI) is common in high-speed flight, and can significantly affect the aerothermodynamic loads.

When a large adverse pressure gradient exists in the inviscid pressure distribution, the viscous effects become important. The interaction between the oncoming boundary layer and the adverse pressure gradient drastically modifies the inviscid pressure distribution and the flow field. Multiple shocks, flow separation, transition to turbulence, unsteadiness, and three-dimensionality appear near the interaction region. This phenomenon appears in transonic flows over airfoils, supersonic flows over compression corners, and flows over steps. Liepmann (1946) and Ackeret, Feldmann & Rott (1947) were the first to investigate experimentally the mutual influences of the compression shocks and the boundary layers at transonic and low supersonic Mach numbers in laminar and turbulent flow regimes. Since then, several experiments, analyses, and computations have been performed to investigate the shock/boundary-layer interactions in detail. Most of the experiments measured or computed aerodynamic quantities such as pressure distribution, skin friction, and heat-transfer rates. The flow separates in front of the shock with a gradual increase of pressure starting upstream of the separation point. This gradual increase in pressure ends with a sharp increase near the shock, and the flow becomes turbulent behind the shock. In both cases, the boundary-layer thickness increases considerably: about 10 times in laminar flows and about four times in turbulent flows. Chapman, Kuehn & Larson (1958) conducted an extensive investigation on flow separation with steps, bases, compression corners and curved surfaces at different Mach numbers, ranging from 0*.*4 to 3*.*6, and at different Reynolds numbers. They observed that the pressure distribution in separated flows depends on the location of the transition point relative to the reattachment and separation points. In laminar separations, the pressure initially rises smoothly, reaches a plateau and, depending on the downstream condition, rises to the final pressure smoothly. In transitional separated flows where the flow starts to become turbulent between the separation and reattachment points, the pressure initially rises smoothly, as in laminar flows, and then increases sharply near the transition region. The pressure distribution also becomes unsteady in this case. In turbulent separated flows, the pressure rise is steep from the start to the end. They also observed that the mixing layer above the separation bubble is stable in supersonic flows, and the stability increases with increasing Mach numbers. The mechanism for the upstream propagation of the disturbances in a boundary layer and in a supersonic free stream from the adverse pressure-gradient region was first explained by Lighthill (1950, 1953a, b) using self-induced separation theory and later by Stewartson & Williams (1969) using asymptotic triple-deck theory. The mechanism is that the separated region near the shock produces an adverse pressure gradient in the outer part of the boundary layer, and this induces further growth of the separated region until they come to an equilibrium state further upstream of the original discontinuity. This theory predicts the initial pressure rise close to the separation point, and the agreement between the calculated and the experimental pressure distribution close to the separation point is excellent. For a more comprehensive discussion of shock-boundary layer interaction phenomena of such base flows, see Delery & Marvin (1986), Smits & Dussauge (1996) and Dolling ´ (2001). One of the current problems of SWBLI is to understand the various physical mechanisms responsible for the unsteady character of this interaction. Whether the flow is laminar, transitional or turbulent, this unsteadiness is likely to strongly modify the various physical characteristics of the SWBLI previously evoked. In the following, some physical mechanisms responsible for SWBLI unsteadiness will be highlighted.

Instability studies of supersonic flows with or without SWBLI have been carried out principally in the transition context. Transition from a laminar to a turbulent flow comprises high aerodynamic loads. It has been a major area of concern over the past few decades and much work has been carried out in order to understand and possibly influence transition. However, although progress has been made, the physics are far from being understood. Much less work has been done on compressible flows, such as hypersonic flows, than on incompressible flows. For the first phase of the transition process, quantitative predictions can be made with compressible linear stability theory, which was formulated, for example, by Mack (1969) and Malik (1990). Kosinov, Maslov & Shevelkov (1990) and Eissler & Bestek (1996) investigated transition to turbulence of flat-plate boundary layers for a large range of Mach numbers, For hypersonic Mach numbers, few studies were carried out in particular with SWBLI (Bedarev et al. 2002; Pagella, Rist & Wagner 2002; Balakumar, Zhao & Atkins 2005).

![](_page_2_Figure_3.jpeg)

Figure 1. Schematic representation of the shock-wave/boundary-layer interaction.

For SWBLI, the only transition mechanisms which were studied, to our knowledge, concern the existence of local instabilities and generally only convective ones. Rare are the cases where the analysis relates to the presence of three-dimensional global instabilities with large transverse wavelengths. These global instabilities, when they exist, can be responsible for certain physical mechanisms producing an intrinsic three-dimensionality of the flow as well as a low-frequency unsteadiness.

The objective of the work presented in this paper is twofold, it is on the one hand to show that SWBLI can become unsteady independently of the laminar or turbulent nature of the flow and on the other hand, to show that the physical mechanisms responsible for this unsteadiness originate in a three-dimensional global instability.

A direct numerical simulation was carried out. The final state reached by simulation is described in a precise way in order to characterize its various space–time characteristics (§ 2). A study of the computational transients was undertaken in order to analyse the various processes leading to this final state. In addition, a study of global linear stability was carried out and compared (§ 3) with the results obtained by the numerical simulation. The last section is dedicated to the conclusions and prospects (§ 4).

# *2. Numerical simulations*

## 2.1. Physical configurations

In the following, only shock-wave/laminar-boundary-layer interaction on a flate plate has been investigated. The test case considered has been experimentally and numerically studied by Degrez, Boccadoro & Wendt (1987). The free-stream inflow Mach number is 2*.*15 for the numerical simulation. The Reynolds number based on the distance *Xsh* between the plate leading edge and the shock impingement point is 105. The shock angle with respect to the horizontal is *θ* = 30*.*8◦, which corresponds to a shock generator angle of 3*.*75◦. This dataset takes into account confinement, three-dimensional effects and measurement approximations; it is not strictly the same as the experimental free-stream conditions (see Degrez et al. 1987 for more details). At this incidence angle, Degrez et al. (1987) indicate that the flow remains stationary and two-dimensional upstream, downstream and in the interaction. Furthermore, it remains laminar at least until the end of the measurement zone. Figure 1 shows a sketch of the shock-wave/laminar-boundary-layer interaction and table 1 gives the different physical parameters.

Parameter Value Free-stream Mach number  $M_{\infty} = 2.15$  Interaction length  $X_{sh} = 8 \times 10^{-2} \text{ m}$  Free-stream Reynolds number  $Re = 10^5$  Incident shock angle  $\theta = [30.8^{\circ}; 33^{\circ}]$  Spanwise length  $\theta = [0.1; 3]$  Prandtl number  $\theta = [0.72]$  Ratio of specific heats  $\theta = [0.1; 3]$ 

TABLE 1. Flow parameters for the SWBLI.

To demonstrate that the low-frequency behaviour observed in some SWBLI configurations can be linked to the intrinsic dynamics of the detached zone induced by the interaction, independently of the turbulent boundary-layer characteristics, the evolution of an incident shock wave impinging on a laminar boundary layer developing over a flat plate is studied when the incident shock angle is gradually increased. The free-stream inflow Mach number and the global Reynolds number remain unchanged. The evolution of the SWBLI when the incident shock angle increases is a complex problem. Indeed, for a particular value of the angle  $\theta$ , the flow becomes transitional in the interaction zone. This transitional state will probably modify substantially the topology and the dynamics of the interaction zone. In addition, no unsteady disturbance, of convective instability type, is introduced at the upstream end of the computational domain in order not to start possible instabilities of a convective nature which could mask and/or modify the existence of a global instability. Three-dimensional numerical simulations will be carried out without taking into account the transitional character of SWBLI. Considering these assumptions, these present computations are meant to show that an SWBLI can become unsteady without taking into account the turbulent character of the flow. In this scenario, the unsteadiness onset is directly linked to the intrinsic dynamics of the detached zone and quickly leads toward a three-dimensional and unsteady flow.

## 2.2. Governing equations

The equations solved are the three-dimensional unsteady compressible Navier-Stokes (N-S) equations in conservation form:

$$\frac{\partial}{\partial t}Q_i + \frac{\partial}{\partial x_i} \left( F_{ji}^{(c)} - F_{ji}^{(v)} \right) = 0, \tag{2.1}$$

Here

$$Q_{i} = \begin{pmatrix} \rho \\ \rho E \\ \rho u \\ \rho v \\ \rho w \end{pmatrix}, \left[ F_{ji}^{(c)} \right] = \begin{pmatrix} \rho u_{j} \\ (\rho E + p) u_{j} \\ \rho u u_{j} + \delta_{1j} p \\ \rho v u_{j} + \delta_{2j} p \\ \rho w u_{j} + \delta_{3j} p \end{pmatrix}, \left[ F_{ji}^{(v)} \right] = \begin{pmatrix} 0 \\ u \tau_{1j} + v \tau_{2j} + w \tau_{3j} - q_{j} \\ \tau_{1j} \\ \tau_{2j} \\ \tau_{3j} \end{pmatrix}. \quad (2.2)$$

Here (x, y, z) are the Cartesian coordinates, (u, v, w) are the velocity components,  $\rho$  is the density, and p is the pressure. E is the total energy given by:

$$E = e + (u^2 + v^2 + w^2)/2, \ e = c_v T, \ p = \rho r T.$$
 (2.3)

Here e is the internal energy, and T is the temperature. The shear stress and the heat flux are given by

$$\tau_{ij} = \mu \left( \frac{\partial u_i}{\partial x_j} + \frac{\partial u_j}{\partial x_i} - \frac{2}{3} \delta_{ij} \frac{\partial u_k}{\partial x_k} \right), \quad q_j = -\kappa \frac{\partial T}{\partial x_j}. \tag{2.4}$$

The viscosity  $\mu$  is computed using Sutherland's law, and the coefficient of conductivity  $\kappa$  is given in terms of the Prandtl number Pr. The variables  $\rho$ , p, T and velocity are non-dimensionalized by their corresponding reference variables  $\rho_{\infty}$ ,  $p_{\infty}$ ,  $T_{\infty}$  and  $U_{\infty}$ , respectively. The reference value for length is  $X_{sh}$ , the distance between the plate leading edge and the shock impingement.

## 2.3. Numerical method

The governing equations are solved using a fifth-order-accurate AUSMPW+ scheme for space discretization initially developed by Liou & Edwards (1998). Kim, Kim & Rho (2001a, b) describe in detail the solution method (the AUSMPW+ scheme) implemented in our computation. These methods are suitable in flows with discontinuities or high-gradient regions. These schemes solve the governing equations discretely in a uniform structured computational domain in which flow properties are known pointwise at the grid nodes. They approximate the spatial derivatives in a given direction to a higher order at the nodes, using the neighbouring nodal values in that direction, and they integrate the resulting equations in time to find the point values as a function of time. Because the spatial derivatives are independent of the coordinate directions, the method can easily add multidimensions. It is well known that approximating a discontinuous function by a higher-order (two or more) polynomial generally introduces oscillatory behaviour near the discontinuity, and this oscillation increases with the order of the approximation. High accuracy of the inviscid numerical fluxes is ensured through the use of a fifth-order MUSCL reconstruction of the primitive variables vector  $(\rho, u, v, w, p)^t$ . Usually, the reconstruction process also involves the use of a slope limiter in order to avoid the numerical oscillations onset in the solution. The main effect of this limiting process is to bring down the accuracy of the scheme to first order in flow regions where it is active: in turn, this reduction of accuracy can substantially alter the flow prediction, so that unsteady phenomena may no longer spontaneously appear. In the present study, the use of such a limiter was not found to be necessary because enough natural dissipation is provided by the viscous terms to prevent the occurrence of numerical oscillations. For more details on the performances of this present solver with or without limiting process, see Alfano et al. (2004).

A time-accurate approximate solution of system (2.1) is obtained using the following implicit second-order linear multi-step method:

$$\mathscr{T}(\mathscr{U}^{n+1}, \mathscr{U}^n, \mathscr{U}^{n-1}) + \mathscr{R}(\mathscr{U}^{n+1}) = 0, \tag{2.5}$$

where  $\mathcal{R}$  gathers the space-discretization operators described in the previous section and  $\mathcal{T}$  is a three-step approximation of  $\mathcal{U}_t$  at time level (n+1) defined by:

$$\mathcal{T}(\mathcal{U}^{n+1}, \mathcal{U}^n, \mathcal{U}^{n-1}) = (1+\phi) \frac{(\mathcal{U}^{n+1} - \mathcal{U}^n)}{\Delta t} - \phi \frac{(\mathcal{U}^n - \mathcal{U}^{n-1})}{\Delta t} = (\mathcal{U})^{n+1} + O(\Delta t^p). \quad (2.6)$$

The choice of  $\phi = 1/2$  in (2.6) allows us to reach second-order accuracy in time (p = 2). Although the order in time is not very high, the studied global instabilities generally have low frequencies and the time step  $\Delta t$  is selected small, around  $\Delta t \sim 10^{-5}$  s, which is largely sufficient to compute required dynamics. In order to solve efficiently

| Parameter                                 | Value                                                         |
|-------------------------------------------|---------------------------------------------------------------|
| $(N_{x}, N_{y}, N_{z})$                   | (600, 180, 60)                                                |
| $(\Delta x, (\Delta y)^{wall}, \Delta z)$ | $(3 \times 10^{-3}, 7.8 \times 10^{-5}, 1.66 \times 10^{-2})$ |
| Geometrical ratio $q$                     | 1.02                                                          |
| $x_{in}, x_{sponge}, x_{out}$             | 0.2, 2, 2.3                                                   |
| $y \in [0, y_{max}]$                      | $y \in [0; 0.94]$                                             |
| $z \in [0, L_{z}]$                        | $z \in [0; 1]$                                                |
| Dual CFL                                  | 50                                                            |
| Physical CFL                              | 6                                                             |
| Time step                                 | $6.82 \times 10^{-6}$                                         |

Table 2. Computational parameters for the SWBLI.

the implicit system R(U*<sup>n</sup>*+1) = 0 where R(U*<sup>n</sup>*+1) = T(U*<sup>n</sup>*+1*,* U*n,* U*<sup>n</sup>*−1) + R(U*<sup>n</sup>*+1), we make use of a dual time technique, well known for incompressible flow calculations (Peyret & Taylor 1983) and made popular by Jameson (1991) for computing compressible flows. Actually, U*<sup>n</sup>*+1 is obtained as a steady solution of an evolution problem with respect to a dual or fictitious time *τ* :

$$\mathscr{U}_{\tau} + \mathscr{R}^{\star}(\mathscr{U}) = 0. \tag{2.7}$$

Solving (2.7) instead of (2.5) allows the use of much larger physical time steps; however, in the meantime, it also requires converging to a pseudosteady state at each physical time step. Therefore, the dual time approach is of interest only if system (2.7) can be efficiently solved. In the present study, a matrix-free point-relaxation method allows us to obtain a steady solution of system (2.7) after a few sub-iterations on the dual time (here around 200); moreover, this implicit treatment induces a low memory storage requirement which makes the treatment of a large number of grid points accessible with moderate computer configurations (see for instance Luo, Baum & Lhner (2001) for more details on this implicit technique). For more details on the numerical procedure used here as well as different test cases validating the various numerical choices, see Boin et al. (2006).

## 2.4. Computational domain and boundary conditions

The numerical method described in the previous section is applied to the computation of an oblique shock wave/laminar boundary layer on a flat plate. The coordinates are non-dimensionalized by the interaction length *Xsh*. The geometry of the threedimensional domain is D = [0*.*2; 2*.*3] × [0; 0*.*94] × [0*, Lz*] with 600 × 180 × 60 points. The grid is uniform in the streamwise and spanwise directions and geometrical in the normal direction. The transverse direction *Lz* lies between 0*.*1 and 3 with a number of planes ranging between 40 and 60 planes. The three-dimensional dimensionless mesh spacing is equal to  *x* = 3 × 10<sup>−</sup>3,  *y* = 7*.*8 × 10<sup>−</sup><sup>5</sup> at the wall and from  *z* = 1*.*05 × 10<sup>−</sup><sup>3</sup> to  *z* = 1*.*66 × 10<sup>−</sup>2. All the numerical parameters necessary to the numerical simulation are given in table 2. Figure 2 shows the dimensions of the geometry and the computational domain.

The steady two-dimensional Navier–Stokes solution is imposed at the inflow (*x* = 0*.*2). This latter is repeated in the spanwise direction. The inflow boundary condition is thus homogeneous according to *z*. At the outflow and at the upper boundary, high-order extrapolations are used as boundary conditions for the conservative variables. The flat plate is assumed to be an adiabatic wall where

![](_page_6_Figure_3.jpeg)

Figure 2. Computational domain and boundary conditions.

the velocity vector is zero (no-slip condition); pressure is also extrapolated from the values just above the plate. A sponge zone is imposed from *x* = 2 to *x* = 2*.*3. At the wall, the simulation uses viscous conditions for the velocities and a constant temperature condition, and it computes density from the continuity equation. In the spanwise direction, the solution can be characterized as a neutral oscillation, which is periodic at its boundaries. In this case, for computational cost reasons, we preferred to use a spectral scheme based on a Fourier decomposition. The Fourier basis is a natural choice for expanding the function with periodic boundary conditions as is the case in the spanwise direction in our problem. For more details on the numerical implementation, see Canuto et al. (1987) and Boyd (1999). These boundary conditions are classically used in direct numerical simulation, but will have important consequences on the results analysed in this study. Indeed, in the case of a flow that naturally produces three-dimensional structures, the dimension *Lz* given to the domain in the *z*-direction will force the wavelength of the spanwise structures. Therefore, the spanwise dimension should ideally be as large as possible to let three-dimensional instabilities appear spontaneously. For traditional DNS studies of convective instabilities, *Lz* is fixed by the user and corresponds to the most unstable wavelength, which appears first and is representative of the real configuration (without lateral boundaries). This wavelength is generally short compared to the other characteristic dimensions of the problem and is given by a linear stability calculation. The present SWBLI study is focused on low-frequency phenomena with corresponding large wavelengths; moreover, there is no way to determine in a reliable way the wavelength of the strongest instability in that case. It was therefore decided to consider *Lz* as a parameter rather than a fixed input data point; for all the threedimensional computations, a parametric study on *Lz* is required. Two types of initial condition are used in order to study the dependence of the solution with respect to the latter. The first condition consists in initializing the computation by a two-dimensional solution in the *z*-direction. The second condition is to impose the incoming supersonic flow *V*<sup>∞</sup> = (*ρ*∞*, u*∞*, v*∞*, w*∞*, p*∞) *<sup>t</sup>* on the lower part of the domain, *y* ∈ [0; *ys*], ∀*x*, while another (supersonic) state *V*down is imposed on the upper part of the domain,

![](_page_7_Figure_3.jpeg)

FIGURE 3. Flow organization according to the incident shock angle and the transverse size of the computation domain.

 $y \in [y_i; y_{max}, \forall x$ . This state is computed so as to satisfy the Rankine-Hugoniot relations across a shock with the upstream state  $V_{\infty}$  and given shockwave angle  $\theta$ .

These computations use the dual CFL number of 50 and the convergence for the dual iterations is obtained after the residual decreases by six orders. The dimensional physical time step is  $\Delta t = 6.82 \times 10^{-6}$  s, which gives a physical CFL number close to 6.

## 2.5. DNS results

### 2.5.1. General results

Many computations were carried out for various values of the incident shock angle  $\theta$  and spanwise length  $L_z$ . In agreement with the experimental results of Degrez, for  $\theta = 30.8^{\circ}$  and all  $L_z$ , the solution obtained is two-dimensional and steady (see Boin et al. 2006). When the incident shock angle is greater, for the same flow conditions (Re, M), the flow is destabilized towards a complex space—time dynamical state. When  $31.7^{\circ} < \theta < 32.8^{\circ}$ , the two-dimensional flow is conditionally stable with respect to the spanwise length  $L_z$ . Indeed, there are two critical spanwise lengths,  $L_{z_{c_1}}(\theta)$  and  $L_{z_{c_2}}(\theta)$ , where the flow bifurcates. If  $L_{z_{c_1}}(\theta) < L_z < L_{z_{c_2}}(\theta)$ , the SWBLI bifurcates towards a three-dimensional and stationary asymptotic state. If  $L_z \geqslant L_{z_{c_2}}(\theta)$ , the asymptotic state corresponds to a three-dimensional and unsteady flow. When  $\theta > 32.7^{\circ}$ , a three-dimensional and unsteady flow is directly reached. Figure 3 synthesizes the results obtained. As shown in figure 4, when the incident shock angle increases, the SWBLI becomes gradually three-dimensional. This three-dimensionality, for all the configurations studied in this paper, remains confined in the interaction zone and more precisely in the separated zone. This three-dimensionality is characterized by the appearance of a secondary recirculation within the primary recirculation and located in the downstream part of the bubble and close to the wall. This creation of secondary recirculation is immediately followed by the creation of a spanwise component of the

![](_page_8_Figure_3.jpeg)

Figure 4. Iso-lines of longitudinal velocity *U*(*x,y*) and streamlines for *Lz* = 0*.*8. (a) *θ* = 30*.*8◦, (b) *θ* = 31*.*7◦, (c) *θ* = 32*.*0◦ and (d) *θ* = 32*.*5◦.

velocity, which is initially localized in the vicinity of the core of the recirculations. This three-dimensionality of the flow remains confined in the recirculation core when the incident shock angle lies in the interval 31*.*7◦ *<θ<* 32*.*8◦. For higher values of *θ*, the separated zone quickly becomes completely three-dimensional and is not limited only to the core of the recirculation zones. In this case the asymptotic solution is always unsteady. In § 3, it will be shown that this first bifurcation (twodimensional-stationary/three-dimensional-stationary) can be comprehended by an analysis of global linear stability of the mean flow. Boin et al. (2006) gives details of the influence of the transverse length.

### 2.5.2. Asymptotic state

In this section, only the final state will be considered. The physical mechanisms generating this asymptotic state will be studied in the following section. In the preceding section, we have shown that the asymptotic state is a three-dimensional and unsteady state if *Lz Lzc*<sup>2</sup> (*θ*) for 31*.*7◦ *<θ<* 33◦ and ∀*Lz* for *θ >* 33◦. In order to characterize this bifurcation, an amplitude parameter is defined as *Amp* = max(*w*(*t*))− min(*w*(*t*)), where max(*w*(*t*)) and min(*w*(*t*)) are the maximum and the minimum of

![](_page_9_Figure_3.jpeg)

Figure 5. The amplitude of the oscillations of *w* as a function of *θ*. , numerical simulations; line, *Amp* <sup>∼</sup> <sup>√</sup> *θ* − 31*.*7 at (*x*0*,y*0*,z*0) = (1*,* 5 × 10<sup>−</sup><sup>2</sup>*,* 0*.*4), *Lz* = 0*.*8.

the spanwise velocity component, *w*(*t*), respectively. Figure 5 shows the amplitude of the oscillations of *w* in a particular point in the SWBLI for the established flow and for *Lz* = 0*.*8. The characteristics of a supercritical Hopf bifurcation are observed. In order to extract the spectral contents from the dynamics of the SWBLI, fast Fourier transforms (FFTs) taken in particular points are carried out. Figure 6 shows such FFT results. The fundamental frequency, *f*0, is close to 700 Hz and this frequency is present in all part of the flow except for the upstream zone of the incident shock wave. The harmonics, *nf*0, of *f*<sup>0</sup> are also in evidence. Inside the bubble (figure 6d), a sub-harmonic frequency (350 Hz) and a broadband frequency around 3000 Hz are also observed. This last characteristic is purely local and observable only inside the bubble. The frequency *f*<sup>0</sup> is related to a breathing of the bubble which is connected to three-dimensional movement with important spanwise components.

### 2.5.3. Transient states

When the incident shock angle and the spanwise length are *θ* = 32◦ and *Lz* = 0*.*8, respectively, the flow is three-dimensional and unsteady. However, the onset of the unsteadiness is not a direct scenario. Figure 7 shows the time evolution of the physical residual based on the conservative *ρw* variable (the CPU time figures in the second *x*-axis). The unsteady state is reached after several characteristic stages which can be connected with different instabilities. Three-dimensional views are presented in Figure 8 for different moments (from A to F). During the first stage (from A to B) from *t* = 0 to *t* = 16 ms, the flow remains two-dimensional and stationary. State A corresponds to the initial state of the computation which is not a solution of the threedimensional equations. State B is a solution to the three-dimensional equations and it corresponds to two-dimensional flow which is very close to the solution obtained by two-dimensional Navier–Stokes equations. After this state, a first bifurcation appears in the residual evolution. A three-dimensional instability is observed with transverse wavelength close to 0*.*8 (from B to C). The three-dimensional character occurs also

![](_page_10_Figure_3.jpeg)

FIGURE 6. Pressure spectra density for different points in the flow. (a)  $(x_0, y_0, z_0) = (0.8053, 0.6, 0.4)$ , (b) (0.8053, 0.28, 0.4), (c) (0.8053, 0.18, 0.4) and (d) (0.8053, 0.042, 0.4) for  $\theta = 32.5^{\circ}$  and  $L_z = 0.8$ .

![](_page_10_Figure_5.jpeg)

Figure 7. Residual time evolution and CPU time on a 2.4 GHz Pentium processor,  $L_z=0.8,\,\theta=32^\circ.$ 

inside the recirculation zone with the existence of two counter-rotating vortex tubes in which some fluid from the wall is transported to the downstream shear layer. From  $t = 35 \,\text{ms}$ , a second instability appears, but with a transverse wavelength close to 0.4

![](_page_11_Figure_3.jpeg)

FIGURE 8. Three-dimensional views inside the bubble at different moments: (a) near B stage, (b) in (B-C) stages, (c) and (d) in (C-D) stage.  $L_z = 0.8$ ,  $\theta = 32^{\circ}$ .

(from C to D). At this time, two instabilities are simultaneously present with different wavelengths, 0.4 and 0.8. From t = 44 ms, the instability with the shortest wavelength disappears (from D to E). The residual decreases then by two orders up to t = 58 ms when a Hopf bifurcation (characterized in the previous section) leads the flow toward an unsteady state (F).

Thereafter, only stage (B–C) is studied. Figure 9 presents, at a given point  $(x_0, y_0, z_0) = (1.1, 2 \times 10^{-2}, 0.4)$  the time flow evolution of the spanwise velocity component in log scale. This run has been initialized by a two-dimensional solution. After a short transient state (before point B), the amplitude of the spanwise velocity component increases exponentially (linear evolution). When this amplitude becomes finite, a nonlinear saturation takes place (point C). From points C to E, the spanwise velocity amplitude remains constant. When t > 60 ms, the Hopf bifurcation, previously described, appears. This characteristic is observed in all flow cases. The linear amplification rate observed in stage (B–C) is similar at any point of the flow. These results suggest the existence of a global instability mechanism. In the next section, this characteristic will be demonstrated.

# 3. Compressible biglobal linear stability theory

## 3.1. Theoretical basics

The analysis of flow stability is based on the compressible equations of motion, (2.1). Central to work on linear flow instability is the concept of decomposition of any

![](_page_12_Figure_3.jpeg)

Figure 9. Time evolution of |w| in logarithmic scale,  $L_z = 0.8$ ,  $\theta = 32^\circ$ . Continuous line, numerical simulation; dashed line, linear amplification rate.

flow quantity into an O(1) steady or time-periodic laminar basic flow upon which small-amplitude three-dimensional disturbances are permitted to develop. The most general framework in which a linear instability analysis can be performed is one in which three inhomogeneous spatial directions are resolved and time-periodic small-amplitude disturbances, inhomogeneous in all three directions, are superimposed upon the underlying steady or time-periodic O(1) basic state. The related three-dimensional global triglobal instability ansatz yields a three-dimensional eigenvalue problem in which all three spatial directions must be resolved simultaneously in a coupled manner. Though this most general ansatz is consistent with the separability in the governing equations of time on the one hand and the three spatial directions on the other, the size of the resulting eigenvalue problem is such that currently available computing hardware and algorithms permit its solution only in a very limited range of Reynolds numbers, of  $Re \sim O(10^2)$ .

In order to proceed, the basic state is assumed to be independent of one spatial coordinate, z, an assumption in line with the two-dimensional cavity geometry. Flow quantities are then decomposed according to

$$q(x, y, z, t) = \overline{Q}(x, y) + \varepsilon \widetilde{q}(x, y, z, t)$$
 (3.1)

with  $\overline{Q} = (\overline{U}, \overline{V}, \overline{W}, \overline{P}, \overline{T})^T$  and  $\widetilde{q} = (\widetilde{u}, \widetilde{v}, \widetilde{w}, \widetilde{p}, \widetilde{T})^T$  representing the steady two-dimensional basic flow and the unsteady three-dimensional infinitesimal perturbations, respectively, the latter being inhomogeneous in x and y and periodic in z. Note also that, unlike the incompressible case, pressure is a predictive variable in, rather than a constraint of, the equations of motion. On substituting (3.1) into the governing equations (2.1), taking  $\varepsilon \ll 1$  and linearizing about  $\overline{Q}$ , we may write

$$\tilde{q}(x, y, z, t) = \hat{q}(x, y) e^{i\Theta_{2D}} + \text{c.c.}$$
 (3.2)

with  $\hat{q} = (\hat{u}, \hat{v}, \hat{w}, \hat{p}, \hat{T})^T$  representing the vector of two-dimensional complex amplitude functions of the infinitesimal three-dimensional perturbations, a complex

eigenvalue and

$$\Theta_{2D} = \beta z - \omega t, \tag{3.3}$$

a complex phase function. 'c.c.' represents the complex conjugate. The linear disturbance equations of biglobal stability analysis are obtained at  $O(\varepsilon)$  by substituting the decomposition (3.1)–(3.3) into the equations of motion, subtracting out the O(1) base-flow terms and neglecting terms at  $O(\varepsilon^2)$ . In the present temporal framework,  $\beta$  is taken to be a real wavenumber parameter describing an eigenmode in the z-direction, while the complex eigenvalue  $\omega$ , and the associated eigenvectors  $\hat{q}$  are sought. The real part of the eigenvalue,  $\text{Re}(\omega)$ , is related to the frequency of the global eigenmode while the imaginary part is its growth/damping rate; a positive value of  $\text{Im}(\omega)$  indicates exponential growth of the instability mode whereas  $\text{Im}(\omega) < 0$  denotes decay of  $\tilde{q}$  in time. The system for the determination of the eigenvalue and the associated eigenfunctions  $\hat{q}$  in its most general form can be written as the complex non-symmetric generalized eigenvalue problem

$$\mathscr{L}\hat{\mathbf{q}} = \omega \mathscr{R}\hat{\mathbf{q}} \tag{3.4a}$$

+ boundary conditions on 
$$\partial \mathcal{D}$$
, (3.4b)

or, more explicitly,

$$\begin{pmatrix} \mathcal{I} \mathcal{L}_{\hat{u}}^{(c)} & \mathcal{I} \mathcal{L}_{\hat{v}}^{(c)} & \mathcal{I} \mathcal{L}_{\hat{w}}^{(c)} & \mathcal{L}_{\hat{p}}^{(Gc)} & \mathcal{I} \mathcal{L}_{\hat{T}}^{(c)} \\ \mathcal{L}_{\hat{u}}^{(x)} & \mathcal{L}_{\hat{v}}^{(x)} & \mathcal{L}_{\hat{w}}^{(x)} & \mathcal{I} \mathcal{L}_{\hat{p}}^{(x)} & \mathcal{L}_{\hat{T}}^{(x)} \\ \mathcal{L}_{\hat{u}}^{(y)} & \mathcal{L}_{\hat{v}}^{(y)} & \mathcal{L}_{\hat{w}}^{(y)} & \mathcal{I} \mathcal{L}_{\hat{p}}^{(y)} & \mathcal{L}_{\hat{T}}^{(y)} \\ \mathcal{L}_{\hat{u}}^{(z)} & \mathcal{L}_{\hat{v}}^{(z)} & \mathcal{L}_{\hat{w}}^{(z)} & \mathcal{I} \mathcal{L}_{\hat{p}}^{(z)} & \mathcal{L}_{\hat{T}}^{(z)} \\ \mathcal{L}_{\hat{u}}^{(e)} & \mathcal{L}_{\hat{v}}^{(e)} & \mathcal{L}_{\hat{w}}^{(e)} & \mathcal{I} \mathcal{L}_{\hat{p}}^{(e)} & \mathcal{L}_{\hat{T}}^{(e)} \end{pmatrix} \begin{pmatrix} \hat{u} \\ \hat{v} \\ \hat{w} \\ \hat{p} \\ \hat{T} \end{pmatrix}$$
 
$$= \omega \begin{pmatrix} 0 & 0 & 0 & \mathcal{R}_{\hat{p}}^{(Gc)} & \mathcal{I} \mathcal{R}_{\hat{T}}^{(c)} \\ \mathcal{R}_{\hat{u}}^{(x)} & 0 & 0 & 0 & 0 \\ 0 & \mathcal{R}_{\hat{v}}^{(y)} & 0 & 0 & 0 \\ 0 & 0 & \mathcal{R}_{\hat{w}}^{(z)} & 0 & 0 \\ 0 & 0 & 0 & \mathcal{I} \mathcal{R}_{\hat{p}}^{(e)} & 0 \end{pmatrix} \begin{pmatrix} \hat{u} \\ \hat{v} \\ \hat{w} \\ \hat{p} \\ \hat{T} \end{pmatrix} , \hfill (3.5)$$

where the operator  $\mathcal{L}$  of the linear two-dimensional eigenvalue problem (3.4a) has the form

$$\mathscr{L} = \mathbf{M}_1 \frac{\partial^2}{\partial x^2} + \mathbf{M}_2 \frac{\partial^2}{\partial y^2} + \mathbf{M}_3 \frac{\partial^2}{\partial x \partial y} + \mathbf{M}_4 \frac{\partial}{\partial x} + \mathbf{M}_5 \frac{\partial}{\partial y} + \mathbf{M}_6.$$
 (3.6)

Details of the operators are presented in the Appendix. Here the linearized equation of state

$$\hat{p}/\overline{P} = \hat{\rho}/\overline{\rho} + \hat{T}/\overline{T} \tag{3.7}$$

has been used, viscosity and thermal conductivity of the medium have been taken as functions of temperature alone, the perturbations of these quantities are written as:

$$\hat{\mu} = \frac{\mathrm{d}\overline{\mu}}{\mathrm{d}\overline{T}}\hat{T}, \ \hat{\kappa} = \frac{\mathrm{d}\overline{\kappa}}{\mathrm{d}\overline{T}}\hat{T}.$$

In addition,  $\mathcal{J}$  and  $\mathcal{J}$  are interpolation arrays transferring data from the Gauss to the Gauss-Lobatto and from the Gauss-Lobatto to the Gauss spectral collocation

grids, respectively. To close the partial differential equation system (3.4a), boundary conditions must be imposed on  $\partial \mathcal{D}$ . At the solid wall, viscous boundary conditions are imposed on all disturbance velocity components  $\hat{u} = \hat{v} = \hat{w} = 0$  and the temperature perturbation is set to zero,  $\hat{T} = 0$ . There are no physical boundary condition for the pressure disturbance. Two computational strategies are possible. Either all unknowns are calculated at the same grid point (collocated grid) and, for the pressure, the compatibility conditions must be imposed

$$\frac{\partial \hat{p}}{\partial x} = \frac{1}{Re} \Delta_{2d} \hat{u} - \overline{U} \frac{\partial \hat{u}}{\partial x} - \overline{V} \frac{\partial \hat{u}}{\partial y}, \tag{3.8a}$$

$$\frac{\partial \hat{p}}{\partial y} = \frac{1}{Re} \Delta_{2d} \hat{v} - \overline{U} \frac{\partial \hat{v}}{\partial x} - \overline{V} \frac{\partial \hat{v}}{\partial y}, \tag{3.8b}$$

derived from the Navier-Stokes equations at the boundary of the domain. The other strategy consists in using a staggered grid, in this case, the momentum and energy equations are computed on the Chebychev Gauss–Lobatto points, while the equation of continuity is computed on the Chebychev Gauss points. In this case, any boundary condition is necessary for the pressure. Although, the two approaches give very similar results, the second one converges better when the number of points increases. In the following, only the second strategy is used. In the free stream, in the normal direction, exponential decay of all disturbance quantities is expected, similar boundary conditions to those imposed on the wall are imposed at a large distance  $(y = y_{max})$  from the wall. At inflow, homogeneous Dirichlet boundary conditions on all disturbances are used; this choice corresponds to studying disturbances generated within the examined basic flow field. At the outflow boundary, quadratic extrapolation of all disturbance quantities from the interior of the integration domain is performed. This computational strategy has been successfully used for the first time in a compressible global eigenproblem by Theofilis & Colonius (2004) for a compressible open cavity problem.

## 3.2. Basic flows

The availability of a two-dimensional basic state  $\overline{Q}$  will be known analytically only in exceptional model flows; in the large majority of cases of industrial interest, it must be determined by numerical or experimental means. An accurate basic state is a prerequisite for reliability of the instability results obtained; if numerical residuals exist in the basic state (at O(1)) they will act as forcing terms in the  $O(\varepsilon)$  disturbance equations and will result in erroneous instability predictions. In laminar flows, current hardware capabilities permit the determination of a basic state using twodimensional DNS at arbitrary high resolution. Thereafter, the basic flow is obtained by the resolution of the two-dimensional equations of motion. In the temporal evolution of the residual, figure 7, the state B corresponds precisely to the twodimensional solution. This characteristic is independent of the initialization of the three-dimensional simulation as long as this one is initialized by a two-dimensional field. This characteristic is observed for  $\theta < 33^{\circ}$ . Beyond that, the residual no longer exhibits this behaviour and reaches the asymptotic state after a transient that is different from the cases where  $\theta < 33^{\circ}$ . Figure 10 shows the topology of the interaction zone for  $\theta = 32^{\circ}$ . The separated zone extends on  $L_s \approx 1$  and is only one piece (not a secondary zone). After computing the basic flow solutions on the computational domain using a high-resolution grid inaccessible to the instability analysis, a cubic spline interpolation scheme is used to transpose the basic flow solution onto the stability grid. The basic flow solutions are converged in time to within a tolerance

![](_page_15_Figure_3.jpeg)

Figure 10. Iso-lines of longitudinal velocity *U*(*x,y*) and streamlines. *θ* = 32◦, Re = 105.

tol ≡ |(*gt*0+*<sup>t</sup>* −*gt*<sup>0</sup> )*/gt*<sup>0</sup> |*<* 10<sup>−</sup>10, where *g* is an integral measure of the flow or the value of a local flow quantity.

## 3.3. Numerical approach

The stability equations (3.4) are solved on a domain identical to that used for the basic flow computation: D*<sup>s</sup>* = [0*.*2; 2*.*3] × [0; 0*.*94]. The choice of numerical method for the biglobal eigenvalue problem is crucial for the success of the computation. In the present biglobal analysis methodology, the amplitude functions of the small-amplitude disturbances develop along two inhomogeneous spatial directions which must be solved simultaneously. Consequently, the resolution requirements for an adequate description of biglobal instabilities can be challenging; a thorough discussion of this point, mainly focusing on incompressible flows, has been presented by Theofilis (2003). Solving compressible BiGlobal analysis accurately requires more memory, not only because of the need to solve the energy equation in addition to those of incompressible flow, but also because increasing Mach numbers result in tighter eigenmode structures to be resolved, compared with their incompressible counterparts. Consequently, numerical methods of high-resolution capacity are essential in this problem. In the present analysis, spectral collocation has been used, based on two sets of Chebychev points for each direction *x* and *y*; the Chebychev Gauss–Lobatto (CGL) points

$$\xi_j = \cos\left(\frac{j\pi}{N}\right) \quad (j = 0, \dots, N),\tag{3.9}$$

the extrema of the *N*th-order Chebychev polynomials *TN* (*ξ* ) = cos(*N*cos<sup>−</sup><sup>1</sup>*ξ* ); and the Chebyshev–Gauss (CG) points

$$\xi_j = \cos\left(\frac{(2j+1)\pi}{2N}\right) \quad (j=0,\dots,N-1),$$
 (3.10)

the roots of the *TN* (*ξ* ), (*N* is *nx* or *ny* ). A similar definition is used for the longitudinal direction *x* where *ξ* is replaced by *ζ* . These two sets of points are introduced in view of the different types of boundary condition pertaining to the stability problem as was mentioned in § 3. Each sub-matrix *M<sup>j</sup>* of the linear two-dimensional eigenvalue problem (3.6) is defined either on the CGL or the CG points. First derivatives in *x* and *y* on the CGL points are calculated using the collocation derivative matrix (Canuto et al. 1987; Boyd 1999)

$$\left. \begin{aligned} \left(D_{\text{GL}}^{(1)}\right)_{ik} &= \frac{c_i}{c_k} \frac{(-1)^{k+i}}{\xi_i - \xi_k}; \quad i \neq k, \\ \left(D_{\text{GL}}^{(1)}\right)_{ii} &= -\frac{\xi_i}{2(1 - \xi_i^2)}, \\ \left(D_{\text{GL}}^{(1)}\right)_{00} &= \frac{2N^2 + 1}{6} = -\left(D_{\text{GL}}^{(1)}\right)_{NN}. \end{aligned} \right\} \tag{3.11}$$

with  $\xi_j$  defined by (3.9) and  $c_0 = c_N = 2$ ,  $c_k = 1$ , for  $k \in [1, ..., N-1]$ . First derivatives in x and y on the CG points are calculated using

$$\left. \begin{aligned} (D_{G}^{(1)})_{ik} &= \frac{(-1)^{k+i}}{\xi_{i} - \xi_{k}} \sqrt{\frac{1 - \xi_{k}^{2}}{1 - \xi_{i}^{2}}}, & i \neq k, \\ (D_{G}^{(1)})_{kk} &= \frac{\xi_{k}}{2(1 - \xi_{k}^{2})}, & i = k. \end{aligned} \right\} \tag{3.12}$$

with  $\xi_j$  defined by (3.10). Higher derivatives on either the CGL or the CG points may be calculated using

$$D_{ik}^{(m)} = \left(D_{ik}^{(1)}\right)^m. \tag{3.13}$$

Contrary to an incompressible formulation, the matrix  $M_3$  is not equal to zero, the global stability operator (3.6) has cross-derivatives, which are calculated by

$$\frac{\partial^2 \hat{\boldsymbol{q}}}{\partial \xi \partial \zeta}(\zeta_i, \xi_j) = \sum_k D_{ik}^{(x)} \left[ \sum_l D_{jl}^{(y)} \hat{\boldsymbol{q}}(\zeta_k, \xi_l) \right], \tag{3.14}$$

where  $D^{(x)}$  (resp.  $D^{(y)}$ ) is the first derivative in streamwise (resp. normal) direction. Data may be transferred between the grids using the interpolation arrays  $\mathscr I$  and  $\mathscr I$  introduced in (3.4) and previously used by Theofilis & Colonius (2004) for cavity flow,

$$\mathscr{I} = \mathscr{C}_G^{-1}\mathscr{C}_{GL},\tag{3.15a}$$

$$\mathcal{J} = \mathcal{C}_{GL}^{-1} \mathcal{C}_G, \tag{3.15b}$$

where, for (i = 0, ..., N),

$$(\mathscr{C}_{GL})_{ik} = \frac{2}{c_i c_k N} \cos\left(\frac{ik\pi}{N}\right), \quad k = 0, \dots, N,$$
(3.16a)

$$(\mathscr{C}_{GL})_{ki}^{-1} = \cos\left(\frac{ik\pi}{N}\right), \quad k = 0, \dots, N,$$
(3.16b)

$$(\mathscr{C}_G)_{ik} = \frac{2}{Nc_i} \cos\left[\frac{i(k+\frac{1}{2})\pi}{N}\right], \quad k = 0, \dots, N-1,$$
 (3.16c)

$$(\mathscr{C}_G)_{ki}^{-1} = \cos\left[\frac{i(k+\frac{1}{2})\pi}{N}\right], \quad k = 0, \dots, N-1.$$
 (3.16d)

Because of the complexity of the basic flow (shock wave, separated boundary layer, etc.), a single-domain algorithm cannot be used to describe accurately the entire domain of this flow. In order to extend the biglobal stability analysis methodology to complex geometries with a certain degree of regularity, the spectral multidomain algorithm of Streett & Macaraeg (1989) is an obvious candidate.

![](_page_17_Figure_3.jpeg)

FIGURE 11. A typical grid, generated by (3.18) and (3.17), for BiGlobal instability analysis of SWBLI at  $\theta = 32^{\circ}$ ,  $n_x = 10 + 50 + 20$ ,  $n_y = 60$ .

In the normal y-direction, a mapping transformation for semi-infinite domains of boundary-layer type is used

$$y = \frac{a_0(1-\xi)}{a_1+\xi}$$
 with  $a_0 = \frac{y_a y_m}{y_m - 2y_a}$ ,  $a_1 = 1 + 2\frac{a_0}{y_m}$ , (3.17)

 $y_m$  is the upper boundary domain and  $y_a$  (here  $\sim 0.05$ ) is the coordinate where between  $[0; y_a]$  there is 50% of total points. In the streamwise x-direction, a spectral multidomain is used

$$x \in [x_0; x_d], \ x = x_0 + \frac{(x_a - x_0)(x_d - x_0)(1 + \zeta)}{(x_d + x_0 - 2x_a)(1 + 2(x_a - x_0)/(x_d + x_0 - 2x_a) - \zeta)}, \ x_a \approx 0.4,$$

$$(3.18a)$$

$$x \in [x_d; x_r], \ x = x_i + x_d \frac{\tan(\frac{1}{2}c\pi\zeta)}{\tan(\frac{1}{2}c\pi)}, \ c = 0.9,$$
 (3.18b)

$$x \in [x_r; x_n], \ x = x_r + \frac{(x_b - x_r)(x_n - x_r)(1 + \zeta)}{(x_n + x_r - 2x_b)(1 + 2(x_b - x_r)/(x_n + x_r - 2x_b) - \zeta)}, \ x_b \approx 1.8.$$
(3.18c)

where  $x_d$ ,  $x_i$  and  $x_r$  correspond, respectively, to the separation, interaction and reattachment points of the basic flow. The grid used is shown in figure 11. At the interface of the domains, continuity and derivability of the disturbances are imposed.

## 3.4. Eigenvalue problem

Using the tools presented, the compressible biglobal linear eigenvalue problem (3.4) is transformed into a discrete matrix eigenvalue problem.

$$[\mathcal{M}(Re,\beta) - \omega \mathcal{N}(Re,\beta)] \hat{\mathbf{Z}} = 0, \tag{3.19}$$

where  $\hat{\mathbf{Z}} = \{\hat{q}_{ij}\}$ . A standard eigenvalue subroutine may now be used to compute the eigenvalues. Two methods were used to solve this algebraic system (3.19): a local method based on a shooting method with a classical Newton-Raphson algorithm and a global method, where the discretized operator spectrum is computed by the QZ algorithm in the absence of prior information on interesting regions of the parameter space. When the interesting zone of the spectrum is identified, a less expensive

![](_page_18_Figure_3.jpeg)

FIGURE 12. Discretized linear stability spectrum: (a)  $\theta = 31^{\circ}$ ,  $\beta = 8.02$ ; (b)  $\theta = 32^{\circ}$ ,  $\beta = 7.86$ .

algorithm, the Arnoldi algorithm, is used to compute only a part of the spectrum as well as the associated eigenfunctions.

## 3.5. Biglobal results

The approach described in the preceding sections is employed to compute linear global stability for various values of the incident shock wave angle from  $\theta = 31^{\circ}$  to  $\theta = 33^{\circ}$ . Certain characteristics observed in the DNS are found. The critical shock angle beyond which the flow becomes unstable is very close:  $\theta_c = 31.8^{\circ}$  for the stability analysis and  $\theta_c = 31.7^{\circ}$  for the DNS. The eigenvalue spectra in the neighbourhood of  $\omega = 0$  and for  $\theta = 31^{\circ}$  and  $32^{\circ}$  are shown in figures 12(a) and 12(b), respectively. The calculation of these spectra is carried out with  $(n_x \times n_y) = (80 \times 60)$  points. Stationary (Re( $\omega$ ) = 0) as well as travelling (Re( $\omega$ )  $\neq$  0) modes are to be found in this window of the spectrum, calculated by the Arnoldi algorithm. The travelling modes appear in symmetric pairs, indicating that there is no preferential direction in z. At this set of parameters (Re, M and  $\theta$ ), the most unstable mode is a three-dimensional stationary perturbation. The most unstable wavelength for example for  $\theta = 32^{\circ}$  is equal to 0.7987, which is very close to that observed in the DNS which is around 0.8. The evolution of the amplification rate,  $\text{Im}(\omega)$ , according to the wavelength  $\lambda$  for various  $\theta$  values is shown on figure 13 for the most unstable mode. With this most unstable wavelength, the amplification rate is equal to  $Im(\omega)_{bg} = 3.85 \times 10^{-4}$  which we must compare with the amplification rate resulting from the DNS:  $Im(\omega)_{dns} = 3.719 \times 10^{-4}$  (see figure 9). The agreement is excellent. When the shock angle increases, a broader wavelength range becomes unstable, mainly towards the high wavelengths. The very short wavelengths seem always very strongly attenuated for all the studied values of  $\theta$ . These characteristics are compatible with those observed in the DNS where the small wavelengths are always stable. With regard to the most unstable wavelength,  $\lambda_m = 2\pi/\beta_m$  such as  $\operatorname{Im}[\omega(\beta_m)] = \operatorname{max}_{\beta} \operatorname{Im}[\omega(\beta)],$  it is slightly dependent on  $\theta$  (increases with  $\theta$ ). If we compare these results with those obtained in figure 3, some differences can nevertheless be observed. The stability results indicate that beyond a critical wavelength  $\lambda_c^{(l)}$ , for example  $\lambda_c^{(l)}(\theta = 32^\circ) \approx 1.2$ , the flow is again stable. That does not seem to be the case for the DNS. For the latter, when  $\theta = 32^{\circ}$ , the basic flow becomes unstable when  $\lambda \geqslant \lambda_c^{(1)}$  which results in a three-dimensional and stationary asymptotic state. However,

![](_page_19_Figure_3.jpeg)

FIGURE 13. Temporal growth rate  $\text{Im}(\omega)$  versus spanwise wavelength  $\lambda = 2\pi/\beta$  for various incident shock angle  $\theta$ . —,  $\theta = 31^\circ$ ; ---,  $32^\circ$ ; —,  $33^\circ$ .

| $\theta = 31^{\circ}$ |                     |           | $\theta = 31.8^{\circ}$ |                     |           | $\theta = 32^{\circ}$ |                     |           |
|-----------------------|---------------------|-----------|-------------------------|---------------------|-----------|-----------------------|---------------------|-----------|
|                       |                     |           |                         |                     |           |                       |                     |           |
| $(n_x \times n_y)$    | $\text{Im}(\omega)$ | $\lambda$ | $(n_x \times n_y)$      | $\text{Im}(\omega)$ | $\lambda$ | $(n_x \times n_y)$    | $\text{Im}(\omega)$ | $\lambda$ |
| $60 \times 50$        | $-2.869(-3)$        | 0.7793    | $60 \times 50$          | $+7.374(-6)$        | 0.7918    | $60 \times 50$        | $+3.61(-4)$         | 0.7954    |
| $70 \times 50$        | $-2.938(-3)$        | 0.7817    | $70 \times 50$          | $+7.409(-6)$        | 0.7932    | $70 \times 50$        | $+3.77(-4)$         | 0.7969    |
| $80 \times 50$        | $-2.942(-3)$        | 0.7832    | $80 \times 50$          | $+7.413(-6)$        | 0.7951    | $80 \times 50$        | $+3.81(-4)$         | 0.7981    |
| $80 \times 60$        | $-2.946(-3)$        | 0.7834    | $80 \times 60$          | $+7.416(-6)$        | 0.7956    | $80 \times 60$        | $+3.85(-4)$         | 0.7987    |

Table 3. Convergence history of the most unstable eigenmode for various shock angles  $\theta$ .  $x(y) \equiv x \times 10^{y}$ .

when  $\lambda \geqslant \lambda_c^{(2)} \simeq 0.5$ , this three-dimensional steady flow itself becomes unstable, finally reaching a three-dimensional and unsteady state. This difference is not understood at present. The summary of the results obtained by the biglobal approach is given in table 3. The real part of the eigenvector  $\hat{q}(x, y) \exp[i(\beta z - \omega t)]$ , at z = 0.4, for the most unstable mode is presented in figures 14 and 15. Most of the activity in all disturbance eigenfunctions is confined within the boundary layer and to some degree in the vicinity of the reflected shock. The upstream zone of SWTBLI is inactive in agreement with DNS results. The neighbourhood of the basic laminar flow separation point is weakly affected, as is clearly demonstrated by the level of activity of all disturbance velocity components and pressure in that region. The peak of both the streamwise and the wall-normal linear disturbance velocity components is to be found in the interaction zone,  $x/X_{sh} \approx 1$ , with higher-level linear activity continuing past the point of primary reattachment for the streamwise component.  $\hat{w}$  (which is the only source of threedimensionality in this linear framework) is mostly distributed within the primary separation bubble and has a tendency to split the latter into two regions of fluid moving in opposite directions. It should be noted that there is strong activity in the vicinity of the reflected shock wave for the normal component of fluctuating velocity. Pressure also has interesting characteristics. There are two zones where the amplitude

![](_page_20_Figure_3.jpeg)

FIGURE 14. Normalized disturbance velocity components and pressure distribution of the unstable stationary global mode in figure 12(b). Note the different y-scales in the different eigenvector components (for the pressure).

![](_page_20_Figure_5.jpeg)

FIGURE 15. Normalized temperature disturbance of the unstable stationary global mode.

of the pressure fluctuation is high, in the vicinity of the reflected shock and the interaction point between the incident shock and the boundary layer. However, these two peaks are of opposite sign and represent a phase difference between the reflected shock and the separated zone. The pressure fluctuation remains high downstream from the interaction and well after the laminar basic flow has reattached. The main activity in temperature disturbance remains confined within the interaction zone and in the downstream boundary layer. Within the bubble, two peaks of opposite sign are present. The eigenfunctions were normalized by the fluctuating streamwise velocity component resulting from the numerical simulation at the point  $x_d = (1, 0.04, 0.4)$ .

![](_page_21_Figure_3.jpeg)

FIGURE 16. Evolution of the fluctuating quantities resulting from the numerical simulation in the (x, y)-plan for t = 0.025 at z = 0.4.

In order to validate the relevance of these eigenfunctions, the fluctuating quantities have been extracted from the direct numerical simulation. In order to obtain the fluctuations of the various physical quantities from the instantaneous solution, this instantaneous solution is selected in order to stay in the linear regime, i.e. when |w(x, y, z, t)| is small compared to other characteristic velocities of the flow. In our case, that corresponds to t = 0.025 at z = 0.4. The fluctuating quantities are obtained by the subtraction of the basic flow from this instantaneous solution. Figures 16 and 17 show the result obtained. The comparison with the disturbances resulting from the stability approach (figures 14 and 15) shows some strong similarities, but also some significant differences. In the two approaches, the streamwise and spanwise velocity fluctuations are very similar qualitatively and quantitatively. The principal difference in  $Re(\tilde{u})$  is in the vicinity of the downstream recompression shock which has a fluctuating activity in numerical simulation whereas no activity of the same order is perceptible in the biglobal approach. As for the spanwise fluctuating velocity, the disturbance decreases more slowly when  $y \to \infty$  in the numerical simulation than in the biglobal approach. This characteristic is generally observed, to some degree, on the whole of the fluctuating quantities. The normal component of the velocity fluctuation resulting from the numerical simulation has almost twice the amplitude of that resulting from the biglobal analysis. The most important difference between the two approaches is localized in the interaction zone where the signs of fluctuating velocity are rather dissimilar. This variation is not understood at present and is observed from the instability onset. This characteristic is also observed for the temperature disturbance. Between the two approaches, the pressure disturbance is very similar. However, the amplitude of the fluctuating pressure in numerical simulation is almost three times stronger near the reflected shock than in the stability analysis.

![](_page_22_Figure_3.jpeg)

Figure 17. Temperature disturbance *Tf* .

Elsewhere, the amplitude remains comparable. This difference can undoubtedly be explained by the lack of points in the stability grid around the shock waves.

# *4. Discussion and prospects*

The main objective of this paper was to highlight that an interaction between an oblique shock wave impacting on a laminar boundary layer developing on a flat plate could be the generating seat of a global instability of low-frequency self-sustained oscillations. Therefore, three-dimensional direct numerical simulations were carried out for a configuration close to that of Degrez et al. (1987) where the incident shock angle is gradually increased.

These numerical simulations highlighted a complex process in the onset of unsteady dynamics when the angle of the incident shock increases. These numerical computations have shown that before becoming unsteady, the SWBLI goes through a phase where the flow becomes three-dimensional and stationary (for *θ >* 31*.*7◦). However, this state is unstable and can lead to a fully three-dimensional and unsteady flow. The final state is reached more quickly when the angle of the incident shock is large.

When the spanwise dimension *Lz* is large enough, the main spanwise wavelength of the disturbance is close to *λ<sup>z</sup>* = 0*.*8. In the interaction, the topology of the separated zone is complex and mainly characterized by cells in the spanwise direction where the flow is alternatively separated and reattached. Within this separated zone, a vortex waterspout which connects the flow from the wall to the downstream shear layer of the interaction is observed (figure 8c). This topologically complex zone exhibits an unsteady self-sustained low-frequency dynamics close to 700 Hz.

Linearized global stability analysis was carried out in order to find the physical origin of the bifurcation generating the three-dimensional character of the flow. This analysis highlighted that beyond a critical angle of the incident shock wave, the flow becomes linearly globally unstable, a stationary three-dimensional mode with characteristics very close to those highlighted in the direct numerical simulation has been observed. The wavelength, the temporal amplification rate and the main space characteristics of the disturbance are found.

However, in a boundary layer there exist starting from a critical Reynolds number, which depends on the local characteristics of the boundary layer, unstable waves which destabilize the flow and involve its transition towards turbulence. These local instabilities are more intense when the boundary layer is separated and/or when a shock wave interacts with this boundary layer.

The existence of upstream disturbances related to mechanisms of instability involved in the transition process have a great influence on the unsteady dynamics of the interaction. It would be interesting to verify whether the results presented in this paper remain unchanged if convective unstable disturbances develop from upstream of the interaction. The main difficulty is to extract both Kelvin–Helmholtz instabilities and global dynamics which are characterized by very different temporal and spatial length scales ( $t_{KH} \sim 5 \times 10^{-5}$  s and  $\lambda_{KH} \sim \delta$  ( $\delta$  is the boundary-layer scale),  $t_G \sim 10^{-3}$  s and  $\lambda_G \sim L_i$ , where  $L_i = x_r - x_d$  is the interaction length). In this case, it would be interesting to study the possible interactions between these two types of local and global instability.

The case of an interaction between a shock wave and a turbulent boundary layer (SWTBLI) represents a configuration closer to the industrial cases. In some configurations SWTBLI can be the seat of low-frequency unsteadiness. A study of global stability similar to that carried out in the laminar case would permit us to understand better the various mechanisms present in a turbulent configuration. Moreover, many experiments on SWTBLI on a flat plate carried out by the *supersonic group* at IUSTI/CNRS (Dupont *et al.* 2003, 2005; Haddad *et al.* 2004; Dussauge, Dupont & Debiève 2006), would enable us to validate such an approach in a more realistic configuration.

Computing time was provided by Institut du Développement et des Ressources en Informatique Scientifique (IDRIS)-CNRS.

# Appendix. Linearized compressible Navier-Stokes matrix coefficients

Disturbance x-momentum

Disturbance x-momentum

$$\mathscr{L}_{\hat{u}}^{(x)} = -\frac{4\overline{\mu}}{3Re} \frac{\partial^2}{\partial x^2} + \left( -\frac{4}{3Re} \frac{d\overline{\mu}}{d\overline{T}} \frac{\partial \overline{T}}{\partial x} + \overline{\rho} \overline{U} \right) \frac{\partial}{\partial x} - \frac{\overline{\mu}}{Re} \frac{\partial^2}{\partial y^2} + \left( -\frac{1}{Re} \frac{d\overline{\mu}}{d\overline{T}} \frac{\partial \overline{T}}{\partial y} + \overline{\rho} \overline{V} \right) \frac{\partial}{\partial y}$$

$$+ \beta^2 \frac{\overline{\mu}}{Re} + 2\overline{\rho} \frac{\partial \overline{U}}{\partial x} + \overline{\rho} \frac{\partial \overline{V}}{\partial y} + i\beta\overline{\rho} \overline{W} + \overline{U} \frac{\partial \overline{\rho}}{\partial x} + \overline{V} \frac{\partial \overline{\rho}}{\partial y},$$

$$\mathscr{L}_{\hat{v}}^{(x)} = -\frac{\overline{\mu}}{3Re} \frac{\partial^2}{\partial x \partial y} - \frac{1}{Re} \frac{d\overline{\mu}}{d\overline{T}} \frac{\partial \overline{T}}{\partial y} \frac{\partial}{\partial x} + \frac{2}{3Re} \frac{d\overline{\mu}}{d\overline{T}} \frac{\partial \overline{T}}{\partial x} \frac{\partial}{\partial y} + \overline{\rho} \frac{\partial \overline{U}}{\partial y},$$

$$\mathscr{L}_{\hat{w}}^{(x)} = -\frac{i\beta\overline{\mu}}{3Re} \frac{\partial}{\partial x} + \frac{2i\beta}{3Re} \frac{d\overline{\mu}}{d\overline{T}} \frac{\partial \overline{T}}{\partial x},$$

$$\mathscr{L}_{\hat{p}}^{(x)} = \frac{1}{\gamma M^2} \frac{\partial}{\partial x} + \frac{1}{\overline{T}} \left( \overline{U} \frac{\partial \overline{U}}{\partial x} + \overline{V} \frac{\partial \overline{U}}{\partial y} \right),$$

$$\mathscr{L}_{\hat{T}}^{(x)} = -\frac{1}{Re} \frac{d\overline{\mu}}{d\overline{T}} \left( \frac{4}{3} \frac{\partial \overline{U}}{\partial x} - \frac{2}{3} \frac{\partial \overline{V}}{\partial y} \right) \frac{\partial}{\partial x} - \frac{1}{Re} \frac{d\overline{\mu}}{d\overline{T}} \left( \frac{\partial \overline{U}}{\partial y} + \frac{\partial \overline{V}}{\partial x} \right) \frac{\partial}{\partial y}$$

$$- \frac{1}{Re} \frac{d\overline{\mu}}{d\overline{T}} \left( i\beta \frac{\partial \overline{W}}{\partial x} + \frac{4}{3} \frac{\partial^2 \overline{U}}{\partial x^2} + \frac{\partial^2 \overline{U}}{\partial y^2} + \frac{1}{3} \frac{\partial^2 \overline{V}}{\partial x \partial y} \right)$$

$$- \frac{1}{Re} \frac{d^2\overline{\mu}}{d\overline{T}^2} \left( \frac{4}{3} \frac{\partial \overline{T}}{\partial x} \frac{\partial \overline{U}}{\partial x} + \frac{\partial \overline{T}}{\partial y} \frac{\partial \overline{U}}{\partial y} + \frac{\partial \overline{T}}{\partial y} \frac{\partial \overline{V}}{\partial x} - \frac{2}{3} \frac{\partial \overline{T}}{\partial x} \frac{\partial \overline{V}}{\partial y} \right)$$

$$- \frac{\overline{\rho}}{\overline{T}} \left( \overline{U} \frac{\partial \overline{U}}{\partial x} + \overline{V} \frac{\partial \overline{U}}{\partial y} \right),$$

$$\mathscr{R}_{\hat{u}}^{(x)} = i\overline{\rho}.$$

Disturbance y-momentum

$$\mathcal{L}_{\hat{u}}^{(y)} = -\frac{\overline{\mu}}{3Re} \frac{\partial^2}{\partial x\partial y} + \frac{2}{3Re} \frac{d\overline{\mu}}{d\overline{T}} \frac{\partial \overline{T}}{\partial y} \frac{\partial}{\partial x} - \frac{1}{Re} \frac{d\overline{\mu}}{d\overline{T}} \frac{\partial \overline{T}}{\partial x} \frac{\partial}{\partial y} + \overline{\rho} \frac{\partial \overline{V}}{\partial x},$$

$$\mathcal{L}_{\hat{v}}^{(y)} = -\frac{\overline{\mu}}{Re} \frac{\partial^2}{\partial x^2} + \left( -\frac{1}{Re} \frac{d\overline{\mu}}{d\overline{T}} \frac{\partial \overline{T}}{\partial x} + \overline{\rho} \overline{U} \right) \frac{\partial}{\partial x} - \frac{4\overline{\mu}}{3Re} \frac{\partial^2}{\partial y^2} + \left( \frac{4}{3Re} \frac{d\overline{\mu}}{d\overline{T}} \frac{\partial \overline{T}}{\partial y} + \overline{\rho} \overline{V} \right) \frac{\partial}{\partial y}$$

$$+ \beta^2 \frac{\overline{\mu}}{Re} + \overline{\rho} \frac{\partial \overline{U}}{\partial x} + 2\overline{\rho} \frac{\partial \overline{V}}{\partial y} + i\beta\overline{\rho} \overline{W} + \overline{U} \frac{\partial \overline{\rho}}{\partial x} + \overline{V} \frac{\partial \overline{\rho}}{\partial y},$$

$$\mathcal{L}_{\hat{w}}^{(y)} = -\frac{i\beta\overline{\mu}}{3Re} \frac{\partial}{\partial y} + \frac{2i\beta}{3Re} \frac{d\overline{\mu}}{d\overline{T}} \frac{\partial \overline{T}}{\partial y},$$

$$\mathcal{L}_{\hat{p}}^{(y)} = \frac{1}{\gamma M^2} \frac{\partial}{\partial y} + \frac{1}{\overline{T}} \left( \overline{U} \frac{\partial \overline{V}}{\partial x} + \overline{V} \frac{\partial \overline{V}}{\partial y} \right),$$

$$\mathcal{L}_{\hat{T}}^{(y)} = -\frac{1}{Re} \frac{d\overline{\mu}}{d\overline{T}} \left( \frac{\partial \overline{U}}{\partial y} + \frac{\partial \overline{V}}{\partial x} \right) \frac{\partial}{\partial x} - \frac{1}{Re} \frac{d\overline{\mu}}{d\overline{T}} \left( \frac{4}{3} \frac{\partial \overline{V}}{\partial y} - \frac{2}{3} \frac{\partial \overline{U}}{\partial x} \right) \frac{\partial}{\partial y}$$

$$- \frac{1}{Re} \frac{d\overline{\mu}}{d\overline{T}} \left( i\beta \frac{\partial \overline{W}}{\partial y} + \frac{4}{3} \frac{\partial^2 \overline{V}}{\partial y^2} + \frac{\partial^2 \overline{V}}{\partial x^2} + \frac{1}{3} \frac{\partial^2 \overline{U}}{\partial x\partial y} \right)$$

$$- \frac{1}{Re} \frac{d^2\overline{\mu}}{d\overline{T}^2} \left( \frac{4}{3} \frac{\partial \overline{T}}{\partial y} \frac{\partial \overline{V}}{\partial y} + \frac{\partial \overline{T}}{\partial x} \frac{\partial \overline{U}}{\partial y} + \frac{\partial \overline{T}}{\partial x} \frac{\partial \overline{V}}{\partial x} - \frac{2}{3} \frac{\partial \overline{T}}{\partial y} \frac{\partial \overline{U}}{\partial x} \right)$$

$$- \frac{\overline{\rho}}{\overline{T}} \left( \overline{U} \frac{\partial \overline{V}}{\partial x} + \overline{V} \frac{\partial \overline{V}}{\partial y} \right),$$

$$\mathcal{R}_{\hat{v}}^{(y)} = i\overline{\rho}.$$

Disturbance z-momentum

$$\begin{aligned} \mathscr{L}_{\hat{u}}^{(z)} &= -\frac{i\beta\overline{\mu}}{3Re}\frac{\partial}{\partial x} - \frac{i\beta}{Re}\frac{d\overline{\mu}}{d\overline{T}}\frac{\partial\overline{T}}{\partial x} + \overline{\rho}\frac{\partial\overline{W}}{\partial x}, \\ \mathscr{L}_{\hat{v}}^{(z)} &= -\frac{i\beta\overline{\mu}}{3Re}\frac{\partial}{\partial y} - \frac{i\beta}{Re}\frac{d\overline{\mu}}{d\overline{T}}\frac{\partial\overline{T}}{\partial y} + \overline{\rho}\frac{\partial\overline{W}}{\partial y}, \\ \mathscr{L}_{\hat{w}}^{(z)} &= -\frac{\overline{\mu}}{Re}\frac{\partial^{2}}{\partial x^{2}} - \frac{\overline{\mu}}{Re}\frac{\partial^{2}}{\partial y^{2}} + \left(-\frac{1}{Re}\frac{d\overline{\mu}}{d\overline{T}}\frac{\partial\overline{T}}{\partial x} + \overline{\rho}\overline{U}\right)\frac{\partial}{\partial x} + \left(-\frac{1}{Re}\frac{d\overline{\mu}}{d\overline{T}}\frac{\partial\overline{T}}{\partial y} + \overline{\rho}\overline{V}\right)\frac{\partial}{\partial y} \\ &\quad + \frac{4\overline{\mu}}{3Re}\beta^{2} + \overline{\rho}\frac{\partial\overline{U}}{\partial x} + \overline{\rho}\frac{\partial\overline{V}}{\partial y} + i\overline{\rho}\overline{W} + \overline{U}\frac{\partial\overline{\rho}}{\partial x} + \overline{V}\frac{\partial\overline{\rho}}{\partial y}, \\ \mathscr{L}_{\hat{p}}^{(z)} &= \frac{i\beta}{\gamma M^{2}} + \frac{1}{\overline{T}}\left(\overline{U}\frac{\partial\overline{W}}{\partial x} + \overline{V}\frac{\partial\overline{W}}{\partial y}\right), \\ \mathscr{L}_{\hat{T}}^{(z)} &= -\frac{1}{Re}\frac{d\overline{\mu}}{d\overline{T}}\frac{\partial\overline{W}}{\partial x}\frac{\partial}{\partial x} - \frac{1}{Re}\frac{d\overline{\mu}}{d\overline{T}}\frac{\partial\overline{W}}{\partial y}\frac{\partial}{\partial y} \\ &\quad - \frac{1}{Re}\frac{d\overline{\mu}}{d\overline{T}}\left(\frac{\partial^{2}\overline{W}}{\partial x^{2}} + \frac{\partial^{2}\overline{W}}{\partial y^{2}} - \frac{2i\beta}{3}\left(\frac{\partial\overline{U}}{\partial x} + \frac{\partial\overline{V}}{\partial y}\right)\right) - \frac{1}{Re}\frac{d^{2}\overline{\mu}}{d\overline{T}^{2}}\left(\frac{\partial\overline{T}}{\partial x}\frac{\partial\overline{W}}{\partial x} + \frac{\partial\overline{T}}{\partial y}\frac{\partial\overline{W}}{\partial y}\right) \\ &\quad - \frac{\overline{\rho}}{\overline{T}}\left(\overline{U}\frac{\partial\overline{W}}{\partial x} + \overline{V}\frac{\partial\overline{W}}{\partial y}\right), \\ \mathscr{R}_{\hat{w}}^{(z)} &= i\overline{\rho}. \end{aligned}$$

Disturbance energy

Disturbance energy

$$\begin{aligned} \mathscr{L}_{\hat{u}}^{(e)} &= \left[ -2 \frac{\gamma (\gamma - 1) M^2}{Re} \overline{\mu} \left( \frac{4}{3} \frac{\partial \overline{U}}{\partial x} - \frac{2}{3} \frac{\partial \overline{V}}{\partial y} \right) + \gamma \overline{\rho} \overline{T} \right] \frac{\partial}{\partial x} \\ &- 2 \frac{\gamma (\gamma - 1) M^2}{Re} \overline{\mu} \left( \frac{\partial \overline{U}}{\partial y} + \frac{\partial \overline{V}}{\partial x} \right) \frac{\partial}{\partial y} - 2 \frac{i \beta \gamma (\gamma - 1) M^2}{Re} \overline{\mu} \frac{\partial \overline{W}}{\partial x} + \overline{T} \frac{\partial \overline{\rho}}{\partial x} + \overline{\rho} \frac{\partial \overline{T}}{\partial x}, \\ \mathscr{L}_{\hat{v}}^{(e)} &= -2 \frac{\gamma (\gamma - 1) M^2}{Re} \overline{\mu} \left( \frac{\partial \overline{U}}{\partial y} + \frac{\partial \overline{V}}{\partial x} \right) \frac{\partial}{\partial x} + \left[ -2 \frac{\gamma (\gamma - 1) M^2}{Re} \overline{\mu} \left( -\frac{2}{3} \frac{\partial \overline{U}}{\partial x} + \frac{4}{3} \frac{\partial \overline{V}}{\partial y} \right) \right. \\ &+ \left. \gamma \overline{\rho} \overline{T} \right] \frac{\partial}{\partial y} - 2 \frac{i \beta \gamma (\gamma - 1) M^2}{Re} \overline{\mu} \frac{\partial \overline{W}}{\partial y} + \overline{T} \frac{\partial \overline{\rho}}{\partial y} + \overline{\rho} \frac{\partial \overline{T}}{\partial y}, \\ \mathscr{L}_{\hat{w}}^{(e)} &= -2 \frac{\gamma (\gamma - 1) M^2}{Re} \overline{\mu} \frac{\partial \overline{W}}{\partial x} \frac{\partial}{\partial x} - 2 \frac{\gamma (\gamma - 1) M^2}{Re} \overline{\mu} \frac{\partial \overline{W}}{\partial y} \frac{\partial}{\partial y} \\ &+ \frac{4}{3} \frac{i \beta \gamma (\gamma - 1) M^2}{Re} \overline{\mu} \left( \frac{\partial \overline{U}}{\partial x} + \frac{\partial \overline{V}}{\partial y} \right) + i \beta \gamma \overline{\rho} \overline{T}, \\ \mathscr{L}_{\hat{T}}^{(e)} &= -\frac{\gamma \overline{\kappa}}{Re Pr} \frac{\partial^2}{\partial x^2} - 2 \frac{\gamma}{Re Pr} \frac{d \overline{\kappa}}{d \overline{T}} \frac{\partial \overline{T}}{\partial x} \frac{\partial}{\partial x} - \frac{\gamma \overline{\kappa}}{Re Pr} \frac{\partial^2}{\partial y^2} - 2 \frac{\gamma}{Re Pr} \frac{d \overline{\kappa}}{d \overline{T}} \frac{\partial \overline{T}}{\partial y} \frac{\partial}{\partial y} + \frac{\beta^2 \gamma \overline{\kappa}}{Re Pr} \\ &- \frac{\gamma}{Re Pr} \frac{d \overline{\kappa}}{d \overline{T}} \left( \frac{\partial^2 \overline{T}}{\partial x^2} + \frac{\partial^2 \overline{T}}{\partial y^2} \right) - \frac{\gamma}{Re Pr} \frac{d^2 \overline{\kappa}}{d \overline{T}^2} \left( \frac{\partial \overline{T}}{\partial x}^2 + \frac{\partial \overline{T}}{\partial y}^2 \right) - \frac{\gamma (\gamma - 1) M^2}{Re} \frac{d \overline{\mu}}{d \overline{T}} \\ &\times \left( \frac{4}{3} \frac{\partial \overline{U}}{\partial x}^2 + \frac{\partial \overline{V}}{\partial x}^2 + \frac{\partial \overline{W}}{\partial x}^2 + \frac{\partial \overline{U}}{\partial y}^2 + \frac{4}{3} \frac{\partial \overline{V}}{\partial y}^2 + \frac{\partial \overline{W}}{\partial y}^2 - \frac{4}{3} \frac{\partial \overline{U}}{\partial x} \frac{\partial \overline{V}}{\partial y} + 2 \frac{\partial \overline{U}}{\partial y} \frac{\partial \overline{V}}{\partial x} \right), \\ \mathscr{L}_{\hat{p}}^{(e)} &= \overline{U} \frac{\partial}{\partial x} + \overline{V} \frac{\partial}{\partial y} + i \beta \overline{W} + \gamma \left( \frac{\partial \overline{U}}{\partial x} + \frac{\partial \overline{V}}{\partial y} \right), \\ \mathscr{R}_{\hat{p}}^{(e)} &= i. \end{aligned}$$

Disturbance continuity

$$\begin{aligned} \mathscr{L}_{\hat{u}}^{(c)} &= \overline{\rho} \overline{T} \frac{\partial}{\partial x} + \overline{T} \frac{\partial \overline{\rho}}{\partial x}, \\ \mathscr{L}_{\hat{v}}^{(c)} &= \overline{\rho} \overline{T} \frac{\partial}{\partial y} + \overline{T} \frac{\partial \overline{\rho}}{\partial y}, \\ \mathscr{L}_{\hat{w}}^{(c)} &= i\beta \overline{\rho} \overline{T} \overline{W}, \\ \mathscr{L}_{\hat{p}}^{(Gc)} &= \overline{U} \frac{\partial}{\partial x} + \overline{V} \frac{\partial}{\partial y} + i\beta \overline{W} + \frac{\partial \overline{U}}{\partial x} + \frac{\partial \overline{V}}{\partial y} - \frac{1}{\overline{T}} \left( \overline{U} \frac{\partial \overline{T}}{\partial x} + \overline{V} \frac{\partial \overline{T}}{\partial y} \right), \\ \mathscr{L}_{\hat{T}}^{(c)} &= -\overline{\rho} \overline{U} \frac{\partial}{\partial x} - \overline{\rho} \overline{V} \frac{\partial}{\partial y} - \overline{\rho} \frac{\partial \overline{U}}{\partial x} - \overline{\rho} \frac{\partial \overline{V}}{\partial y} - i\beta \overline{\rho} \overline{W} - \overline{U} \frac{\partial \overline{\rho}}{\partial x} - \overline{V} \frac{\partial \overline{\rho}}{\partial y} + \frac{\overline{\rho}}{\overline{T}} \left( \overline{U} \frac{\partial \overline{T}}{\partial x} + \overline{V} \frac{\partial \overline{T}}{\partial y} \right), \\ \mathscr{R}_{\hat{T}}^{(c)} &= -i\overline{\rho}, \\ \mathscr{R}_{\hat{p}}^{(c)} &= i. \end{aligned}$$

# REFERENCES

Ackeret, J., Feldmann, F. & Rott, N. 1947 Investigation of compression shocks and boundary layers in gases moving at high speed. *NACA TM* 1113.

- Alfano, D., Corre, C., de la Motte, P., Joubert, P. N. & Lerat, A. 2004 Assesment of numerical methods for unsteady shock/boundary layer interaction. Boundary and Interior Layers – Computational and Asymptotic Methods (BAIL 2004) Conference, Toulouse, July.
- Balakumar, P., Zhao, H. & Atkins, H. 2005 Stability of hypersonic boundary layers over a compression corner. AIAA J. **43**, 760–767.
- Bedarev, I. A., Maslov, A., Sidorenko, A., Fedorova, N. N. & Shiplyuk, A. 2002 Experimental and numerical study of hypersonic separated flow in the vicinity of a cone-flare model. J. Appl. Mech. Tech. Phys. **43**, 867–876.
- Boin, J. P., Robinet, J.-Ch., Corre, Ch. & Deniau, H. 2006 3D steady and unsteady bifurcations in a shock-wave/laminar boundary layer interaction; a numerical study. Theoret. Comput. Fluid Dyn. **20**, 163–180.
- Boyd, J. P. 1999 Chebyshev and Fourier Spectral Methods. Dover.
- Canuto, C., Hussaini, M. Y., Quarteroni, A. & Zang, T. A. 1987 Spectral Methods in Fluid Dynamics. Springer.
- Chapman, D. R., Kuehn, D. M. & Larson, H. K. 1958 Investigation of separated flow in supersonic and subsonic streams with emphasis on the effect of transition. NACA Rep. 1356.
- Degrez, G., Boccadoro, C. H. & Wendt, J. F. 1987 The interaction of an oblique shock wave with a laminar boundary layer revisited. An experimental and numerical study. J. Fluid Mech. **177**, 247–263.
- Delery, J. & Marvin, J. G. ´ 1986 Shock-wave boundary layer interactions. AGARDograph.
- Dolling, D. S. 2001 Fifty years of shock-wave/boundary-layer interaction research: What next? AIAA J. **39** (8), 1517–1531.
- Dupont, P., Debieve, J. F., Ardissone, J. P. & Haddad, C. ` 2003 Some time properties in shock boundary layer interaction. West East High Speed Flows, CIMNE 1st edn, January.
- Dupont, P., Haddad, C., Ardissone, J. P. & Debieve, J.-F. ` 2005 Space and time organisation of a shock wave/turbulent boundary layer interaction. Aerospace Sci. Technol. **9**, 561–572.
- Dussauge, J.-P., Dupont, P. & Debieve, J.-F. ` 2006 Unsteadiness in shock wave boundary layer interaction with separation. Aerospace Sci. Technol. **10**, 85–91.
- Eißler, W. & Bestek, H. 1996 Spatial numerical simulations of linear and weakly nonlinear wave instabilities in supersonic boundary layers. Theoret. Comput. Fluid Dyn. **8**, 219–235.
- Haddad, C., Ardissone, J. P., Dupont, P. & Debieve, J. F. ` 2004 Space and time organization of a shock wave/turbulent boundary layer interaction. Congres AAAF, A ` erodynamiques ´ instationnaires, Paris.
- Jameson, A. 1991 Time-dependent calculations using multigrid with applications to unsteady flows past airfoils and wings. AIAA Paper 1991-1596.
- Kim, K. H., Kim, C. & Rho, O.-H. 2001a Methods for the accurate computations of hypersonic flows i. AUSMPW+ scheme. J. Comput. Phys. **174**, 38–80.
- Kim, K. H., Kim, C. & Rho, O.-H. 2001b Methods for the accurate computations of hypersonic flows ii. Shock-aligned grid technique. J. Comput. Phys. **174**, 81–119.
- Kosinov, A. D., Maslov, A. A. & Shevelkov, S. G. 1990 Experiments on the stability of supersonic boundary layers. J. Fluid Mech. **219**, 621–633.
- Liepmann, H. W. 1946 The interaction between boundary layer and shock waves in transonic flow. J. Aeronaut. Sci. **13**, 623–637.
- Lighthill, M. J. 1950 Reflection at a laminar boundary-layer of a weak steady disturbance to a supersonic stream neglecting viscosity and heat conduction. Q. J. Mech. Appl. Maths **3**, 303.
- Lighthill, M. J. 1953a On the boundary layer upstream influence I. A comparison between subsonic and supersonic flows. Proc. R. Soc. A **217**, 344–357.
- Lighthill, M. J. 1953b On the boundary layer upstream influence II. Supersonic flows without separation. Proc. R. Soc. A **217**, 478–507.
- Liou, M. S. & Edwards, J. 1998 Low-diffusion flux-splitting methods for flows at all speeds. AIAA J. **36**, 1610–1617.
- Luo, H., Baum, J. D. & Lhner, R. 2001 An accurate, fast, matrix-free implicit method for computing unsteady flows on unstructured grids. Comput. Fluids **30**, 137–159.
- Mack, L. 1969 Boundary layer stability theory. Tech. Rep. 900-277. Jet Propulsion Laboratory, Pasadena.
- Malik, M. R. 1990 Numerical methods for hypersonic boundary layer stability. J. Comput. Phys. **86**, 376–413.

- Pagella, A., Rist, U. & Wagner, S. 2002 Numerical investigations of small-amplitude disturbances in a boundary layer with impinging shock wave at Ma = 4*.*8. Phys. Fluids **14** (7), 2088–2101.
- Peyret, R. & Taylor, T. 1983 Computational Methods for Fluid Flows. Springer.
- Smits, A. J. & Dussauge, J.-P. 1996 Turbulent Shear Layers in Supersonic Flow. AIP Press, New York.
- Stewartson, K. & Williams, P. G. 1969 Self-induced separation. Proc. R. Soc. Lond. A **312** (1508), 181206.
- Streett, C. L. & Macaraeg, M. G. 1989 Spectral multi-domain for large-scale fluid dynamic simulations. Appl. Numer. Maths **6**, 123–139.
- Theofilis, V. 2003 Advances in global linear instability analysis of nonparallel and three-dimensional flows. Prog. Aerospace Sci. **39**, 249–315.
- Theofilis, V. & Colonius, T. 2004 Three-dimensional instabilities of compressible flow over open cavities: direct solution of the biglobal eigenvalue problem. AIAA Paper 2004–2544.