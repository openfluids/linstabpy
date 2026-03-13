# Analysis of the dynamics of subharmonic flow structures via the harmonic resolvent: application to vortex pairing in an axisymmetric jet

Alberto Padovan<sup>∗</sup> and Clarence W. Rowley† Department of Mechanical and Aerospace Engineering, Princeton University, Princeton, NJ 08544, USA (Dated: June 30, 2022)

In this paper we extend the harmonic resolvent analysis to study the dynamics of subharmonic perturbations about a periodically time-varying base flow. In doing so, we recover an input-output operator that is closely related to the harmonic transfer function introduced in [Wereley, 1991], and we also elucidate the nature of the cross-frequency interactions between subharmonic flow structures in the proximity of the base flow. We first demonstrate the use of this method on the R¨ossler system, under conditions for which the dynamics are sensitive to period-doubling perturbations. We then apply it to a forced incompressible axisymmetric jet, and we study how the jet's sensitivity to subharmonic perturbations varies as a function of the Reynolds number. This analysis suggests that as the Reynolds number is increased, the cross-frequency interactions between subharmonic structures with period 2T and the T-periodic base flow become increasingly important. Remarkably, we also demonstrate that the well-known nonlinear vortex pairing phenomenon is driven by the spatio-temporal structures contained within the first right singular vector (or input mode) of the harmonic resolvent evaluated at the 1/2-subharmonic of the fundamental frequency. In particular, we show that if the nonlinear flow is forced with an input that is orthogonal to the first right singular vector of the harmonic resolvent, then no pairing will occur.

## I. INTRODUCTION

Over the past few decades, model-based approaches rooted in linear systems theory have helped shed light on the physics of complex, high-dimensional fluid flows. While there are a number of linear techniques that can be used to analyze the dynamics of fluid flows in the proximity of a solution of the Navier-Stokes equation, here we focus on the so-called input-output methods. Perhaps the most well-known is resolvent analysis [1–3], a frequency-domain technique that allows one to study the input-output dynamics of perturbations about a steady base flow (typically either a steady solution of the Navier-Stokes equation or the temporal mean of a data set of interest). This method is particularly helpful for the analysis of non-normal systems that exhibit large-amplitude transient energy growth in response to external perturbations [2, 3], but it has also been used for controller design in unsteady turbulent flows [4, 5]. While resolvent analysis is an appealing analysis tool due to its (usually) low computational cost, it is well-known that it loses its predictive capabilities when the nonlinear dynamics evolve far away from the temporal mean or from a steady solution of the Navier-Stokes equation. This is because the steady nature of the base flow is such that this analysis neglects all the cross-frequency interactions in the flow. A successful effort to improve the performance of resolvent analysis by accounting for some of the triadic coupling that dominates the dynamics in highly oscillatory flows may be found in [6]. Another noteworthy improvement of resolvent analysis applied to wall-bounded flows was proposed in [7], where the authors treated the action of the nonlinearity as a structured uncertainty that forces the linear part of the dynamics.

Recently, resolvent analysis was extended to flows that exhibit large-amplitude deviations from the temporal mean or from a steady solution of the governing equations. More specifically, in the harmonic resolvent analysis described in [8] the governing equations are linearized about a periodically time-varying base flow that accounts for the aforementioned large-amplitude oscillations about the mean. This leads to a frequency-domain system of coupled equations where the coupling is introduced by the time-varying nature of the base flow. Given its ability to capture the first-order triadic interactions in the proximity of the base flow, the harmonic resolvent analysis can be seen as a first step towards understanding the nature of the cross-frequency coupling in fluid flows that are dominated by nonlinear effects. In this paper, we present an extension of the harmonic resolvent framework that considers subharmonic perturbations about a base flow with period T. While previous work in [8] focused exclusively on the dynamics of T-periodic perturbations, here we consider perturbations with period nT, for an integer n. These perturbations are of particular interest in shear flows such as mixing layers and jets, whose selective sensitivity to subharmonic flow structures lead to phenomena such

<sup>∗</sup> apadovan@princeton.edu

<sup>†</sup> cwrowley@princeton.edu

as vortex merging and pairing [9, 10]. Recently, the authors in [11] showed that vortex pairing in an incompressible axisymmetric jet can be understood as a Floquet instability arising from an underlying (unstable) periodic solution of the Navier-Stokes equation. However, they also observed that the pairing can happen at Reynolds numbers much lower than the critical Reynolds number, and they attributed this to the highly non-normal nature of the flow due to the presence of high-shear regions along the shear layer. We will show how the harmonic resolvent framework can be used to confirm the highly non-normal nature of the problem, and also to obtain additional information regarding the driving mechanism behind vortex pairing. In particular, we shall see that as the Reynolds number is increased, the input-output dynamics are dominated by cross-frequency interactions between subharmonic perturbations with period 2T and the T-periodic base flow. Furthermore, we will use this analysis to compute the spatio-temporal structures that drive the vortex pairing mechanism.

The harmonic resolvent framework discussed herein lies at the intersection of harmonic balancing methods and linear time-periodic systems theory. Harmonic balancing methods are becoming an increasingly popular technique for solving partial differential equations thanks to the fact that they convert time-dependent problems into constant-coefficient algebraic systems of equations in the frequency domain. A remarkable use of harmonic balancing is presented in [12], where the authors studied the transition to turbulence in a volumetrically-forced incompressible boundary layer. As far as linear time-periodic systems theory is concerned, historic and recent applications to incompressible flows in the time domain can be found in [13–15]. In [13], the authors studied the stability properties of the two-dimensional wake of a circular cylinder via a Floquet analysis, and they detected a three-dimensional instability at Reynolds number Re ≈ 189. More recently, the authors in [14] sought the optimal inflow forcing profile that would maximize spreading in an incompressible jet linearized about an axisymmetric time-periodic solution of the Navier-Stokes equation. Similarly, in [15], the authors studied the linear energy-amplification mechanisms in pulsatile channels and pipes. Examples of the use of the frequency domain for the analysis of periodic fluid flows may be found in [16–18]. The authors in [16] proposed a computationally efficient approach to compute the H<sup>2</sup> norm of linear time-periodic systems where the periodic component of the dynamics is small. In particular, by expanding the periodic dynamics in powers of a parameter ε 1, they showed that the H<sup>2</sup> norm of the system can be approximated to second order by solving a sequence of simplified Lyapunov and Sylvester equations in the frequency domain. They then demonstrated their approach on a pressure-driven two-dimensional channel subject to streamwise (small-amplitude) time-periodic oscillations of the bottom wall. The theoretical developments presented in [16] were also used in [17] and [18]. In reference [17] the authors performed a receptivity analysis of the linearized Navier-Stokes equations to inform the design of small-amplitude streamwise travelling waves to suppress transition to turbulence in a channel. Similarly, in [18], the authors designed small-amplitude transverse wall oscillations to achieve turbulent drag reduction in channel flow. These references stand out as examples of how to develop flow control strategies by leveraging the frequencydomain structure of linearized models that are periodic in time, but where the periodic component is small. Finally, a formalism similar to the ones described herein was used in [19] and [20] to account for the effect of spanwise-periodic riblets in a turbulent channel and to design spanwise-periodic riblets for turbulent drag reduction in a channel, respectively. In these references, the spatially-periodic structure of the riblets was not assumed to be small, so the analysis tools used in [19] and [20] may be understood as the spatial analog of the temporal harmonic resolvent that we discuss in this paper. For a comprehensive summary of resolvent-like methods for the space-time analysis of fluid flows, we refer the reader to the recent review [21].

The structure of this paper is as follows. In section II we present the extension of the harmonic resolvent to study the dynamics of subharmonic perturbations. In particular, we will see that if we seek a (nT)-periodic solution of the linearized Navier-Stokes equation about a T-periodic base flow, the problem decouples into n subproblems that can be solved separately at a much lower computational cost. This will lead to the definition of an input-output operator that is closely related to the harmonic transfer function discussed in [22]. Henceforth, we will refer to this input-output operator as harmonic resolvent operator. It will become clear in section II that the operator in [8] is a special case of the harmonic resolvent discussed herein when the period of the perturbations is the same as the period of the base flow (γ = 0, in the notation of section II). In section III we show that, similarly to the resolvent analysis in [2, 3], the singular value decomposition of the harmonic resolvent operator provides information about the dominant input-output mechanisms of the flow in the proximity of the periodic orbit. Moreover, since the harmonic resolvent operator supports coupling between selected frequency pairs, we show that the singular value decomposition of the blocks of the harmonic resolvent operator can shed light on the cross-frequency input-output dynamics. In section IV we demonstrate the use of this method on the R¨ossler equations under conditions for which the system is highly sensitive to period-doubling perturbations. Finally, in section V we apply the harmonic resolvent framework to the periodically-forced axisymmetric jet considered in [11]. Here, we identify the mechanisms by which the flow selectively amplifies subharmonic perturbations and we use these results to explain why the authors in [11] observed vortex pairing in an otherwise linearly-stable flow configuration. In particular, we demonstrate that if the nonlinear flow is forced with an input that is orthogonal to the optimal input mode of the harmonic resolvent evaluated at the 1/2-subharmonic of the fundamental frequency, then no vortex pairing occurs. We therefore conclude that the

optimal input mode defines the spatio-temporal signal that is responsible for vortex pairing. In both examples we also provide a comparison with the results obtained by performing the mean-based resolvent analysis.

## II. MATHEMATICAL FORMULATION

In this section we extend the harmonic resolvent framework [8] by considering perturbations with period nT about a T-periodic base flow. This will lead to the definition of an input-output operator that is closely related to Wereley's harmonic transfer function [22]. As mentioned in the introduction, we refer to this input-output operator as the harmonic resolvent operator and it will become clear that the operator described in [8] arises as a special case when the perturbations have the same period as the base flow. The reader who is familiar with the harmonic transfer function literature [22] may be used to working in the space of exponentially modulated periodic (EMP) signals. In this section, we choose to proceed in a different fashion, and we rely exclusively on the Fourier transform. We favor this approach because it does not require a-priori knowledge of EMPs, and because it sheds light on the reason why EMPs are appropriate test signals for the frequency-domain analysis of periodic systems. The equivalence between our derivation and the derivation using EMPs is clarified in Remark 1 in section II C.

### A. Frequency-domain representation of linear time-periodic dynamics

Let us consider a nonlinear dynamical system with state  $q(t) \in \mathbb{R}^N$ , whose time evolution is governed by

$$\frac{\mathrm{d}}{\mathrm{d}t}q(t) = g(q(t)). \tag{1}$$

In our present context, this may be understood as the spatially-discretized incompressible Navier-Stokes equation, and the state vector contains the three-dimensional velocity and the pressure at the cell faces (or cell centers) of the computational grid. Normally, formula (1) is also equipped with an algebraic constraint on the velocity field (i.e., the continuity equation). However, it is well known that this constraint can often be removed from the formulation by expressing the pressure as a function of the velocity by means of a Poisson equation. Therefore, we move forward by working with the constraint-free system (1). Algebraic constraints on the state q(t) can be handled with a straightforward modification of the harmonic resolvent framework discussed in this paper. Letting Q(t) = Q(t+T) denote a T-periodic base flow, we perform the state decomposition

$$\mathbf{q}(t) = \mathbf{Q}(t) + \mathbf{q}'(t),\tag{2}$$

where q'(t) denotes a perturbation about Q(t). We observe that, in general, the perturbation q'(t) need not be periodic. Upon substitution of (2) into the governing equation (1), we obtain

$$\frac{\mathrm{d}}{\mathrm{d}t} \boldsymbol{q}'(t) = \underbrace{D_{\boldsymbol{q}} \boldsymbol{g}(\boldsymbol{Q}(t))}_{\boldsymbol{A}(t)} \boldsymbol{q}'(t) + \boldsymbol{f}'(t), \tag{3}$$

where f'(t) contains higher-order terms  $O(\|q'\|^2)$  as well as an additional error term if the base flow does not satisfy (1) exactly. By virtue of the T-periodicity of the base flow, the Jacobian A(t) is also periodic with period T. We henceforth assume that A(t) is a differentiable function of time; this guarantees the existence of a unique solution to (3) (see, e.g., Theorem 3.2 in [23]), and furthermore it implies nice properties of the Fourier series of A(t) (e.g., pointwise convergence, and square-summability of the Fourier coefficients). A simple rescaling of time allows us to take  $T = 2\pi$ , so the Fourier series of A(t) has the form

$$\mathbf{A}(t) = \sum_{k \in \widetilde{\Omega}} \hat{\mathbf{A}}_k e^{ikt}, \quad \widetilde{\Omega} = \{\dots, -2, -1, 0, 1, 2, \dots\}.$$

$$(4)$$

We observe that the frequency set  $\widetilde{\Omega}$  is potentially infinite. In practice, however, it will contain a few frequencies associated with the energetically-dominant coherent structures that we observe in the flow. While in [8] we were interested in studying the dynamics of perturbations q'(t) with the same period as the base flow, here we consider perturbations with period  $2\pi n$ , where n is a natural number. Since solutions of (3) are differentiable with respect to time, we may expand q'(t) in a square-summable Fourier series with fundamental frequency 1/n,

$$\mathbf{q}'(t) = \sum_{\alpha \in \Omega} \hat{\mathbf{q}}_{\alpha} e^{i\alpha t}, \quad \Omega = \left\{ \cdots, -\frac{2}{n}, -\frac{1}{n}, 0, \frac{1}{n}, \frac{2}{n}, \cdots \right\}. \tag{5}$$

We refer to  $\Omega$  as the frequency set associated with the perturbations q'(t). It is convenient, from a notational standpoint, to let the set  $\Omega$  be infinite. In practice, however, we truncate  $\Omega$  at some appropriate frequency r/n that depends on how accurately we wish to resolve the temporal evolution of the perturbations q'(t). Plugging formulas (5) and (4) into (3), and writing f'(t) as a Fourier series over  $\Omega$  as in (5), we obtain

$$i\alpha\hat{\boldsymbol{q}}_{\alpha} = \sum_{\substack{\beta \in \Omega \\ (\alpha - \beta) \in \widetilde{\Omega}}} \hat{\boldsymbol{A}}_{\alpha - \beta}\hat{\boldsymbol{q}}_{\beta} + \hat{\boldsymbol{f}}_{\alpha}, \quad \forall \alpha \in \Omega.$$

$$(6)$$

Formula (6) is a system of linearly-coupled equations, where structures at frequency  $\alpha$  are coupled to structures at frequency  $\beta$  through the base flow at the frequency difference  $(\alpha - \beta) \in \widetilde{\Omega}$ . However, we will see shortly that this equation hides n decoupled systems of equations that can be solved separately at a much lower computational cost. Uncovering the decoupled nature of the problem will also help us understand the nature of the triadic frequency coupling in the proximity of the base flow.

### B. Understanding the cross-frequency coupling

In this section we show how the convolution-like sum in formula (6) couples structures at different frequencies. We will do so using a simple example. Consider the base flow set below, and a perturbation set with fundamental frequency 1/n, where we take n = 3,

$$\widetilde{\Omega} = \{-1, 0, 1\}, \quad \Omega = \left\{ \cdots, -\frac{7}{3}, -\frac{6}{3}, -\frac{5}{3}, \cdots, -\frac{1}{3}, 0, \frac{1}{3}, \cdots, \frac{5}{3}, \frac{6}{3}, \frac{7}{3}, \cdots \right\}. \tag{7}$$

We now show that the three disjoint subsets of  $\Omega$  given below contain the frequencies that are non-trivially coupled to one another through the base flow set  $\widetilde{\Omega}$ .

$$\Omega_{-1/3} = \left\{ \cdots, -\frac{7}{3}, -\frac{4}{3}, -\frac{1}{3}, \frac{2}{3}, \frac{5}{3}, \cdots \right\}, \quad \Omega_0 = \left\{ \cdots, -2, -1, 0, 1, 2, \cdots \right\}, \quad \Omega_{1/3} = \left\{ \cdots, -\frac{5}{3}, -\frac{2}{3}, \frac{1}{3}, \frac{4}{3}, \frac{7}{3}, \cdots \right\}.$$
(8)

As previously discussed, the coupling between frequencies  $\alpha, \beta \in \Omega$  is supported by the base flow at the frequency difference  $(\alpha - \beta) \in \widetilde{\Omega}$ . For example, it is easy to verify that for every  $\alpha \in \Omega_{1/3}$ , there exists a  $\beta \neq \alpha$  in the same set such that  $(\alpha - \beta) \in \widetilde{\Omega}$ . We therefore say that the frequencies in  $\Omega_{1/3}$  are coupled through the base flow. Conversely, there is no frequency  $\alpha \in \Omega_{1/3}$  and no frequency  $\beta \in \Omega_0$  such that  $(\alpha - \beta) \in \widetilde{\Omega}$ . So the frequencies in  $\Omega_{1/3}$  are fully decoupled from those in  $\Omega_0$ . A schematic of the cross-frequency coupling is shown in figure 1. It is also useful to observe that all the sets in (8) may be written as

$$\Omega_{\gamma} = \gamma + \{\dots, -2, -1, 0, 1, 2, \dots\}, \quad \gamma \in \left\{-\frac{1}{3}, 0, \frac{1}{3}\right\},$$
(9)

where the + in the definition of  $\Omega_{\gamma}$  denotes element-wise addition. In general, the following rule can be used to determine whether any two frequencies are coupled.

**Property II.1.** Let  $\widetilde{\Omega}$  and  $\Omega$  be given as in (4) and (5), respectively. Two frequencies  $\alpha$ ,  $\beta \in \Omega$  are coupled if  $\alpha = \beta \pmod{1}$ . Letting  $\mathbb{Z}$  denote the set of integers, it follows that the set  $\Omega$  can be partitioned into n disjoint subsets  $\Omega_{\gamma}$  of the form

$$\Omega_{\gamma} = \gamma + \mathbb{Z}, \quad \gamma \in \Gamma = \left\{ -\frac{1}{n} \left\lfloor \frac{(n-1)}{2} \right\rfloor, \cdots, -\frac{1}{n}, 0, \frac{1}{n}, \cdots, \frac{1}{n} \left\lfloor \frac{n}{2} \right\rfloor \right\}. \tag{10}$$

Observe that  $-1/2 < \gamma \le 1/2$  for every value of n. Furthermore, when n is even we see that  $\Gamma$  contains the element 1/2, but it does not contain -1/2. This is because  $\Omega_{1/2} = \Omega_{-1/2}$ , so including -1/2 would be redundant.

We can now uncover a second property. It is easy to see using our simple example that  $\Omega_{-1/3} = -\Omega_{1/3}$ . Thus, the Fourier coefficients of q'(t) corresponding to the frequencies in  $\Omega_{-1/3}$  are merely the complex conjugates of the Fourier coefficients corresponding to the frequencies in  $\Omega_{1/3}$ . For our example in (7), this means that it suffices to explicitly solve for the Fourier coefficients of q'(t) over the sets  $\Omega_0$  and  $\Omega_{1/3}$ , while the Fourier coefficients over  $\Omega_{-1/3}$  may be obtained by complex-conjugating the coefficients over  $\Omega_{1/3}$ . In general, we have the following property.

![](_page_4_Figure_1.jpeg)

FIG. 1. Two frequencies  $\alpha$ ,  $\beta \in \Omega$  are linked if  $\alpha - \beta \in \widetilde{\Omega}$ . Two frequencies  $\alpha$ ,  $\beta \in \Omega$  share the same color (hence are coupled) if  $\alpha = \beta \pmod{1}$ .

**Property II.2.** Let the frequency sets  $\Omega_{\gamma}$  be given as in property II.1. The signal q'(t) in (3) is fully determined by its Fourier coefficients over the sets  $\Omega_{\gamma}$  with  $0 \le \gamma \le 1/2$ .

While property II.1 states that the original problem (6) decouples into n separate problems, property II.2 states that only  $\lceil n/2 \rceil$  of these need to be solved explicitly. This is of course beneficial from a computational standpoint, since solving  $\lceil n/2 \rceil$  smaller decoupled problems is much easier than solving one large coupled system of equations. We are now equipped to define the harmonic resolvent operator.

### C. The harmonic resolvent operator

In this section, we adopt the following notation. Given a periodic signal q'(t) with period  $2\pi n$ , we define

$$\mathcal{F}_{\gamma}(\mathbf{q}'(t)) = (\dots, \hat{\mathbf{q}}_{\gamma-1}, \hat{\mathbf{q}}_{\gamma}, \hat{\mathbf{q}}_{\gamma+1}, \dots), \quad \gamma \in \Gamma, \tag{11}$$

where  $\hat{q}_{\alpha}$  are the Fourier coefficients as given in (5). This may be understood as an infinite-dimensional vector containing a subset of the Fourier coefficients of q'(t), namely those corresponding to frequencies in the set  $\Omega_{\gamma} = \gamma + \mathbb{Z}$ . We now use the properties discussed in the previous section to introduce the harmonic resolvent operator. Let us recall that property II.1 states that problem (6) decouples into n separate problems over the sets  $\Omega_{\gamma}$ . We can therefore constrain our attention to the Fourier coefficients of q'(t) and f'(t) corresponding to the frequencies in  $\Omega_{\gamma}$ , for some fixed  $\gamma \in [0, 1/2]$  according to property II.2. Moreover, any  $\alpha \in \Omega_{\gamma}$  may be written as  $(\gamma + k)$  for some integer  $k \in \mathbb{Z}$ . We then proceed as follows. We start from formula (6), replace  $\Omega$  with  $\Omega_{\gamma}$  and write  $\alpha = \gamma + k$  and  $\beta = \gamma + j$ , to obtain

$$i(\gamma + k)\hat{\mathbf{q}}_{\gamma+k} = \sum_{\substack{j \in \mathbb{Z} \\ (k-j) \in \widetilde{\Omega}}} \hat{\mathbf{A}}_{k-j}\hat{\mathbf{q}}_{\gamma+j} + \hat{\mathbf{f}}_{\gamma+k}, \quad \forall k \in \mathbb{Z}.$$
 (12)

Formula (12) is a system of coupled linear equations, and it may be written compactly be defining appropriate linear operators. In particular, letting  $\hat{q} = \mathcal{F}_{\gamma}(q'(t))$ , we introduce the operator

$$\left[\hat{T}\hat{q}\right]_{k+\gamma} = -ik\hat{q}_{\gamma+k} + \sum_{\substack{j \in \mathbb{Z} \\ (k-j) \in \widetilde{\Omega}}} \hat{A}_{k-j}\hat{q}_{\gamma+j}. \tag{13}$$

It is important to observe that  $\hat{T}$  does not depend on  $\gamma$ , but only on the base flow set  $\widetilde{\Omega}$ . Formula (12) may finally be written compactly as

$$(i\gamma \mathbf{I} - \hat{\mathbf{T}})\hat{\mathbf{q}} = \hat{\mathbf{f}},\tag{14}$$

where I is the identity operator. Before moving forward, it is instructive to discuss the structure of the operator  $\hat{T}$ . First, we observe that it is infinite-dimensional since it acts on the infinite-dimensional vector  $\hat{q}$ . Second, it may be

visualized in matrix form as follows

matrix form as follows

$$\hat{T} = \begin{bmatrix}
\ddots & \ddots & \ddots & \ddots & & \\
\ddots & \hat{R}_{-2} & \hat{A}_{-1} & \hat{A}_{-2} & \ddots & \\
\ddots & \hat{A}_{1} & \hat{R}_{-1} & \hat{A}_{-1} & \hat{A}_{-2} & \ddots & \\
\ddots & \hat{A}_{2} & \hat{A}_{1} & \hat{R}_{0} & \hat{A}_{-1} & \hat{A}_{-2} & \ddots \\
& \ddots & \hat{A}_{2} & \hat{A}_{1} & \hat{R}_{1} & \hat{A}_{-1} & \ddots \\
& & \ddots & \hat{A}_{2} & \hat{A}_{1} & \hat{R}_{2} & \ddots \\
& & & \ddots & \ddots & \ddots & \ddots
\end{bmatrix}, \quad \hat{R}_{k} = (-ik I + \hat{A}_{0}) \in \mathbb{C}^{N \times N}, \tag{15}$$

where we recall that N is the dimension of the state vector  $\mathbf{q}'(t)$ . The number of off-diagonal bands in  $\hat{\mathbf{T}}$  is equal to the number of non-zero frequencies in the base flow set  $\tilde{\Omega}$ . Moreover,  $\hat{\mathbf{A}}_{-k}$  is the complex-conjugate of  $\hat{\mathbf{A}}_k$ , since the matrix  $\mathbf{A}(t)$  is a real-valued function of time. If the base flow is steady, then the off-diagonal blocks are zero and  $\hat{\mathbf{T}}$  is a block-diagonal operator. Consequently, there no longer is any cross-frequency coupling between structures at different frequencies. It is also important to observe that given a diagonal block  $\hat{\mathbf{R}}_k$ , the operator  $-\hat{\mathbf{R}}_k^{-1}$  is the resolvent operator discussed in [2, 3] and evaluated at frequency k. Thus, when the base flow is steady, the operator  $(i\gamma \mathbf{I} - \hat{\mathbf{T}})$  is block diagonal with blocks  $-\hat{\mathbf{R}}_{(\gamma+k)}$ , and we recover the well-known resolvent analysis.

Resuming from formula (14), we now wish to define the harmonic resolvent operator as a linear operator that maps inputs  $\hat{f}$  to outputs  $\hat{q}$ . However, properly defining this operator requires some care and we first need to address a subtlety. Specifically, it is straightforward to show that if the nonlinear dynamics in (1) admit time-periodic solutions  $Q(t+\varepsilon)$  for any phase shift  $\varepsilon$ , then  $\hat{T}$  (assembled about the base flow Q(t)) is singular and its nullspace is spanned by the Fourier coefficients of  $\mathbf{v}(t) \coloneqq \mathrm{d}\mathbf{Q}(t)/\mathrm{d}t$ . In other words,  $\hat{\mathbf{T}}\hat{\mathbf{v}} = 0$ , where  $\hat{\mathbf{v}} = \mathcal{F}_0(\mathbf{v}(t))$ . This scenario is very common, and it is most often encountered when the homogeneous nonlinear dynamics (1) admit time-periodic solutions (e.g., incompressible flow past a body at moderately low Reynolds numbers and periodic orbits in turbulent regimes). Conversely, if the periodic solution arises in response to some external time-periodic forcing (e.g., flows in turbomachinery and rotorcraft), then the latter usually sets the phase of the solution and the nullspace of  $\hat{T}$  will be trivial. We henceforth refer to the time-derivative of the base flow as the direction of phase shift about the base flow. This singularity of  $\hat{T}$  is problematic because the direction of phase shift is physically uninteresting and, at the same time, it prevents us from defining the harmonic resolvent as the inverse of  $(i\gamma I - T)$  (in fact, this inverse is undefined for  $\gamma = 0$ ). Since we want our definition to hold for all  $\gamma$ , we need to remove this singularity by restricting the domain of  $\hat{T}$  to a physically meaningful subspace  $\Sigma$  on which  $\hat{T}$  is bijective. In order to define this subspace  $\Sigma$ , we begin by requiring that for all times, all time-periodic solutions of the linear dynamics in (3) lie in the range of the  $2\pi$ -periodic oblique projection

$$P(t) = I - v(t)u(t)^*. \tag{16}$$

Here,  $\boldsymbol{u}(t)$  is a  $2\pi$ -periodic signal such that  $\boldsymbol{u}(t)^*\boldsymbol{v}(t)=1$  for all times, so that  $\boldsymbol{P}(t)$  is a projection (i.e.,  $\boldsymbol{P}(t)^2=\boldsymbol{P}(t)$ ). In particular, we choose  $\boldsymbol{u}(t)$  such that  $\hat{\boldsymbol{T}}^*\hat{\boldsymbol{u}}=0$ , where  $\hat{\boldsymbol{u}}=\mathcal{F}_0(\boldsymbol{u}(t))$ . In other words, the Fourier coefficients of  $\boldsymbol{u}(t)$  span the left nullspace of  $\hat{\boldsymbol{T}}$ . The efficient computation of  $\hat{\boldsymbol{u}}$  is discussed in appendix A. Clearly, the projection  $\boldsymbol{P}(t)$  annihilates any component along the phase shift direction, since  $\boldsymbol{P}(t)\boldsymbol{v}(t)=0$  for all times. Since  $\boldsymbol{P}(t)$  is  $2\pi$ -periodic, it may be written in a Fourier series

$$\mathbf{P}(t) = \sum_{k \in \mathbb{Z}} \hat{\mathbf{P}}_k e^{ikt}, \quad \hat{\mathbf{P}}_k = \mathbf{I}\delta_{k,0} - \sum_{j \in \mathbb{Z}} \hat{\mathbf{v}}_{k+j} \hat{\mathbf{u}}_j^*, \tag{17}$$

where  $\delta_{k,0} = 1$  if k = 0 and  $\delta_{k,0} = 0$  otherwise. We may then define the frequency-domain representation of P(t) as

$$[\hat{\boldsymbol{P}}\hat{\boldsymbol{q}}]_{k+\gamma} = \sum_{j \in \mathbb{Z}} \hat{\boldsymbol{P}}_{k-j} \hat{\boldsymbol{q}}_{\gamma+j}.$$
 (18)

The desired subspace  $\Sigma$  is defined as the range of  $\hat{P}$ . It can be shown that  $\Sigma$  is an invariant subspace of  $\hat{T}$  and, by design,  $\hat{T}$  is bijective on  $\Sigma$ . Thus, letting  $\hat{T}_{\Sigma}$  denote the restriction of  $\hat{T}$  onto  $\Sigma$ , the harmonic resolvent operator is given by

$$H(i\gamma) = (i\gamma I - \hat{T}_{\Sigma})^{-1}.$$
(19)

This is a linear input-output operator that maps the Fourier coefficients of inputs  $f'(t) \in \text{Range}(P(t)^*)$  with period  $T/\gamma$  and frequency content  $\Omega_{\gamma}$  to the Fourier coefficients of outputs  $q'(t) \in \text{Range}(P(t))$  with period  $T/\gamma$  and frequency content  $\Omega_{\gamma}$ . It is important to observe that above we have used a slightly different projection than that used in [8]: in that formulation, the perturbations  $\mathcal{F}_0(q'(t))$  were constrained to the orthogonal complement of  $\hat{v}$ , while here they are constrained to the range of  $\hat{P}$ . Consequently, when  $\gamma = 0$ , we do not recover the previous definition of harmonic resolvent. While the definition in [8] is mathematically correct and physically meaningful, the orthogonal complement of  $\hat{v}$  may not be invariant under  $\hat{T}$  (unless  $\hat{T}$  is a normal operator, which is never the case in high-shear flows), so performing a projection onto  $\hat{v}^{\perp}$  loses its physical meaning when  $\gamma \neq 0$ . In contrast, the projection as defined here is meaningful for all choices of  $\gamma$ , including the case  $\gamma = 0$  (i.e., the harmonic resolvent defined in [8]). Finally, we observe for the sake of completeness that it is not necessary to evaluate  $H(i\alpha)$  for  $\alpha > 1/2$ . It is easy to verify using sets of the form (10) that for any set  $\Omega_{\alpha}$  with  $\alpha > 1/2$  there exists a  $\gamma \in [0, 1/2]$  such that either  $\Omega_{\alpha} = \Omega_{\gamma}$  or  $\Omega_{\alpha} = -\Omega_{\gamma}$ . Thus, either  $H(i\alpha) = H(i\gamma)$  or  $H(i\alpha) = \overline{H(i\gamma)}$ , where the overline indicates complex conjugation. For a computationally efficient way to compute with  $H(i\gamma)$ , we point the reader to appendix B.

**Remark 1.** As mentioned at the beginning of this section, the harmonic resolvent operator (19) can be derived using exponentially modulated periodic (EMP) signals [22]. More precisely, this can be done by writing q'(t) in (5) as

$$\mathbf{q}'(t) = e^{i\gamma t} \sum_{\substack{k \in \mathbb{Z} \\ := \widetilde{\mathbf{q}}(t)}} \hat{\mathbf{q}}_{\gamma+k} e^{ikt}, \tag{20}$$

where the  $2\pi$ -periodic component  $\tilde{q}(t)$  of q'(t) is modulated by  $e^{i\gamma t}$ . Formula (12) (and therefore (19)) may then be obtained by substituting formulas (20) and (4) into (3). While the derivation of the harmonic resolvent  $H(i\gamma)$  using EMPs may appear less cumbersome than the one presented in this paper, it is precisely the frequency decoupling of subharmonics discussed in section IIB that makes EMPs an appropriate class of signals for the analysis of the input-output dynamics of time-periodic systems. For completeness, we observe that (if  $\gamma$  is a rational number) the Fourier coefficients of the signal q'(t) in (20) are precisely given by  $\mathcal{F}_{\gamma}(q'(t))$ , and the associated frequency set is  $\Omega_{\gamma}$ .

## III. AMPLIFICATION MECHANISMS

The spectral analysis of the harmonic resolvent operator can provide insightful information about the physics of the flow under consideration. In the first subsection, we discuss how to extract information about the global amplification mechanisms, while in the second subsection we discuss how to probe the harmonic resolvent for information regarding the cross-frequency interactions that dominate the flow dynamics.

### A. Global amplification mechanisms

Here we consider the problem of finding the input  $\hat{f}$  over the set of frequencies  $\Omega_{\gamma}$  that excites the most energetic response through the harmonic resolvent  $H(i\gamma)$ . Mathematically, we seek

$$\hat{\psi}_1 = \arg\max_{\|\hat{f}\|=1} \|\boldsymbol{H}(i\gamma)\hat{f}\|. \tag{21}$$

It can be shown that the optimal forcing input  $\hat{\psi}_1$  is the first right singular vector of  $\mathbf{H}(i\gamma)$ , while the corresponding output, which we denote  $\hat{\phi}_1$ , is the first left singular vector of  $\mathbf{H}(i\gamma)$ . The first singular value  $\sigma_1(\gamma)$  can be understood as a gain, or amplification factor, on the leading input-output pair. This provides a measure of the sensitivity of the flow to perturbations with a given frequency content  $\Omega_{\gamma}$ . In particular, if  $\sigma_1(\gamma) > \sigma_1(\alpha)$  then we say that the flow is more sensitive to perturbations over  $\Omega_{\gamma}$  than to perturbations over  $\Omega_{\alpha}$ . We also remark that the optimal input and the optimal output identify spatially- and temporally-coherent signals whose temporal evolution can be reconstructed as follows,

$$\psi_1(t) = \sum_{\alpha \in \Omega_{\gamma}} \hat{\psi}_{1,\alpha} e^{i\alpha t} + c.c., \quad \phi_1(t) = \sum_{\alpha \in \Omega_{\gamma}} \hat{\phi}_{1,\alpha} e^{i\alpha t} + c.c.$$
 (22)

where c.c. denotes complex conjugate. It is worth pointing out that the complex conjugate is not necessary if  $\gamma = 1/2$  or  $\gamma = 0$ , since for any  $\alpha \in \Omega_{1/2}$  (resp.  $\Omega_0$ ) we also have  $-\alpha \in \Omega_{1/2}$  (resp.  $\Omega_0$ ). By recognizing that any element

 $\alpha \in \Omega_{\gamma}$  may be written as  $\alpha = \gamma + k$  for some  $k \in \mathbb{Z}$ , we also observe that the input and output modes in (22) may be expressed as EMP signals,

$$\psi_1(t) = e^{i\gamma t} \sum_{k \in \mathbb{Z}} \hat{\psi}_{1,\gamma+k} e^{ikt} + c.c., \quad \phi_1(t) = e^{i\gamma t} \sum_{k \in \mathbb{Z}} \hat{\phi}_{1,\gamma+k} e^{ikt} + c.c.$$
 (23)

The importance of the optimal input and output signals and of the corresponding singular value can be understood by writing the response  $\hat{q}$  to any input  $\hat{f}$  in terms of the singular value decomposition

$$\boldsymbol{H}(i\gamma) = \sum_{k} \sigma_{k} \hat{\boldsymbol{\phi}}_{k} \hat{\boldsymbol{\psi}}_{k}^{*}. \tag{24}$$

Assuming  $\sigma_1 \gg \sigma_2$  for simplicity, we can write

$$\hat{\mathbf{q}} = \mathbf{H}(i\gamma)\hat{\mathbf{f}} \approx \sigma_1 \hat{\phi}_1 \hat{\psi}_1^* \hat{\mathbf{f}}. \tag{25}$$

If the input  $\hat{f}$  has a non-zero projection onto  $\hat{\psi}_1$ , then  $\hat{q}$  will have a component in the direction of  $\hat{\phi}_1$ . In other words, the spatio-temporal evolution of the optimal response mode  $\phi_1(t)$  describes the flow structures that one expects to see in experiment or simulation when external perturbations force the system. The signal  $\psi_1(t)$ , on the other hand, identifies the spatio-temporal structures that are responsible for exciting the response  $\phi_1(t)$ . These modes can be used to understand the physics of the flow, but also to develop a control strategy to meet some control objective. For instance, if one wants to excite the flow response  $\phi_1(t)$ , then they would select a control input that aligns well with  $\psi_1(t)$ . Similarly, if one wants to suppress the spatio-tempotal waveform  $\phi_1(t)$ , they could design a controller that rejects disturbances that align with  $\psi_1(t)$ . Finally, it is worth observing that if  $\sigma_1 \gg 1$  (i.e., the flow is very sensitive to disturbances over the frequency set  $\Omega_{\gamma}$ ), then any  $\hat{f}$  with a small component along the direction  $\hat{\psi}_1$  will suffice to trigger a large response  $\hat{q}$  in the direction  $\hat{\phi}_1$ . We will see in section V that this is precisely why we observe vortex pairing in the jet flow.

It is important to remark that the cost function in the optimization problem in (21) may be expressed in terms of a weighted norm  $\|\hat{u}\|_{\hat{W}}^2 = \hat{u}^* \hat{W} \hat{u}$ , for some positive-definite weight  $\hat{W}$ . It is then straightforward to show that the solution  $\hat{\psi}_1$  is equal to the first right singular vector of the linear operator

$$\widetilde{\boldsymbol{H}}(i\gamma) = \hat{\boldsymbol{W}}^{1/2} \boldsymbol{H}(i\gamma) \hat{\boldsymbol{W}}^{-1/2}, \tag{26}$$

where  $\hat{W}^{1/2}$  denotes the square-root of  $\hat{W}$ . The physical interpretation of the optimal input and output modes and corresponding singular value remains unchanged.

### B. Cross-frequency amplification mechanisms

We have seen in the previous sections that the time-varying nature of the base flow introduces coupling between flow structures over the set  $\Omega_{\gamma}$ . We can therefore consider the problem of finding the input  $\hat{f}_{\beta}$  at frequency  $\beta \in \Omega_{\gamma}$ that excites the most energetic response  $\hat{q}_{\alpha}$  at frequency  $\alpha \in \Omega_{\gamma}$ . Letting  $H_{\alpha,\beta}(i\gamma)$  denote the block of the harmonic resolvent operator that maps inputs  $\hat{f}_{\beta}$  to outputs  $\hat{q}_{\alpha}$ , we seek

$$\hat{\psi}_{\beta,1} = \arg\max_{\|\hat{f}_{\beta}\|=1} \|\boldsymbol{H}_{\alpha,\beta}(i\gamma)\hat{f}_{\beta}\|. \tag{27}$$

As before, the optimal forcing input  $\psi_{\beta,1}$  is the first right singular vector of  $\mathbf{H}_{\alpha,\beta}(i\gamma)$ , the corresponding response, which we denote  $\hat{\phi}_{\alpha,1}$ , is the first left singular vector of  $\mathbf{H}_{\alpha,\beta}(i\gamma)$ , while the first singular value  $\sigma_{(\alpha,\beta),1}$  is a gain on the leading input-output pair. This singular value can be understood as a measure of how strongly inputs at frequency  $\beta$  affect the response of the flow at frequency  $\alpha$ . If  $\sigma_{(\alpha,\beta),1} > \sigma_{(\alpha,\eta),1}$  then we say that disturbances at frequency  $\beta \in \Omega_{\gamma}$  are more effective than disturbances at frequency  $\eta \in \Omega_{\gamma}$  at exciting an energetic response at  $\alpha \in \Omega_{\gamma}$ . The optimal input and output vectors identify spatially-coherent structures that oscillate at frequencies  $\beta$  and  $\alpha$ , respectively,

$$\psi_{1,\beta}(t) = \hat{\psi}_{1,\beta}e^{i\beta t} + c.c., \quad \phi_{1,\alpha}(t) = \hat{\phi}_{1,\alpha}e^{i\alpha t} + c.c.$$
 (28)

As in the previous section, the optimization problem (27) may be written with respect to a weighted inner product.

![](_page_8_Figure_1.jpeg)

![](_page_8_Figure_2.jpeg)

FIG. 2. In (a) we show the periodic solution of (29) with period  $T \approx 2\pi$ . The black marker indicates the temporal mean. In (b) we show the leading singular value of the harmonic resolvent  $H(i\gamma)$  for  $\gamma \in [0, \omega/2]$ . In (c) we show the leading singular value of the mean-based resolvent operator evaluated at frequency  $\gamma$ .

## IV. APPLICATION TO A TOY MODEL: THE RÖSSLER SYSTEM

In this section we demonstrate the harmonic resolvent framework on the Rössler system under conditions for which the dynamics are sensitive to period-doubling perturbations. We also present a comparison with the well-known resolvent analysis (i.e., linearization about the temporal mean) and we show how this analysis can fail to capture the qualitative sensitivity of the system to subharmonic perturbations. Consider the Rössler system in the form

$$\dot{x} = -y - z 
\dot{y} = x + 0.1y 
\dot{z} = 0.1 + z(x - c).$$
(29)

As discussed in [24], these dynamics undergo a cascade of period-doubling bifurcations as we sweep the bifurcation parameter c. Here, we fix c=5.3 and, using a Newton-based harmonic balancing method, we find that (29) admits a periodic solution with period  $T\approx 2\pi$ . This solution is shown in figure 2a. Our choice of parameter c is close to the critical value for one of the aforementioned period-doubling bifurcations, so we expect the system to be very sensitive to any disturbance with frequency  $\omega/2$ , where  $\omega=2\pi/T$  is the fundamental frequency. We are interested in verifying whether the harmonic resolvent analysis can be used to identify the frequency sets over which the base flow is sensitive to perturbations. We also wish to check if this framework accurately predicts the input-output dynamics of the flow under small-amplitude forcing. The harmonic resolvent results are compared against the predictions provided by the mean-based resolvent analysis, which is performed by linearizing the dynamics about the temporal mean of the T-periodic solution (see black marker in figure 2a).

Throughout this section, we let our periodic base flow for the harmonic resolvent analysis be given by the curve shown in figure 2a. The base flow has frequency content in  $\widetilde{\Omega} = \{-20\omega, \cdots, 0, \cdots, 20\omega\}$ , and we let the perturbations evolve over sets of frequencies  $\Omega_{\gamma} = \gamma + \{-30\omega, \cdots, 0, \cdots, 30\omega\}$ , where  $0 \le \gamma \le \omega/2$  by property II.2. We begin by computing the singular value decomposition of the harmonic resolvent operator  $\boldsymbol{H}(i\gamma)$ , defined in (19), and a plot of the first singular value over  $0 \le \gamma \le \omega/2$  is shown in figure 2b. Figure 2b shows a clear peak at  $\gamma = \omega/2$ , which suggests that the flow is very sensitive to disturbances over the set  $\Omega_{\omega/2}$ . By contrast, in figure 2c, the curve associated with the leading singular value of the mean-based resolvent operators suggests that the flow is sensitive to perturbations that oscillate at frequency  $\gamma \approx \omega$ .

We now try to verify whether the results provided by the harmonic resolvent framework convey reliable information about the nonlinear dynamics of the flow. In particular, we are interested in checking whether the flow is more sensitive to perturbations at frequency  $\omega$ , as indicated by the mean-resolvent analysis, or if it is more sensitive to perturbations over the set  $\Omega_{\omega/2}$ , as indicated by the harmonic resolvent analysis. We perform a numerical test by introducing a forcing term f'(t) to the governing equations (29),

$$\mathbf{f}'(t) = \varepsilon \mathbf{B} \cos(\alpha t). \tag{30}$$

Here  $\varepsilon = 3 \times 10^{-4}$ ,  $\mathbf{B} \in \mathbb{R}^3$  is a column vector of ones and we take  $\alpha$  to be either equal to  $5\omega/2 \in \Omega_{\omega/2}$  or to  $\omega$ . The results are shown in figures 3a and 3b for values  $\alpha = 5\omega/2$  and  $\alpha = \omega$ , respectively. We can see that the forcing

![](_page_9_Figure_1.jpeg)

![](_page_9_Figure_2.jpeg)

FIG. 3. We show (a) the forced nonlinear response (purple), the harmonic resolvent prediction (orange) and the mean-based resolvent prediction (green) for a forcing input (30) with  $\varepsilon = 3 \times 10^{-4}$  and  $\alpha = 5\omega/2$ . Panel (b) is the analog of (a) with  $\alpha = \omega$ .

input f'(t) triggers a much more energetic response when it oscillates at frequency  $5\omega/2$ . The same forcing input at frequency  $\omega$  excites a response that is roughly two orders of magnitude less energetic. This is consistent with the information provided by the singular values of the harmonic resolvent operators evaluated at different values of  $\gamma$ . In particular, the leading singular value  $\sigma_1(\omega/2)$  is roughly two orders of magnitude larger than  $\sigma_1(0)$  (see figure 2b). Figures 3a and 3b highlight additional details. First, the mean-based resolvent prediction shown in green, is only active at the forcing frequencies  $\alpha = 5\omega/2$  and  $\alpha = \omega$ , respectively. This is due to the fact that the mean-based resolvent operator does not describe cross-frequency interactions, since it is based on a linearization about a steady base flow. On the other hand, the harmonic resolvent prediction shown in orange is active at multiple frequencies over the sets  $\Omega_{\omega/2}$  and  $\Omega_0$ , respectively. We also observe that that the nonlinear response in figure 3a is active (although very small) at frequencies  $k\omega$  with k an integer. This is exclusively due to nonlinear interactions between perturbations. For instance, a perturbation at frequency  $\omega/2 \in \Omega_{\omega/2}$  can nonlinearly interact with itself to produce a perturbation at frequency  $\omega \notin \Omega_{\omega/2}$ . Nonlinear (i.e., second-order) interactions between perturbations are disregarded in the harmonic resolvent formulation, so the harmonic resolvent prediction in figure 3a is inactive at frequencies  $k\omega$ with integer k. We take this opportunity to observe that if the forcing amplitude  $\varepsilon$  in (30) becomes too large, then these second-order nonlinear effects start dominating the response of the system and the harmonic resolvent may lose predictive accuracy.

## V. APPLICATION TO AN INCOMPRESSIBLE FORCED AXISYMMETRIC JET

### A. Flow configuration

In this section we consider an incompressible axisymmetric jet forced by an axial inflow velocity. Throughout, velocities are non-dimensionalized by the jet centerline velocity and lengths are non-dimensionalized by the jet diameter. The flow is governed by the incompressible Navier-Stokes equation along with the continuity equation over the spatial domain  $\mathcal{D} = \{(z, r) | z \in [0, L_z], r \in [0, L_r]\}$ , with  $L_z = 20$  and  $L_r = 6$ . In particular, given the (dimensionless) axial velocity u, the radial velocity v and the pressure p, we have

$$\begin{aligned} &\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial z} + v \frac{\partial u}{\partial r} = -\frac{\partial p}{\partial z} + \frac{1}{Re} \left( \frac{1}{r} \frac{\partial}{\partial r} \left( r \frac{\partial u}{\partial r} \right) + \frac{\partial^2 u}{\partial z^2} \right) \\ &\frac{\partial v}{\partial t} + u \frac{\partial v}{\partial z} + v \frac{\partial v}{\partial r} = -\frac{\partial p}{\partial r} + \frac{1}{Re} \left( \frac{1}{r} \frac{\partial}{\partial r} \left( r \frac{\partial v}{\partial r} \right) - \frac{v}{r^2} + \frac{\partial^2 v}{\partial z^2} \right) \quad (31) \\ &\frac{\partial u}{\partial z} + \frac{1}{r} \frac{\partial (rv)}{\partial r} = 0. \end{aligned}$$

At the centerline r = 0 we impose axisymmetric boundary conditions, at the outflow and at the top boundary we impose a zero normal gradient boundary condition on both velocity components, and at the inflow we consider the

![](_page_10_Figure_1.jpeg)

![](_page_10_Figure_2.jpeg)

FIG. 4. Vorticity snapshots from the T-periodic solution at time t = 0 for (a) Reynolds number Re = 500 and (b) Reynolds number Re = 1250. The colorbar ranges from 0 to 9 and it has been slightly saturated to better visualize downstream vortices.

axial velocity profile

$$u(r, z = 0, t) = g(r) (1 + A\cos\omega t),$$
 (32)

where A is the non-dimensional forcing amplitude,  $\omega$  is the forcing frequency, and

$$g(r) = \frac{1}{2} \left\{ 1 - \tanh \left[ \frac{1}{4\theta_0} \left( r - \frac{1}{4r} \right) \right] \right\}. \tag{33}$$

The parameter  $\theta_0$  may be understood as a non-dimensional vorticity thickness of the incoming profile. The spatial domain is discretized on a fully-staggered grid of size  $N_z \times N_r = 800 \times 300$  using second-order finite differences. The advective term is treated using a third-order upwind-biased scheme. Given the fully-staggered nature of the grid, we do not require explicit pressure boundary conditions. The time-stepping Navier-Stokes solver was validated by reproducing the Floquet analysis in [11] for Re = 1000, and we registered less than 0.5% error on the least stable Floquet multiplier.

Throughout this section, we consider fixed values A = 0.05,  $\theta_0 = 0.025$  and  $\omega = 6\pi/5$ , but we consider several different Reynolds numbers: Re = 500, 750, 1000, 1250. This choice is motivated by the recent work in [11], where the authors studied the stability and modal/transient growth dynamics of the flow for a wide range of parameters. For the parameters considered herein, they observed that for  $Re \lesssim 1350$ , the Navier-Stokes equations admit a stable T-periodic solution (where  $T=2\pi/\omega$ ) characterized by unpaired vortices. Snapshots from two such solutions at Re = 500 and Re = 1250 are shown in figures 4a and 4b. For larger Reynolds number values, the T-periodic solution becomes linearly unstable and the flow settles onto a 2T-periodic orbit characterized by vortex pairing. However, the authors also observed that vortex pairing can be easily triggered by low-amplitude noise at Reynolds numbers well-below the critical Reynolds number for modal instability, and they attributed this behavior to the large-amplitude transient growth due to the non-normality of the underlying linear operator. Here, we confirm the non-normal nature of the linearized dynamics by studying the spectral proprties of the harmonic resolvent operator over all frequency sets  $\Omega_{\gamma}$ , for  $\gamma \in [0, \omega/2]$ . We also offer an additional explanation regarding the driving mechanisms behind vortex pairing. In particular, we will use the harmonic resolvent to uncover the fact that the pairing mechanism is driven by complicated frequency interactions between the subharmonic frequency  $\omega/2$  and its odd harmonics. Finally, we use nonlinear direct numerical simulations to demonstrate that vortex pairing is driven by the spatio-temporal structure defined by the first right singular vector of the harmonic resolvent evaluated at  $\gamma = 1/2$ . We also perform a mean-based resolvent analysis for comparison.

### B. Sensitivity analysis via the harmonic resolvent and mean-based resolvent

In this section we study the sensitivity of the flow to subharmonic perturbations in the proximity of the T-periodic base flow at four different Reynolds numbers: Re = 500, 750, 1000, 1250. We do so by performing the global (section III A) and cross-frequency (section III B) spectral analyses of the harmonic resolvent operators  $\mathbf{H}(i\gamma)$  for  $\gamma \in [0, \omega/2]$ . Recall that, as discussed at the end of section II C, the harmonic resolvent operators only need to be evaluated over the interval  $[0, \omega/2]$  in order to fully characterize the dynamics of perturbations. Throughout this section, we will take the base flow set to be  $\widetilde{\Omega} = \omega\{-4, \cdots, 0, \cdots, 4\}$ , and we will consider perturbations over frequency sets  $\Omega_{\gamma} = \gamma + \omega\{-6, \cdots, 0, \cdots, 6\}$ . We also perform the mean-based resolvent analysis by computing the singular value decomposition of the resolvent operator  $\mathbf{R}(i\gamma)$  for  $\gamma \in [0, \infty)$ . For all four Reynolds numbers, the periodic base flow (and hence the temporal mean) was computed via numerical integration of the Navier-Stokes equation using

the time-delayed feedback technique described in [25] to suppress transient growth and to accelerate convergence to the desired solution. It is also worth observing that since the inflow boundary condition determines the phase of the solution, the operator  $\hat{T}$  evaluated about the T-periodic solution of the Navier-Stokes has a trivial nullspace. Therefore, the harmonic resolvent in (19) can be safely defined in terms of  $\hat{T}$  instead of  $\hat{T}_{\Sigma}$ . Moving forward, we also restrict our attention to flow structures over the spatial domain  $\tilde{D} = \{(r, z) \mid 0 \le r \le 2.5, 0 \le z \le 20\}$ . We do so by defining two operators B and C that are equal to the identity on  $\tilde{D}$  and zero otherwise. This is done in order to mitigate spurious amplification mechanisms associated to mass entrainment from the top boundary located at r = 6. These manifest themselves as energetic zero-frequency input-output structures supported in close proximity of the top boundary. While entrainment is a physical phenomenon, we believe that the high sensitivity of the flow at the top boundary detected by the harmonic resolvent and by the mean-based resolvent is simply an artifact of domain truncation in the radial direction. In any case, we are not interested in amplification mechanisms far away from the shear layer so, for this example, we consider the input-output operator

$$H(i\gamma) = \hat{C}(i\gamma I - \hat{T})^{-1}\hat{B}, \tag{34}$$

where  $\hat{B} = I \otimes B$  and  $\hat{C} = I \otimes C$  (that is, block diagonal matrices formed from B and C). Similarly, for the mean-based resolvent analysis we consider the operator

$$\mathbf{R}(i\gamma) = \mathbf{C} \left( i\gamma \mathbf{I} - \hat{\mathbf{A}}_0 \right)^{-1} \mathbf{B},\tag{35}$$

where  $\hat{A}_0$  denotes the linearization of the Navier-Stokes equation about the temporal mean. In the presence of the input and output matrices  $\hat{B}$  and  $\hat{C}$ , the operator  $H(i\gamma)$  in (34) is precisely the harmonic transfer function discussed in [22], while  $R(i\alpha)$  in (35) is the well-known linear time-invariant transfer function.

The spectral analysis of  $\mathbf{H}(i\gamma)$  and  $\mathbf{R}(i\alpha)$  is performed by solving the optimization problem (21) in the radially-weighted norm

$$\|\boldsymbol{u}\|^2 = \int_{\mathcal{D}} |\boldsymbol{u}(r,z)|^2 r \, \mathrm{d}r \, \mathrm{d}z. \tag{36}$$

In other words, letting W denote the matrix representation of the weight, and letting  $\hat{W} = I \otimes W$ , we compute the singular values of

$$\widetilde{\boldsymbol{H}}(i\gamma) = \hat{\boldsymbol{W}}^{1/2} \boldsymbol{H}(i\gamma) \hat{\boldsymbol{W}}^{-1/2}, \quad \widetilde{\boldsymbol{R}}(i\gamma) = \boldsymbol{W}^{1/2} \boldsymbol{R}(i\gamma) \boldsymbol{W}^{-1/2}.$$
(37)

Plots of the first singular value of  $\widetilde{\boldsymbol{H}}(i\gamma)$  and  $\widetilde{\boldsymbol{R}}(i\gamma)$  across the considered Reynolds numbers are shown in figures 5a and 5b, respectively. For all Reynolds numbers, the singular value  $\sigma_1$  of the harmonic transfer function  $\widetilde{\boldsymbol{H}}(i\gamma)$  peaks at  $\gamma = \omega/2$ , which suggests that the flow is most sensitive to perturbations over the frequency set  $\Omega_{\omega/2} = \omega/2 + \omega\{-6,\ldots,0,\ldots,6\}$ . On the other hand, for all Reynolds numbers, the singular value of the mean-based transfer function  $\widetilde{\boldsymbol{R}}(i\gamma)$  peaks at  $\gamma \approx \omega/2$ , and this suggests that the flow is sensitive to flow structures that oscillate at the single frequency  $\gamma \approx \omega/2$ . Both analyses show an increase in sensitivity as the Reynolds number is increased, but the harmonic transfer function shows a much steeper increase in the magnitude of the singular value. In order to explain this discrepancy, we must study how frequencies within the set  $\Omega_{\omega/2}$  interact with one another, and we do so by computing the cross-frequency singular values of  $\widetilde{\boldsymbol{H}}(i\omega/2)$  (see section III B).

Plots of the cross-singular values of  $\dot{H}(i\omega/2)$  for Re=500, 750, 1000, 1250 are shown in figure 6. The blocks are color-coded according to the quantity

$$E_{k,j} = \frac{\sigma_{(k,j),1}^2}{\max_{(l,m)} \sigma_{(l,m),1}^2},$$
(38)

where  $\sigma_{(k,j),1}$  denotes the leading singular value of the block  $\widetilde{H}_{(k,j)}(i\omega/2)$  of the harmonic transfer function. Large values (i.e., darker color) of  $E_{k,j}$  indicate that forcing the flow at frequency  $(1/2+j)\omega$  excites a very energetic response at frequency  $(1/2+k)\omega$ . At Re=500, in figure 6a, we see that self interactions at frequency  $\omega/2$  dominate the input-output behavior, while other cross-frequency interactions are weaker. As we increase the Reynolds number, we observe that self interactions at  $\omega/2$  remain strong, but the cross interaction between  $\omega/2$  and its odd harmonics becomes progressively stronger and unidirectional. By unidirectional we mean that while, for instance, forcing at  $3\omega/2$  excites a very amplified response at  $\omega/2$ , the converse is not true.

![](_page_12_Figure_1.jpeg)

![](_page_12_Figure_2.jpeg)

FIG. 5. Panel (a) shows the first singular value of the harmonic transfer function  $\widetilde{H}(i\gamma)$  (37) as a function of  $\gamma$  for four different Reynolds numbers Re = 500, 750, 1000, 1250. Panel (b) shows the first singular value of the mean-based transfer function  $\widetilde{R}(i\gamma)$  (37) as a function of  $\gamma$  for four different Reynolds numbers Re = 500, 750, 1000, 1250.

![](_page_12_Figure_4.jpeg)

FIG. 6. For Reynolds numbers Re = 500, 750, 1000, 1250, we plot the quantity  $E_{k,j}$  defined in (38).

The information contained within these plots sheds light on the reason why the harmonic transfer function  $\widetilde{H}(i\omega/2)$  exhibited larger peaks in its singular value compared to the mean-based transfer function  $\widetilde{R}(i\omega/2)$  (see figures 5a and 5b). In particular, while the peak in  $\widetilde{R}(i\omega/2)$  is symptomatic of strong self interactions at  $\omega/2$ , the much larger peaks at  $\widetilde{H}(i\omega/2)$  are due to the fact that other cross-frequency interactions (i.e., those highlighted in figure 6) are contributing the input-output dynamics of the flow. We also offer the conclusions drawn from figure 6 as an explanation for the observations made in [11], where the authors observed how low-amplitude noise could easily trigger vortex pairing even in linearly-stable configurations. First and foremost, we remark that paired vortices are energetic flow structures that oscillate at frequency  $\omega/2$ , and we also recall that a noisy signal has frequency content across the entire frequency spectrum. Given the information in figures 6b through 6d, it becomes clear that the flow amplifies many frequencies from the spectrum of the noisy forcing profile, and it selectively injects energy into flow structures with frequency  $\omega/2$ . In other words, a non-trivial amount of the energy of the noisy forcing profile gets transferred to flow structures at frequency  $\omega/2$ , and we therefore observe vortex pairing.

In order to better understand how the sensitivity of the flow varies as a function of the Reynolds number, it is instructive to look at the so-called input modes (i.e., the first right singular vector) of the harmonic transfer function and of the mean-based transfer function. These are shown in figures 7 and 8 for Reynolds numbers Re = 500 and

![](_page_13_Figure_1.jpeg)

FIG. 7. On the left we show the vorticity field from the  $\omega/2$ -,  $3\omega/2$ - and  $5\omega/2$ -entries of the first right singular vector (i.e., the first input mode) of  $\widetilde{\boldsymbol{H}}(i\omega/2)$  at Re=500. The right panels are the analog of the left ones for the mean-based transfer functions  $\widetilde{\boldsymbol{R}}(i\omega/2)$ ,  $\widetilde{\boldsymbol{R}}(i3\omega/2)$  and  $\widetilde{\boldsymbol{R}}(i5\omega/2)$ . The colorbar is not shown on the right because these frequencies are decoupled and the modes are scaled arbitrarily.

Re = 1250, respectively. In particular, on the left-hand side panels we show the vorticity fields from the entries  $\omega/2$ ,  $3\omega/2$  and  $5\omega/2$  entries of the first input mode of  $\hat{H}(i\omega/2)$ . On the right-hand side panels we show the vorticity field from the first right singular vector of the mean-based transfer functions  $\vec{R}(i\gamma)$  evaluated at  $\gamma = \omega/2$ ,  $3\omega/2$ ,  $5\omega/2$ . It is worth recalling that the structures in the left-hand side panels (i.e., those associated with the harmonic transfer function) are spatially- and temporally-coherent since they are coupled to one another via the base flow. On the other hand, structures associated with the mean-based transfer function are temporally decoupled, so their relative magnitude is arbitrary. Comparing these two sets of figures allows us to draw several conclusions regarding the amplification mechanisms of the flow under consideration. First, we observe that at low Reynolds numbers, there is no substantial qualitative difference between the mean-based and harmonic modes at low frequencies. This suggests that the amplification mechanisms are primarily driven by self interactions via the mean flow. This can also be inferred from the cross-frequency plot in figure 6a, where we see that the input-output dynamics are dominated by self interactions of structures at frequency  $\omega/2$ . Conversely, at higher Reynolds numbers, we see that the mean-based and harmonic input structures are qualitatively different at all frequencies. This is most likely due to the fact that cross-frequency interactions (see, e.g., figure 6d) start dominating the input-output dynamics of the flow and the temporal mean no longer provides qualitatively meaningful information regarding the dynamics of the flow. It is also useful to observe that as the Reynolds number is increased, the spatial support of the input modes becomes more concentrated near the shear layer at r = 1/2 and  $0 \le z \le 2$ .

### C. The driving mechanism behind vortex pairing: case study at Re = 1000 and Re = 1250

In this section we explore the driving mechanism behind vortex pairing, and in particular, we show that this nonlinear phenomenon is driven primarily by a linear mechanism. Specifically, although nearly all perturbations of this flow lead to vortex pairing [11], we show that if one projects out the component of the forcing aligned with the leading input mode of the harmonic transfer function at the pairing frequency, then no pairing is observed.

We consider Re=1000 and Re=1250, and we let  $\hat{\psi}=\left(\dots,\hat{\psi}_{(-1+1/2)\omega},\hat{\psi}_{(0+1/2)\omega},\hat{\psi}_{(1+1/2)\omega},\dots\right)$  denote the first right singular vector (i.e., the optimal input mode) of the harmonic transfer function  $\widetilde{H}(i\omega/2)$ . We remark that, at Re=1250, the vorticity fields from the Fourier coefficients  $\omega/2$ ,  $3\omega/2$  and  $5\omega/2$  in  $\hat{\psi}$  are shown in the left-hand side panels of figure 8. Similarly, we let  $\hat{\xi}_{\omega/2}$  denote the first right singular vector of the mean-based transfer function  $\widetilde{R}(i\omega/2)$ . The vorticity field corresponding to this mode at Re=1250 is shown in the top right panel of figure 8. We now demonstrate that, for this flow, the spatio-temporal signal defined by  $\hat{\psi}$  drives the vortex-pairing mechanism. In order to do so, we perform two numerical experiments at each Reynolds number.

![](_page_14_Figure_1.jpeg)

FIG. 8. Analog of figure 7 for Re = 1250.

1. In the first numerical experiment, we first embed  $\hat{\boldsymbol{\xi}}_{\omega/2}$  into a vector  $\hat{\boldsymbol{\xi}} = \left(\dots, 0, \hat{\boldsymbol{\xi}}_{-\omega/2}, \hat{\boldsymbol{\xi}}_{\omega/2}, 0, \dots\right)$  with the same dimension as  $\hat{\boldsymbol{\psi}}$ , and we scale  $\hat{\boldsymbol{\xi}}$  so that  $\|\hat{\boldsymbol{\xi}}\|_E = \varepsilon$ . Here, the subscript E denotes the Euclidean norm, since the weight  $\boldsymbol{W}$  is already accounted for in the definition of  $\tilde{\boldsymbol{R}}$  in (37). We then compute the nonlinear forced response of the Navier-Stokes equation to the forcing input

$$\boldsymbol{\xi}(t) = \sum_{\alpha \in \Omega_{\omega/2}} \hat{\boldsymbol{\xi}}_{\alpha} e^{i\alpha t} = 2 \operatorname{Real}(\hat{\boldsymbol{\xi}}_{\omega/2} e^{i\omega t/2}). \tag{39}$$

2. In the second numerical experiment, we project out the component of  $\hat{\xi}$  in the direction of  $\hat{\psi}$  according to

$$\hat{\boldsymbol{\xi}}_{\perp} = \hat{\boldsymbol{\xi}} - \hat{\boldsymbol{\psi}} \langle \hat{\boldsymbol{\xi}}, \hat{\boldsymbol{\psi}} \rangle_E, \tag{40}$$

and then we scale the resulting vector  $\hat{\boldsymbol{\xi}}_{\perp}$  so that  $\|\hat{\boldsymbol{\xi}}_{\perp}\|_{E} = \varepsilon$ . Finally, we compute the nonlinear forced response of the Navier-Stokes equation to the forcing input

$$\boldsymbol{\xi}_{\perp}(t) = \sum_{\alpha \in \Omega_{\omega/2}} \hat{\boldsymbol{\xi}}_{\perp,\alpha} e^{i\alpha t}. \tag{41}$$

In other words, in the first experiment we compute the nonlinear response of the flow to a 2T-periodic forcing input that has a non-zero component along the direction  $\hat{\psi}$ . It is worth mentioning that, at both Reynolds numbers, the projection of the unit vector in the direction of  $\hat{\boldsymbol{\xi}}$  onto  $\hat{\boldsymbol{\psi}}$  is approximately 0.77. In the second experiment we compute the nonlinear response to a 2T-periodic input that has no component along  $\hat{\psi}$ . In both cases, the forcing inputs are normalized to have the same magnitude  $\varepsilon$ . The nonlinear response is computed by integrating the Navier-Stokes equations to time t = 40T with initial condition given by the base flow at time t = 0. The results are shown in figures 9a and 9b for Re = 1000 and  $\varepsilon = 5 \times 10^{-3}$ . Figures 9c and 9d show the analog for Re = 1250 and  $\varepsilon = 10^{-3}$ . Remarkably, in both cases we see that if the forcing input has no component along the direction of  $\hat{\psi}$  then no vortex pairing is observed. We must therefore conclude that vortex pairing is triggered by the component of  $\xi$  along the direction of  $\psi$ . In other words, the spatio-temporal signal defined by  $\psi$  drives the pairing mechanism. The implications of this result are far-reaching. First and foremost, it means that a fundamentally nonlinear phenomenon such as vortex pairing is driven by a linear mechanism. Secondly, vortex pairing is driven by the exchange of energy between the unsteady structures in the base flow and the perturbations, and a single input mode seems to be responsible for nearly all the energy exchange. By contrast, if it were driven solely by the exchange of energy between the perturbations and the temporal mean then the mean-based input mode  $\hat{\boldsymbol{\xi}}$  would be perfectly aligned with the harmonic input mode  $\hat{\boldsymbol{\psi}}$ . Finally, we wish to stress the fact that this information can be used to develop control strategies. In fact, we have just demonstrated that if one wished to suppress vortex pairing, then one would have to design a controller that rejects disturbances that align with  $\tilde{\psi}$ .

![](_page_15_Figure_1.jpeg)

FIG. 9. In panels (a) and (b) we show the vorticity field from the nonlinear response to inputs (39) and (41), respectively, at time t = 40T, Re = 1000 and forcing amplitude  $\varepsilon = 5 \times 10^{-3}$ . Panels (c) and (d) show the analog for Re = 1250 and forcing amplitude  $\varepsilon = 10^{-3}$ . The colorbar is shown in figure 4a.

## VI. CONCLUSION

We presented an extension of the harmonic resolvent framework to study the dynamics of subharmonic perturbations about a time-periodic base flow. In particular, by linearizing the Navier-Stokes equations about a T-periodic base flow and by seeking perturbed solutions with period nT, one may define a family of input-output operators  $\boldsymbol{H}(i\gamma)$  parameterized by a scalar  $\gamma \in [0, \omega/2]$ , where  $\omega = 2\pi/T$ . The operator  $\boldsymbol{H}(i\gamma)$  is the harmonic resolvent operator over the frequency set  $\Omega_{\gamma} = \gamma + \omega\{\dots, -1, 0, 1, \dots\}$ , and it is closely related to the harmonic transfer function discussed in [22]. When  $\gamma = 0$  (i.e., when the perturbations have the same period as the base flow), we recover the operator discussed in [8] (however, here we have used a slightly different projection operator). The singular value decomposition of  $\boldsymbol{H}(i\gamma)$  at different values of  $\gamma$  can be used to study the input-output properties of the flow in the proximity of the base flow. Moreover, since  $\boldsymbol{H}(i\gamma)$  accounts for the first-order triadic frequency coupling between the perturbations and the base flow, it is also possible to study the input-output dynamics of the flow at selected frequencies of interest. We demonstrated this method on a simple toy model and then we used it to shed light on the driving mechanism behind vortex pairing in an incompressible forced axisymmetric jet.

## ACKNOWLEDGMENTS

The authors wish to thank Samuel Otto for insightful comments that helped improve the presentation of this paper. We also wish to thank an anonymous reviewer who suggested a more thorough inspection of the input modes, which led to the results of section V C. We gratefully acknowledge support from the Air Force Office of Scientific Research, award FA9550-19-1-0005.

## Appendix A: Computing the vectors $\hat{v}$ and $\hat{u}$

In this appendix we show how to compute the vectors  $\hat{\boldsymbol{v}}$  and  $\hat{\boldsymbol{u}}$  that define the oblique projection  $\hat{\boldsymbol{P}}$  in (18). As discussed in section II C, when the base flow  $\boldsymbol{Q}(t)$  and its phase shift  $\boldsymbol{Q}(t+\varepsilon)$  satisfy (1),  $\hat{\boldsymbol{T}}$  is singular with its nullspace spanned by the phase shift direction  $\mathcal{F}_0(\mathrm{d}\boldsymbol{Q}(t)/\mathrm{d}t)$ . However, truncation errors and numerical errors are likely to perturb this singularity so that, in practice, the operator  $\hat{\boldsymbol{T}}$  is not singular to machine precision. Here we seek a procedure to calculate  $\hat{\boldsymbol{v}}$  and  $\hat{\boldsymbol{u}}$  that is independent of the singular/non-singular nature of the matrix  $\hat{\boldsymbol{T}}$ . The choice of  $\hat{\boldsymbol{v}}$  is very natural: since we are ultimately interested in removing the direction of phase shift, it is appropriate to simply define  $\hat{\boldsymbol{v}} \coloneqq \mathcal{F}_0(\mathrm{d}\boldsymbol{Q}(t)/\mathrm{d}t)$ . That is, we let  $\hat{\boldsymbol{v}}$  be the vector containing the Fourier coefficients of the time-derivative of the base flow. When  $\hat{\boldsymbol{T}}$  is exactly singular,  $\hat{\boldsymbol{v}}$  spans the nullspace of  $\hat{\boldsymbol{T}}$ . When  $\hat{\boldsymbol{T}}$  is not exactly singular due to numerical and truncation errors we have  $\|\hat{\boldsymbol{T}}\hat{\boldsymbol{v}}\| \ll 1$ .

Computing the vector  $\hat{\boldsymbol{u}}$  is slightly more involved. As discussed in section II C, when  $\hat{\boldsymbol{T}}$  is singular we would like to choose  $\hat{\boldsymbol{u}}$  so that  $\hat{\boldsymbol{T}}^*\hat{\boldsymbol{u}} = 0$ . When  $\hat{\boldsymbol{T}}$  is not exactly singular, we seek an appropriate  $\hat{\boldsymbol{u}}$  such that  $\|\hat{\boldsymbol{T}}^*\hat{\boldsymbol{u}}\| \ll 1$ . Both

scenarios may be handled by letting  $\hat{u}$  be the solution of the optimization problem below

$$\hat{\boldsymbol{u}} = \arg\min_{\hat{\boldsymbol{q}}} \frac{1}{2} ||\hat{\boldsymbol{T}}^* \hat{\boldsymbol{q}}||^2$$
s.t.  $\langle \hat{\boldsymbol{q}}, \hat{\boldsymbol{v}} \rangle = 1$ . (A1)

The equality constraint is chosen such that  $\hat{u}$  and  $\hat{v}$  may be used to define an oblique projection. It can be shown that the optimal solution  $\hat{u}$  solves the problem

$$\hat{\boldsymbol{T}}\hat{\boldsymbol{T}}^*\hat{\boldsymbol{u}} = \lambda\hat{\boldsymbol{v}} 
\hat{\boldsymbol{u}}^*\hat{\boldsymbol{v}} = 1,$$
(A2)

where  $\lambda \geq 0$  is a Lagrange multiplier. In matrix form, we may write the system of equations above as

$$\begin{bmatrix} \hat{T}\hat{T}^* & -\hat{v} \\ \hat{v}^* & 0 \end{bmatrix} \begin{bmatrix} \hat{u} \\ \lambda \end{bmatrix} = \begin{bmatrix} 0 \\ 1 \end{bmatrix}. \tag{A3}$$

Thus, the desired  $\hat{u}$  is obtained by solving (A3). The structure of (A3) is reminiscent of the method discussed in [26] for the computation of Landau constants. It is instructive to observe that the augmented matrix in (A3) is invertible even when  $\hat{T}$  is singular with its nullspace spanned by  $\hat{v}$ . It is also reassuring to observe that when  $\hat{T}$  is singular, the optimal solution  $\hat{u}$  lies in the nullspace of  $\hat{T}^*$ , and the objective function attains its minimum at zero.

## Appendix B: Computing with the harmonic resolvent $H(i\gamma)$

In this appendix we discuss the efficient computation of the singular value decomposition (SVD) of the operator  $\boldsymbol{H}(i\gamma)$  in (19). Computing the SVD of  $\boldsymbol{\widetilde{H}}(i\gamma)$  in (26) may be done in an analogous fashion. Henceforth, we assume that all the infinite-dimensional linear operators introduced in section II have been truncated, and they may therefore be understood as matrices. The SVD is computed using one of the randomized SVD algorithms in [27], and it requires solving linear systems of the form

$$(i\gamma I - \hat{T}_{\Sigma})\hat{q} = \hat{f}, \quad (i\gamma I - \hat{T}_{\Sigma})^*\hat{z} = \hat{g},$$
 (B1)

where  $\hat{f}$ ,  $\hat{q} \in \Sigma := \text{Range}(\hat{P})$  and  $\hat{g}$ ,  $\hat{z} \in \text{Range}(\hat{P}^*)$ , and  $\hat{P}$  is defined in (18). It is straightforward to verify that

$$\hat{T}_{\Sigma} = \hat{T}\hat{P} = \hat{P}\hat{T}\hat{P},\tag{B2}$$

since  $\Sigma$  is an invariant subspace of  $\hat{T}$ . The linear systems in (B1) are not particularly well-suited for computation since  $\hat{T}_{\Sigma}$  is a dense matrix. Instead, we would like to obtain the solutions  $\hat{q}$  and  $\hat{z}$  in terms of  $\hat{T}$ , which is a very sparse matrix that can be assembled and stored in memory. Substituting  $\hat{T}_{\Sigma} = \hat{T}\hat{P}$  in the first linear system (B1), using the fact that  $\hat{P} = I$  on  $\Sigma$ , and observing that  $\hat{f} = \hat{P}\hat{f}$  (since  $\hat{f} \in \Sigma$ ), we obtain

$$(i\gamma I - \hat{T})\hat{P}\hat{q} = \hat{P}\hat{f}.$$
 (B3)

In a similar fashion, the second linear system in (B1) may be written in the form

$$(i\gamma \mathbf{I} - \hat{\mathbf{T}})^* \hat{\mathbf{P}}^* \hat{\mathbf{z}} = \hat{\mathbf{P}}^* \hat{\mathbf{g}}.$$
 (B4)

To solve (B3) for  $\hat{q}$  when  $\hat{f}$  is a random vector (as in the randomized SVD algorithm mentioned above), we proceed as follows:

- Compute  $\hat{f}_{\Sigma} = \hat{P}\hat{f}$ . This operation is performed without assembling  $\hat{P}$  explicitly. The action of  $\hat{P}$  on a vector may be easily obtained using vector-vector products (see definition of  $\hat{P}$  in (18) and appendix A).
- Solve  $(i\gamma I \hat{T})q = \hat{f}_{\Sigma}$  for q. This can be done either iteratively or via direct solution using an LU decomposition. We recommend the use of MUMPS for large-scale LU decompositions. When  $\gamma = 0$  and  $\hat{T}$  is exactly singular, an LU decomposition cannot be performed, but Krylov-based methods will have no problems finding a solution, since  $\hat{f}_{\Sigma} \in \Sigma \subseteq \text{Range}(\hat{T})$ .

• Compute <sup>q</sup><sup>ˆ</sup> <sup>=</sup> P q<sup>ˆ</sup> . This step is not necessary if (iγ<sup>I</sup> <sup>−</sup>Tˆ) is full-rank, since <sup>f</sup><sup>Σ</sup> <sup>∈</sup> Σ so <sup>q</sup> must also lie in Σ (since Σ is an invariant subspace of Tˆ). However, it might be useful to perform this operation anyways, in order to have <sup>q</sup><sup>ˆ</sup> <sup>∈</sup> Σ to machine precision. It is important to observe that this step is necessary if <sup>γ</sup> = 0 and <sup>T</sup><sup>ˆ</sup> is exactly singular, since the solution to (B3) is not unique and Krylov-based solvers may find one with some component along the nullspace of Tˆ (which is contained in the nullspace of Pˆ).

Solving (B4) may be done in a similar way.

- [1] P. J. Schmid and D. S. Henningson, Stability and transition in shear flows (Springer, 2001).
- [2] M. R. Jovanovi´c and B. Bamieh, Componentwise energy amplification in channel flows, J. Fluid Mech. 534, 145 (2005).
- [3] B. J. McKeon and A. S. Sharma, A critical-layer framework for turbulent pipe flow, J. Fluid Mech. 658, 336 (2010).
- [4] M. Luhar, A. S. Sharma, and B. J. McKeon, Opposition control within the resolvent analysis framework, J. Fluid Mech. 749, 597–626 (2014).
- [5] C.-A. Yeh and K. Taira, Resolvent-analysis-based design of airfoil separation control, J. Fluid Mech. 867, 572–610 (2019).
- [6] K. Rosenberg, S. Symon, and B. J. McKeon, Role of parasitic modes in nonlinear closure via the resolvent feedback loop, Phys. Rev. Fluids 4, 052601 (2019).
- [7] C. Liu and D. F. Gayme, Structured input–output analysis of transitional wall-bounded flows, J. Fluid Mech. 927, A25 (2021).
- [8] A. Padovan, S. E. Otto, and C. W. Rowley, Analysis of amplification mechanisms and cross-frequency interactions in nonlinear flows via the harmonic resolvent, J. Fluid Mech. 900, A14 (2020).
- [9] C.-M. Ho and L.-S. Huang, Subharmonics and vortex merging in mixing layers, J. Fluid Mech. 119, 443 (1981).
- [10] H. S. Husain, Experiments on subharmonic resonance in a shear layer, J. Fluid Mech. 304, 343 (1995).
- [11] L. Shaabani-Ardali, D. Sipp, and L. Lesshafft, Vortex pairing in jets as a global Floquet instability: modal and transient dynamics, J. Fluid Mech. 862, 951 (2019).
- [12] G. Rigas, D. Sipp, and T. Colonius, Nonlinear input/output analysis: application to boundary layer transition, J. Fluid Mech. 911, A15 (2021).
- [13] D. Barkley and R. D. Henderson, Three-dimensional Floquet stability analysis of the wake of a circular cylinder, J. Fluid Mech. 322, 215 (1996).
- [14] L. Shaabani-Ardali, D. Sipp, and L. Lesshafft, Optimal triggering of jet bifurcation: an example of optimal forcing applied to a time-periodic base flow, J. Fluid Mech. 885, A34 (2020).
- [15] B. Pier and P. J. Schmid, Optimal energy growth in pulsatile channel and pipe flows, J. Fluid Mech. 926, A11 (2021).
- [16] M. R. Jovanovi´c and M. Fardad, H<sup>2</sup> norm of linear time-periodic systems: A perturbation analysis, Automatica 44, 2090 (2008).
- [17] R. Moarref and M. R. Jovanovi´c, Controlling the onset of turbulence by streamwise travelling waves. part 1. receptivity analysis, Journal of Fluid Mechanics 663, 70–99 (2010).
- [18] R. Moarref and M. R. Jovanovi´c, Model-based design of transverse wall oscillations for turbulent drag reduction, Journal of Fluid Mechanics 707, 205–240 (2012).
- [19] A. Chavarin and M. Luhar, Resolvent analysis for turbulent channel flow with riblets, AIAA Journal 58, 589 (2020).
- [20] W. Ran, A. Zare, and M. R. Jovanovi´c, Model-based design of riblets for turbulent drag reduction, Journal of Fluid Mechanics 906, A7 (2021).
- [21] M. R. Jovanovi´c, From bypass transition to flow control and data-driven turbulence modeling: An input–output viewpoint, Annual Review of Fluid Mechanics 53, 311 (2021).
- [22] N. M. Wereley, Analysis and control of linear periodically time varying systems, Ph.D. thesis, Massachusetts Institute of Technology (1991).
- [23] H. K. Khalil, Nonlinear Systems, 3rd ed. (Prentice Hall, 2002).
- [24] J. J. Bramburger, J. N. Kutz, and S. L. Brunton, Data-driven stabilization of periodic orbits, IEEE Access 9, 43504–43521 (2021).
- [25] L. Shaabani-Ardali, D. Sipp, and L. Lesshafft, Time-delayed feedback technique for suppressing instabilities in time-periodic flow, Phys. Rev. Fluids 2, 113904 (2017).
- [26] J. Crouch and T. Herbert, A note on the calculation of Landau constants, Physics of Fluids A: Fluid Dynamics 283 (1993).
- [27] N. Halko, P. G. Martinsson, and J. A. Tropp, Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions, SIAM Review 53, 217 (2011).