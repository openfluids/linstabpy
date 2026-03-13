# Efficient harmonic resolvent analysis via time-stepping

Ali Farghadan<sup>1</sup> , Jung Jung<sup>1</sup> , Rutvij Bhagwat<sup>1</sup> , Aaron Towne<sup>∗</sup><sup>1</sup>

<sup>1</sup>Department of Mechanical Engineering, University of Michigan, Ann Arbor, MI, USA

## Abstract.

We present an extension of the RSVD-∆t algorithm initially developed for resolvent analysis of statistically stationary flows to handle harmonic resolvent analysis of timeperiodic flows. The harmonic resolvent operator, as proposed by [Padovan et al.](#page-25-0) [\[2020\]](#page-25-0), characterizes the linearized dynamics of time-periodic flows in the frequency domain, and its singular value decomposition reveals forcing and response modes with optimal energetic gain. However, computing harmonic resolvent modes poses challenges due to (i) the coupling of all N<sup>ω</sup> retained frequencies into a single harmonic resolvent operator and (ii) the singularity or near-singularity of the operator, making harmonic resolvent analysis considerably more computationally expensive than a standard resolvent analysis. To overcome these challenges, the RSVD-∆t algorithm leverages time stepping of the underlying time-periodic linearized Navier-Stokes operator, which is N<sup>ω</sup> times smaller than the harmonic resolvent operator, to compute the action of the harmonic resolvent operator. We develop strategies to minimize the algorithm's CPU and memory consumption, and our results demonstrate that these costs scale linearly with the problem dimension. We validate the RSVD-∆t algorithm by computing modes for a periodically varying Ginzburg-Landau equation and demonstrate its performance using the flow over an airfoil.

# 1 Introduction

Model reduction plays a critical role in the study of fluid mechanics due to the complex and highdimensional nature of fluid flows, especially turbulent ones. In particular, modal decompositions, both data-driven and equation-based, have proven to be effective at identifying low-dimensional sets of modes associated with interpretable coherent structures that significantly influence quantities of engineering interest such as kinetic energy, heat and mass transfer, and noise emissions [\[Taira](#page-25-1) [et al.,](#page-25-1) [2017,](#page-25-1) [Towne et al.,](#page-25-2) [2018\]](#page-25-2). In particular, resolvent analysis has been extensively employed to comprehend, model, and control statistically stationary turbulent flows such as wall-bounded turbulence [\[Dawson and McKeon,](#page-25-3) [2019\]](#page-25-3), turbulent boundary layer [\[Sipp and Marquet,](#page-25-4) [2013\]](#page-25-4), and jet flows [\[Pickering et al.,](#page-25-5) [2021\]](#page-25-5). However, real-world fluid systems often exhibit non-stationary

<sup>∗</sup>Email address for correspondence: towne@umich.edu

behavior, including periodicity, rendering resolvent analysis less effective. Examples of flows exhibiting periodic behavior include internal waves in a stratified fluid [\[Troy and Koseff,](#page-25-6) [2005\]](#page-25-6), vortex shedding in the wake of a bluff body [\[Giannenas et al.,](#page-25-7) [2022\]](#page-25-7), and pulsatile blood flow [\[Farghadan](#page-25-8) [and Arzani,](#page-25-8) [2019\]](#page-25-8). Recently, an extension of resolvent analysis, known as harmonic resolvent analysis, has been developed by [Padovan et al.](#page-25-0) [\[2020\]](#page-25-0). This extension characterizes the perturbation forcing and responses around a periodic base flow, providing a more effective tool for analyzing, modeling, and controlling periodically time-varying flows. The harmonic resolvent operator can be considered a special case of the harmonic transfer function introduced by [Wereley](#page-26-0) [\[1990\]](#page-26-0), wherein the operator exhibits a dominant frequency.

Investigating the harmonic resolvent operator is one way to identify the optimal spatio-temporal perturbations and their corresponding responses in fluid systems. Unlike the resolvent operator, the harmonic resolvent operator takes into consideration the periodicity of the base flow by representing it in the form of a Fourier series. This allows for characterization of the dynamic behavior of the system, particularly with respect to its spectral components and it provides a means to understand and explore the interactions between input and output modes through the linearized dynamics. Harmonic resolvent analysis determines the triadic interactions between the input frequency ω1, the base flow frequency ω1−ω2, and the response frequency ω2. The number of Fourier modes of the base flow is determined by its power spectral density (PSD), which in turn determines the number of triads interacting with the forcing. Typically, periodic flows with a dominant periodicity exhibit a multimodal nature with their most energetic modes being harmonics of the base frequency. When the base flow is statistically stationary, linearization around the temporal average is sufficient, and traditional resolvent analysis arises as a special case.

Recent studies have expanded on this research area to model and control the dynamics of turbulent flows with periodic motions. [Lin et al.](#page-26-1) [\[2023\]](#page-26-1) applied the harmonic resolvent framework for linear time-periodic systems with more than one dominant frequency, offering a tool to study the flow control of a plunging cylinder. In a distinct approach, [Franceschini et al.](#page-26-2) [\[2022\]](#page-26-2) proposed a novel method aimed at capturing the phase dependency of small-scale turbulent phenomena in flows exhibiting a periodic limit cycle. This method builds upon classical spectral proper orthogonal decomposition (SPOD) and resolvent analyses but introduces a quasi-steady approximation that effectively separates the high-frequency turbulent component from the slow periodic oscillation. Moreover, [Heidt and Colonius](#page-26-3) [\[2023\]](#page-26-3) have pioneered the theory and algorithm for cyclostationary SPOD (CS-SPOD). This innovative approach extends the conventional SPOD algorithm to effectively capture the characteristics of flows characterized by periodic motions. CS-SPOD is closely tied to harmonic resolvent analysis, akin to the relationship between SPOD and resolvent analysis [\[Towne et al.,](#page-25-2) [2018\]](#page-25-2).

Harmonic resolvent analysis offers a promising avenue for new discoveries in physics. For example, recent research conducted by [Wu et al.](#page-26-4) [\[2022\]](#page-26-4) explored the response of a turbulent separation bubble to zero-net-mass-flux actuation, with a base flow characterized by periodicity. The noteworthy finding revealed an agreement between the reduction in separation bubble size identified through both harmonic resolvent optimal response and direct numerical simulation. Another contribution in this realm comes from the work of [Padovan and Rowley](#page-26-5) [\[2022\]](#page-26-5), who investigated the driving mechanisms underlying vortex pairing in an incompressible forced axisymmetric jet. They showed that the optimal forcing of the subharmonic resolvent operator featured spatiotemporal structures leading to the nonlinear vortex pairing phenomenon. In general, harmonic resolvent analysis can facilitate the understanding of energy transfer mechanisms [\[Jin et al.,](#page-26-6) [2021\]](#page-26-6) and is a potentially powerful tool for secondary stability analyses involving periodic additions to the base flow, with applications such as boundary layer stability and transition [\[Herbert,](#page-26-7) [1984\]](#page-26-7), screeching jets [\[Edgington-Mitchell et al.,](#page-26-8) [2021\]](#page-26-8), and the effect of mass injection on secondary instability

## [\[Kumar and Prakash,](#page-26-9) [2022\]](#page-26-9).

From a computational standpoint, computing harmonic resolvent modes presents similar challenges to resolvent analysis, including the need to invert sparse matrices in Fourier space. This challenge is further exacerbated in harmonic resolvent analysis due to expanded system size, where all relevant frequencies are merged into a unified operator, in contrast to resolvent analysis, where each frequency has its own independent system. The second shared challenge involves performing the singular value decomposition (SVD). Turbulent and other complex flows characterized by periodic motions require high-resolution meshes and potentially large computational domains. This results in a sizable linearized operator, introducing formidable computational challenges when it comes to matrix inversion and performing SVD. A modified randomized SVD (RSVD) can be used to simultaneously overcome both computational challenges [\[Moarref et al.,](#page-26-10) [2013,](#page-26-10) [Ribeiro et al.,](#page-26-11) [2020,](#page-26-11) [Padovan et al.,](#page-25-0) [2020\]](#page-25-0). Nevertheless, the modified RSVD, referred to as "RSVD-LU" in this study, becomes impractical for tackling large-scale turbulent flows since it requires the lower-upper (LU) decomposition of the harmonic resolvent operator. Recently, RSVD-∆t, an improvement to RSVD-LU, addresses bottlenecks in resolvent analysis [\[Farghadan et al.,](#page-26-12) [2023\]](#page-26-12). The approach of time-stepping, initially developed by [Monokrousos et al.](#page-26-13) [\[2010\]](#page-26-13) and further optimized by [Martini](#page-26-14) [et al.](#page-26-14) [\[2021\]](#page-26-14), allows for computing the action of an operator on a vector without solving a linear system in Fourier space, making it a key component of RSVD-∆t.

The main objective of this study is to expand the applicability of the RSVD-∆t algorithm, initially crafted for resolvent analysis, to facilitate the computation of harmonic resolvent modes. A streaming approach is adopted for time integration, reducing memory usage while maintaining computational efficiency. Moreover, an efficient transient removal strategy is introduced, cutting the temporal length of the integration and thereby minimizing the CPU cost of RSVD-∆t for achieving a desired accuracy. The RSVD-∆t algorithm for periodic flows maintains linear scalability in both CPU time and memory consumption. This scalability enables the efficient computation of harmonic resolvent modes for large-scale turbulent flows that were previously out of reach.

The structure of our paper unfolds as follows. We begin by explaining resolvent analysis in §[2,](#page-2-0) followed by the formulation of harmonic resolvent analysis in §[3,](#page-3-0) where the computation of harmonic resolvent modes using RSVD-LU is briefly described. The RSVD-∆t algorithm is introduced in §[4.](#page-9-0) A comparison of the computational complexity of algorithms and suggested improvements appear in §[5.](#page-12-0) Further details on performance and error analysis are presented in §[6.](#page-14-0) In §[7,](#page-17-0) we conduct two test cases to demonstrate the accuracy and cost efficiency of RSVD-∆t for two periodic systems, and we conclude with final remarks in §[8.](#page-23-0)

# <span id="page-2-0"></span>2 Resolvent analysis

Before considering the time-periodic case, we briefly review standard resolvent analysis for stationary flows. The Navier-Stokes equations can be linearized around a base flow via Reynolds decomposition of the state, q = q¯+ q ′ , where q¯ represents the temporal average of the flow, and q ′ denotes the time-varying fluctuations. The dynamics of perturbations are then described by the linear time-invariant system

<span id="page-2-1"></span>
$$\frac{d\mathbf{q}'}{dt} = \mathbf{A}(\bar{\mathbf{q}})\mathbf{q}' + \mathbf{f}'(\bar{\mathbf{q}}, \mathbf{q}'), \tag{2.1}$$

where A ∈ C <sup>N</sup>×<sup>N</sup> is the linearized Navier-Stokes (LNS) operator, N is the state dimension of the discretized system, and f ′ represents an external forcing as well as nonlinear terms that are treated as an additional forcing, as suggested by [McKeon and Sharma](#page-26-15) [\[2010\]](#page-26-15). It should be noted that A is time-independent because the base flow is time-invariant. By taking the Fourier transform

$$\mathcal{F}(\cdot) = (\hat{\cdot})(\omega) = \int_{-\infty}^{+\infty} (\cdot)e^{-i\omega t} dt$$
 (2.2)

in time, we obtain

$$\hat{q}(\omega) = R(\omega)\hat{f}(\omega), \tag{2.3}$$

where  $\omega$  represents the frequency and  $(\hat{\cdot})$  denotes the frequency-domain representation of the time-domain vector. The resolvent operator

$$\mathbf{R} = (\mathrm{i}\omega \mathbf{I} - \mathbf{A})^{-1} \in \mathbb{C}^{N \times N}$$
(2.4)

is a transfer function that maps the input forcing to the output response. Here,  $i = \sqrt{-1}$  and I is the identity matrix of appropriate dimension.

To analyze the perturbation characteristics of a system, the transfer function can be examined across a range of frequencies. Typically, the most amplified responses are identified by computing the left singular vectors  $U_R$  of the resolvent operator,

<span id="page-3-1"></span>
$$\mathbf{R} = \mathbf{U}_R \mathbf{\Sigma}_R \mathbf{V}_R^*, \tag{2.5}$$

where  $(\cdot)^*$  denotes complex conjugate transpose. The optimal forcing inputs that lead to the highest amplifications comprise the right singular vectors  $V_R$ , and the degree of amplifications are determined by the singular values  $\Sigma_R$ .

The SVD in 2.5 implies that disturbance amplitudes are measured in a Euclidean norm. More generally,  $\|x\|_f^2 = \langle x, x \rangle_f = x^* W_f x$  computes the f-norm of any vector x, and input and output norms can be different, i.e.,  $\|\cdot\|_q = \|\cdot\|_f$  is not necessary. Additionally, one can define an input matrix B to restrict the forcing and an output matrix C to extract the output of interest from the state. For the sake of notational brevity, we assumed the weight, input, and output matrices to be the identity, i.e., W = B = C = I, throughout this paper. However, our algorithm, with minor adjustments, can accommodate non-identity weight, input, and output matrices by computing the modes of the modified resolvent operator

$$\tilde{R} = W_q^{1/2} CRBW_f^{-1/2}.$$
 (2.6)

For more details, please refer to Farghadan et al. [2023].

# <span id="page-3-0"></span>3 Harmonic resolvent analysis

In this section, we provide an overview of harmonic resolvent analysis. After reviewing its basic formulation, we discuss the singularity issue that can arise in some special flows, the truncation of the harmonic resolvent operator, and two variations of harmonic resolvent analysis, namely cross-frequency amplification and subharmonic resolvent. Lastly, a brief overview is provided for the state-of-the-art algorithm used to compute the harmonic resolvent modes and gains.

## 3.1 Formulation

Analogous to resolvent analysis, by inserting the generalized Reynolds decomposition  $q(t) = \bar{q}(t) + q'(t)$  into the Navier-Stokes equations, the linearization around a periodic base flow  $\bar{q}(t) = \bar{q}(t+T)$  leads to the linear time-periodic system

<span id="page-3-2"></span>
$$\frac{d\mathbf{q'}}{dt} = \mathbf{A}_p \mathbf{q'} + \mathbf{f},\tag{3.1}$$

where  $A_p(t) = A_p(t+T) \in \mathbb{C}^{N \times N}$  is the periodic LNS operator,  $\omega_f$  is the fundamental frequency, and  $T = 2\pi/\omega_f$  is defined as the fundamental cycle length. As both  $A_p$  and q' are time-periodic, the Fourier transform of (3.1) incorporates interactions between all pairs of frequencies. For simplicity, we drop the prime notation for fluctuations from here on out.

Expanding  $A_p(t)$  and q(t) in terms of the Fourier series

$$\begin{aligned} &\boldsymbol{A}_p(t) = \sum_{j=-\infty}^{\infty} \hat{\boldsymbol{A}}_{p,j} e^{\mathrm{i} j \omega_f t}, \\ &\boldsymbol{q}(t) = \sum_{j=-\infty}^{\infty} \hat{\boldsymbol{q}}_j e^{\mathrm{i} j \omega_f t}, \end{aligned} \qquad \text{(3.2)}$$

where  $(\cdot)_j$  denotes  $j^{th}$  harmonic of  $\omega_f$ , and substituting these expansions into (3.1), we obtain

<span id="page-4-0"></span>
$$[T\hat{q}]_k = ik\omega_f \hat{q}_k - \sum_{j=-\infty}^{\infty} \hat{A}_{k-j} \hat{q}_j = \hat{f}_k.$$
(3.3)

Here, T is an infinite dimensional block matrix of the form

$$T = \begin{bmatrix} \ddots & \vdots & \vdots & \vdots & \vdots & \ddots \\ \dots & L_{-1} & -\hat{A}_{-1} & -\hat{A}_{-2} & -\hat{A}_{-3} & \dots \\ \dots & -\hat{A}_{1} & L_{0} & -\hat{A}_{-1} & -\hat{A}_{-2} & \dots \\ \dots & -\hat{A}_{2} & -\hat{A}_{1} & L_{1} & -\hat{A}_{-1} & \dots \\ \dots & -\hat{A}_{3} & -\hat{A}_{2} & -\hat{A}_{1} & L_{2} & \dots \\ \ddots & \vdots & \vdots & \vdots & \vdots & \ddots \end{bmatrix}, \tag{3.4}$$

where the diagonal entries are in fact the inverse of resolvent operators at various frequencies,

$$\boldsymbol{L}_{j} = \boldsymbol{R}_{j}^{-1} = ij\omega_{f}\boldsymbol{I} - \hat{\boldsymbol{A}}_{p,0}, \forall j \in \mathbb{Z}.$$
(3.5)

The harmonic resolvent operator

<span id="page-4-2"></span>
$$\boldsymbol{H} = \boldsymbol{T}^{-1} \tag{3.6}$$

transfers the harmonic inputs to the outputs for periodic base flows,

<span id="page-4-3"></span>
$$\hat{q} = H\hat{f}. \tag{3.7}$$

The SVD of the harmonic resolvent operator

$$\boldsymbol{H} = \mathbf{U}_H \boldsymbol{\Sigma}_H \mathbf{V}_H^* \tag{3.8}$$

unveils the most amplified responses  $\mathbf{U}_H$  that correspond to optimal forcing  $\mathbf{V}_H$ , each accompanied by associated amplification magnitudes  $\mathbf{\Sigma}_H$ . The matrix  $\mathbf{H}$  encompasses the base flow, the fundamental frequency, and higher harmonics. Whereas individual frequencies can be analyzed separately in a standard resolvent analysis, all frequencies are coupled within  $\mathbf{H}$  in harmonic resolvent analysis and must be considered simultaneously. Furthermore, the modes of the modified harmonic resolvent operator, defined as

<span id="page-4-1"></span>
$$\tilde{\boldsymbol{H}} = \boldsymbol{W}_q^{1/2} \boldsymbol{C} \boldsymbol{H} \boldsymbol{B} \boldsymbol{W}_f^{-1/2}, \tag{3.9}$$

can be computed using our algorithm with the same minor adjustments as detailed in Farghadan et al. [2023]. In a special case where the base flow is time-independent, i.e.,  $\hat{A}_{p,j} = 0$  for  $j \neq 0$ , H reduces to

$$H = \begin{bmatrix} \ddots & \vdots & \vdots & \vdots & \mathinner{\mkern2mu\raise1pt\hbox{.}\mkern2mu\raise4pt\hbox{.}\mkern2mu\raise7pt\hbox{.}\mkern1mu} \\ \dots & \boldsymbol{R}_{-1} & 0 & 0 & \dots \\ \dots & 0 & \boldsymbol{R}_0 & 0 & \dots \\ \dots & 0 & 0 & \boldsymbol{R}_1 & \dots \\ \mathinner{\mkern1mu\raise7pt\vbox{\kern7pt\hbox{.}}\mkern2mu\raise4pt\hbox{.}\mkern2mu\raise1pt\hbox{.}\mkern1mu} & \vdots & \vdots & \vdots & \ddots \end{bmatrix}, \tag{3.10}$$

so each frequency is decoupled. Therefore, resolvent analysis can be perceived as a special case of harmonic resolvent analysis.

<span id="page-5-2"></span>

## 3.2 Singular or near-singular nature of T

The operator T tends to be ill-conditioned for many periodic systems [Padovan et al., 2020]. A system is considered ill-conditioned when the ratio of the largest and smallest singular values is large ( $\kappa = \sigma_{\text{max}}/\sigma_{\text{min}} \sim O(10^4)$ ) or higher). The smallest singular value of T can be shown to reach machine precision zero if the periodic base flow satisfies the Navier-Stokes equations [Padovan et al., 2020, Leclercq and Sipp, 2023], *i.e.*, when

<span id="page-5-1"></span>
$$\frac{d\bar{q}}{dt} = \mathcal{N}(\bar{q}),\tag{3.11}$$

where  $\mathcal{N}$  is the nonlinear Navier-Stokes operator. Taking a time derivative of both sides,

<span id="page-5-0"></span>
$$\frac{d\left(\frac{d\bar{q}}{dt}\right)}{dt} = \frac{d\left(\mathcal{N}(\bar{q})\right)}{dt} = \frac{\partial \mathcal{N}}{\partial \bar{q}} \frac{d\bar{q}}{dt} = A_p \frac{d\bar{q}}{dt}.$$
(3.12)

Taking a Fourier transform of (3.12) and recalling that T is obtained as shown in (3.3),  $\widehat{dq}/\widehat{dt}$  is a non-trivial solution of

$$Tq = 0, (3.13)$$

residing within the null space of T, thereby proving the singularity of T. In systems where the base flow approximately satisfies (3.11), T becomes nearly singular. Consequently, irrespective of the specific periodic system under consideration, T might be poorly conditioned, presenting even more challenges when dealing with large systems.

Padovan et al. [2020] demonstrated that defining the harmonic resolvent operator in a manner that eliminates the phase shift imparted by the Fourier coefficients of  $d\bar{q}/dt$  is effective, without adversely affecting the other dominant amplification mechanisms. By constraining T to a subspace  $U^{\perp}$  that is orthogonal to the direction of the phase shift  $d\bar{q}/dt$ , the range of the harmonic resolvent operator is limited to  $U^{\perp}$ . The key concept here is to eliminate the singular vectors associated with the smallest singular value of T without affecting the other singular values and vectors of T. Given that the phase shift resides in the null space of T, the desired right singular vector  $v = \frac{d\bar{q}}{dt}/||\hat{d}||$  is readily available. We can also solve

$$T^*\tilde{\boldsymbol{u}} = \boldsymbol{v} \tag{3.14}$$

and compute the corresponding left singular vector as  $\mathbf{u} = \tilde{\mathbf{u}}/\|\tilde{\mathbf{u}}\|$ . Upon obtaining  $\mathbf{v}$  and  $\mathbf{u}$ , the process becomes straightforward: projecting out these modes from the forcing and response terms effectively restricts  $\mathbf{T}$  to  $\mathbf{U}^{\perp}$ . These steps are elucidated in detail in Padovan et al. [2020].

To compute the action of the harmonic resolvent operator on a given vector (or matrix), our goal is to determine the vector (or matrix) q that satisfies q = Hf. In brief, to compute the action of a singular H on a vector  $\tilde{f}$ , the component along u must be projected out as

$$\tilde{\mathbf{f}}_{in} = \tilde{\mathbf{f}} - \mathbf{u}(\mathbf{u}^* \tilde{\mathbf{f}}), \tag{3.15}$$

and this will provide a response

$$T\tilde{q} = \tilde{f}_{in} \rightarrow \tilde{q} = H\tilde{f}_{in},$$
 (3.16)

which is orthogonal to the direction of the phase shift. Nevertheless, to mitigate potential round-off errors, it is advisable to refine the output by projecting out the component along v as

$$\tilde{\mathbf{q}}_{out} = \tilde{\mathbf{q}} - \mathbf{v}(\mathbf{v}^* \tilde{\mathbf{q}}). \tag{3.17}$$

Similarly, when computing the action of singular  $H^*$  on a vector, the initial step involves projecting out the component along v. After the response is obtained, to mitigate round-off errors, the component along v is projected out.

## 3.3 Truncating the harmonic resolvent operator

Computing harmonic resolvent modes is contingent upon T being of finite size. The size and block sparsity pattern of T are determined by two key variables: the number of frequencies within the base flow,  $N_b$ , and within the response,  $N_{\omega}$ . Here, we elucidate the impact of each variable on the operator.

The rows of T consist of  $N_b$ -stencil blocks, dictated by the presence of non-zero elements in  $\hat{A}_{p,j}$ . In the context of periodic flows, the time-dependent base flow can be expressed using a Fourier series expansion

$$\bar{q}(t) = \sum_{j\omega_f \in \Omega_{\bar{q}}} \hat{\bar{q}}_j e^{ij\omega_f t}.$$
 (3.18)

The set  $\Omega_{\bar{q}}$  encompasses the relevant frequencies associated with the dominant flow structures. Typically,  $\Omega_{\bar{q}}$  is a subset of  $j\omega_f$  with  $j \in \mathbb{Z}$  and encapsulates the majority of energy within the periodic flows. By retaining the base periodicity and a limited number of higher harmonics, it becomes possible to simplify the base flow representation, resulting in  $N_b$  frequencies within  $\Omega_{\bar{q}}$  and  $N_b$  non-zero elements within each row of T.

On the other hand, we aim to resolve perturbations occurring at temporal frequencies  $\omega \in \Omega_q$ , where  $\Omega_q$  represents a subset of  $j\omega_f$  with  $j \in \mathbb{Z}$ , although the specific choice of harmonics for  $\Omega_q$  may vary, as further elucidated in §3.5. Typically,  $\Omega_{\bar{q}}$  is a subset of  $\Omega_q$ , *i.e.*, the perturbation frequency content spans a wider range of harmonics [Padovan et al., 2020]. The number of frequencies  $N_\omega$  expanding both  $\boldsymbol{f}$  and  $\boldsymbol{q}$  determines the dimensions of the block matrices, yielding a size of  $(N_\omega N) \times (N_\omega N)$  for  $\boldsymbol{T}$ .

We provide a simple example to visually observe the harmonic resolvent operator. Suppose we use  $\Omega_{\bar{q}} = \{-2, -1, 0, 1, 2\}\omega_f$  to expand the LNS operator and  $\Omega_q = \{-4, -3, \dots, 3, 4\}\omega_f$  to expand the perturbations. In this case, the stencil length would be  $N_b = 5$ , and the size of the block matrix

would be  $(9N) \times (9N)$ . The harmonic resolvent operator is then

$$\boldsymbol{H} = \begin{bmatrix} \boldsymbol{L}_{-4} & -\hat{\boldsymbol{A}}_{-1} & -\hat{\boldsymbol{A}}_{-2} & 0 & 0 & 0 & 0 & 0 & 0 \\ -\hat{\boldsymbol{A}}_{1} & \boldsymbol{L}_{-3} & -\hat{\boldsymbol{A}}_{-1} & -\hat{\boldsymbol{A}}_{-2} & 0 & 0 & 0 & 0 & 0 \\ -\hat{\boldsymbol{A}}_{2} & -\hat{\boldsymbol{A}}_{1} & \boldsymbol{L}_{-2} & -\hat{\boldsymbol{A}}_{-1} & -\hat{\boldsymbol{A}}_{-2} & 0 & 0 & 0 & 0 \\ 0 & -\hat{\boldsymbol{A}}_{2} & -\hat{\boldsymbol{A}}_{1} & \boldsymbol{L}_{-1} & -\hat{\boldsymbol{A}}_{-1} & -\hat{\boldsymbol{A}}_{-2} & 0 & 0 & 0 \\ 0 & 0 & -\hat{\boldsymbol{A}}_{2} & -\hat{\boldsymbol{A}}_{1} & \boldsymbol{L}_{0} & -\hat{\boldsymbol{A}}_{-1} & -\hat{\boldsymbol{A}}_{-2} & 0 & 0 \\ 0 & 0 & 0 & -\hat{\boldsymbol{A}}_{2} & -\hat{\boldsymbol{A}}_{1} & \boldsymbol{L}_{1} & -\hat{\boldsymbol{A}}_{-1} & -\hat{\boldsymbol{A}}_{-2} & 0 \\ 0 & 0 & 0 & 0 & -\hat{\boldsymbol{A}}_{2} & -\hat{\boldsymbol{A}}_{1} & \boldsymbol{L}_{2} & -\hat{\boldsymbol{A}}_{-1} & -\hat{\boldsymbol{A}}_{-2} \\ 0 & 0 & 0 & 0 & 0 & -\hat{\boldsymbol{A}}_{2} & -\hat{\boldsymbol{A}}_{1} & \boldsymbol{L}_{3} & -\hat{\boldsymbol{A}}_{-1} \\ 0 & 0 & 0 & 0 & 0 & 0 & -\hat{\boldsymbol{A}}_{2} & -\hat{\boldsymbol{A}}_{1} & \boldsymbol{L}_{4} \end{bmatrix}^{-1} . \qquad (3.19)$$

The forcing and response vectors can be represented as

$$\hat{\mathbf{f}} = \begin{bmatrix} \hat{\mathbf{f}}_{-4} \\ \hat{\mathbf{f}}_{-3} \\ \hat{\mathbf{f}}_{-2} \\ \hat{\mathbf{f}}_{-1} \\ \hat{\mathbf{f}}_{0} \\ \hat{\mathbf{f}}_{1} \\ \hat{\mathbf{f}}_{2} \\ \hat{\mathbf{f}}_{3} \\ \hat{\mathbf{f}}_{4} \end{bmatrix}, \hat{\mathbf{q}} = \begin{bmatrix} \hat{\mathbf{q}}_{-4} \\ \hat{\mathbf{q}}_{-3} \\ \hat{\mathbf{q}}_{-2} \\ \hat{\mathbf{q}}_{-1} \\ \hat{\mathbf{q}}_{0} \\ \hat{\mathbf{q}}_{1} \\ \hat{\mathbf{q}}_{2} \\ \hat{\mathbf{q}}_{3} \\ \hat{\mathbf{q}}_{4} \end{bmatrix}. \tag{3.20}$$

The accuracy of the obtained resolvent system in Fourier space depends on how well the chosen frequencies capture the relevant flow information, and it is affected by the truncation limit imposed on the frequency range. Although it is feasible to reduce the infinite-dimensional harmonic operator to a finite dimension based on perturbation frequencies, the choice of frequency range is consequential. To demonstrate the influence of truncation on response modes, consider a periodic flow with a dominant first harmonic, such that the base flow can be expressed as

$$\bar{\boldsymbol{q}}(t) = \hat{\bar{\boldsymbol{q}}}_0 + \hat{\bar{\boldsymbol{q}}}_1 e^{i\omega t} + \hat{\bar{\boldsymbol{q}}}_{-1} e^{-i\omega t}. \tag{3.21}$$

Assume that the LNS equations are subjected to a constant forcing  $f(t) = \hat{f}_0$ . The LNS operator exhibits frequency content at 0 and  $\pm \omega$ , indicating that a constant forcing will directly induce a response at these frequencies. However, higher harmonics are also indirectly stimulated, and the triggering of responses at higher harmonics follows a cascade pattern, with norms gradually decaying towards zero, i.e.,  $\lim_{j\to\infty} ||\hat{q}_j|| = 0$ . Hence, the truncation must be set at a sufficiently high level to preserve the response norms.

In practice, by defining  $\Omega_q$  for both input perturbation and the response, we compute accurate harmonic resolvent modes as long as the norms of higher frequency modes become less relevant within the spectrum. In other words, if we set both forcing and response frequencies sufficiently high and, upon computing harmonic resolvent modes, find that the norms of input and output modes with higher frequency content are relatively small, it indicates convergence, and including higher frequencies has negligible impact on the results. In a case, for instance, where response modes with higher frequency content remain relatively important, extending the frequency range of the output modes is necessary without changing the input frequency content. In our algorithm, the input and output frequency ranges are adjustable as needed.

## 3.4 Cross-frequency harmonic resolvent analysis

Harmonic resolvent analysis captures the interaction between various frequencies within Ωq¯ and Ωq, and the typical objective is to identify the optimal forcing mode, potentially spanning various frequencies, that produces the optimal response across a range of frequencies. Alternatively, one can compute the unit-norm forcing at one frequency ω<sup>1</sup> ∈ Ω<sup>q</sup> that triggers the most amplified response at another frequency ω<sup>2</sup> ∈ Ωq. This is achieved by defining

$$H_{\omega_2,\omega_1} = CHB \tag{3.22}$$

and computing the modes and gains following the same procedure as before. Here, matrices B and C are constructed to extract the ω<sup>1</sup> and ω<sup>2</sup> frequencies from the forcing and response modes, respectively. This represents a special case of the modified harmonic resolvent operator in [\(3.9\)](#page-4-1).

## <span id="page-8-0"></span>3.5 Subharmonic resolvent analysis

The frequency content of the forcing inputs is typically composed of harmonics of the base flow frequency. However, [Padovan and Rowley](#page-26-5) [\[2022\]](#page-26-5) have shown that the harmonic resolvent system may exhibit sensitivity to subharmonic inputs. For instance, in the case where the fundamental base flow frequency is denoted by ω<sup>f</sup> , the linearized system may be sensitive to forcing inputs with γ < ω<sup>f</sup> , such as ω<sup>f</sup> /2. More generally, if the frequency content of a periodic LNS operator is limited to the frequency range Ωq¯, it may be desirable to compute input-output modes with frequencies outside of this range, i.e., , γ ∉ Ωq¯. In such cases, the subharmonic resolvent modes become relevant.

To identify unique subharmonic inputs, we can take advantage of the linear nature of the harmonic resolvent operator, which allows interaction with the fundamental frequency ω<sup>f</sup> and its harmonics. It can be demonstrated that the interval (−ω<sup>f</sup> /2, ω<sup>f</sup> /2] encompasses all sets of subharmonic frequencies where the input-output modes are unique. As discussed in §[3,](#page-3-0) a forcing with a frequency γ ∈ (−ω<sup>f</sup> /2, ω<sup>f</sup> /2] can only trigger responses at frequencies γ + jω<sup>f</sup> , where j is an integer. Therefore, the input and output modes exist within the frequency range Ω<sup>γ</sup> = γ ⊕ Ωq¯, where ⊕ denotes element-wise addition. Harmonics of γ trigger a unique set of outputs as long as jγ ≤ ∣ω<sup>f</sup> ∣. Due to the linear relationship, we can show that all harmonics are isolated sets that need to be studied separately. Note that −ω<sup>f</sup> /2 is excluded since Ω−ω<sup>f</sup> /<sup>2</sup> = Ωω<sup>f</sup> /<sup>2</sup> , resulting in redundancy. For γ ∈ Ωγ, one can derive the subharmonic resolvent system as [\[Padovan and Rowley,](#page-26-5) [2022\]](#page-26-5)

$$\hat{\mathbf{q}} = (i\gamma \mathbf{I} - \mathbf{T})^{-1}\hat{\mathbf{f}}.$$
(3.23)

To further elucidate the point, consider an example where the subharmonic frequency of interest is ω<sup>f</sup> /5 along with its harmonics. Note that the valid harmonics are the ones that fulfill the condition jω<sup>f</sup> /5 ∈ (−ω<sup>f</sup> /2, ω<sup>f</sup> /2], which includes the set {−2/5,−1/5, 0, 1/5, 2/5}. While Ω<sup>0</sup> = 0 ⊕ {−Nb/2,⋯, Nb/2}, where N<sup>b</sup> is the number of active frequencies within the base flow, is identical to harmonic inputs, the sets Ω−2/<sup>5</sup> , Ω−1/<sup>5</sup> , Ω1/<sup>5</sup> , and Ω2/<sup>5</sup> represent distinct input-output modes. In general, for any two distinct subharmonic frequencies γ<sup>1</sup> and γ<sup>2</sup> within the interval (−ω<sup>f</sup> /2, ω<sup>f</sup> /2], the corresponding subharmonic modes in Ωγ<sup>1</sup> are decoupled from those in Ωγ<sup>2</sup> as there exists no common frequency γ<sup>1</sup> − γ<sup>2</sup> ∈ Ωq¯. While our algorithm described in §[4](#page-9-0) is described for harmonic resolvent analysis, it can be easily extended to compute subharmonic resolvent modes, as described in Appendix [A.](#page-25-9)

The approach used to eliminate the singularity of T described in §[3.2](#page-5-2) is not applicable for subharmonic resolvent analysis (γ ≠ 0). [Padovan and Rowley](#page-26-5) [\[2022\]](#page-26-5) formulated an alternative approach in which the domain of T is constrained to a physically meaningful subspace for all choices of γ. In this approach, an oblique projection operator is defined as

$$P(t) = I - v(t)u^*(t), \tag{3.24}$$

where u and v are defined the same as in §[3.2.](#page-5-2) The restricted operator can be written as

$$T_p = T\hat{P}. (3.25)$$

The resulting range of T<sup>p</sup> is deliberately designed to be invariant under T. This projection facilitates steps within the RSVD-LU algorithm, effectively addressing singularities. For more details on oblique projection properties and the efficient computation of u and v, we refer the reader to [Padovan and Rowley](#page-26-5) [\[2022\]](#page-26-5).

## 3.6 Computing harmonic resolvent modes using RSVD-LU

The RSVD algorithm [\[Halko et al.,](#page-26-17) [2011\]](#page-26-17) is a randomized linear algebra technique developed to efficiently identify the singular vectors with the highest gains in a given matrix. In the context of harmonic resolvent analysis, the matrix of interest is the harmonic resolvent operator. This operator, similar to the resolvent operator, relies on the inverse of a sparse matrix, as shown in [\(3.6\)](#page-4-2), making the computation of modes challenging. Modifying the original RSVD algorithm is necessary to address these challenges, enabling harmonic resolvent analysis. The modifications for resolvent analysis have been extensively documented in the literature by [Ribeiro et al.](#page-26-11) [\[2020\]](#page-26-11) and [Farghadan et al.](#page-26-12) [\[2023\]](#page-26-12), while the outline for the harmonic resolvent operator can be found in the appendix of [Padovan et al.](#page-25-0) [\[2020\]](#page-25-0).

In brief, the computation involves computing the action of H and H<sup>∗</sup> for sketching the range and image of the harmonic resolvent operator, respectively, and approximating the leading harmonic modes. Computing actions of H and H<sup>∗</sup> , however, requires solving linear systems

<span id="page-9-2"></span>
$$T\hat{q} = \hat{f},$$

$$T^*\hat{q} = \hat{f},$$
(3.26)

respectively, in Fourier space. These linear systems are solved via LU-decomposition of T , but due to its large size and ill-conditioned nature, computing its LU decomposition can be a formidable computational obstacle, particularly for flows with three inhomogeneous directions. We propose an alternative way in the following section.

# <span id="page-9-0"></span>4 Computing harmonic resolvent modes using time stepping

In this section, we show how time stepping can be used to efficiently compute harmonic resolvent modes. The same concept was first used by [Monokrousos et al.](#page-26-13) [\[2010\]](#page-26-13) in the context of the resolvent analysis and further optimized by [Martini et al.](#page-26-14) [\[2021\]](#page-26-14) and [Farghadan et al.](#page-26-12) [\[2023\]](#page-26-12). Here, we introduce the extension of our algorithm to compute the harmonic resolvent modes and gains.

## <span id="page-9-1"></span>4.1 Computing the action of H using time stepping

Computing the action of H in the Fourier space poses a bottleneck within the RSVD-LU algorithm. To overcome this limitation, we propose to use time-stepping as an effective surrogate approach. Given that Ωq¯ represents the frequency content of the base flow around which T is constructed, we define the time-domain linearized operator as

$$\mathbf{A}_{p}(t) = \sum_{j\omega_{f} \in \Omega_{\bar{q}}} \hat{\mathbf{A}}_{p,j} e^{ij\omega_{f}t}.$$
(4.1)

![](_page_10_Picture_0.jpeg)

Figure 1: Flowchart depicting the action of H on a forcing comprised of N<sup>ω</sup> frequencies within the RSVD-LU (top route) and the RSVD-∆t (bottom route) algorithms. Both routes produce the same result, but the bottom route is computationally advantageous for large systems.

<span id="page-10-0"></span>The action of H on fˆ in Fourier space is expressed in [\(3.7\)](#page-4-3), where the frequency content of the forcing lies within Ωq.

The steady-state solution of [\(3.1\)](#page-3-2) is given by

$$\hat{\boldsymbol{q}}_s = \boldsymbol{H}\hat{\boldsymbol{f}},\tag{4.2}$$

when subjected to the forcing represented as

$$f = \sum_{j\omega_f \in \Omega_q} \hat{f}_j e^{ij\omega_f t}. \tag{4.3}$$

The steady-state response can be obtained through time-stepping before taking a Fourier transform. The time-domain forcing is constructed to include frequencies identical to those present in fˆ. Since the time-domain and frequency-domain frequencies are the same, both approaches yield identical results, i.e., qˆ<sup>s</sup> = qˆ. In practice, the time-stepping method is inherently discrete, and the frequencies are retained up to the maximum limit determined by the chosen time step (below the Nyquist frequency [\[Nyquist,](#page-26-18) [1928\]](#page-26-18)). By employing a suitably small time step, we can capture frequencies up to the desired limit.

The equivalence between computing the action of H in both the RSVD-LU and RSVD-∆t algorithms is depicted in figure [1.](#page-10-0) In the upper route, within the RSVD-LU algorithm, the LNS equations are first transformed into Fourier space before solving a coupled linear system in the frequency domain for a given forcing. In the lower route, within the RSVD-∆t algorithm, the LNS equations are initially integrated in the time domain before being transformed into Fourier space. Both routes produce identical outputs so long as numerical artifacts are minimized.

## 4.2 RSVD-∆t for harmonic resolvent analysis

By recognizing the equivalence between time stepping and solving linear systems in the frequency domain, we can effectively address the limitations of the RSVD-LU algorithm, leading to the development of RSVD-∆t. The time-stepping aspect of RSVD-∆t relies on the linearized operator  $A_p$  without explicitly constructing H in Fourier space. The algorithmic steps for extending RSVD- $\Delta t$  to harmonic systems are presented in Algorithm 1.

## <span id="page-11-0"></span>**Algorithm 1** RSVD- $\Delta t$ for harmonic resolvent analysis

```
1: Inputs: A_p, k, q, \Omega_q, \Omega_{\bar{q}}, TSS, dt, T_t
 2: \hat{\boldsymbol{\Theta}} \leftarrow \operatorname{randn}(NN_{\omega}, k)

 3: \hat{Y} \leftarrow \text{DirectAction}(A_p, \hat{\Theta}, TSS, dt, T_t)
                                                                                                                                                    \triangleright Sample the range of H
 4: if q > 0 then
                                                                                                                                               > Optional power iteration
             \hat{\boldsymbol{Y}} \leftarrow \text{PI}(\boldsymbol{A}_p, \hat{\boldsymbol{Y}}, q, \text{TSS}, dt, T_t)
                                                                                                                          ▷ Power iteration with time-stepping
 6: \hat{\boldsymbol{Q}}_{\Omega} \leftarrow \operatorname{qr}(\hat{\boldsymbol{Y}}_{\Omega})
                                                                                                                         \triangleright Build the orthonormal subspace Q_{\Omega}
 7: \hat{\boldsymbol{S}} \leftarrow \texttt{AdjointAction}(\boldsymbol{A}_p^*, \hat{\boldsymbol{Q}}, TSS, dt, T_t)
                                                                                                                                                   \triangleright Sample the image of H
 8: (\hat{\boldsymbol{U}}_H, \boldsymbol{\Sigma}_H, \boldsymbol{V}_H) \leftarrow \operatorname{svd}(\hat{\boldsymbol{S}}_H)
                                                                                                                                                                   \triangleright Obtain \Sigma_H, V_H
 9: U_H \leftarrow Q_{\Omega} U_H
                                                                                                                                                                        \triangleright Recover U_H
10: Outputs: U_H, \Sigma_H, V_H
```

Algorithm 1. Inputs: linearized operator  $A_p$ , number of modes k, number of power iterations q, frequency range of the perturbations and LNS operator  $\Omega_q$  and  $\Omega_{\bar{q}}$ , respectively, time-stepping scheme abbreviated as TSS (e.g., backward Euler), time step dt, and the transient length  $T_t$ . Outputs: k response modes  $U_H$ , k forcing modes  $V_H$  and the corresponding gains  $\Sigma_H$ . Here,  $k, q, \Omega_q, \Omega_{\bar{q}}$  are common parameters with RSVD-LU. ( $\cdot$ ) $\Omega$  indicates all frequencies are merged into a single column. DirectAction and AdjointAction are functions that solve the direct and adjoint LNS equations, respectively, with a predefined forcing. PI is a function that performs the power iteration.

The algorithm proposed in this study is based on the RSVD- $\Delta t$  algorithm introduced by Farghadan et al. [2023], and we will provide a concise overview of the key steps. Initially, a random matrix  $\hat{\boldsymbol{\Theta}} \in \mathbb{C}^{NN_{\omega} \times k}$  is generated (line 2) to compute the sketch of  $\boldsymbol{H}$  through time stepping (line 3). The dimensions of the random matrix are determined by the state dimension N, the number of frequencies to resolve  $N_{\omega}$ , and the desired number of modes to compute k. DirectAction is a function that computes the action of H onto a given forcing using time-stepping and transforms the steady-state responses to Fourier space. As outlined in §4.1, DirectAction solves (3.1) with zero initial condition over a sufficiently long time interval, causing the initial transient response to dissipate.  $T_t$  represents the duration of this interval before the transient diminishes. The forcing term in (3.1) is constructed via inverse Fourier transform of  $\Theta$  and the steady-state response undergoes a Fourier transform to obtain Y. Next, an optional power iteration (PI) is performed (lines 4 and 5) via q successive applications of DirectAction and AdjointAction, enhancing the accuracy of harmonic resolvent modes. QR decomposition is performed to obtain  $Q_{\Omega}$  (line 6) before constructing S via time stepping (line 7). AdjointAction is the function that computes the action of  $H^*$  onto a given forcing using time-stepping and transforms the steady-state responses to Fourier space. This function is similar to DirectAction, but it evolves the adjoint system. The forcing term in AdjointAction is constructed via the inverse Fourier transform of  $\hat{Q}$ . An SVD (line 8) is conducted to obtain the optimal forcing modes  $V_H \in \mathbb{C}^{NN_\omega \times k}$  and gains  $\Sigma_H \in \mathbb{R}^{k \times k}$  of H. Lastly, the optimal response modes  $U_H \in \mathbb{C}^{NN_\omega \times k}$  are recovered in line 9.

Compared to the RSVD- $\Delta t$  algorithm for resolvent analysis Farghadan et al. [2023], two notable differences are present. In the extension of RSVD- $\Delta t$ , the response modes at various frequencies are merged prior to both the QR decomposition and the SVD steps. This deviation from the original algorithm is motivated by the nature of the problem that requires interactions between frequencies, unlike in resolvent analysis where the QR decomposition and SVD are performed separately for each individual frequency of interest. The second difference is the generation of LNS operators over

time, whereas in resolvent analysis, the LNS operator remains constant throughout the integration.

If T approaches machine precision singularity, it is necessary to project out the forcing along the u direction before the DirectAction and along v before the AdjointAction, as described in §[3.2.](#page-5-2) Moreover, to mitigate numerical artifacts, it is advisable to project out the response along v and u after the DirectAction and AdjointAction, respectively.

# <span id="page-12-0"></span>5 Computational complexity

A brief comparison is conducted between memory requirements and CPU cost of the RSVD-LU and RSVD-∆t algorithms. We also propose strategies to minimize the memory consumption. Throughout, we assume that the linearized operator A<sup>p</sup> is sparse, i.e., that its number of non-zero entries scales as nnz(Ap) ∼ O(N). Sparse operators are obtained when using sparse discretization schemes such as finite differences, finite volume, or finite elements. For more detailed information on the CPU and memory scaling of the LU decomposition of the resolvent operator, we refer readers to [Farghadan et al.](#page-26-12) [\[2023\]](#page-26-12). To offer empirical insights and confirmation of the theoretical scalings discussed below, we present a two-dimensional test case in §[7.](#page-17-0)

## <span id="page-12-1"></span>5.1 Memory benefits of RSVD-∆t

In the case of resolvent analysis, the memory scaling of the LU decomposition is empirically O(N1.<sup>2</sup> ) and O(N1.<sup>6</sup> ) for two- and three-dimensional systems, respectively [\[Towne et al.,](#page-27-0) [2022,](#page-27-0) [Farghadan](#page-26-12) [et al.,](#page-26-12) [2023\]](#page-26-12). The memory scaling of LU decomposition is expected to be worse for the harmonic resolvent operator, as the lower- and upper-triangular matrices are denser. When T is singular or nearly singular, the computational task becomes even more challenging, as discussed in [Padovan](#page-25-0) [et al.](#page-25-0) [\[2020\]](#page-25-0). An alternative approach for solving [\(3.26\)](#page-9-2) involves leveraging iterative solvers, such as Krylov subspace methods [\[Rigas et al.,](#page-27-1) [2021,](#page-27-1) [Wu et al.,](#page-26-4) [2022\]](#page-26-4). This approach eliminates the need for computing the LU decomposition and becomes more memory-efficient, especially for large systems. However, finding a suitable preconditioner remains a challenging task.

On the other hand, the memory usage of RSVD-∆t for harmonic resolvent analysis is contingent on the sizes of various matrices, specifically the sparse LNS operators Aˆ <sup>p</sup> ∈ C N×N×N<sup>b</sup> , and the dense forcing matrix Fˆ ∈ C NNω×k and response matrix Qˆ ∈ C NNω×k , all of which are stored in Fourier space. The scaling for resolvent analysis is O(NNω), and the sole extra space is for the storage of LNS operators.

One approach to generating LNS operators for all time points involves retaining N<sup>b</sup> coefficients in Fourier space and subsequently constructing A<sup>Ω</sup> = {Ap,1, Ap,2, Ap,3, . . . , Ap,Nt} once for all time steps within a period. However, this approach can be memory-intensive for large systems, especially when N<sup>t</sup> ∼ O(10<sup>3</sup> − 10<sup>5</sup> ). An alternative strategy to reduce memory consumption is to create LNS operators on the fly, as shown in figure [2.](#page-13-0) As a result, no time-domain LNS operator is permanently stored in memory. Each LNS operator is created on the fly using the discrete Fourier transform (DFT) of N<sup>b</sup> Fourier coefficients,

$$\mathbf{A}_{p}(t) = \sum_{j=-N_b/2}^{N_b/2} \hat{\mathbf{A}}_{p,j} e^{\mathrm{i}j\omega_f t}.$$
(5.1)

Since A<sup>p</sup> is very sparse, nnz(Ap) ∼ O(N), resulting in an overall memory consumption of O(NNb). The methods employed for generating LNS operators, both in the streaming and memory-intensive approaches, mirror those used for creating forcing terms in resolvent analysis, as elaborated in Farghadan et al. [2023]. Additionally, the CPU cost of this procedure scales as  $O(\text{nnz}(A_p)N_b)$  or  $O(NN_b)$ .

![](_page_13_Figure_1.jpeg)

<span id="page-13-0"></span>Figure 2: Schematic of the action of H with streaming DFT/iDFT to transform between the Fourier and time domains.

Streaming DFT is indeed effective in reducing memory consumption; nevertheless, additional memory savings can be specifically achieved for real-valued LNS operators. In many cases,  $A_p$  is real-valued, *i.e.*, confined to  $\mathbb{R}^{N\times N}$ , yielding  $\hat{A}_{p,j} = \bar{\hat{A}}_{p,-j}$ , where  $\bar{(\cdot)}$  denotes the complex conjugate. Hence,  $\hat{A}_{p,j}$  with  $j \geq 0$  suffices to construct  $A_p$  as

$$\mathbf{A}_{p}(t) = \hat{\mathbf{A}}_{p,0} + 2\mathcal{R}e\left(\sum_{j=1}^{N_{b}/2} \hat{\mathbf{A}}_{p,j}e^{\mathrm{i}j\omega_{f}t}\right),\tag{5.2}$$

resulting in halving the number of Fourier coefficients of the LNS operator to  $\lfloor N_b/2 \rfloor + 1$ . Here,  $\Re$  represents the real part.

Another opportunity to reduce memory usage for real-valued LNS matrices arises from the symmetry between positive and negative frequencies. Rewriting (3.3) for  $-k\omega_f$  yields

<span id="page-13-1"></span>
$$i(-k\omega_f)\hat{q}_{-k} - \sum_{j=-\infty}^{\infty} \hat{A}_{-k+j}\hat{q}_{-j} = \hat{f}_{-k}, \qquad (5.3)$$

and the complex conjugate version of (3.3) becomes

<span id="page-13-2"></span>
$$-ik\omega_f \bar{\hat{q}}_k - \sum_{j=-\infty}^{\infty} \bar{\hat{A}}_{k-j} \bar{\hat{q}}_j = \bar{\hat{f}}_k.$$
 (5.4)

Both equations (5.3) and (5.4) yield the same output for a given  $\bar{f}_k = \hat{f}_{-k}$  since  $\bar{A}_{k-j} = \hat{A}_{-k+j}$ , inducing the harmonic resolvent response and forcing modes containing  $U_{H,j} = \bar{U}_{H,-j}$  and  $V_{H,j} = \bar{V}_{H,-j}$  for  $j \neq 0$ , respectively. Hence, we only keep  $\lfloor N_{\omega}/2 \rfloor + 1$  Fourier coefficients of the forcing and response modes, reducing memory consumption by half when storing these dense matrices.

## 5.2 CPU benefits of RSVD- $\Delta t$

The computational cost of computing harmonic resolvent modes using RSVD-LU is predominantly determined by the LU decomposition of T. In the case of resolvent analysis, the CPU scaling of the LU decomposition is  $O(N^{1.5})$  and  $O(N^2)$  for two- and three-dimensional systems, respectively [Towne et al., 2022, Farghadan et al., 2023]. When, in general, block matrix T contains  $\hat{A}_{p,i}$  terms,

and yields a denser matrix than block diagonal with a complex sparsity pattern, the scaling is expected to be  $O((N_{\omega}N)^{1.5})$  or  $O((N_{\omega}N)^2)$  or worse. Alternatively, iterative solvers may outperform LU decomposition in terms of CPU scalability, contingent on improving the condition number of T via applying an appropriate preconditioner [Rigas et al., 2021, Wu et al., 2022], which itself may or may not be inexpensive to obtain.

The cost of using RSVD- $\Delta t$  is directly tied to the cost of computing the actions of H and  $H^*$ . In resolvent analysis, the total CPU cost is proportional to the state dimension N. This cost can be broken down into three main components: (i) time-stepping, e.g., Adams-Bashforth methods, which scales as O(N) when A is sparse, (ii) creating forcing from Fourier space to the time domain, which scales as  $O(NN_tN_\omega)$ , (iii) taking the response from the time domain to Fourier space, which scales as  $O(NN_t^2)$ . Among these components, the time-stepping part is the most significant contributor to the overall cost. The CPU cost of harmonic resolvent analysis can be assessed similarly to resolvent analysis, with one exception. For periodic flows, the LNS operator varies over time, whereas it remains constant for statistically stationary flows. We showed in §5.1 that the cost of creating LNS operators over time scales as  $O(NN_b)$  for sparse matrices.

In summary, the CPU and memory costs of RSVD- $\Delta t$  exhibit linear scaling with respect to the state dimension N, regardless of whether the base flow is steady or periodic. Moreover, the dominant terms in the CPU cost are independent of the number of retained frequencies  $N_{\omega}$ . Both of these properties make the RSVD- $\Delta t$  algorithm considerably more scalable than RSVD-LU for harmonic resolvent analysis.

## <span id="page-14-0"></span>6 Minimizing the CPU cost of RSVD- $\Delta t$

Similar to time stepping in the context of resolvent analysis, periodic systems may also experience a slow decay of transient response, which is undesirable, since we require the steady-state response in isolation to compute the action of the harmonic resolvent operator using time stepping. This phenomenon might be linked to instances when T is near singularity. Regardless, in cases where the LNS equations are absolutely stable but exhibit slowly decaying modes, we propose a strategy that can effectively reduce the CPU cost of time stepping.

## 6.1 Stability analysis: Floquet theorem and transient response

The steady-state response of the LNS equations, subject to forcing as described in equation (3.1), is of interest for our analysis. In the case of a linear time-invariant system where A is independent of time, the decay rate is controlled by the least-damped eigenvalue of A. This decay rate can be estimated by integrating a homogeneous system over a sufficiently long period. However, for a linear time-periodic system, the decay rate is determined by the least-damped Floquet exponent.

Assuming  $\Phi(t)$  is the fundamental solution of (3.1), the monodromy matrix is constructed as

$$\mathbf{M} = \mathbf{\Phi}(0)^{-1}\mathbf{\Phi}(T),\tag{6.1}$$

where T is the period of the flow. When  $\Phi(0) = I, \Phi(t)$  is referred to as the principal fundamental matrix, and the monodromy matrix simplifies to  $\Phi(T)$ . In other words, we evaluate the principal fundamental matrix at the end of the first period. The eigenvalues of M, known as Floquet multipliers  $\mu_j$ , allow us to determine the Floquet exponents  $\lambda_j = \log(\mu_j)/T$ . The least-damped Floquet exponent has the smallest real part and it determines the decay rate of the transient response. If any mode is located in the unstable half-plane, i.e.,  $\Re(\lambda_j) > 0$ , the system is globally unstable.

It is worth noting that the transient length remains independent of the steady-state period. This characteristic is particularly important as it ensures that the length of integration does not rely on T, thereby avoiding potential deterioration in the performance of RSVD-∆t for flows with low periodicity. In practice, both finding the principal fundamental matrix and determining the Floquet exponents can be computationally expensive, especially for large-scale systems. As with autonomous cases, it is possible to run a homogeneous simulation to analyze the long-term behavior of the transient response for periodic flows. This equivalence has been illustrated on a test case in §[7.](#page-17-0)

If T is singular but all other modes are stable, then the Floquet exponent with the smallest real part will be zero with d̂q¯/dt as the corresponding Floquet mode. This is a well-known fact from Floquet theory [\[Wereley,](#page-26-0) [1990\]](#page-26-0) and can also be seen from [\(3.12\)](#page-5-0), where the phase shift direction is a non-trivial solution of the homogenous system. [Leclercq and Sipp](#page-26-16) [\[2023\]](#page-26-16) have also shown that dq¯/dt is associated with a zero Floquet exponent under the assumption that the base flow satisfies the Navier-Stokes equations as in [\(3.11\)](#page-5-1).

## 6.2 Efficient transient removal

Our strategy leverages the periodic nature of the steady-state part and the linear evolution of the transient part of the solution to directly compute and eliminate the undesired transient portion. For a given pair of solutions q<sup>1</sup> and q<sup>2</sup> at times t<sup>1</sup> and t<sup>2</sup> = t1+T, respectively, they can be expressed as a sum of their steady-state and transient components as

<span id="page-15-0"></span>
$$q_1 = q_{s,1} + q_{t,1},$$
  
 $q_2 = q_{s,2} + q_{t,2},$  (6.2)

where qt,<sup>1</sup> and qt,<sup>2</sup> represent the transient part which decays to zero as t → ∞ and qs,<sup>1</sup> and qs,<sup>2</sup> denote the steady-state part which evolves periodically. The time distance between two snapshots is one period, hence

<span id="page-15-1"></span>
$$\boldsymbol{q}_{s,2} = \boldsymbol{q}_{s,1}. \tag{6.3}$$

Also, the evolution of the transient part can be expressed as

<span id="page-15-2"></span>
$$\mathbf{q}_{t,2} = \mathbf{\Phi}(T)\mathbf{q}_{t,1},\tag{6.4}$$

where Φ is the principal fundamental matrix of [\(3.1\)](#page-3-2). Simplifying [\(6.2\)](#page-15-0), [\(6.3\)](#page-15-1), and [\(6.4\)](#page-15-2) leads to

<span id="page-15-3"></span>
$$(\mathbf{\Phi}(T) - \mathbf{I})\mathbf{q}_{t,1} = \mathbf{b},\tag{6.5}$$

where b = q<sup>2</sup> − q<sup>1</sup> is known.

By obtaining qt,<sup>1</sup> from solving [\(6.5\)](#page-15-3), the steady-state solution can be recovered as qs,<sup>1</sup> = q1−qt,1. The central challenge in removing the transient is the computational cost associated with solving the linear system in [\(6.5\)](#page-15-3). Instead, we employ Petrov-Galerkin (or Galerkin) projection to obtain an approximate solution to [\(6.5\)](#page-15-3) in a more cost-effective manner.

A low-dimensional representation of the transient part can be expressed as

<span id="page-15-4"></span>
$$\mathbf{q}_{t,1} = \boldsymbol{\phi} \boldsymbol{\beta}_1, \tag{6.6}$$

where ϕ ∈ C N×r is an orthonormal low-dimensional trial basis with r ≪ N, and β<sup>1</sup> ∈ C r represents the coefficients describing the transient response in this basis. Substituting [\(6.6\)](#page-15-4) into [\(6.5\)](#page-15-3), the linear system

<span id="page-15-5"></span>
$$(\mathbf{\Phi}(T) - \mathbf{I})\phi\beta_1 = \mathbf{b} \tag{6.7}$$

![](_page_16_Figure_0.jpeg)

<span id="page-16-1"></span>Figure 3: Flowchart depicting the action of H using time stepping (a) with efficient transient removal strategy and (b) with natural transient decay.

is overdetermined. We use Petrov-Galerkin projection with a low-dimensional test basis ψ ∈ C N×r to close [\(6.7\)](#page-15-5), yielding

<span id="page-16-0"></span>
$$\tilde{\boldsymbol{M}}\boldsymbol{\beta}_1 = \boldsymbol{\psi}^* \boldsymbol{b},\tag{6.8}$$

where

$$\tilde{\boldsymbol{M}} = \boldsymbol{\psi}^* (\boldsymbol{\Phi}(T) - \boldsymbol{I}) \boldsymbol{\phi} \tag{6.9}$$

is the map between coefficients. Solving [\(6.8\)](#page-16-0) for β<sup>1</sup> is inexpensive due to its reduced dimension, and from [\(6.6\)](#page-15-4) the transient estimate is obtained as

$$q_{t,1} = \phi(\psi^*\phi - \tilde{M})^{-1}\psi^*b.$$
 (6.10)

The reduced operator M˜ is obtained by integrating the columns of ϕ for one period, giving Φ(T)ϕ, and then projecting (Φ(T) − I)ϕ against the columns of ψ. This integration happens once and its CPU cost is equivalent to integrating the LNS equations for r periods. The analogous process occurs for the adjoint equations.

How should the trial and test bases be selected? We have found a trial basis spanning the least-damped modes to be effective. Rather than determining these modes through an exhaustive Floquet analysis, we employ a homogenous simulation as illustrated in the top row of the flowchart in figure [3.](#page-16-1) By initiating a sufficiently long simulation from a normalized random initial condition, the system asymptotically converges to the least-damped modes. The longer the integration length, the lower-dimensional the space becomes. As demonstrated in a test case in §[7,](#page-17-0) the dimension of the trial bases can, in some cases, be as small as a single column. Our test cases employ Galerkin projection with identical trial and test bases.

The obtained trial basis effectively captures the transient response at a specific phase θ. Utilizing ϕ and performing Galerkin projection, we obtain M˜ . The time stepping is then carried out for a sufficiently long duration to allow the initial transients to vanish before applying [\(6.8\)](#page-16-0) to determine the transient at phase  $\theta$  as shown in the middle row of the flowchart in figure 3. Once the steady-state response at the same phase is updated as  $q_{s,1} = q_1 - q_{t,1}$ , it is used as a new initial condition, which is synchronized with the forcing and will not excite a transient response (within the span of the bases used to construct  $\tilde{M}$ ). Then, integration for one period is sufficient, and  $N_{\omega}$  steady-state snapshots are collected as desired. The resulting steady-state snapshots are identical to those obtained by a prolonged wait in the case of natural decay, as illustrated in the bottom row of the flowchart in figure 3. This procedure is also applicable to the adjoint LNS equations. Given that the adjoint equations differ from the LNS equations, a new basis needs to be constructed for them. Implementing this strategy can reduce the integration length by a factor of 10 or more, depending on the desired accuracy.

<span id="page-17-0"></span>

## 7 Test cases

We assess the effectiveness of RSVD- $\Delta t$  in periodic systems using two test cases. First, we use a modified Ginzburg-Landau equation to validate the RSVD- $\Delta t$  algorithm and to illustrate the transient removal strategy. Second, we consider a periodic flow around an airfoil, similar to a test case from Padovan et al. [2020], to evaluate the accuracy and cost-savings of the RSVD- $\Delta t$  algorithm compared to RSVD-LU.

## 7.1 Periodically varying complex Ginzburg-Landau equation

The one-dimensional complex Ginzburg-Landau equations is a widely used model for understanding and controlling the non-modal growth of instabilities in transitional and turbulent shear flows [Chomaz et al., 1988, Hunt and Crighton, 1991, Bagheri et al., 2009, Chen and Rowley, 2011, Cavalieri et al., 2019]. Here, a modified system is used as an inexpensive test case to validate our algorithm. The original complex Ginzburg-Landau system follows the form of (2.1) with

$$\begin{aligned} \mathcal{A} &= -\nu \frac{\partial}{\partial x} + \gamma \frac{\partial^2}{\partial x^2} + \mu(x), \\ \mu(x) &= (\mu_0 - c_\mu^2) + \frac{\mu_2}{2} x^2, \end{aligned} \tag{7.1}$$
  
$$\mathbf{B} = \mathbf{C} = \mathbf{I}.$$

Following Bagheri et al. [2009], we set  $c_{\mu} = 0.2, \mu_2 = -0.01, \gamma = 1$  – i, and  $\nu = 2 + 0.2$ i. This system is globally stable when  $\mu_0 < \mu_{0,cr} \approx 0.3977$  [Bagheri et al., 2009]. By substituting  $\mu_0(t) = \bar{\mu}_0 + \mu_p \sin(\omega_f t - \pi/2)$  in place of  $\mu_0$ , we transform the Ginzburg-Landau equations to a periodically varying system with fundamental frequency  $\omega_f$ , following the form of (3.1). We set  $\mu_0 = 0.395$ ,  $\mu_p = 0.1$ , and  $\omega_f = 0.1$ ; the mean value of  $\mu(t)$  is equal to  $\mu_0$ , whose value is close to the critical value  $\mu_{0,cr}$  to promote significant growth. The periodic linearized operators  $A_{p,j}$  are constructed using a central finite-difference method for  $x \in [-50, 50]$  with N = 1000 grid points, and we set W = I on account of the uniform grid.

Nine operators are built within the interval [0,T), with  $T = 2\pi/\omega_0 \approx 62.83$ , such that  $A_p(t) = A_p(t+T)$ , before conducting a DFT to obtain  $\hat{A}_p$ . Figure 4(a) depicts the Frobenius norm of  $\hat{A}_p$  up to the fourth harmonic, normalized relative to the maximum norm. The spectrum reveals that only the mean and the first harmonic are active, as expected, since  $\mu_0(t)$  oscillates at a single frequency. Accordingly,  $A_p$  can be reconstructed at any given time using  $\hat{A}_{p,0}$ , and  $\hat{A}_{p,\pm 1}$ , and the harmonic resolvent operator is constructed around  $\Omega_{\bar{q}} = \{-1,0,1\}\omega_f$ , i.e.,  $N_b = 3$ . Notably, the first harmonic's norm is several orders of magnitude smaller than that of the base flow. Nonetheless, it should not be disregarded, as we will elucidate its significance in subsequent discussions.

![](_page_18_Figure_0.jpeg)

<span id="page-18-0"></span>Figure 4: Ginzburg-Landau test case: (a) the normalized Frobenius norm of  $\hat{A}_{p,\omega}$  up to the fourth harmonic. (b) The norm of the transient response over time (blue) along with the expected decay rate from Floquet analysis (red).

An essential initial step involves ensuring the stability of  $A_p$ . Typically, this is achieved by integrating the homogeneous system, *i.e.*, integrating (3.1) with zero forcing, starting from a normalized random initial condition. However, the compact size of this system allows us to validate the time-stepping approach for computing the least-damped decaying mode against a Floquet stability analysis. The least-damped Floquet exponent, computed as  $\lambda_l = -0.0021 - 0.0477i$ , is expected to govern the decay rate of the transient response. Figure 4(b) displays the norm of snapshots of the homogeneous system response over time, along with a reference line representing  $e^{\lambda_l t}$ . The natural decay rate aligns with the reference line, as expected, ensuring that the transient run will ultimately converge to the least-damped Floquet mode. Additionally, the system is not singular, as the transient dynamics never reach a naturally stable state, and none of the Floquet exponents have a zero real value.

Before proceeding to the computation of harmonic resolvent modes and gains using RSVD- $\Delta t$  and RSVD-LU, the transient removal strategy is demonstrated, and computing the actions of  $\boldsymbol{H}$  and  $\boldsymbol{H}^*$  are validated for a given random forcing containing  $\Omega_q \in \{-10, -9, ..., 9, 10\}\omega_f$ , *i.e.*,  $\omega \in [-1, 1]$  with  $\Delta \omega = 0.1$ . The responses have been computed up to the  $20^{th}$  harmonic, *i.e.*,  $\omega \in [-2, 2]$ , using both time stepping and directly in Fourier space as outlined in (3.7). We used a  $4^{th}$  order Runge–Kutta (RK4) integration scheme and a time step of dt = 0.001 and  $T_t = 15000$  for this purpose.

Figure 5(a) depicts the spectrum of response norms. In order to highlight the substantial difference in output generated by the harmonic resolvent operator around the base flow ( $\Omega_{\bar{q}} = \{0\}\omega_f, N_b = 1$ ), the norm of the responses for the same input forcing is also presented. Figure 5(b) displays the relative errors. The harmonic resolvent with  $N_b = 1$  produces inaccurate results across the entire spectrum and has zero response for  $|\omega| > 1$  since the modes at different frequencies are entirely decoupled. Therefore, neglecting  $\hat{A}_{p,\pm 1}$  is impractical, as mentioned earlier, and induces significant alterations in the harmonic resolvent modes and gains. On the other hand, the time-stepping output exhibits almost machine-precision accuracy within  $|\omega| \le 1$ , and its accuracy diminishes as response norms decrease within  $|\omega| > 1$ . The validation of  $H^*$  follows a similar procedure but is omitted here for brevity.

The transient norm decay rate, as shown in figure 4(b), follows  $e^{-0.0021t}$ . This indicates that approximately 40 periods are necessary for the transient norm to decay to 1%. However, we can significantly reduce this duration by employing our transient removal strategy. To begin, we precompute the trial basis by conducting a sufficiently long homogeneous simulation, and we find that a single mode suffices. Next, we integrate the orthogonalized basis for one period to obtain  $\tilde{M}$  and complete the preprocessing step.

![](_page_19_Figure_0.jpeg)

<span id="page-19-0"></span>Figure 5: Ginzburg-Landau test case: (a) the norms of the Fourier space response for  $N_b = 3$  (black) and  $N_b = 1$  (blue), along with the norm of the steady-state response obtained via time-stepping (red). (b) The relative errors of the norms, with the norms from Fourier space with  $N_b = 3$  (black line in (a)) serving as the reference.

Employing the same forcing as the validation case, we obtain the responses in time at the identical phase (initiated at t = 0 or  $\theta = 0$  with a time interval of  $\Delta t = T$ ), and compare them with the solutions after the removal of transients. Figure 6(a) presents the norm of the responses, which illustrates the effectiveness of our strategy commencing from the second period. This plot suggests that after two periods using our transient removal strategy with Galerkin projection, we obtain an accurate steady-state snapshot which can be used as the synchronized initial condition for the forced equations. Figure 6(b) demonstrates that the snapshots ultimately align perfectly with the updated snapshot. Hence, using our strategy, a total of three periods are required – two periods before transient removal and one period after the initial condition is obtained – to acquire all  $N_{\omega}$  steady-state snapshots, a 10-fold speed-up compared to the natural decay. A similar observation was made regarding the adjoint equations. In brief, our strategy can significantly reduce computational costs, potentially by one or more orders of magnitude, depending on the desired level of accuracy.

![](_page_19_Figure_3.jpeg)

<span id="page-19-1"></span>Figure 6: Ginzburg-Landau test case: (a) the response norms before (black) and after (red) transient removal at the same phase, *i.e.*, snapshots with  $\Delta t = T$  interval. (b) Comparison of the normalized response at the second period in blue, the  $50^{th}$  period in black, and the second period after transient removal in red.

Finally, RSVD- $\Delta t$  employing an RK4 integration scheme with dt = 0.003 and  $T_t = 2T$  is carried out along with our efficient transient removal strategy to compute the harmonic resolvent modes for k = 5 and q = 1. To establish a reference, the harmonic resolvent modes are also computed

using RSVD-LU with an equivalent number of test vectors and power iterations. Figure 7(a) illustrates that RSVD- $\Delta t$  computes the five leading gains identically to RSVD-LU. The gain relative error in figure 7(b) confirms that relative errors remain below  $10^{-9}$  across all modes. The inner products of response and forcing modes, computed via RSVD- $\Delta t$  and RSVD-LU, demonstrate parallel directions as the relative errors remain below  $10^{-8}$  across all modes. The inner-product error between two unit-norm vectors  $\mathbf{v}_1$  and  $\mathbf{v}_2$  is defined as

<span id="page-20-1"></span>
$$e_{ip} = 1 - \langle \boldsymbol{v_1}, \boldsymbol{v_2} \rangle. \tag{7.2}$$

![](_page_20_Figure_2.jpeg)

<span id="page-20-0"></span>Figure 7: Ginzburg-Landau test case: (a) the five leading gains using RSVD- $\Delta t$  (blue) and RSVD-LU (red). (b) The corresponding gain relative error (purple) and relative inner product errors (7.2) between the response modes (cyan) and the forcing modes (green) computed via RSVD- $\Delta t$  and RSVD-LU.

## 7.2 Flow over an airfoil

Our second test case consists of the flow over a NACA0012 airfoil at a low Reynolds number of Re = 200 and a steep angle of attack of  $\alpha = 20^{\circ}$ , which is dominated by periodic motions. This problem serves as a benchmark for computational cost and accuracy comparisons. A direct numerical simulation is conducted using the "CharLES" compressible flow solver. To mimic the incompressible simulations by Padovan et al. [2020], we set the Mach number to 0.05. The simulation employs a C-shaped mesh with a total of 62,000 cells. The leading edge of the airfoil is located at the origin  $(x/L_c, y/L_c) = (0,0)$ , where  $L_c = 1$  is the chord length. The computational domain spans  $x/L_c \times y/L_c \in [-49,50] \times [-50,50]$ . A constant time step of  $\Delta t U_{\infty}/L_c = 6.88 \times 10^{-5}$  is used, corresponding to a Courant-Friedrichs-Lewy number of 0.91, where  $U_{\infty}$  is the inflow streamwise velocity. Integration continues for a sufficient duration until the flow reaches a periodic state. Figure 8(a) displays the power spectral density computed from the transverse velocity at  $(x/L_c, y/L_c) = (3.0, -0.43)$ , where a pronounced vortex shedding occurs behind the trailing edge. The shedding frequency is  $St_f = 0.114$ , where the Strouhal number is defined as  $St = \frac{\omega L_c sin(\alpha)}{2\pi U_{\infty}}$ . The consistency between the dominant frequency identified in the power spectral density and obtained from our linearized code has been confirmed in our previous work [Jung et al., 2023].

The linearized operators are constructed at 100 time points within  $T = 2\pi/St_f \approx 55$ . Performing a DFT on the  $A_i$ , we generate 100 LNS operators  $\hat{A}_j$  in Fourier space to be used for harmonic resolvent analysis. The Frobenius norms  $||\hat{A}_{p,St}||_F$ , depicted in figure 8, indicate that  $A_p(t)$  can be accurately reconstructed using up to the  $5^{th}$  harmonic. Given that the linearized operator  $A_p(t)$  is real-valued for this problem, we retained only  $\lfloor N_b/2 \rfloor + 1 = 6$  coefficients, *i.e.*, the zeroth and 5 positive harmonics. For time stepping, dt = 0.0045 is chosen to ensure the stability and accuracy of

![](_page_21_Figure_0.jpeg)

<span id="page-21-0"></span>Figure 8: Airfoil test case: (a) the PSD spectrum based on transverse velocity at  $(x/L_c, y/L_c) = (3.0, -0.43)$ . (b) The normalized Frobenius norm of  $\hat{A}_{p,St}$  up to the eighth harmonic.

the RK4 integration scheme. The input and output perturbation frequency range spans up to the  $7^{th}$  harmonic, i.e.,  $N_{\omega} = 15$ . The domain of interest is  $x/L_c \times y/L_c \in [-4, 12] \times [-2.5, 2.5]$ , identical to that in Padovan et al. [2020]. We seek optimal forcing and response under the Chu energy norm [Chu, 1965], which has been used in several previous studies [Towne et al., 2018, Schmidt et al., 2018, Heidt and Colonius, 2023]. The number of test vectors is set to k = 5, and q = 2 power iteration proves sufficient for the convergence of both gains and modes.

A homogeneous simulation of the time-periodic system exhibits a slow decay rate, so we employ the transient removal strategy. Three snapshots constitute the trial basis, obtained after a sufficiently long time interval where most initial transients are eliminated. A random forcing, encompassing the set  $\Omega_q \in \{-7, -6, ..., 6, 7\} St_f$  frequencies, is applied to the LNS equations for over 30 periods. The snapshot norms slowly approach towards the steady-state norm but remain far from converging, as illustrated in figure 9. Utilizing our strategy, the relative norm error falls below the 1% threshold after 20-30 periods in most cases with different random forcings. The efficacy of our strategy is further emphasized in figure 9(b), where the 1% relative error is achieved before 10 periods for the optimal forcing (from RSVD- $\Delta t$  output). Ultimately, we set  $T_t = 29T$  and conducted our simulations for a total duration of 30 periods to compute the actions of H and  $H^*$ . Extrapolating the natural decay rate suggests that, in the absence of our transient-removal strategy, the norms reach the 1% threshold after around 2000 periods, demonstrating over a 60-fold acceleration in time stepping and overall algorithm performance with our strategy. This observation holds true for the adjoint equations as well.

![](_page_21_Figure_4.jpeg)

<span id="page-21-1"></span>Figure 9: Airfoil test case: (a) the response norms before (black) and after (red) transient removal, and the true steady-state norm (blue), at the same phase, *i.e.*, snapshots with  $\Delta t = T$  interval, for a random forcing. (b) The relative error between the steady-state response norm and the response norms before (black) and after (red) transient removal.

The optimal forcing and response modes (apart from the phase shift mode) are forcing and response modes are shown in figure [10.](#page-23-1) The vorticity patterns of the first output mode of the harmonic resolvent closely predict vortical structures observed in simulations that are induced by sinusoidal perturbations to the periodic base flow across all frequencies[\[Padovan et al.,](#page-25-0) [2020\]](#page-25-0). This close match is attributed to the low-rank nature of the harmonic resolvent operator of the airfoil. This agreement also implies that regardless of the forcing type, resultant flow perturbations resemble the suboptimal output modes. Moreover, the proximity of forcing modes to the airfoil suggests sensitivity to perturbations near it. The significance of harmonic resolvent analysis lies in its ability to uncover flow structures that would remain hidden when linearizing around the temporal mean. It reveals the intricate interplay between different frequencies, the base frequency and its harmonics, which collectively contribute to the observed flow patterns during real perturbation analysis.

The results from RSVD-∆t closely align with the existing data obtained from RSVD-LU by [Padovan et al.](#page-25-0) [\[2020\]](#page-25-0), despite variations in the CFD solver, boundary conditions, domain setup, and energy norm. The gain plot in figure [11\(](#page-24-0)a) exhibits a similar pattern to [Padovan et al.](#page-25-0) [\[2020\]](#page-25-0), with the exception that we have retained the optimal gain associated with the phase shift (the exact values differ due to differences in the problem setup). We noted that the leading mode and gain, i.e., the mode associated with the phase shift, converged without power iterations due to a substantial two-orders-of-magnitude gap between the optimal and suboptimal gains.

From a computational standpoint, a fair comparison between the RSVD-LU and RSVD-∆t algorithms is feasible when both utilize the same parameters. However, the use of N<sup>b</sup> = 11 proves to be excessively memory-intensive for the RSVD-LU algorithm, exceeding the available 3.5 TB of memory on our cluster. The CPU and memory scaling plots in figure [11\(](#page-24-0)b) and (c), respectively, depict the LU decomposition of T with N<sup>b</sup> = 3, maintaining a constant N while varying the number of blocks N<sup>ω</sup> from 5 to 15 with ∆N<sup>ω</sup> = 2. On the same plots, we present the cost of computing the action of H on a vector using time-stepping. The total duration is set to 30 periods to obtain an accurate solution. We only vary N<sup>ω</sup> while keeping N<sup>b</sup> = 3, the overall cost of RSVD-∆t remains constant and unaffected by changes in Nω. This implies that the creation of LNS operators and the time-stepping process incur significantly higher costs compared to the transformations between Fourier space and the time domain for forcing and response. The memory of RSVD-∆t to store LNS operators remains independent of Nω, as expected (see §[5.1\)](#page-12-1), while storing Q and F matrices grows linearly with the number of frequencies Nω.

RSVD-LU is estimated to require 1543 CPU-hours for the LU decomposition of T , and an additional 15 CPU-hours for solving the LU-decomposed system for each test vector, yielding a total cost of CPURSVD-LU ≈ 1543 + 15 × 5 × 2 × 2 = 1843 CPU-hours for k = 5, q = 1. On the other hand, employing RSVD-∆t with the aforementioned time-stepping parameters and N<sup>b</sup> = 11 and N<sup>ω</sup> = 15 incurs a cost of approximately 7 CPU-hours per period for a test vector, resulting in a total cost of CPURSVD-∆<sup>t</sup> ≈ 30 × 7 × 5 × 2 × 2 = 4200 CPU-hours. The cost of QR decomposition and the final SVD are ignored as they are orders of magnitude faster. In terms of memory consumption, RSVD-LU peaks at RAMRSVD-LU ≈ 2036GB for the LU decomposition of T . RSVD-∆t , on the other hand, stores ⌊Nb/2⌋ + 1 = 6 sparse matrices, each of size 0.4 GB, and three dense matrices of size N k(⌊Nω/2⌋+1) = 313630×5×8, totaling RAMRSVD-∆<sup>t</sup> ≈ 0.4×6+0.185×3 ≈ 3GB. This translates to a memory saving of approximately three orders of magnitude, even with an unbalanced number of base frequencies. Overcoming the memory consumption hurdle, which is typically the limiting factor in practice, RSVD-∆t emerges as a viable tool for analyzing larger-scale flows compared to RSVD-LU.

![](_page_23_Figure_0.jpeg)

<span id="page-23-1"></span>Figure 10: Airfoil test case: real part of the vorticity field computed from (a, c, e, g) the input mode and (b, d, f, h) the output mode associated with the first suboptimal gain.

# <span id="page-23-0"></span>8 Conclusions

This paper presents an extension of the RSVD-∆t algorithm that was originally developed for resolvent analysis of steady base flows to handle harmonic resolvent analysis of periodic flows. Specifically, we demonstrate that the time-stepping technique employed within RSVD-∆t can effectively replace the actions of H and its complex conjugate H<sup>∗</sup> in Fourier space. In terms of CPU cost, the RSVD-∆t algorithm exhibits a scaling of O(N) for both resolvent and harmonic resolvent analyses. This offers a significant advantage, considering that the LU decomposition of T scales as O((NωN) c ) with c ≥ 1.5. Regarding memory usage, our algorithm only stores relevant matrices in Fourier space and exhibits O(NNω) scaling; in contrast, the memory peak of LU decomposition of T empirically scales with O((NωN) 1.5 ) or worse. One difference between applying RSVD-∆t to resolvent and harmonic resolvent analyses lies in the necessity to update the LNS operators during the time-stepping process for the latter. These operators are efficiently generated on the fly from their N<sup>b</sup> Fourier components.

The error sources in our algorithm align with those in resolvent analysis, extensively investigated in [Farghadan et al.](#page-26-12) [\[2023\]](#page-26-12). A unique contribution of this paper is the introduction of a novel transient removal strategy for harmonic resolvent analysis. While reminiscent of our approach for resolvent operators, the challenge in harmonic resolvent analysis lies in the intertwining of all

![](_page_24_Figure_0.jpeg)

<span id="page-24-0"></span>Figure 11: Airfoil test case: (a) the five leading gains using RSVD- $\Delta t$ . (b) The CPU-hour, and (c) memory usage scaling of RSVD-LU (red) and RSVD- $\Delta t$  (blue) to compute the action of  $\boldsymbol{H}$  onto a vector, *i.e.*, k=1. The memory usage of RSVD- $\Delta t$  is decomposed into memory required to store LNS operators (solid) and forcing and response matrices (dashed).

retained frequencies. Our strategy takes advantage of the differing evolution of the steady-state and transient components of the response and Petrov-Galrkin or Galerkin projections.

A validation of RSVD- $\Delta t$  against RSVD-LU is conducted using a periodic Ginzburg-Landau system. Additionally, we verify the governing role of the Floquet exponent in determining the decay rate of the least-damped mode of the system. Extending the application of our algorithm, we analyze a two-dimensional flow passing an airfoil, providing insights into the CPU and memory complexities. Despite dealing with a mid-sized flow scenario, a substantial memory gap persists between RSVD-LU and RSVD- $\Delta t$  algorithms. The computed forcing and response modes corresponding to the first suboptimal gain closely resemble those obtained by Padovan et al. [2020]. The adaptation of the transient removal strategy for these test cases significantly enhanced the performance of RSVD- $\Delta t$ , resulting in a 10-fold speed-up for the Ginzburg-Landau problem and a 60-fold speed-up for the airfoil to reach a 1% relative error. The speed-up is even greater for lower error tolerances.

Moreover, our algorithm exhibits versatility, accommodating non-identity weight, input, and output matrices. This capability extends to computing modes for the modified harmonic resolvent operator, and in particular, the analysis of cross-frequency amplification mechanisms via  $H_{\omega_2,\omega_1}$ . Our algorithm tackles the computation of subharmonic resolvent modes if desired. Implementation of the RSVD- $\Delta t$  algorithm within Petsc [Balay et al., 2019] and Slepc [Hernandez et al., 2005] environments leverage parallel computations for enhanced efficiency. Our time-stepping approach allows for a matrix-free application. This can be implemented using any code equipped with linear direct and adjoint capabilities, bypassing the explicit formation of T [de Pando et al., 2012, Martini et al., 2021]. Finally, the time stepping within our algorithm can also be used along with other SVD algorithms, e.q., Arnoldi and power iteration, if desired.

In turbulent flows, where the state dimension N can become very large due to higher resolution requirements, CPU cost and memory requirements can swiftly impose constraints on the applicability of RSVD-LU. This limitation applies even to steady problems and is further exacerbated for harmonic resolvent analysis due to the inflated frequency-domain operators necessitated by triadic frequency coupling. Our algorithm circumvents this issue by strategically operating in the time domain, avoiding the need for limiting operations like LU decomposition, leading to linear cost scaling. Because of this key distinction, the RSVD- $\Delta t$  could enable harmonic resolvent analysis of previous intractable turbulent flows.

# Acknowledgements

This work was funded in part by Air Force Office of Scientific Research (AFOSR) grant no. FA9550- 20-1-0214 and by a Catalyst Grant from the Michigan Institute for Computational Discovery and Engineering (MICDE). We acknowledge the University of Michigan's Great Lakes cluster for providing the essential computational resources to conduct all the simulations in this study.

# <span id="page-25-9"></span>Appendix A RSVD-∆t for the subharmonic resolvent operator

In §[3.5,](#page-8-0) we provided an overview of subharmonic resolvent analysis. In this appendix, we briefly outline the application of RSVD-∆t in computing subharmonic resolvent modes and gains. Consider the frequency of interest as γ ∈ Ωγ. This set specifies the perturbation frequency, while the base flow frequency content is confined to Ωq¯.

To compute the actions of H and H<sup>∗</sup> using time-stepping, we must adhere to both the base flow frequency, enforcing a duration of T = 2π/ω<sup>f</sup> , and the perturbation frequency, enforcing a duration of T<sup>p</sup> = 2π/γ, in order to obtain N<sup>ω</sup> steady-state solutions to [\(3.1\)](#page-3-2). Thus, we need to integrate for Tsub such that Tsub/T ∈ N and Tsub/T<sup>p</sup> ∈ N, during which N<sup>ω</sup> equidistant snapshots are saved. For instance, if we consider γ = ω<sup>f</sup> /5, requiring T<sup>p</sup> = 2π/(ω<sup>f</sup> /5) = 5T, integrating for Tsub = 5T is sufficient to obtain the steady-state solutions. All the other steps remain the same as introduced in Algorithm [1.](#page-11-0)

# References

- <span id="page-25-0"></span>A. Padovan, S. E. Otto, and C. W. Rowley. Analysis of amplification mechanisms and crossfrequency interactions in nonlinear flows via the harmonic resolvent. Journal of Fluid Mechanics, 900, 2020.
- <span id="page-25-1"></span>K. Taira, S. L. Brunton, S. T. M. Dawson, C. W. Rowley, T. Colonius, B. J. McKeon, O. T. Schmidt, S. Gordeyev, V. Theofilis, and L. S. Ukeiley. Modal analysis of fluid flows: An overview. AIAA Journal, 55(12):4013–4041, 2017.
- <span id="page-25-2"></span>A. Towne, O. T. Schmidt, and T. Colonius. Spectral proper orthogonal decomposition and its relationship to dynamic mode decomposition and resolvent analysis. Journal of Fluid Mechanics, 847:821–867, 2018.
- <span id="page-25-3"></span>S. T. M. Dawson and B. J. McKeon. On the shape of resolvent modes in wall-bounded turbulence. Journal of Fluid Mechanics, 877:682–716, 2019.
- <span id="page-25-4"></span>D. Sipp and O. Marquet. Characterization of noise amplifiers with global singular modes: the case of the leading-edge flat-plate boundary layer. Theoretical and Computational Fluid Dynamics, 27(5):617–635, 2013.
- <span id="page-25-5"></span>E. Pickering, A. Towne, P. Jordan, and T. Colonius. Resolvent-based modeling of turbulent jet noise. The Journal of the Acoustical Society of America, 150(4):2421–2433, 2021.
- <span id="page-25-6"></span>C. D. Troy and J. R. Koseff. The instability and breaking of long internal waves. Journal of Fluid Mechanics, 543:107–136, 2005.
- <span id="page-25-7"></span>A. E. Giannenas, S. Laizet, and G. Rigas. Harmonic forcing of a laminar bluff body wake with rear pitching flaps. Journal of Fluid Mechanics, 945:A5, 2022.
- <span id="page-25-8"></span>A. Farghadan and A. Arzani. The combined effect of wall shear stress topology and magnitude on cardiovascular mass transport. International Journal of Heat and Mass Transfer, 131:252–260, 2019.

- <span id="page-26-0"></span>N. M. Wereley. Analysis and control of linear periodically time varying systems. PhD thesis, Massachusetts Institute of Technology, 1990.
- <span id="page-26-1"></span>C.-T. Lin, M.-L. Tsai, and H.-C. Tsai. Flow control of a plunging cylinder based on resolvent analysis. Journal of Fluid Mechanics, 967:A41, 2023.
- <span id="page-26-2"></span>L. Franceschini, D. Sipp, O. Marquet, J. Moulin, and J. Dandois. Identification and reconstruction of high-frequency fluctuations evolving on a low-frequency periodic limit cycle: application to turbulent cylinder flow. Journal of Fluid Mechanics, 942:A28, 2022.
- <span id="page-26-3"></span>L. Heidt and T. Colonius. Spectral proper orthogonal decomposition of harmonically forced turbulent flows. arXiv preprint arXiv:2305.05628, 2023.
- <span id="page-26-4"></span>W. Wu, C. Meneveau, R. Mittal, A. Padovan, C. W. Rowley, and L. Cattafesta. Response of a turbulent separation bubble to zero-net-mass-flux jet perturbations. Physical Review Fluids, 7 (8):084601, 2022.
- <span id="page-26-5"></span>A. Padovan and C. W. Rowley. Analysis of the dynamics of subharmonic flow structures via the harmonic resolvent: Application to vortex pairing in an axisymmetric jet. Physical Review Fluids, 7(7):073903, 2022.
- <span id="page-26-6"></span>B. Jin, S. Symon, and S. J. Illingworth. Energy transfer mechanisms and resolvent analysis in the cylinder wake. Physical Review Fluids, 6(2):024702, 2021.
- <span id="page-26-7"></span>T. Herbert. Analysis of the subharmonic route to transition in boundary layers. In 22nd Aerospace Sciences Meeting, page 9, 1984.
- <span id="page-26-8"></span>D. Edgington-Mitchell, T. Wang, P. Nogueira, O. Schmidt, V. Jaunet, D. Duke, P. Jordan, and A. Towne. Waves in screeching jets. Journal of Fluid Mechanics, 913:A7, 2021.
- <span id="page-26-9"></span>C. Kumar and A. Prakash. Effect of mass injection on secondary instability of hypersonic boundary layer over a blunt cone. Physics of Fluids, 34(6), 2022.
- <span id="page-26-10"></span>R. Moarref, A. S. Sharma, J. A. Tropp, and B. J. McKeon. Model-based scaling of the streamwise energy density in high-Reynolds-number turbulent channels. Journal of Fluid Mechanics, 734: 275–316, 2013.
- <span id="page-26-11"></span>J. H. M. Ribeiro, C.-A. Yeh, and K. Taira. Randomized resolvent analysis. Physical Review Fluids, 5(3):033902, 2020.
- <span id="page-26-12"></span>A. Farghadan, E. Martini, and A. Towne. Scalable resolvent analysis for three-dimensional flows. arXiv preprint arXiv:2309.04617, 2023.
- <span id="page-26-13"></span>A. Monokrousos, E. ˚Akervik, L. Brandt, and D. S. Henningson. Global three-dimensional optimal disturbances in the blasius boundary-layer flow using time-steppers. Journal of Fluid Mechanics, 650:181–214, 2010.
- <span id="page-26-14"></span>E. Martini, D. Rodr´ıguez, A. Towne, and A. V. G. Cavalieri. Efficient computation of global resolvent modes. Journal of Fluid Mechanics, 919, 2021.
- <span id="page-26-15"></span>B. J. McKeon and A. S. Sharma. A critical-layer framework for turbulent pipe flow. Journal of Fluid Mechanics, 658:336–382, 2010.
- <span id="page-26-16"></span>C. Leclercq and D. Sipp. Mean resolvent operator of a statistically steady flow. Journal of Fluid Mechanics, 968:A13, 2023.
- <span id="page-26-17"></span>N. Halko, P. Martinsson, and J. A. Tropp. Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions. SIAM Review, 53(2):217–288, 2011.
- <span id="page-26-18"></span>H. Nyquist. Certain topics in telegraph transmission theory. Transactions of the American Institute of Electrical Engineers, 47(2):617–644, 1928.

- <span id="page-27-0"></span>A. Towne, G. Rigas, O. Kamal, E. Pickering, and T. Colonius. Efficient global resolvent analysis via the one-way Navier-Stokes equations. Journal of Fluid Mechanics, 948:A9, 2022.
- <span id="page-27-1"></span>G. Rigas, D. Sipp, and T. Colonius. Nonlinear input/output analysis: application to boundary layer transition. Journal of Fluid Mechanics, 911:A15, 2021.
- <span id="page-27-2"></span>J. M. Chomaz, P. Huerre, and L. G. Redekopp. Bifurcations to local and global modes in spatially developing flows. Physical Review Letters, 60(1):25, 1988.
- <span id="page-27-3"></span>R. E. Hunt and D. G. Crighton. Instability of flows in spatially developing media. Proceedings of the Royal Society of London. Series A: Mathematical and Physical Sciences, 435(1893):109–128, 1991.
- <span id="page-27-4"></span>S. Bagheri, D. S. Henningson, J. Hoepffner, and P. J. Schmid. Input-output analysis and control design applied to a linear model of spatially developing flows. Applied Mechanics Reviews, 62(2), 2009.
- <span id="page-27-5"></span>K. K. Chen and C. W. Rowley. H2 optimal actuator and sensor placement in the linearised complex Ginzburg–Landau system. Journal of Fluid Mechanics, 681:241–260, 2011.
- <span id="page-27-6"></span>A. V. G. Cavalieri, P. Jordan, and L. Lesshafft. Wave-packet models for jet dynamics and sound radiation. Applied Mechanics Reviews, 71(2), 2019.
- <span id="page-27-7"></span>J. Jung, R. Bhagwat, and A. Towne. Resolvent-based estimation of laminar flow around an airfoil. AIAA Paper #2023-0077, 2023.
- <span id="page-27-8"></span>B.-T. Chu. On the energy transfer to small disturbances in fluid flow (part i). Acta Mechanica, 1 (3):215–234, 1965.
- <span id="page-27-9"></span>O. T. Schmidt, A. Towne, G. Rigas, T. Colonius, and G. A. Br`es. Spectral analysis of jet turbulence. Journal of Fluid Mechanics, 855:953–982, 2018.
- <span id="page-27-10"></span>S. Balay, S. Abhyankar, M. Adams, J. Brown, P. Brune, K. Buschelman, L. Dalcin, A. Dener, V. Eijkhout, W. Gropp, et al. PETSc users manual. Argonne National Laboratory, 2019.
- <span id="page-27-11"></span>V. Hernandez, J. E. Roman, and V. Vidal. Slepc: A scalable and flexible toolkit for the solution of eigenvalue problems. ACM Transactions on Mathematical Software (TOMS), 31(3):351–362, 2005.
- <span id="page-27-12"></span>M. F. de Pando, D. Sipp, and P. J. Schmid. Efficient evaluation of the direct and adjoint linearized dynamics from compressible flow solvers. Journal of Computational Physics, 231(23):7739–7755, 2012.