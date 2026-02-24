set(groot, 'defaultAxesTickLabelInterpreter','latex'); set(groot, 'defaultLegendInterpreter','latex'); set(groot,'defaultTextInterpreter','latex')
clear variables
close all
clc
addpath(genpath('aux_matlab'));
addpath(genpath('BaseFlows'));

savepath = "Results_St0.1_beta0.0_NeumannOutBotTemp_SS_200-500_100eig_4thOrder_LargeMesh_Half/NoSponge_NoShock_NoFilter/";

% Physical parameters
% Reynolds number
baseFlow.Re = 395000;
% Mach number
baseFlow.Ma = 2.0;
% Prandtl number
baseFlow.Pr = 0.747;
% heat capacity ratio
baseFlow.kappa = 1.31;
% temperature
baseFlow.T_0 = 1540.0;

% Perturbation & EVP parameters
%ratio between the freestream velocity and the characteristic velocity used to non-dimensionalize the problem
ratio_Uinfty_to_characVelocity = 1.0;
%ratio between the length that the strouhal is defined (L) and the characteristic length used to non-dimensionalize the problem (Lx)
ratio_L_to_Lx = 0.1;
%Strouhal based on L
St = 0.1;
% azimuthal wave number
beta = 0.0;

%RESOLVENT
freq = St*ratio_Uinfty_to_characVelocity/ratio_L_to_Lx; % frequency
omega_resolvent = freq*2*pi;              % angular frequency
% Arnoldi method number of eigenvalues RESOLVENT
nEig = 10;

%STABILITY
omega_target    = omega_resolvent;  % value around which the eigenvalues will be seek
% Arnoldi method number of eigenvalues GSA
nEig_GS = 100;

% setup eigs options
opts.tol = 1e-8;
opts.disp = 2;
opts.p = 2*nEig_GS+3;

% Domain & grid
% finite difference order of accuracy
FDorder = 4;

% Flags
% visualize grid, base flow and results
verbose = false;
%%%% run resolvent
resolvent_flag = false;
%%% run stability analysis
stability_analysis_flag = true;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Create mesh and obtain differentiation matrices                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Cartesian mesh in computational domain
% use symmetry on y coordinate around y=0 (for axysymmetric problems) false
y_symmetry = false;
% use periodic b.c. on x false
x_periodicity = false;
% spatial filter coefficient 0.0
alphaFilter = 0.0;
% domain range in ksi [0 2];  
xrange = [0 1];
% domain range in eta [0 1];
yrange = [0 0.5];

load('mesh.mat')            
X               = x.' ;    % X,Y: Parabolic grid coordinates
Y               = y.';
[Nr Nz] = size(X);

cmesh           = CreateMesh(xrange,yrange,Nz,Nr,FDorder, ...     
                          y_symmetry,x_periodicity,alphaFilter); %construct mesh

mesh            = DeformMesh(cmesh,X,Y);


% Plot meshes in comutational and physical domains
if verbose
    figure
    
    subplot(1,2,1)
    title('Computatinal domain')
    hold on
    plot(cmesh.X, cmesh.Y, '-k', cmesh.X', cmesh.Y', '-k','HandleVisibility','off');
    for bondaries= fields(mesh.idx)'
        ids = mesh.idx.(bondaries{1});
        plot(cmesh.X(ids), cmesh.Y(ids),'linewidth',2);
    end
    xlabel('$\xi$');
    ylabel('$\eta$');
    legend(fields(mesh.idx))
    axis equal tight
    
    subplot(1,2,2)
    title('Physical domain')
    hold on
    plot(mesh.X, mesh.Y, '-k', mesh.X', mesh.Y', '-k','HandleVisibility','off');
    
    for bondaries= fields(mesh.idx)'
        ids = mesh.idx.(bondaries{1});
        plot(mesh.X(ids), mesh.Y(ids),'linewidth',2);
        
    end
    legend(fields(mesh.idx),'Location', 'Best')
    xlabel('$x$');
    ylabel('$y$');
    axis equal tight
end

% Custumize the next function to read your baseflow
baseFlow    = readbaseflow(mesh,baseFlow); 

% Viscosity (via Sutherland) and heat conductivity 
% (via constant Prandtl number)
[baseFlow]  = sutherland_air(baseFlow);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Sponge                                                                 %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% sponge defined in the original, non-deformed, mesh.
load('sponge')
spongeAmp   = 1.0;
mesh.sponge = spongeAmp*sponge_aux.';
if verbose
    figure(1001);
    pcolor(X,Y,mesh.sponge); shading interp
    colorbar;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Visualize base flow                                                    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if verbose
    figure('name','Base Flow')
    vars = {baseFlow.RHO ,'$\rho$';     baseFlow.U,'$U$';
            baseFlow.V,'$V$';           baseFlow.W,'$W$';
            baseFlow.T,'$T$';           mesh.sponge, 'sponge'
            baseFlow.MU,'$\mu$';        baseFlow.dmudT,'$\frac{d\mu}{dT}$';
            baseFlow.d2mudT2                                ,'$\frac{d^2\mu}{dT^2}$';
            reshape(mesh.Dy*(baseFlow.W(:)),Nr,Nz)          ,'$\frac{dW}{dy}$';
            reshape(mesh.Dx*(baseFlow.W(:)),Nr,Nz)         ,'$\frac{dW}{dx}$';
            reshape(mesh.Dx*(mesh.Dy*baseFlow.W(:)),Nr,Nz) ,'$\frac{d^2W}{dydx}$'};
    plotFlow(X,Y,vars,4,3);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Set up linear operator and boundary conditions                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%    dqdt = L0q
%%
[L0,idx] 	= GetLinProblem(mesh,baseFlow,'2D',beta);

% Enforce Dirichlet b,c on the bottom and top boundaries, for rho,u,v,w
penalty = 1e8;

%%% config BC suction side
borders     = 'lt';  vars = 'ruvwT';
[L0,idx_dirichlet_lt] ...
            = BC_Dirichlet(L0,idx,borders,vars,penalty);

borders     = 'b';  vars = 'ruvw';
[L0,idx_dirichlet_b] ...
            = BC_Dirichlet(L0,idx,borders,vars,penalty);

borders     = 'b';  vars = 'T';
[L0,idx_neumann_b] ...
           = BC_Neumann(L0,mesh,idx,borders,vars,penalty);

borders     = 'r';  vars = 'ruvwT';
[L0,idx_neumann_r] ...
           = BC_Neumann(L0,mesh,idx,borders,vars,penalty); 


%%% config BC pressure side
% borders     = 'rt';  vars = 'ruvwT';
% [L0,idx_dirichlet_lt] ...
%             = BC_Dirichlet(L0,idx,borders,vars,penalty);
% 
% borders     = 'b';  vars = 'ruvw';
% [L0,idx_dirichlet_b] ...
%             = BC_Dirichlet(L0,idx,borders,vars,penalty);
% 
% borders     = 'b';  vars = 'T';
% [L0,idx_neumann_b] ...
%            = BC_Neumann(L0,mesh,idx,borders,vars,penalty);
% 
% borders     = 'l';  vars = 'ruvwT';
% [L0,idx_neumann_r] ...
%            = BC_Neumann(L0,mesh,idx,borders,vars,penalty); 

%        
[W,invW]    = GetCompEnergyNorm(mesh,baseFlow,'2D');

%% save variables indexes
save(sprintf('../%s/idx.mat',savepath), 'idx')



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Resolvent analysis                                                     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% dq/dt = L*q + B*v
% u = C*q
% v: input/forcing, u:output/response
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if resolvent_flag
    % Input/forcing matrix
    B = ones(mesh.ngp*5,1);
    B([idx_dirichlet_lt]) = 0; % no forcing on dirichlet b.c.
    B([idx_dirichlet_b]) = 0; % no forcing on dirichlet b.c.
    B([idx_neumann_b]) = 0; % no forcing on neumann b.c.
    B([idx_neumann_r]) = 0; % no forcing on neumann b.c.
    % B([idx.w_j]) = 0; % no w forcing
    % B([idx.T_j;idx.rho_j]) = 0; % no temperature and density forcing

    B = spdiags(B,0,mesh.ngp*5,mesh.ngp*5);

    %Output/observable matrix
    C = B;    

    save(sprintf('../%s/LO.mat',savepath), 'L0')
    save(sprintf('../%s/W.mat',savepath), 'W')
    save(sprintf('../%s/invW.mat',savepath), 'invW')
    save(sprintf('../%s/B.mat',savepath), 'B')
    save(sprintf('../%s/C.mat',savepath), 'C')
    save(sprintf('../%s/mesh.mat',savepath), 'mesh')
    
    %Compute optimal forcings and responses 
    [gains,responses,forces] = resolvent(L0,omega_resolvent,nEig,W,invW,B,C,mesh.filters);
    
    %%
    % Plot modes and gains
    if verbose
       figure('name','Mode gains')
       bar(gains);
       xlabel('mode');
       ylabel('gain');
       title('Resolvent gains')

       figure('name','Resolvent forcing and response modes')
       vars = {real(forces(idx.u_j,1)) ,'$f_v^{(1)}$'; real(responses(idx.u_j,1)) ,'$v^{(1)}$'
               real(forces(idx.u_j,2)) ,'$f_v^{(2)}$'; real(responses(idx.u_j,2)) ,'$v^{(2)}$';
               real(forces(idx.u_j,3)) ,'$f_v^{(3)}$'; real(responses(idx.u_j,3)) ,'$v^{(3)}$';};
       plotFlow(mesh.X,mesh.Y,vars,3,2);

    end
    %%

    save(sprintf('../%s/ResolventModes/gains.mat',savepath), 'gains')

    [nx, nmodes] = size(forces);  % Get the dimensions of the structure

    for col = 1:nmodes
        % Extract the column from the structure
        forces_save = forces(:, col);
        responses_save = responses(:, col);

        rho_f = forces_save(idx.rho_j);
        u_f = forces_save(idx.u_j);
        v_f = forces_save(idx.v_j);
        w_f = forces_save(idx.w_j);
        T_f = forces_save(idx.T_j);

        rho_r = responses_save(idx.rho_j);
        u_r = responses_save(idx.u_j);
        v_r = responses_save(idx.v_j);
        w_r = responses_save(idx.w_j);
        T_r = responses_save(idx.T_j);

        % Create a filename
        file_forcing = sprintf('../%s/ResolventModes/forcing_mode_%d.mat',savepath, col);
        file_response = sprintf('../%s/ResolventModes/response_mode_%d.mat',savepath, col);

        % Save the column structure to the .mat file
        save(file_forcing, 'rho_f', 'u_f', 'v_f', 'w_f', 'T_f');
        save(file_response, 'rho_r', 'u_r', 'v_r', 'w_r', 'T_r');
    end
end


% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %% Global modes 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if stability_analysis_flag
    % Divide by -1i to convert eigenvalues from s to omega (=-1i s)
    L0 = L0/-1i;
    save(sprintf('../%s/LO.mat',savepath), 'L0')

    % compute direct and adjoint modes
    tic;
    [V    ,omega    ] = eigs(L0  ,nEig_GS,omega_target,opts);
    % omega=diag(omega);
    time_eigsMatlab = toc;

    save(sprintf('../%s/EigModes/spectrum.mat',savepath), 'omega')

    [nx, nmodes] = size(V);  % Get the dimensions of the structure

    for col = 1:nmodes
        % Extract the column from the structure
        columnStruct = V(:, col);

        rho = columnStruct(idx.rho_j);
        u = columnStruct(idx.u_j);
        v = columnStruct(idx.v_j);
        w = columnStruct(idx.w_j);
        T = columnStruct(idx.T_j);

        % Create a filename
        file_eigvec = sprintf('../%s/EigModes/eigenvector_%d.mat',savepath, col);

        % Save the column structure to the .mat file
        save(file_eigvec, 'rho', 'u', 'v', 'w', 'T');
    end
end