set(groot, 'defaultAxesTickLabelInterpreter','latex'); set(groot, 'defaultLegendInterpreter','latex'); set(groot,'defaultTextInterpreter','latex')
% run_ex3_paper_*.m sets Lx, Nx, nPeriods, ... in the caller workspace before run().
% A bare "clear" would wipe those; in batch mode keep driver overrides.
if ~isempty(getenv('MARTINI2024_BATCH'))
    clearvars -except martini_case_tag Lx Ly nPeriods Nx Ny temporalAna spatialAna
else
    clear variables
end
close all
clc
martini2024_setup_paths();
martini2024_configure_batch();
if isempty(getenv('MARTINI2024_BATCH'))
    verbose = true;
else
    verbose = false;
end
if ~exist('martini_case_tag', 'var') || isempty(martini_case_tag)
    martini_case_tag = 'bundled_Lxpi_nP2';
end

% Physical parameters
baseFlow.Re     = 28000;          % Reynolds number
baseFlow.Ma     = 6;          % Mach number
baseFlow.Pr     = 0.7;          % Prandtl number
baseFlow.kappa  = 1.4;          % heat capacity ratio
baseFlow.T_0    = 273.15 ;       % temperature


% Perturbation & EVP parameters
freq            = 0;            % frequency
omega_spatialAn = freq*2*pi;    % angular frequency for spatial stability
alpha_tempAn    = .6283;        % stream-wize wavenumber for temporal stability
nEig            = 50;           % Arnoldi method number of eigenvalues

% Domain & grid
FDorder     = 4;            % finite difference order of accuracy

% Flags (override martini_* workspace variables before run for paper cases)
if ~exist('temporalAna', 'var'), temporalAna = true; end
if ~exist('spatialAna', 'var'), spatialAna = false; end
if ~exist('Lx', 'var'), Lx = pi; end
if ~exist('Ly', 'var'), Ly = 5; end
if ~exist('nPeriods', 'var'), nPeriods = 2; end
if ~exist('Nx', 'var'), Nx = 125 * (Lx / pi); end
if ~exist('Ny', 'var'), Ny = 250; end

if nPeriods == 0
    floquetExp = 0;
else
    floquetExp = 2*pi / (Lx * nPeriods);
end

if ~verbose
    fprintf('[martini2024] example_3 tag=%s Lx=%.4g Nx=%d Ny=%d nPeriods=%d temporal=%d spatial=%d\n', ...
        martini_case_tag, Lx, Nx, Ny, nPeriods, temporalAna, spatialAna);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Create mesh and obtain differentiation matrices                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Cartesian mesh in computational domain
y_symmetry      = false;         % use symmetry on y coordinate around y=0 (for axysymmetric problems)
x_periodicity   = true;          % use periodic b.c. on x
alpha_filter    = .0;            % spatial filter coefficient
xrange          = [0  Lx ];  % domain range in x
yrange          = [ -1   1 ];  % domain range in y

cmesh           = CreateMesh(xrange,yrange,Nx,Ny,FDorder, ...     
                             y_symmetry,x_periodicity,alpha_filter); %construct mesh
                     
X   = cmesh.X;           % x,y: Cartesian grid coordinates
Y   = cmesh.Y;

% Mapping function as in Hanifi 1996 and mesh deformation
yi = 2.0                ; ymax = Ly ; 
a  = yi*ymax/(ymax-2*yi); b    = 1+2*a/ymax;
Y=a*(1+Y)./(b-Y);  

mesh    = DeformMesh(cmesh,[],Y,true);

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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Base flow import                                                       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Import base flow

baseFlow    = example_3_readbaseflow(mesh,baseFlow); % Custumize the function to read your baseflow

% Viscosity (via Sutherland) and heat conductivity (via constant Prandtl number)
% [baseFlow]  = sutherland_air(baseFlow);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Sponge                                                                 %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% sponge defined in the origial, non-deformed, mesh.
ys_trans    = .5;
ys          = ymax-1;
spongeAmp   = .1;
mesh.sponge = spongeAmp/2*(tanh( (Y-ys)/ys_trans)+1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Visualize base flow                                                    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if verbose
    figure('name','Base Flow')
    vars = {baseFlow.RHO ,'$\rho$';     baseFlow.U,'$U$';
            baseFlow.V,'$V$';           baseFlow.W,'$W$';
            baseFlow.T,'$T$';           mesh.sponge,'sponge';
            baseFlow.MU,'$\mu$';        baseFlow.dMUdT,'$\frac{d\mu}{dT}$';
            baseFlow.d2MUdT2                                ,'$\frac{d^2\mu}{dT^2}$';
            reshape(mesh.Dy*(baseFlow.W(:)),Nx,Ny)          ,'$\frac{dW}{dy}$';
            reshape(mesh.Dx*(baseFlow.W(:)),Nx,Ny)         ,'$\frac{dW}{dx}$';
            reshape(mesh.Dx*(mesh.Dy*baseFlow.W(:)),Nx,Ny) ,'$\frac{d^2W}{dydx}$'};
    plotFlow(mesh.X,mesh.Y,vars,4,3);
    drawnow
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Temporal Analysis
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if temporalAna

    % Get linear opreator
    [L,idx]     = GetLinProblem(mesh,baseFlow,'2D',alpha_tempAn,floquetExp);
    % Enforce Dirichlet b,c on the top, right and left boundaries, for u,v,w
    % and T
    borders='tb';  vars = 'uvwT';
    [L,idx_dirchlet] = BC_Dirichlet(L,idx,borders,vars);

    %%     Compute eigenvalues       %%
    omega_tar   = 0.5498+1.1i;   % value around which the eigenvalues will be seek

    %-------- Classical approach ------

    % Remove constrains lines/rows and get L1 : \omega \hat q = L1 \hat q
    L1 = L/-1i;

    %setup eigs options
    opts.tol    = 1e-8;
    if verbose
        opts.disp = 2;
    else
        opts.disp = 0;
    end
    opts.p      = 1500;

    % compute eigs
    if ~verbose
        fprintf('[martini2024] example_3 temporal eigs (ndof=%d, p=%d) ...\n', size(L1, 1), opts.p);
    end
    tic;
    [V,omega] = eigs(L1,nEig,omega_tar,opts);
    omega=diag(omega);
    time_eigsMatlab = toc;

    [~, iplot] = sort(imag(omega), 'descend');
    iplot = iplot(1:min(4, numel(iplot)));

    %% Plot and compare eigenvalues. Show some eigen vectors
    if verbose

    %     [~,iplot]=min(abs(lambda-(.9*alpha+.10i)));
        figure('name','Temporal Spectra');
            plot(real(omega),imag(omega),'o')
            xlabel('$\omega_r$')
            ylabel('$\omega_i$')
            
            leg{1} = 'Spectra';
            hold on
            for iiplot=iplot'
                plot(real(omega(iiplot)),imag(omega(iiplot)),'o');
                leg{end+1} = num2str(omega(iiplot));
            end
            grid on
            xlabel('$\omega_r$');
            ylabel('$\omega_i$');
            legend(leg)

        ndofs = size(L,1);
        
        [nDoms,extX,extY,extV,ext_idx,ext_usedInd] = ...
            ReconstructFloquetSol(X,Y,V,idx,mesh.usedInd,floquetExp);
        
        U = zeros(ndofs*nDoms,1);
        for iiplot=iplot'
            U=extV(:,iiplot);
            figure('name',['Eigenmode alpha = ' num2str(omega(iiplot),'%.3f')])
            vars = {real(U(ext_idx.rho_j)) ,'$\rho$'; 
                    real(U(ext_idx.u_j  )) ,'$u$'; 
                    real(U(ext_idx.v_j  )) ,'$v$'; 
                    real(U(ext_idx.w_j  )) ,'$w$'; 
                    real(U(ext_idx.T_j  )) ,'$T$' };
            title(['$\alpha' num2str(omega(iiplot)) '$']);
            axs = plotFlow(extX,extY,vars,3,2,ext_usedInd,101,'linecolor','none');
            for i=1:length(axs)
                axs(i).YLim=[0,4];
            end
        end
    end

    meta_temporal = struct( ...
        'alpha_tempAn', alpha_tempAn, ...
        'Lx', Lx, ...
        'Ly', Ly, ...
        'nPeriods', nPeriods, ...
        'floquetExp', floquetExp, ...
        'Nx', Nx, ...
        'Ny', Ny, ...
        'time_eigsMatlab', time_eigsMatlab);
    results_StreakyBL_temporal = martini2024_pack_parity_export( ...
        'temporal', omega, V, iplot, meta_temporal, mesh, idx, nEig);
    save_martini2024_bundle( ...
        ['example_3_', martini_case_tag, '_results_temporal'], ...
        ['example_3_', martini_case_tag, '_logs'], ...
        struct('results_StreakyBL_temporal', results_StreakyBL_temporal), ...
        struct('case_id', martini_case_tag, 'analysis', 'temporal'));
end

%% Spatial Stability Analysis
if spatialAna
    for wi = linspace(0,0.03,10)
    [Lc,Lw,R,idx]   = GetSpatialLinProblem(mesh,baseFlow,'2D',floquetExp);

    % Enforce Dirichlet bc on the top, right and left boundaries, for u,v,w
    % and T
    borders='tb';  vars = 'uvwT';
    
    if exist('omega', 'var')
        % use most unstable mode from temporal analysis for validation. Spatial
        % analysis should yieald a mode with alpha corresponding to the input
        % of the temporal analysis
        w = real(omega(iplot(1)))+wi*1i; 
    else
        %if temporal analysis was not performed, use the omega value
        %provided on the top of this file
        w = omega_spatialAn;
    end
    
    % Use the value of alpha used in the temporal analysis as target
    % (seachers for corresponidng modes between the two analysis types)
    alpha_target = alpha_tempAn;  
    L = Lc+Lw*w ;
    [L,R,~] = BC_Dirichlet_SpatialStability(L,R,idx,borders,vars);

    alphas = eigs(L,R,nEig,alpha_target,opts);


    if verbose
        figure('name','Spatial spectra')
            plot(real(alphas),imag(alphas),'ob');
            hold on;
            plot(real(alpha_tempAn),imag(alpha_tempAn),'r^','markerSize',15);
            xlabel('$\alpha_r$');
            ylabel('$\alpha_i$');
            legend('Spatial spectra','target');
            grid on;
            title(sprintf('wi=%0.4f',wi))
    end
    
    
    
    [~, iplot] = sort(abs(alphas), 'ascend');
    iplot = iplot(1:min(5, numel(iplot)));

    if verbose
        for iiplot = iplot'
            U = V(:, iiplot);
            figure('name', ['Eigenmode alpha = ' num2str(alphas(iiplot), '%.3f')])
            vars = {real(U(idx.rho_j(1:end/2))), '$\rho$';
                    real(U(idx.u_j(1:end/2))), '$u$';
                    real(U(idx.v_j(1:end/2))), '$v$';
                    real(U(idx.w_j(1:end/2))), '$w$';
                    real(U(idx.T_j(1:end/2))), '$T$'};
            title(['$\alpha' num2str(alphas(iiplot)) '$']);
            plotFlow(mesh.X, mesh.Y, vars, 4, 3);
        end
    end

    meta_spatial = struct('Lx', Lx, 'nPeriods', nPeriods, 'floquetExp', floquetExp);
    results_StreakyBL_spatial = martini2024_pack_parity_export( ...
        'spatial', alphas, alpha_tempAn, w, iplot, wi, meta_spatial, mesh, idx);
    if abs(wi) < 1e-12
        save_martini2024_bundle( ...
            ['example_3_', martini_case_tag, '_results_spatial'], ...
            ['example_3_', martini_case_tag, '_logs_spatial'], ...
            struct('results_StreakyBL_spatial', results_StreakyBL_spatial), ...
            struct('case_id', martini_case_tag, 'analysis', 'spatial', 'wi', wi));
    end
    
end
end