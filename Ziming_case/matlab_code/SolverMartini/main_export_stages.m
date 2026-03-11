set(groot, 'defaultAxesTickLabelInterpreter','latex');
set(groot, 'defaultLegendInterpreter','latex');
set(groot, 'defaultTextInterpreter','latex');
clear variables
close all
clc

addpath(genpath('aux_matlab'));
addpath(genpath('BaseFlows'));

% -------------------------------------------------------------------------
% Export directory for operator stages
% -------------------------------------------------------------------------
this_dir   = fileparts(mfilename('fullpath'));
export_dir = fullfile(this_dir, 'MATLAB_exports_Ziming');
if exist(export_dir, 'dir') ~= 7
    mkdir(export_dir);
end

% -------------------------------------------------------------------------
% Physical parameters (match existing main.m)
% -------------------------------------------------------------------------
baseFlow.Re    = 100000;
baseFlow.Ma    = 2.15;
baseFlow.Pr    = 0.72;
baseFlow.kappa = 1.4;
baseFlow.T_0   = 153.33;

% Perturbation & EVP parameters
ratio_Uinfty_to_characVelocity = 1.0;
ratio_L_to_Lx                  = 0.9;
St                             = 0.0;
beta                           = 19.2;

freq            = St*ratio_Uinfty_to_characVelocity/ratio_L_to_Lx;
omega_resolvent = freq*2*pi;
omega_target    = omega_resolvent;
nEig_GS         = 20;

opts.tol = 1e-8;
opts.disp = 2;
opts.p    = 2*nEig_GS+3;

% -------------------------------------------------------------------------
% Domain, grid, mesh
% -------------------------------------------------------------------------
FDorder       = 2;
verbose       = false;
y_symmetry    = false;
x_periodicity = false;
alphaFilter   = 0.0;
xrange        = [0 2];
yrange        = [0 1];

% Load mesh from precomputed file (created by make_baseflow.py)
load('mesh.mat');   % provides x, y
X = x.';            % X,Y: Parabolic grid coordinates
Y = y.';
[Nr, Nz] = size(X);

cmesh = CreateMesh(xrange, yrange, Nz, Nr, FDorder, ...
                   y_symmetry, x_periodicity, alphaFilter);
mesh  = DeformMesh(cmesh, X, Y);

% -------------------------------------------------------------------------
% Base flow and viscosity
% -------------------------------------------------------------------------
baseFlow = readbaseflow(mesh, baseFlow);
[baseFlow] = sutherland_air(baseFlow);

% -------------------------------------------------------------------------
% Sponge (same as main.m)
% -------------------------------------------------------------------------
load('sponge');        % provides sponge_aux
spongeAmp   = 1.0;
mesh.sponge = spongeAmp * sponge_aux.';

% -------------------------------------------------------------------------
% Build linear operator and apply BCs, exporting stages
% -------------------------------------------------------------------------
penalty = 1e8;

fprintf('--- Assembling raw operator with GetLinProblem ...\n');
[L0, idx] = GetLinProblem(mesh, baseFlow, '2D', beta);

% Stage 1: pre-BC operator
L0_preBC = L0;
save(fullfile(export_dir, 'L0_preBC.mat'), 'L0_preBC', '-v7');

% Dirichlet on left+top for r,u,v,w,T
borders = 'lt';  vars = 'ruvwT';
[L0, idx_dirichlet_lt] = BC_Dirichlet(L0, idx, borders, vars, penalty);

% Stage 2: after Dirichlet on left+top
L0_after_dirichlet_lt = L0;
save(fullfile(export_dir, 'L0_after_dirichlet_lt.mat'), ...
     'L0_after_dirichlet_lt', '-v7');

% Dirichlet on bottom for r,u,v,w
borders = 'b';   vars = 'ruvw';
[L0, idx_dirichlet_b] = BC_Dirichlet(L0, idx, borders, vars, penalty);

% Stage 3: after Dirichlet on bottom
L0_after_dirichlet_b = L0;
save(fullfile(export_dir, 'L0_after_dirichlet_b.mat'), ...
     'L0_after_dirichlet_b', '-v7');

% Neumann on bottom for T only
borders = 'b';   vars = 'T';
[L0, idx_neumann_b] = BC_Neumann(L0, mesh, idx, borders, vars, penalty);

% Stage 4: after Neumann on bottom (T)
L0_after_neumann_bT = L0;
save(fullfile(export_dir, 'L0_after_neumann_bT.mat'), ...
     'L0_after_neumann_bT', '-v7');

% Neumann on right for r,u,v,w,T
borders = 'r';   vars = 'ruvwT';
[L0, idx_neumann_r] = BC_Neumann(L0, mesh, idx, borders, vars, penalty);

% Stage 5: final operator after all BCs (evolution form dq/dt = L0 q)
L0_postBC = L0;
save(fullfile(export_dir, 'L0_postBC.mat'), ...
     'L0_postBC', '-v7');

% Also save the temporal operator used for eigs, for completeness
L0_temporal = L0 / -1i;   % matches L0 = L0/-1i in main.m
save(fullfile(export_dir, 'L0_temporal.mat'), 'L0_temporal', '-v7');

% Save idx and some metadata to help Python reconstruct things
save(fullfile(export_dir, 'idx_and_meta.mat'), ...
     'idx', 'penalty', 'beta', 'St', 'omega_resolvent', 'omega_target', ...
     '-v7');

fprintf('Exported operator stages to: %s\n', export_dir);

