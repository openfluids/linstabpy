set(groot, 'defaultAxesTickLabelInterpreter','latex');
set(groot, 'defaultLegendInterpreter','latex');
set(groot, 'defaultTextInterpreter','latex');
clear variables
close all
clc

addpath(genpath('aux_matlab'));
addpath(genpath('BaseFlows'));

% -------------------------------------------------------------------------
% Export directory for pre-BC L0 (match GetLinProblem_export)
% -------------------------------------------------------------------------
this_dir    = fileparts(mfilename('fullpath'));
internal_dir = fullfile(this_dir, 'MATLAB_exports_Ziming_Internal');
if exist(internal_dir, 'dir') ~= 7
    mkdir(internal_dir);
end

% -------------------------------------------------------------------------
% Physical parameters (same as main.m)
% -------------------------------------------------------------------------
baseFlow.Re    = 100000;
baseFlow.Ma    = 2.15;
baseFlow.Pr    = 0.72;
baseFlow.kappa = 1.4;
baseFlow.T_0   = 153.33;

% Perturbation parameters
ratio_Uinfty_to_characVelocity = 1.0;
ratio_L_to_Lx                  = 0.9;
St                             = 0.0;
beta                           = 19.2;

freq            = St*ratio_Uinfty_to_characVelocity/ratio_L_to_Lx;
omega_resolvent = freq*2*pi;
omega_target    = omega_resolvent;

% -------------------------------------------------------------------------
% Domain & grid (same as main.m)
% -------------------------------------------------------------------------
FDorder       = 2;
verbose       = false; %#ok<NASGU>
y_symmetry    = false;
x_periodicity = false;
alphaFilter   = 0.0;
xrange        = [0 2];
yrange        = [0 1];

% Load mesh from make_baseflow outputs
load('mesh.mat');   % provides x, y
X = x.';            % X,Y: Parabolic grid coordinates
Y = y.';
[Nr, Nz] = size(X); %#ok<NASGU>

cmesh = CreateMesh(xrange, yrange, Nz, Nr, FDorder, ...
                   y_symmetry, x_periodicity, alphaFilter);
mesh  = DeformMesh(cmesh, X, Y);

% -------------------------------------------------------------------------
% Base flow and viscosity (same as main.m)
% -------------------------------------------------------------------------
baseFlow = readbaseflow(mesh, baseFlow);
[baseFlow] = sutherland_air(baseFlow);

% -------------------------------------------------------------------------
% Sponge
% -------------------------------------------------------------------------
load('sponge');              % provides sponge_aux
spongeAmp   = 1.0;
mesh.sponge = spongeAmp * sponge_aux.';

% -------------------------------------------------------------------------
% Assemble raw operator only (no BCs)
% -------------------------------------------------------------------------
fprintf('--- Assembling raw operator with GetLinProblem_export (pre-BC only) ...\n');
[L0, idx] = GetLinProblem_export(mesh, baseFlow, '2D', beta);

% Save the pre-BC operator explicitly as well, for convenience
L0_preBC = L0;
save(fullfile(internal_dir, 'L0_preBC_only.mat'), 'L0_preBC', 'idx', ...
     'beta', 'St', 'omega_resolvent', 'omega_target', '-v7');

fprintf('Saved pre-BC operator to: %s\n', fullfile(internal_dir, 'L0_preBC_only.mat'));
fprintf('Done. No boundary conditions or eigensolves were run.\n');

