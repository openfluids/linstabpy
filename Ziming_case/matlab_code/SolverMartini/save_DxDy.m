set(groot, 'defaultAxesTickLabelInterpreter','latex'); set(groot, 'defaultLegendInterpreter','latex'); set(groot,'defaultTextInterpreter','latex')
clear variables
close all
clc
addpath(genpath('aux_matlab'));
addpath(genpath('BaseFlows'));

savepath = "../BaseFlow/InterpolateBaseFlow/half_mesh";

% Domain & grid
% finite difference order of accuracy
FDorder = 2;


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
xrange = [0 2];
% domain range in eta [0 1];
yrange = [0 1];

load('mesh.mat')            
X               = x.' ;    % X,Y: Parabolic grid coordinates
Y               = y.';
[Nr Nz] = size(X);

cmesh           = CreateMesh(xrange,yrange,Nz,Nr,FDorder, ...     
                          y_symmetry,x_periodicity,alphaFilter); %construct mesh

mesh            = DeformMesh(cmesh,X,Y);

%save mesh derivatives to implement Ducros in CreateBaseFlow
Dx  = mesh.Dx;
save(sprintf('%s/Dx.mat',savepath), 'Dx')
Dy  = mesh.Dy;
save(sprintf('%s/Dy.mat',savepath), 'Dy')
