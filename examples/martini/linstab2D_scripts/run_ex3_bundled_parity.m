% Bundled Example 3 case for compare_martini_examples (Lx=pi, nP2, temporal+spatial).
% Saves example_3_Streaky_BL_results_{temporal,spatial}.mat
clear variables;
close all;
clc;
martini_case_tag = 'Streaky_BL';
Lx = pi;
Ly = 5;
nPeriods = 2;
Nx = 125;
Ny = 250;
temporalAna = true;
spatialAna = true;
run('example_3_Streaky_BL.m');
