% Paper Fig. 11 — spatial stability, Lx = pi, n_floq = 2 (requires temporal block).
clear variables;
close all;
clc;
martini_case_tag = 'paper_spatial_Lxpi_nP2';
Lx = pi;
Ly = 5;
nPeriods = 2;
Nx = 100;
Ny = 200;
temporalAna = true;
spatialAna = true;
run('example_3_Streaky_BL.m');
