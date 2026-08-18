% Paper Fig. 10 — one streak, Floquet n_floq = 0 (nPeriods = 0).
clear variables;
close all;
clc;
martini_case_tag = 'paper_temporal_Lxpi_nfloq0';
Lx = pi;
Ly = 5;
nPeriods = 0;
Nx = 100;
Ny = 200;
temporalAna = true;
spatialAna = false;
run('example_3_Streaky_BL.m');
