% Paper Fig. 10 — one streak, Floquet n_floq = 2 (default bundled case).
clear variables;
close all;
clc;
martini_case_tag = 'paper_temporal_Lxpi_nfloq2';
Lx = pi;
Ly = 5;
nPeriods = 2;
Nx = 100;
Ny = 200;
temporalAna = true;
spatialAna = false;
run('example_3_Streaky_BL.m');
