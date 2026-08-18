% Paper Fig. 10 — temporal spectrum, domain with two streaks (Lx = 2*pi).
clear variables;
close all;
clc;
martini_case_tag = 'paper_temporal_Lx2pi_nP2';
Lx = 2 * pi;
Ly = 5;
nPeriods = 2;
Nx = 200;
Ny = 200;
temporalAna = true;
spatialAna = false;
run('example_3_Streaky_BL.m');
