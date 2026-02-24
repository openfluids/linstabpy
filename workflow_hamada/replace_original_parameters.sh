#!/bin/bash

#### matlab path
export PATH=$PATH:/usr/local/MATLAB/R2017a/bin/

### this code is for substituting the values to the original ones so we can guarantee that only the variables
### changed in this code is modified
##### call replace_original_parameters.sh 


####################################################
###### SOLVER PARAMETERS ###########################
####################################################
Re=395000;     # Reynolds number
Ma=2.0;       # Mach number
Pr=0.747;       # Prandtl number
kappa=1.31;     # heat capacity ratio
T_0=1540.0;    # temperature

nEig=10 ### number of modes for Resolvent analysis
nEig_GS=100 ### number of modes for LSA
tol=1e-8 ####tolerance of the LSA solver

FDorder=2  #finite difference order of accuracy

####################################### SUPER IMPORTANT #####################################
verbose=false ###visualize grid, base flow and results
resolvent_flag=true ##run resolvent
stability_analysis_flag=false  ##run stability analysis
#############################################################################################

#### mesh parameters
y_symmetry=false   ###use symmetry on y coordinate around y=0 
x_periodicity=false #### use periodic b.c. on x
alphaFilter=0.0 ####spatial filter coefficient
xrange="[0 1]"  ###domain range in ksi
yrange="[0 0.5]"  ###domain range in eta

###########
#### boundary condition is hard coded from line 154~
###########
###########
#### Do not forget to update the input-output matrix with the boundary conditions applied
###########


### replace in Solver
matlab_full_file="./SolverMartini/main.m"
matlab_path="./SolverMartini/"
matlab_file="main.m"

cd $matlab_path
sed -i "s/^baseFlow.Re = .*/baseFlow.Re = $Re;/" "$matlab_file"
sed -i "s/^baseFlow.Ma = .*/baseFlow.Ma = $Ma;/" "$matlab_file"
sed -i "s/^baseFlow.Pr = .*/baseFlow.Pr = $Pr;/" "$matlab_file"
sed -i "s/^baseFlow.kappa = .*/baseFlow.kappa = $kappa;/" "$matlab_file"
sed -i "s/^baseFlow.T_0 = .*/baseFlow.T_0 = $T_0;/" "$matlab_file"

sed -i "s/^nEig = .*/nEig = $nEig;/" "$matlab_file"
sed -i "s/^nEig_GS = .*/nEig_GS = $nEig_GS;/" "$matlab_file"
sed -i "s/^opts.tol = .*/opts.tol = $tol;/" "$matlab_file"

sed -i "s/^FDorder = .*/FDorder = $FDorder;/" "$matlab_file"

sed -i "s/^verbose = .*/verbose = $verbose;/" "$matlab_file"
sed -i "s/^resolvent_flag = .*/resolvent_flag = $resolvent_flag;/" "$matlab_file"
sed -i "s/^stability_analysis_flag = .*/stability_analysis_flag = $stability_analysis_flag;/" "$matlab_file"

sed -i "s/^y_symmetry = .*/y_symmetry = $y_symmetry;/" "$matlab_file"
sed -i "s/^x_periodicity = .*/x_periodicity = $x_periodicity;/" "$matlab_file"
sed -i "s/^alphaFilter = .*/alphaFilter = $alphaFilter;/" "$matlab_file"
sed -i "s/^xrange = .*/xrange = $xrange;/" "$matlab_file"
sed -i "s/^yrange = .*/yrange = $yrange;/" "$matlab_file"


### replace in function to obtain the DxDy matrix
DxDy_file="save_DxDy.m"

sed -i "s/^FDorder = .*/FDorder = $FDorder;/" "$DxDy_file"

sed -i "s/^y_symmetry = .*/y_symmetry = $y_symmetry;/" "$DxDy_file"
sed -i "s/^x_periodicity = .*/x_periodicity = $x_periodicity;/" "$DxDy_file"
sed -i "s/^alphaFilter = .*/alphaFilter = $alphaFilter;/" "$DxDy_file"
sed -i "s/^xrange = .*/xrange = $xrange;/" "$DxDy_file"
sed -i "s/^yrange = .*/yrange = $yrange;/" "$DxDy_file"
cd ..
