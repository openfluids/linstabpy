#!/bin/bash

# activate virtual environment
# source .venv/bin/activate

#### matlab path
# export PATH=$PATH:~/MATLAB/R2017a/bin/
export PATH=$PATH:/usr/local/MATLAB/R2017a/bin/

### this code is for substituting the values to the original ones so we can guarantee that only the variables
### changed in this code is modified
##### call replace_original_parameters.sh 
./replace_original_parameters.sh

#######################################################################
###### PARAMETERS THAT ARE NOT IN replace_original_parameters.sh ######
#######################################################################
#### ratio between the freestream velocity and the characteristic velocity used to non-dimensionalize the problem
ratio_Uinfty_to_characVelocity=1.0 
#### ratio between the length that the strouhal is defined (L) and the characteristic length used to non-dimensionalize the problem (Lx)
ratio_L_to_Lx=0.1
#### Strouhal number defined for L
St=0.1
####spanwise wavenumber defined for Lx
beta=0.0
########## used in plot_resolvent_ut_un_p
gamma=1.31
Mach=2.0

####################################################
###### BASEFLOW PAMETERS ###########################
####################################################
#### define which mesh will be used
### put here the mesh defined (original_mesh, half_mesh, double_mesh, doubleX_mesh, doubleY_mesh, meshY_nmax200_cut2, suction_side, pressure_side)
casename="suction_side"

### define sponge parameters
#### top, right, bot, left direction 
sponge_flag=False
sponge_type="constant"  ### "constant", "parabolic", "localized"
##how many points to apply the sponge
pts_top=10
pts_right=0
pts_bot=0
pts_left=10
###amplitude of the sponge in each direction
amp_top=1000
amp_right=0
amp_bot=0
amp_left=1000

## flag to see if heaviside or specific shock sponge is used
shock_flag=False
threshold_heaviside=-8.0
amplitude_shock=0.01

### filter parameters
filter_flag=False
#how many times the filter will pass in x and y directions
nfilter=20
#minimum point in y direction which the filter will pass
nyminfilter=0

####################################################
###### SOLVER PARAMETERS ###########################
####################################################
tol=1e-8 ####tolerance of the LSA solver
FDorder=4  #finite difference order of accuracy

####################################### SUPER IMPORTANT #####################################
verbose=false ###visualize grid, base flow and results
resolvent_flag=true ##run resolvent
stability_analysis_flag=false  ##run stability analysis
#############################################################################################

###########
#### boundary condition is hard coded from line 165~ (around this line)
###########
###########
#### Do not forget to update the input-output matrix with the boundary conditions applied  (around line 200, right below % Input/forcing matrix)
###########

####################################################
###### SAVEPATH AND SAVENAME #######################
####################################################
####### define base savename
savename_base="Results_St${St}_beta${beta}_NeumannOutBotTemp_SS_200-500_100eig_4thOrder_LargeMesh_Half"


####### definitions based on the baseflow flags
if [ "$sponge_flag" = "True" ]; then
    sponge_pts="[${pts_top},${pts_right},${pts_bot},${pts_left}]"
    sponge_amp="[${amp_top},${amp_right},${amp_bot},${amp_left}]"
    sponge_name="Sponge_${sponge_type}_${pts_top}-${amp_top}top_${pts_right}-${amp_right}right_${pts_bot}-${amp_bot}bot_${pts_left}-${amp_left}left"
else
    sponge_pts="[0,0,0,0]"
    sponge_amp="[0,0,0,0]"
    sponge_name="NoSponge"
fi

if [ "$shock_flag" = "True" ]; then
    derivative_matrix="save_derivative_matrix.sh"
    sed -i "s/^FDorder=.*/FDorder=$FDorder/" "$derivative_matrix"
    sed -i "s/^casepath=.*/casepath=\"$casename\"/" "$derivative_matrix"
    ./"$derivative_matrix"

    shock_name="Shock_${threshold_heaviside}threshold_${amplitude_shock}amp"
else
    shock_name="NoShock"
fi

if [ "$filter_flag" = "True" ]; then
    filter_name="Filter_${nfilter}times_${nyminfilter}nymin"
else
    filter_name="NoFilter"
fi

# Create a unique savepath name
SAVEPATH="${savename_base}\/\\${sponge_name}_${shock_name}_${filter_name}\/\\"
TITLE="${sponge_name}_${shock_name}_${filter_name}"
echo "$SAVEPATH"
echo "$TITLE"



### replace in Solver
matlab_full_file="./SolverMartini/main.m"
matlab_path="./SolverMartini/"
matlab_file="main.m"

sed -i "s/^ratio_Uinfty_to_characVelocity = .*/ratio_Uinfty_to_characVelocity = $ratio_Uinfty_to_characVelocity;/" "$matlab_full_file"
sed -i "s/^ratio_L_to_Lx = .*/ratio_L_to_Lx = $ratio_L_to_Lx;/" "$matlab_full_file"
sed -i "s/^St = .*/St = $St;/" "$matlab_full_file"
sed -i "s/^beta = .*/beta = $beta;/" "$matlab_full_file"

sed -i "s/^opts.tol = .*/opts.tol = $tol;/" "$matlab_full_file"

sed -i "s/^FDorder = .*/FDorder = $FDorder;/" "$matlab_full_file"

sed -i "s/^verbose = .*/verbose = $verbose;/" "$matlab_full_file"
sed -i "s/^resolvent_flag = .*/resolvent_flag = $resolvent_flag;/" "$matlab_full_file"
sed -i "s/^stability_analysis_flag = .*/stability_analysis_flag = $stability_analysis_flag;/" "$matlab_full_file"




####################################################
###### SUBSTITUTE IN PLOT FILES ####################
####################################################
spectrum_file="plot_spectrum.py"
resolvent_file="plot_resolvent_modes.py"
ut_un_p_file="plot_resolvent_ut_un_p.py"
makeflow_path="./BaseFlow/CreateBaseFlow/"
makeflow_file="make_baseflow.py"

# replace Strouhal
sed -i "s/^St = .*/St = $St/" "$spectrum_file"
sed -i "s/^St = .*/St = $St/" "$resolvent_file"
sed -i "s/^St = .*/St = $St/" "$ut_un_p_file"
# Replace beta
sed -i "s/^beta = .*/beta = $beta/" "$spectrum_file"
sed -i "s/^beta = .*/beta = $beta/" "$resolvent_file"
sed -i "s/^beta = .*/beta = $beta/" "$ut_un_p_file"

sed -i "s/^gamma = .*/gamma = $gamma/" "$ut_un_p_file"
sed -i "s/^Mach = .*/Mach = $Mach/" "$ut_un_p_file"

### substitute in spectrum and resolvent plot .py files
sed -i "s/^savepath = .*/savepath = \"$SAVEPATH\"/" "$spectrum_file"
sed -i "s/^savepath = .*/savepath = \"$SAVEPATH\"/" "$resolvent_file"
sed -i "s/^savepath = .*/savepath = \"$SAVEPATH\"/" "$ut_un_p_file"
sed -i "s/^title = .*/title = \"$TITLE\"/" "$spectrum_file"
sed -i "s/^title = .*/title = \"$TITLE\"/" "$resolvent_file"
sed -i "s/^title = .*/title = \"$TITLE\"/" "$ut_un_p_file"


# Modify parameters in get_BF.py
cd $makeflow_path

sed -i "s/^savepath = .*/savepath = \"$SAVEPATH\"/" "$makeflow_file"
sed -i "s/^casename = .*/casename = \"$casename\"/" "$makeflow_file"

sed -i "s/^sponge_flag = .*/sponge_flag = $sponge_flag/" "$makeflow_file"
sed -i "s/^sponge_type = .*/sponge_type = \"$sponge_type\"/" "$makeflow_file"
sed -i "s/^sponge_pts = .*/sponge_pts = $sponge_pts/" "$makeflow_file"
sed -i "s/^sponge_amp = .*/sponge_amp = $sponge_amp/" "$makeflow_file"

sed -i "s/^shock_flag = .*/shock_flag = $shock_flag/" "$makeflow_file"
sed -i "s/^threshold_heaviside = .*/threshold_heaviside = $threshold_heaviside/" "$makeflow_file"
sed -i "s/^amplitude_shock = .*/amplitude_shock = $amplitude_shock/" "$makeflow_file"

sed -i "s/^filter_flag = .*/filter_flag = $filter_flag/" "$makeflow_file"
sed -i "s/^nfilter = .*/nfilter = $nfilter/" "$makeflow_file"
sed -i "s/^nyminfilter = .*/nyminfilter = $nyminfilter/" "$makeflow_file"

#### run python code to create the base flow
python3 $makeflow_file
cd ../../

# #### obtain the grid metrics for the conversion of cartesian velocity to tangential and normal velocity
# cd ./GridMetrics
# sed -i "s/^casename = .*/casename = \"$casename\"/" "grid_metrics2D.py"
# python3 grid_metrics2D.py
# cd ../

##### substitute save path and solve the problem using matlab
sed -i "s/^savepath = .*/savepath = \"$SAVEPATH\";/" "$matlab_full_file"
cd $matlab_path
matlab -nodisplay -nosplash -r "main; exit;"
cd ..
if [ "$stability_analysis_flag" = "true" ]; then
    python3 plot_spectrum.py
fi
if [ "$resolvent_flag" = "true" ]; then
    python3 plot_resolvent_modes.py
    python3 plot_resolvent_ut_un_p.py
fi
