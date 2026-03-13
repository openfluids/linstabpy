For all images:
Latex for mathematical expressions
when using colormap for the modes, make it white for magnitude 0, not the light creme color

example_0_curvilinear_validation: 
horizontal spacing between the columns
title too above
increase fontsize of the equation below
separate more the two equations
try to apply latex math font to make it visually appealing 
for small values, use scientific notation 
not for this step but as an improvement for next version, see about controlling the oscillation that appears close to the boundary.

example_0_curvilinear_validation_baseflow:
I understand not putting all the computational grid since it is cartesian, but put all the physical grid, so I can see the whole physical domain
Close the physical domain with the lines
close the spacing between the two columns
use latex in the axes 

example_0_curvilinear_validation_support_error:
latex for the axes
scientific notation for small numbers

example_1_blasius:
overall well structured image
latex for the axes and any mathematical expression
increase the space between the first and second column, so the y axis title of the second column do not invade the first column
the title is too above

example_1_blasius_baseflow:
align the U_infty arrow in horizontal
decrease the distance between the colorbar and the plot
latex for the axes
increase U_infty font

example_1_blasius_support_components:
just the latex font 
and decrease the separation between the two columns
also, plot another image but with every plot using the same limits of colorbar, so we can identify where there are more energy concentrated
reduce the distance of the title

example_2_jet:
decrease the distance of the title
latex fonts 

example_2_jet_baseflow:
make the U_infty arrow horizontal
increase the font of U_infty
latex fonts

example_2_jet_support_components:
make colorbar legends equally spaced
the title is too high
the font of the plot is too small
use latex fonts for the numbers and axes

example_3_streaky_bl:
If it's not too time and memory demanding, run the spectrum for more eigenvalues in python (in matlab is too time demanding but I can try to run it during the weekend)
clearly we are also not finding the second most unstable mode, so we need to increase the number of bases and saving modes
plotwise, latex fonts for the axes, either increase the distance of the second column plots or put the x axis in another place, now the x_axis legend is trespassing the other plot
Another way to plot matlab results is by using the martini2024 paper image, where it has way more eigenvalues, consider that 
put the spectrum plot legend, the python and matlab, in a place where there`s no eigenvalue, right now it's over the results

example_3_streaky_bl_baseflow:
I think you plotted one of the eigenvectors, not the baseflow. 
plotwise, just change to latex fonts
I found the error, the streamwise velocity for him is W, not U, so plot W please

example_3_streaky_bl_spatial:
I have no clue if this plot is good, I'll run the code again to first converge the temporal spectrum before looking at the spatial one

example_3_streaky_bl_spectra:
the same for this plot, I'll first guarantee that the temporal spectrum is good before 

example_4_cavity:
adjust the x_axis so it doesn't extend to other plots
adjust the title, make it closer to the plot
latex fonts
physics wise, the plot seems really good, the only thing was the oscillations, but martini seems to also have it

example_4_cavity_adjoint:
make the colorbar legends with the same spacing
It seems physical, but why do you think these oscillations occur?
latex fonts
spacing between the two columns

example_4_cavity_baseflow:
the lower part of the cavity seems strange compared to matlab plot, in matlab it goes to -1, but here it's truncated before
I'm talking about the blue region, not the gray one
Distance between the colorbar and the plot is too large
latex fonts in the axes

example_4_cavity_direct:
Same recomendations as the example_4_cavity_adjoint
It seems physical, but one question, why mode 18? why you chose this mode?
also, scientifc notation for small numbers

example_5_paraboloid:
just a little more distance between the first and second column, the y axis is going into the first column
adjust the latex fonts
put the title closer to the plot

example_5_paraboloid_baseflow:
why the baseflow seems a little bit off? It doesn`t fill the domain 
colorbar too far from plot
latex fonts

example_5_paraboloid_support_components:
title is too far from plot
colorbar legends must be equally spaced
fonts are too small
use latex fonts





