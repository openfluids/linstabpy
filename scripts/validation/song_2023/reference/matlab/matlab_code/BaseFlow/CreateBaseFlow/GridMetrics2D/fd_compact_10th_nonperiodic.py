import numpy as np
import GridMetrics2D.FD_COMPACT_10TH_NONPERIODIC as fd

def compact_10th_nonperiodic(variable: np.ndarray, spacing: np.ndarray) -> np.ndarray:
    '''########################################################################
    function to evaluate the first derivative of a variable using 10th order compact finite differente scheme
    !!! only 1D arrays
    !!! only constant spacing between two points (need to change the fortran routine to allow stretched mesh)
    -----------------
    Input parameters:
    -----------------
        variable: np.ndarray parameter that will be derivated
        spacing: np.ndarray with the spatial points of the variable
    -----------------
    Output parameters:
    -----------------
       derivative: the derivative of the variable
    ########################################################################''' 

    ## obtain the stencil size
    n = len(variable)

    ### obtain the spacing between two points
    h = spacing[1] - spacing[0]

    derivative = fd.compact_scheme_10th(variable,h,n)

    return derivative
