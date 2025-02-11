import numpy as np
import matplotlib.pyplot as plt


def find_leading_spatial_modes(a:float, b:float, d:float, length:float, number_of_modes:int):
    """
    Finds the leading spatial modes from mode 0 to mode N based on given parameters.
    Parameters:
        a: float
        b: float
        d: float
            Parameters in the Gierer-Meinhardt model
        length: float
            length of the 1D region we are observing
        number_of_modes: int
            The number of modes we are going to analyze for stability/instability
    Returns:
        A list with the leading spatial modes n.
    """
    #partial derivatives (computed by hand) evaluated at the fixed points (when f and g equal 0)
    fu = 2 * b / (a + 1) - b
    fv = -((b / (a + 1)) ** 2)
    gu = 2 * (a + 1) / b
    gv = -1.0

    jacobian = np.array([[fu, fv], [gu, gv]])

    max_real_temp_eigvals = []
    for x in range(number_of_modes):
        for n in range(number_of_modes):
            lambda_x = (x * np.pi / length_x) ** 2
            lambda_y = (y * np.pi / length_y) ** 2

    
    range = np.arange(0, number_of_modes)
    max_real_temp_eigvals = []
    for n in range: #finding the temporal eigenvalues
        lambda_n = ((np.pi * (n+1)) / length) ** 2 #assuming dirichlet boundary conditions
        D_matrix = np.diag([1, d])
        A_n = jacobian - lambda_n * D_matrix
        egival1, eigval2 = np.linalg.eigvals(A_n)
        real1 = egival1.real
        real2 = eigval2.real
        max_eigval = max(real1, real2)
        max_real_temp_eigvals.append(max_eigval)
    
    sorted = np.argsort(max_real_temp_eigvals)[::-1] #from biggest to smallest
    #print((np.array(max_real_temp_eigvals) > 0).sum())
    unstable_modes_indices = sorted[np.array(max_real_temp_eigvals)[sorted] > 0] 
    return unstable_modes_indices