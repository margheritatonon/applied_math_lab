import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def is_turing_instability(a: np.array, b:float, d:np.array):
    """
    A function that returns True if we have the necessary Turing instability conditions based on parameters a, b, d in the Gierer Meinhardt model, and False otherwise.

    Parameters:
        a: np.array
        b: float
        d: np.array
        
    Returns:
        np.array
        An array with elements "True" if the combination of parameters forms the necessary conditions for Turing instability and elements "False" if otherwise.
    """

    #first, we need to define u and v and the partial derivatives: 
    #defining u and v
    uv = np.ones((2,1000))
    uv = uv + np.random.uniform(0, 1, (2, 1000))/100 #adding noise
    u, v = uv

    #partial derivatives (computed by hand)
    fu = - b + 2*u / v
    fv = - (u**2)/(v**2)
    gu = 2*u
    gv = -1

    #defining the conditions
    trace = fu + gv
    det = fu*gv - fv*gu
    leftside = gv + (d * fu)
    rightside = 2 * ((d)**(1/2)) * ((det)**(1/2))


    #now we check if the conditions are met (also need d > 0):
    truth_array = (d > 0) & (trace < 0) & (det > 0) & (leftside > rightside) & (rightside > 0)

    return truth_array


#now we plot a heatmap 
#we let b = 1
a_vals = np.linspace(0, 1, 1000)
d_vals = np.linspace(0, 100, 1000)
#for each of these combinations of values, we want to check if we have a turing instability.

#create a mesh grid to compute the turing instability in the entire plane
mesh_a, mesh_d = np.meshgrid(a_vals, d_vals)
#1000 by 1000 arrays: 1000 arrays with 1000 elements each
mask_turing = is_turing_instability(mesh_a, 1, mesh_d)
print(mask_turing.shape)
