import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def is_turing_instability(a: float, b:float, d:float):
    """
    A function that returns True if we have the necessary Turing instability conditions based on parameters a, b, d in the Gierer Meinhardt model, and False otherwise.

    Parameters:
        a: float
        b: float
        d: float
        
    Returns:
        True if the necessary conditions for a Turing instability are met.
        False otherwise
    """

    #first, we need to solve the PDE: 
    #defining u and v
    uv = np.ones((2,40))
    uv = uv + np.random.uniform(0, 1, (2, 40))/100 #adding noise
    u, v = uv
    #computing laplacians
    u_plus = np.roll(u, shift = 1)
    u_min = np.roll(u, shift = -1)
    v_plus = np.roll(v, shift = 1)
    v_min = np.roll(v, shift = -1)
    lap_u = u_plus - 2*u + u_min
    lap_v = v_plus - 2*v + v_min
    #the functions from Gierer Meinhardt model:
    f = a - b*u + (u**2)/v
    g = u**2 - v
    #the pdes:
    ut = lap_u + f #D1 = 1, gamma = 1 (as stated in the problem of the assignment)
    vt = d * lap_v + g
    #ut and vt are the PDEs (numerical values)



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
    if d >= 0 and trace < 0 and det > 0 and leftside > rightside and rightside > 0:
        return True
    
    else:
        return False
    