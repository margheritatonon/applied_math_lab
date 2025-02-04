import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backend_bases import MouseEvent

uv = np.ones((2,40))
uv = uv + np.random.uniform(0, 1, (2, 40))/100 #adding noise

#parameters
a = 4
d = 20
d1 = 1
d2 = d
gam = 1
b = 1

def pde(t, uv):
    u, v = uv

    #computing laplacians
    lap_u = []
    lap_v = []
    for i in range(len(u)):
        if i != 0 and i != (len(u)-1):
            lapu = u[i-1] - 2* u[i] + u[i+1]
            lap_u.append(lapu)
    for i in range(len(v)):
        if i != 0 and i != (len(v)-1):
            lapv = v[i-1] - 2* v[i] + v[i+1]
            lap_v.append(lapv) 

    #the functions:
    f = a - b*u + (u**2)/v
    g = u**2 - v

    #the pdes:
    ut = d1 * lap_u + gam * f
    vt = d2 * lap_v + gam * g

    return (ut, vt)