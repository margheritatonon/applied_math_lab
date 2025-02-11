import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backend_bases import MouseEvent

uv = np.ones((2,40)) #homogeneous stationary solution 
uv = uv + np.random.uniform(0, 1, (2, 40))/100 #1% amplitude additive noise

#parameters
a = 0.4
d = 30
d1 = 1
d2 = d
gam = 1
b = 1

def pde(t, uv):
    u, v = uv

    #computing laplacians
    u_plus = np.roll(u, shift = 1)
    u_min = np.roll(u, shift = -1)
    v_plus = np.roll(v, shift = 1)
    v_min = np.roll(v, shift = -1)
    lap_u = u_plus - 2*u + u_min
    lap_v = v_plus - 2*v + v_min

    #the functions:
    f = a - b*u + (u**2)/v
    g = u**2 - v

    #the pdes:
    ut = d1 * lap_u + gam * f
    vt = d2 * lap_v + gam * g

    return (ut, vt)

#integrating the system numerically
#uv are the initial conditions

def eulers_method_pde(uv):
    """
    Numerically integrates array uv using Explicit Euler's method.
    Returns a tuple of lists with 100 elements (frames) each.
    """
    dt = 0.01
    uarr_updates = []
    varr_updates = []
    for i in range(50000):
        ut, vt = pde(1, uv) #t = 1 because it doesnt play a role in computing the pdes
        #updating with explicit eulers method
        if i % 500 == 0: #appending every 500 iterations
            uarr_updates.append(uv[0])
        uv[0] = uv[0] + ut * dt

        if i % 500 == 0:
            varr_updates.append(uv[1])
        uv [1] = uv[1] + vt * dt

        #boundary conditions:
        uv[:, 0] = uv[:, 1]
        uv[:, -1] = uv[:, -2]
    
    return (uarr_updates, varr_updates)

uarr_updates, varr_updates = eulers_method_pde(uv)
print(uv[0].shape)
print(len(uarr_updates[-1]))
print(len(uarr_updates))

def plot_static():
    """
    Creates a static plot of the last frame of animation of x versus v. 
    """
    #static plot:
    x_arr = np.linspace(0, 40, 40)
    print(f"x_arr = {x_arr.shape}")
    print(f"varr_updates[-1] = {varr_updates[-1].shape}")
    fig, ax = plt.subplots(1, 1)
    plt.plot(x_arr, varr_updates[-1])
    plt.title(f"Final Animation Frame for d = {d}", fontsize = 20)
    ax.set_xlim((0, 40))
    ax.set_ylim((0, 5))
    plt.xlabel("x", fontsize = 20)
    plt.ylabel("v(x)", fontsize = 20)
    plt.show()

plot_static()