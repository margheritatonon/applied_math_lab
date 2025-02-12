import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backend_bases import MouseEvent


#parameters
a = 0.4
d = 40
d1 = 1
d2 = d
gam = 1
b = 1

def create_array(N:int):
    """
    Returns an initial condition 2D array with noise of length N.
    Parameters:
        N: the number of spatial points in the discretization
    """
    uv = np.ones((2, N)) #homogeneous stationary solution 
    uv = uv + np.random.uniform(0, 1, (2, N))/100 #1% amplitude additive noise
    return uv

uv = create_array(40)

def spatial_part(uv:np.array, dx:float = 1):
    """
    Implements a 1D finite difference numerical approximation to integrate the spatial part of the reaction-diffusion equations.
    Parameters:
        uv: a 2D array of initial conditions for u and v
        dx: the spatial step
    Returns:
        A tuple (ut, vt) of the PDEs
    """
    u, v = uv
    #computing laplacians - we are applying the 1D finite difference numerical scheme
    u_plus = np.roll(u, shift = dx)
    u_min = np.roll(u, shift = -dx)
    v_plus = np.roll(v, shift = dx)
    v_min = np.roll(v, shift = -dx)
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

ut, vt = spatial_part(uv)
print(f"uv.shape = {uv.shape}")
print(f"uv[0].shape = {uv[0].shape}")
print(f"ut.shape = {ut.shape}")
print(f"vt.shape = {vt.shape}")
print(f"uv[0] + ut * dt shape = {(uv[0] + ut * 0.01).shape}")

def eulers_method_pde(dt:float=0.01):
    """
    Numerically integrates array uv obtained from spatial_part function using Explicit Euler's method.
    Parameters:
        dt: float specifying the time step for numerical integration.
    Returns a tuple of lists with 100 elements (frames) each.
    """
    uarr_updates = []
    varr_updates = []
    for i in range(50000): 
        ut, vt = spatial_part(uv)
        #updating with explicit eulers method
        if i % 500 == 0: #appending every 500 iterations
            uarr_updates.append(np.copy(uv[0]))
        uv[0] = uv[0] + ut * dt

        if i % 500 == 0:
            varr_updates.append(np.copy(uv[1]))
        uv[1] = uv[1] + vt * dt

        #boundary conditions:
        uv[:, 0] = uv[:, 1]
        uv[:, -1] = uv[:, -2]
    
    return (uarr_updates, varr_updates)

uarr_updates, varr_updates = eulers_method_pde()
print(f"uv[0].shape = {uv[0].shape}")
print(f"len(uarr_updates[-1]) = {len(uarr_updates[-1])}")
print(f"len(uarr_updates) = {len(uarr_updates)}")

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
    ax.set_xlim((0, 40))
    ax.set_ylim((0, 5))
    plt.xlabel("x", fontsize = 15)
    plt.ylabel("v(x)", fontsize = 15)
    plt.title(f"Final Animation Frame for d = {d}", fontsize = 20)
    plt.show()

plot_static()

print(varr_updates[0])
print(varr_updates[1])
print(varr_updates[2])
print(varr_updates[0] == varr_updates[10])

#now we want to animate the 100 frames we have instead of statically plotting the last frame.
def animate_plot():
    """
    Animates the plot of the numerically integrated solution.
    """
    fig, ax = plt.subplots(1, 1)
    x_arr = np.linspace(0, 40, 40) 
    (plot_v,) = ax.plot(x_arr, varr_updates[0]) 

    def update(frame):
        plot_v.set_ydata(varr_updates[frame])  
        return plot_v,

    ani = animation.FuncAnimation(fig, update, frames=len(varr_updates), interval=100, blit=True)
    plt.xlabel("x")
    plt.ylabel("v")
    ax.set_xlim((0, 40))
    ax.set_ylim((0, 4))
    plt.show()

animate_plot()