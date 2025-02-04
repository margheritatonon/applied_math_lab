import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backend_bases import MouseEvent

uv = np.ones((2,40))
uv = uv + np.random.uniform(0, 1, (2, 40))/100 #adding noise

#parameters
a = 0.4
d = 20
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

dt = 0.01

x_arr = np.linspace(0, 40, 40)

fig, ax = plt.subplots(1, 1)
(plot_v,) = ax.plot(x_arr, uv[1])
ax.set_xlim(0, 40)
ax.set_ylim(0, 5)

def animate(frame):
    #nonlocal uv
    ut, vt = pde(1, uv)
    #eulers
    for i in range(50000):
        #updating with eulers method
        uv[0] = uv[0] + ut * dt
        uv[1] = uv[1] + vt * dt

        #boundary conditions:
        uv[:, 0] = uv[:, 1]
        uv[:, -1] = uv[:, -2]
    
    plot_v.set_data(x_arr, uv[1])
    return (plot_v, )

ani = animation.FuncAnimation(fig, animate, interval=100, blit=False)
plt.show()
