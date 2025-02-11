import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backend_bases import MouseEvent

d1 = 1
gamma = 1
b = 1
length_x = 20
length_y = 50

uv = np.ones((2, 20, 50)) #homogeneous stationary solution 
uv = uv + np.random.uniform(0, 1, uv.shape)/100

def gierer_meinhardt_2d(t, uv, a=0.4, d = 20):
    u, v = uv
    #moving up
    u_up = np.roll(u, shift=1, axis=0)
    v_up = np.roll(v, shift=1, axis=0)
    #moving down
    u_down = np.roll(u, shift=-1, axis=0)
    v_down = np.roll(v, shift = -1, axis = 0)
    #moving left
    u_left = np.roll(uv, shift=1, axis=1)
    v_left = np.roll(uv, shift=1, axis=1)
    #moving right
    u_right = np.roll(uv, shift=-1, axis=1)
    v_right = np.roll(uv, shift=-1, axis = 1)

    # 5 point stencil
    lap_u_5 = u_up + u_down + u_left + u_right - 4*u 
    lap_v_5 = v_up + v_down + v_left + v_right - 4*v

    #the functions:
    f = a - b*u + (u**2)/v
    g = u**2 - v

    #the pdes:
    ut = d1 * lap_u_5 + gamma * f
    vt = d * lap_v_5 + gamma * g

    return (ut, vt)






