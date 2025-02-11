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

def gierer_meinhardt_2d(uv, a=0.4, d = 30):
    u, v = uv
    #moving up
    u_up = np.roll(u, shift=1, axis=0)
    v_up = np.roll(v, shift=1, axis=0)
    #moving down
    u_down = np.roll(u, shift=-1, axis=0)
    v_down = np.roll(v, shift = -1, axis = 0)
    #moving left
    u_left = np.roll(u, shift=1, axis=1)
    v_left = np.roll(v, shift=1, axis=1)
    #moving right
    u_right = np.roll(u, shift=-1, axis=1)
    v_right = np.roll(v, shift=-1, axis = 1)

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

num_iters = 50000
dt = 0.01

ut, vt = gierer_meinhardt_2d(uv)
print(uv[0].shape)
print(ut.shape)

uarr_updates = []
varr_updates = []
for i in range(50000): 
    ut, vt = gierer_meinhardt_2d(uv)
    #updating with explicit eulers method
    if i % 500 == 0: #appending every 500 iterations
        uarr_updates.append(np.copy(uv[0]))
        uv[0] = uv[0] + ut * dt

    if i % 500 == 0:
        varr_updates.append(np.copy(uv[1]))
        uv[1] = uv[1] + vt * dt

        #Neumann boundary conditions:
        uv[:, 0, :] = uv[:, 1, :]
        uv[:, -1, :] = uv[:, -2, :]
        uv[:, :, 0] = uv[:, :, 1]
        uv[:, :, -1] = uv[:, :, -2]


def animate_plot():
    """
    Animates the plot of the numerically integrated solution.
    """
    fig, ax = plt.subplots(1, 1)
    im = ax.imshow(
    	varr_updates[0],
    	interpolation="bilinear",
    	vmin=0,
    	vmax=10,
    	origin="lower",
    	extent=[0, length_y, 0, length_x],
	)

    def update(frame):
        im.set_array(varr_updates[frame])
        return (im, )

    ani = animation.FuncAnimation(
    	fig, update, interval=150, blit=True,
	)
    plt.show()

animate_plot()
