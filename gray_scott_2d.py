import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

d1 = 0.1
d2 = 0.05
F =  0.014
k = 0.047
dx = 1
dt = 2

#discretization with N = 250, dx = 1
#we start with the homogeneous and stationary solution (u, v) = (1, 0)
uv = np.zeros((2, 250, 250))
uv[0, :, :] = 1 #because u = 1

#we perturb this by changing the values on a (20, 20) square where (u, v) = (0.5, 0.5) 
#plus an additive noise of 0.1 that value
np.random.seed(13) #for reproducibility
u_new = 0.5 * (1 + 0.1 * np.random.randn())
v_new = 0.5 * (1 + 0.1 * np.random.randn())
zero_start = np.random.randint(0, 230) #we do this so we have an index where to start and place the 20x20 square on
one_start = np.random.randint(0, 230)
print(zero_start)
uv[0, zero_start:zero_start+20, one_start:one_start+20] = u_new
uv[1, zero_start:zero_start+20, one_start:one_start+20] = v_new
#print(uv[0, zero_start-2:zero_start+2, one_start-2:one_start+2])

def gray_scott_2d(uv, dx:float = dx):
    """
    Sets up the Gray-Scott 2D model for array uv, returning dudt and dvdt.
    """
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
    lap_u_5 = (u_up + u_down + u_left + u_right - 4*u) / dx**2
    lap_v_5 = (v_up + v_down + v_left + v_right - 4*v) / dx**2

    #the pdes:
    ut = d1 * lap_u_5 - u*v*v + F*(1-u)
    vt = d2 * lap_v_5 + u*v*v - (F+k)*v

    return (ut, vt)

num_iters = 50000

uarr_updates = []
varr_updates = []
for i in range(num_iters): 
    #updating with explicit eulers method
    ut, vt = gray_scott_2d(uv)
    uv[0] = uv[0] + ut * dt
    uv[1] = uv[1] + vt * dt

    #periodic boundary conditions: because we use np.roll, these are already implemented before.

    if i % 200 == 0: #appending every 200 iterations
        uarr_updates.append(np.copy(uv[0]))
        varr_updates.append(np.copy(uv[1]))

def animate_plot():
    """
    Animates the plot of the numerically integrated solution.
    """
    fig, ax = plt.subplots(1, 1)
    im = ax.imshow(
    	varr_updates[0],
    	interpolation="bilinear",
    	#vmin=0,
    	#vmax=10,
    	origin="lower",
    	extent=[0, 250, 0, 250],
	)

    def update(frame):
        im.set_array(varr_updates[frame])
        im.set_clim(vmin=np.min(varr_updates[frame]), vmax=np.max(varr_updates[frame]) + 0.01)
        return (im, )
    

    ani = animation.FuncAnimation(
    	fig, update, interval=100, blit=True, frames = len(varr_updates), repeat = False
	)
    plt.show()

animate_plot()


def plot_static():
    """
    Creates a static plot of the last frame of animation of x versus v. 
    """
    #static plot:
    fig, ax = plt.subplots(1, 1)
    im = ax.imshow(
    	varr_updates[0],
    	interpolation="bilinear",
    	#vmin=0,
    	#vmax=10,
    	origin="lower",
    	extent=[0, 250, 0, 250],
	)
    im.set_array(varr_updates[-2])

    im.set_clim(vmin=np.min(varr_updates[-1]), vmax=np.max(varr_updates[-1]) + 0.01)
    plt.xlabel("x", fontsize = 20)
    plt.ylabel("y", fontsize = 20)
    plt.title(f"Gray-Scott Model for F = {F}, k = {k}", fontsize = 18)
    

    plt.show()

plot_static()