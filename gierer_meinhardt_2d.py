import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backend_bases import MouseEvent
from numpy import linalg

d1 = 1
gamma = 1
b = 1
length_x = 20
length_y = 50
d = 20

uv = np.ones((2, 20, 50)) #homogeneous stationary solution 
uv += uv * np.random.randn(2, length_x, length_y) / 100

def gierer_meinhardt_2d(uv, d, a=0.4):
    """
    Sets up the Gierer_Meinhardt 2D model for array uv and parameters d, a, returning dudt and dvdt.
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
    lap_u_5 = u_up + u_down + u_left + u_right - 4*u 
    lap_v_5 = v_up + v_down + v_left + v_right - 4*v

    # 9 point stencil
    #lap_u_9 = (-20*u + 4 * (u_right + u_left + u_up + u_down))/6 + 
    #lap_v_9 = (-20*v + 4 * (v_right + v_left + v_up + v_down))/6

    #the functions:
    f = a - b*u + (u**2)/v
    g = u**2 - v

    #the pdes:
    ut = d1 * lap_u_5 + gamma * f
    vt = d * lap_v_5 + gamma * g

    return (ut, vt)

num_iters = 50000
dt = 0.001

ut, vt = gierer_meinhardt_2d(uv, d = d)
print(uv[0].shape)
print(ut.shape)

uarr_updates = []
varr_updates = []
for i in range(50000): 
    ut, vt = (gierer_meinhardt_2d(uv, d = d))
    uv[0] = uv[0] + ut * dt
    uv[1] = uv[1] + vt * dt
    #updating with explicit eulers method

    if i % 500 == 0: #appending every 500 iterations
        uarr_updates.append(np.copy(uv[0]))
        varr_updates.append(np.copy(uv[1]))

    #Neumann boundary conditions:
    uv[:, 0, :] = uv[:, 1, :]
    uv[:, -1, :] = uv[:, -2, :]
    uv[:, :, 0] = uv[:, :, 1]
    uv[:, :, -1] = uv[:, :, -2]

print(len(varr_updates))


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
    	extent=[0, length_y, 0, length_x],
	)

    def update(frame):
        im.set_array(varr_updates[frame])
        im.set_clim(vmin=np.min(varr_updates[frame]), vmax=np.max(varr_updates[frame]))
        return (im, )
    
    #im.set_clim(vmin=uv[1].min(), vmax=uv[1].max() + 0.1)

    ani = animation.FuncAnimation(
    	fig, update, interval=100, blit=False, frames = len(varr_updates), repeat = False
	)

    plt.show()

animate_plot()


def find_leading_spatial_modes(number_of_modes:int, a:float=0.4, b:float=1, d:float=30):
    """
    Finds the leading spatial modes for 2D Gierer Meinhardt from mode 0 to mode N based on given parameters.
    Parameters:
        a: float
        b: float
        d: float
            Parameters in the Gierer-Meinhardt model
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

    n_values = np.arange(1, number_of_modes)
    max_eigs = np.zeros((number_of_modes, number_of_modes))
    for x in n_values:
        for y in n_values:
            lambda_x = (x * np.pi / length_x) ** 2 #neumman boundary conditions
            lambda_y = (y * np.pi / length_y) ** 2
            D_matrix = np.diag([1, d])
            A_n = jacobian - (lambda_x + lambda_y) * D_matrix

            eig1, eig2 = np.linalg.eigvals(A_n)
            eig1, eig2 = eig1.real, eig2.real
            max_eigs[x, y] = max(eig1, eig2)

    positives = max_eigs > 0
    positive_indices = np.argwhere(positives)
    pos_vals = max_eigs[positives]
    sorted_indices =  np.argsort(-pos_vals)
    leading_modes = positive_indices[sorted_indices]
    leading_eigenvalues = pos_vals[sorted_indices]
    print(leading_eigenvalues) #just to check that they are all indeed positive
    return (leading_modes)

print(find_leading_spatial_modes(10, d = d))


