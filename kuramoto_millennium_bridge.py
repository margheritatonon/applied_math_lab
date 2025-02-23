import numpy as np
import matplotlib.pyplot as plt
from kuramoto_model import initialize_oscillators


#defining parameters:
M = 1000  #bridge mass
D = 10    #damping
K = 500   #stiffness
F = 100   #amplitude of Fcos(theta_i(t))
beta = 0.01
n_pedestrians = 500


time = 100
dt = 0.01
time_steps = int(time / dt)
time_pts = np.linspace(0, time, time_steps)


omegas = np.random.normal(0, 1, n_pedestrians) #we are going to draw the omegas from a normal distribution, for simplicity (but this could also be Cauchy)

#initial conditions:
x1_0 = 0
x2_0 = 0
thetas_0 = np.random.uniform(0, 2*np.pi, n_pedestrians)




def order_parameter_r(theta):
    """
    Calculates the order parameter r based on array of phase angles theta.
    """
    return np.abs(np.sum(np.exp(1j * theta)) / len(theta))

def bridge_odes(t, x1t, x2t, thetas):
    """
    Returns x1_dot, x2_dot, thetas_dot based on x1t, x2t, and thetas.
    """
    x1_dot = x2t
    x2_dot = (1/M) * (-D*x2t - K*x1t + F*np.sum(np.cos(thetas)))
    thetas_dot = omegas - beta * x2t * np.sin(thetas)
    return (x1_dot, x2_dot, thetas_dot)


r_vals_eulers = []
#first, we plot a graph of r versus time for different n_oscillators.
#this means that we need to evolve the system over time - using eulers method.
initialx1, initialx2, initialthetas = x1_0, x2_0, thetas_0
for t in time_pts:
    x1_dot, x2_dot, thetas_dot = bridge_odes(1, initialx1, initialx2, initialthetas)
    initialx1 = initialx1 + x1_dot * dt
    initialx2 = initialx2 + x2_dot * dt
    initialthetas = initialthetas + thetas_dot * dt
    initialthetas = np.mod(initialthetas, 2 * np.pi)
    r_obtained = order_parameter_r(initialthetas)
    r_vals_eulers.append(r_obtained)

#now we have a list of r values for every time step.
fig, ax = plt.subplots(1, 1)
ax.plot(time_pts, np.array(r_vals_eulers))
ax.set_title(f"Order parameter r versus time for n = {n_pedestrians}")
ax.set_ylabel("r")
ax.set_xlabel("Time (t)")
plt.show()



#then, we plot a graph of r versus n (like we did r vs k and averaging over the last 300 time steps)
