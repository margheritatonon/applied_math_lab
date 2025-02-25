import numpy as np
import matplotlib.pyplot as plt


#defining parameters:
M = 1000  #bridge mass
D = 10    #damping
K = 500   #stiffness
F = 100   #amplitude of Fcos(theta_i(t))
beta = 0.01
n_pedestrians = 5


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

def bridge_odes(t, x1t, x2t, thetas, omegas = omegas):
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



#creating a graph of the amplitude of the bridge sway with respect to the number of pedestrians driving it
#N versus amplitude 
#we should observe a sharp increase of amplitude of bridge oscillatory motion above a critical number of pedestrians

#first approach for amplitude: we just look at the minimum and the maximum
def calculate_max_amplitude(n, time_pts = time_pts, dt = dt, x1_0 = x1_0, x2_0 = x2_0):
    """
    Returns the maximum r value for n pedestrians (as a somewhat measure of amplitude).
    """
    omegas = np.random.normal(0, 1, n)
    thetas_0 = np.random.uniform(0, 2*np.pi, n)
    initialthetas = thetas_0
    initialx1, initialx2, initialthetas = x1_0, x2_0, thetas_0
    r_vals = []
    for t in time_pts:
        x1_dot, x2_dot, thetas_dot = bridge_odes(1, initialx1, initialx2, initialthetas, omegas)
        initialx1 = initialx1 + x1_dot * dt
        initialx2 = initialx2 + x2_dot * dt
        initialthetas = initialthetas + thetas_dot * dt
        initialthetas = np.mod(initialthetas, 2 * np.pi)
        r_obtained = order_parameter_r(initialthetas)
        r_vals.append(r_obtained)
    return np.max(r_vals)

if __name__ == "__main__":
    #now we have a list of r values for every time step.
    fig, ax = plt.subplots(1, 1)
    ax.plot(time_pts, np.array(r_vals_eulers))
    ax.set_title(f"Order parameter r versus time for n = {n_pedestrians}")
    ax.set_ylabel("r")
    ax.set_xlabel("Time (t)")
    #plt.show()
    
    n_pedestrians = np.arange(1, 1002, 10) #from 1 to 500 pedestrians, sampling every 10
    print(n_pedestrians)

    max_r_vals = []
    for p in n_pedestrians:
        max_r_vals.append(calculate_max_amplitude(p))

    figur, ax_rs = plt.subplots()
    ax_rs.scatter(n_pedestrians, max_r_vals, label='Amplitude of sway')
    ax_rs.set_title("Maximum r Value Versus Number of Pedestrians", size = 17)
    ax_rs.set_xlabel("Number of Pedestrians (N)", size = 15)
    ax_rs.set_ylabel("Max Amplitude (r)", size = 15)
    plt.show()
