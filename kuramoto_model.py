import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def initialize_oscillators(n:int, sigma:float = 1.0, concentration:str = "dispersed", distribution:str = "cauchy"):
    """
    Initializes n oscillators, either dispersed or concentrated, from a cauchy or normal distribution.
    """
    #draw theta from uniform distribution
    if concentration == "dispersed":
        thetas = np.random.uniform(0, 2*np.pi, n)
    elif concentration == "concentrated":
        thetas = np.random.uniform(0, np.pi, n) #the initial phases will only span half of the circle
    else:
        raise ValueError("Invalid concentration value. You must input \"dispersed\" or \"concentrated\"")
    
    if distribution == "cauchy":
        #draw omega from normal distribution
        omegas = np.random.standard_cauchy(n)
    elif distribution == "normal":
        #draw omega from normal distribution
        omegas = np.random.normal(0, sigma, n)
    else:
        raise ValueError("Invalid distribution. Enter \"normal\" or \"cauchy\"")

    return (thetas, omegas)

def mean_field_odes(t, thetas, omegas, K):
    thetas = np.mod(thetas, 2 * np.pi)
    #frame of reference: 0
    r = np.abs((1/len(thetas)) * np.sum(np.exp(thetas * 1j)))
    thetas_dot = omegas - K*r*np.sin(thetas)
    return thetas_dot

def pairwise_odes(t, thetas, omegas, K):
    thetas = np.mod(thetas, 2*np.pi)
    thetas_dot = []
    for i in range(len(thetas)):
        gamma_js = []
        for j in range(len(thetas)):
            if i == j:
                next
            else:
                gamma_ij = (K / len(thetas)) * np.sin(thetas[j] - thetas[i])
                gamma_js.append(gamma_ij)
        thetai_dot = omegas[i] + np.sum(np.array(gamma_ij))
        thetas_dot.append(thetai_dot)
    return np.array(thetas_dot)


#parameters:
K = 1
n = 100
sigma = 1
dt = 0.01
conc = "dispersed"
distr = "cauchy"

#initializing oscillators
thetas, omegas = initialize_oscillators(n, sigma, concentration=conc, distribution=distr)

#plots:
fig, (ax_phase, ax_r_time,) = plt.subplots(1, 2, figsize=(12, 6))
ax_phase.set_title(f"Kuramoto Model for k = {K} and {distr.capitalize()} Distribution")
ax_phase.set_xlabel("Cos(theta)")
ax_phase.set_ylabel("Sin(theta)")
ax_phase.set_xlim(-1.1, 1.1)
ax_phase.set_ylim(-1.1, 1.1)
ax_phase.set_aspect("equal")
ax_phase.grid(True)

#drawing unit circle
circle =plt.Circle((0, 0), 1, color="lightgray", fill=False)
ax_phase.add_artist(circle)

scatter = ax_phase.scatter([], [], s=50, color="blue", alpha=0.5)
#so we can plot the centroid of the distribution
(centroid_line,) = ax_phase.plot([], [], color = "red", lw = 2)
(centroid_point,) = ax_phase.plot([], [], color = "red", markersize=8)


#now we want a funciton that computes r and phi so that we can have that at the center of the circle "following" the points
def other_params(thetas):
    """
    Finds r, phi, rcosphi, and rsinphi based on the theta array.
    """
    r = np.abs((1/len(thetas)) * np.sum(np.exp(thetas * 1j)))
    phi = np.angle((1/len(thetas)) * np.sum(np.exp(thetas * 1j)))
    rcosphi = np.real((1/len(thetas)) * np.sum(np.exp(thetas * 1j)))
    rsinphi = np.imag((1/len(thetas)) * np.sum(np.exp(thetas * 1j)))
    return r, phi, rcosphi, rsinphi

#we also want a graph of r with respect to time.
ax_r_time.set_title("Order Parameter r Versus Time")
ax_r_time.set_ylabel("r")
ax_r_time.set_xlabel("Time")
ax_r_time.set_ylim(0, 1)
ls_order_param = [] #[0] * 500
ls_t = [] #np.arange(0, 500) * dt
(line_order_param,) = ax_r_time.plot(ls_t, ls_order_param, color="red")

def update(frame:int):
    global thetas

    sol = solve_ivp(mean_field_odes, (0, dt), thetas, args=(omegas, K))
    thetas = sol.y[..., -1]

    thetas = np.mod(thetas, 2 * np.pi)

    #update scatter plot on the unit circle
    x = np.cos(thetas)
    y = np.sin(thetas)
    data = np.vstack((x, y)).T
    scatter.set_offsets(data)

    #so that we can plot the red line in the middle of the circle
    r, phi, rcosphi, rsinphi = other_params(thetas)
    centroid_line.set_data([0, rcosphi], [0, rsinphi])
    centroid_point.set_data([rcosphi], [rsinphi])

    ls_order_param.append(r)
    #ls_order_param.pop(0)
    ls_t.append(frame*dt)
    line_order_param.set_data(ls_t, ls_order_param)
    #ax_r_time.set_xlim(0, max(ls_t)) if max(ls_t) > 1 else 1
    ax_r_time.relim()
    ax_r_time.autoscale_view()
    #fig.canvas.draw_idle()
	
    return [scatter, centroid_line, centroid_point, line_order_param]

def animate_circle():
    ani = animation.FuncAnimation(fig, update, blit=True, interval=1)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    animate_circle()
