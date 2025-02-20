import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def initialize_oscillators(n:int, sigma:float = 1.0, concentration:str = "dispersed"):
    #draw theta from uniform distribution
    if concentration == "dispersed":
        thetas = np.random.uniform(0, 2*np.pi, n)
    elif concentration == "concentrated":
        thetas = np.random.uniform(0, np.pi, n) #the initial phases will only span half of the circle
    else:
        raise ValueError("Invalid concentration value. You must input \"dispersed\" or \"concentrated\"")
    
    #draw omega from normal distribution
    omegas = np.random.normal(0, sigma, n)

    return (thetas, omegas)

def mean_field_odes(t, thetas, omegas, K):
    thetas = np.mod(thetas, 2 * np.pi)
    #frame of reference: 0
    r = (1/len(thetas)) * np.sum(np.exp(thetas * 1j))
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
K = 7
n = 100
sigma = 1
dt = 0.01
conc = "dispersed"

#initializing oscillators
thetas, omegas = initialize_oscillators(n, sigma, concentration=conc)

#plots:
fig, (ax_phase, ax_r_time,) = plt.subplots(1, 2, figsize=(12, 6))
ax_phase.set_title("Kuramoto Model")
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
    ax_r_time.relim()
    ax_r_time.autoscale_view()
	
    return [scatter, centroid_line, centroid_point, line_order_param]

def animate_circle():
    ani = animation.FuncAnimation(fig, update, blit=True, interval=1)
    plt.tight_layout()
    plt.show()

#animate_circle()



#BIFURCATION DIAGRAM
def prob_distribution(omega, sigma:float=sigma):
    return (1 / np.sqrt(2*np.pi*(sigma**2))) * np.exp(-(omega**2) / (2 * sigma**2))

g_zero = prob_distribution(0)

k_critical = 2 / (np.pi * g_zero) #this is the theoretical k critical value
print(k_critical)
kmin = k_critical/3
kmax = 3*k_critical
print(f"kmin, kmax = ({kmin}, {kmax})")

#we sample 20 values of k between kmin and kmax:
kvalues = np.linspace(kmin, kmax, 20)
print(kvalues)

#for each of these values of k, we need to find r_inf

#plotting theoretical values:
ranges = np.linspace(0, 5, 100)
to_plot = []
for k in ranges:
    if k < k_critical:
        to_plot.append(0)
    else:
        to_plot.append(np.sqrt(1 - k_critical/k))
arr_to_plot = np.array(to_plot)

#empirical:
def integrate_for_r(num_iters, K, dt:float = dt):
    theta, omega = initialize_oscillators(1000, sigma)
    thetas_dot = mean_field_odes(1, theta, omega, K)
    rs = []
    for i in range(num_iters):
        theta = theta + dt * thetas_dot
        r = np.abs((1/len(theta)) * np.sum(np.exp(theta * 1j))) #the absolute value is the modulus
        rs.append(r)
    #print(len(rs[-11:-1]))
    return np.sum(np.array(rs[-11:-1])) / 10

avg_rs_for_k = []
for k in ranges:
    rs = integrate_for_r(1000, k)
    avg_rs_for_k.append(rs)

#print(avg_rs_for_k)

#print(integrate_for_r(1000, 0.5))
#print(integrate_for_r(1000, 1))
#print(integrate_for_r(1000, 2))
#print(integrate_for_r(1000, 5))

fig, ax_bifurcation = plt.subplots(1, 1, figsize=(12, 6))
plt.plot(ranges, arr_to_plot, label = "Theoretical")
plt.scatter(ranges, np.array(avg_rs_for_k), label = "Empirical", color = "red")
plt.title("Bifurcation Diagram")
plt.xlabel("Coupling Strength (K)")
plt.ylabel("Order Parameter (r)")
#plt.show()