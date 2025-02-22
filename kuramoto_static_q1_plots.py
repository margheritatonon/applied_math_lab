import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from kuramoto_model import K, n, sigma, dt, conc, distr, other_params

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


#initializing oscillators
thetas, omegas = initialize_oscillators(n, sigma, concentration=conc, distribution=distr)

#here we compute the final and initial values for the plots:
t_final = 100
t_eval = np.arange(0, t_final, dt)
print(t_eval.shape)
sol = solve_ivp(mean_field_odes, (0, t_final), thetas, args=(omegas, K), t_eval=t_eval)
thetas_final = sol.y[:, -1]
r_values = [other_params(sol.y[:, i])[0] for i in range(len(sol.t))] #finds r over the time we set

fig, (ax_initial, ax_final, ax_time_final) = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(f"Initial and Final Frames for k = {K}, {distr.capitalize()} Distribution, and {conc.capitalize()} Concentration", fontsize=23)

def plot_phase(ax, thetas, title):
    ax.set_title(title, size = 20)
    ax.set_xlabel("Cos(theta)", size = 13)
    ax.set_ylabel("Sin(theta)", size = 13)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    ax.grid(True)

    #drawing circle
    circle = plt.Circle((0, 0), 1, color="lightgray", fill=False)
    ax.add_artist(circle)

    #points
    x = np.cos(thetas)
    y = np.sin(thetas)
    ax.scatter(x, y, s=50, color="blue", alpha=0.5)

    r, phi, rcosphi, rsinphi = other_params(thetas)
    ax.plot([0, rcosphi], [0, rsinphi], color="red", lw=2)
    ax.plot(rcosphi, rsinphi, 'ro', markersize=8)

plot_phase(ax_initial, thetas, "Initial State")
plot_phase(ax_final, thetas_final, "Final State")

ax_time_final.plot(sol.t, r_values, color="red")
ax_time_final.set_title("Order Parameter r vs Time", size = 17)
ax_time_final.set_xlabel("Time", size = 15)
ax_time_final.set_ylabel("r", size = 15)
ax_time_final.set_ylim(0, 1)

plt.tight_layout()
plt.show()