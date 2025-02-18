import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def initialize_oscillators(n:int, sigma:float = 1.0):
    #draw theta from uniform distribution
    thetas = np.random.uniform(0, 2*np.pi, n)
    
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
K = 1
n = 100
sigma = 1
dt = 0.01

#initializing oscillators
thetas, omegas = initialize_oscillators(n, sigma)

#plots:
fig, ax_phase = plt.subplots(1, 1, figsize=(12, 6))
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
	
    return [scatter]

ani = animation.FuncAnimation(fig, update, blit=True, interval=1)

plt.tight_layout()
plt.show()



