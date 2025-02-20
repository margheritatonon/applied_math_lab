import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#from kuramoto_model import initialize_oscillators, mean_field_odes, pairwise_odes, other_params, update, animate_circle, sigma, n, dt, conc
from kuramoto_model import sigma, n, dt, conc, initialize_oscillators, mean_field_odes

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

#I will create a static plot for these k values so we see the evolution of r versus t for large values of t
#now we need to compute the r for all of these different values of t 
num_iters = 10000
t_range = np.linspace(0, num_iters, num_iters)


#fig, axes = plt.subplots(5, 4, figsize = (8,8))
#axes = axes.flatten()

ls_r_q10 = np.zeros_like(kvalues)
ls_r_q50 = np.zeros_like(kvalues)
ls_r_q90 = np.zeros_like(kvalues)
t_span = (0, 1000)
t_eval = np.arange(0, 1000, dt)
idx_end = int(len(t_eval) * 0.25)
t_eval = t_eval[-idx_end:]

theta, omega = initialize_oscillators(n, sigma, conc)
print(f"theta.shape = {theta.shape}")
print(f"theta.flatten().shape = {theta.flatten().shape}")
print(f"omega.shape = {omega.shape}")

for i, k in enumerate(kvalues):
    #rs = []
    print(k)
    sol = solve_ivp(mean_field_odes, t_span, theta, t_eval, args=(omega, k))
    theta = sol.y
    #for n in range(num_iters):
        #theta = theta + dt * thetas_dot #integrating numerically
        #r = np.abs((1/len(theta)) * np.sum(np.exp(theta * 1j)))
        #rs.append(r)
    theta = np.mod(theta, 2 * np.pi)
    r = np.abs((1/len(theta)) * np.sum(np.exp(theta * 1j))) #the absolute value is the modulus
    ls_r_q10[i] = np.percentile(r, 10)
    ls_r_q50[i] = np.percentile(r, 50)
    ls_r_q90[i] = np.percentile(r, 90)
    theta = theta[:, -1]
    #for plotting:
    #ax = axes[i]  
    #ax.scatter(t_range, np.array(rs), s=2)
    #ax.set_ylim(0, 1)
    #ax.set_title(f"k = {k:.2f}") 
    #ax.set_ylabel("r")
    #ax.set_xlabel("t")


to_plot = []
for k in kvalues:
    if k >= k_critical:
        to_plot.append(np.sqrt(1 - k_critical/k))
    else:
        to_plot.append(0)
arr_to_plot = np.array(to_plot)


fig, ax = plt.subplots()
ax.plot(kvalues, arr_to_plot, label="Theoretical", color="blue")
    # Plot the empirical order parameter as points with error bars
ax.errorbar(
        kvalues,
        ls_r_q50,
        yerr=[ls_r_q50 - ls_r_q10, ls_r_q90 - ls_r_q50],
        fmt="o",
        label="Empirical",
        color="red",
    )
ax.set_xlabel("Coupling strength (K)")
ax.set_ylabel("Order parameter (r)")
ax.set_title("Kuramoto model")
ax.legend()


plt.tight_layout()
plt.show()