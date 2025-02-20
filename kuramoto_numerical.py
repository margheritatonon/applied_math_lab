import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from kuramoto_model import initialize_oscillators, mean_field_odes, pairwise_odes, other_params, update, animate_circle, sigma, n, dt, conc

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


for k in kvalues:
    animate_circle()