import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import quad
from scipy.optimize import brentq


#from kuramoto_model import initialize_oscillators, mean_field_odes, pairwise_odes, other_params, update, animate_circle, sigma, n, dt, conc
from kuramoto_model import sigma, n, dt, conc, distr, initialize_oscillators, mean_field_odes

#BIFURCATION DIAGRAM
def prob_distribution(omega, sigma:float=sigma, dist:str = "cauchy"):
    """
        Returns the probability distribution values of either a normal or cauchy distribution
    """
    if dist == "normal":
        return (1 / np.sqrt(2*np.pi*(sigma**2))) * np.exp(-(omega**2) / (2 * sigma**2))
    elif dist == "cauchy": #we assume the scale parameter gamma = 1
        return (1/np.pi) * (1/(omega**2 + 1))
    else:
        raise ValueError("Invalid distribution. Enter \"normal\" or \"cauchy\"")

g_zero = prob_distribution(0, dist=distr)


k_critical = 2 / (np.pi * g_zero) #this is the theoretical k critical value for any distribution
print(f"k_critical = {k_critical}")
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


fig, axes = plt.subplots(5, 4, figsize = (8,8))
axes = axes.flatten()




theta, omega = initialize_oscillators(n, sigma, conc, distribution=distr)
print(f"theta.shape = {theta.shape}")
print(f"theta.flatten().shape = {theta.flatten().shape}")
print(f"omega.shape = {omega.shape}")

all_rs = [] #EMPIRICAL
print(f"Value of n (1): {n}")
for i, k in enumerate(kvalues):
    rs = []
    #print(k)
    #sol = solve_ivp(mean_field_odes, t_span, theta, t_eval, args=(omega, k),)
    #theta = sol.y
    #print(theta)
    for q in range(num_iters):
        thetas_dot = mean_field_odes(1, theta, omega, k)
        theta = theta + dt * thetas_dot #integrating numerically
        theta = np.mod(theta, 2 * np.pi)
        r = np.abs((1/len(theta)) * np.sum(np.exp(theta * 1j)))
        rs.append(r)
    #for plotting:
    ax = axes[i]  
    ax.scatter(t_range, np.array(rs), s=2)
    ax.set_ylim(0, 1)
    ax.set_title(f"k = {k:.2f}") 
    ax.set_ylabel("r")
    ax.set_xlabel("t")
    all_rs.append(rs)

#plt.savefig("20plots.png")
plt.close()



means = []
stds = []
for i, k in enumerate(kvalues):
    mean_r = np.mean(all_rs[i][-300:])
    std_r = np.std(all_rs[i][-300:])
    means.append(mean_r)
    stds.append(std_r)

print(np.mean(all_rs[i][-300:]))
print(stds)



#if we were to solve the function numerically, we have the 1 = integral, so integral - 1 = 0 and we can use newtons method

def integral(theta, k, r, sigma = sigma, dist = distr):
    return k * (((np.cos(theta))**2) *  prob_distribution(k*r*np.sin(theta), sigma = sigma, dist = distr))

def root_find(r, k, sigma = sigma, d = distr):
    result, _ = quad(integral, -np.pi/2, np.pi/2, args=(k, r, sigma, d))
    return result - 1

r_solution = brentq(root_find, 0, 1, args=(k, sigma, distr)) #r is between 0 and 1, always

#print(f"r_solution = {r_solution}")

#now we do this for all values of k
theoretical_r_values = []
print("entering ks loop:")
for ks in kvalues:
    print(ks)
    try:
        r_solution = brentq(root_find, 0, 1, args=(ks, sigma, distr)) #r is between 0 and 1, always
        theoretical_r_values.append(r_solution)
        print(r_solution)
    except:
        theoretical_r_values.append(0)
        print(0)
    print("---\n")

print(f"r_values = {theoretical_r_values}") #these are always the same!!!!

#this is solving it analytically
#need this because need it for the rinf theoretical computations
def normal_second_derivative(omega, sigma:float = sigma): 
    return (((omega**2)/(sigma**2)) - 1) * (1/(np.sqrt(2*np.pi*sigma**6))) * np.exp(-(omega**2)/(2*sigma**2))

if distr == "cauchy":
    to_plot = []
    for k in kvalues:
        if k >= k_critical:
            to_plot.append(np.sqrt(1 - k_critical/k))
        else:
            to_plot.append(0)
    arr_to_plot = np.array(to_plot)
    #print(to_plot)
    #print(len(to_plot))

elif distr == "normal":
    to_plot = []
    for k in kvalues:
        if k < k_critical:
            to_plot.append(0)
        else:
            mu = (k - k_critical)/k_critical
            g0 = -1 / (sigma**3 * np.sqrt(2 * np.pi)) #because we are already evaluating at omega = 0
            #r = np.sqrt(16/(np.pi*(k_critical**3))) * np.sqrt(mu / (-1*normal_second_derivative(0)))
            r =  np.sqrt(mu / (-1*g0)) * np.sqrt(16 / np.pi * (k_critical**3))
            r = np.minimum(r, 1)
            to_plot.append(r)
    arr_to_plot = np.array(to_plot)


theoretical_rs_from_integration = np.array(theoretical_r_values)

fig, ax_bifurcation = plt.subplots(1, 1, figsize=(12, 6))
ax_bifurcation.plot(kvalues, theoretical_rs_from_integration, label = "Theoretical")
#ax_bifurcation.scatter(kvalues, np.array(means), label = "Empirical", color = "red")
ax_bifurcation.errorbar(kvalues, np.array(means), yerr=[stds, stds], label = "Empirical", color = "red", fmt = "o")
#ax_bifurcation.set_title(f"Empirical and Theoretical r Versus k for {distr.capitalize()} Distribution")
ax_bifurcation.set_title(f"{distr.capitalize()} Distribution with n = {n}", size = 40)
ax_bifurcation.set_xlabel("k", size = 30)
ax_bifurcation.set_ylabel("r", size = 30)

#we now need to identify the approximate value of kc and compare it with the theoretical value


plt.tight_layout()
plt.show()
#plt.savefig("kuramoto.png")

