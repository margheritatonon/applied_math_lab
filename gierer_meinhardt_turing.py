import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches


def is_turing_instability(a: np.array, b:float, d:np.array):
    """
    A function that returns True if we have the necessary Turing instability conditions based on parameters a, b, d in the Gierer Meinhardt model, and False otherwise.

    Parameters:
        a: np.array
        b: float
        d: np.array
        
    Returns:
        np.array
        An array with elements "True" if the combination of parameters forms the necessary conditions for Turing instability and elements "False" if otherwise.
    """

    #first, we need to define u and v and the partial derivatives: 
    #defining u and v
    uv = np.ones((2,1000))
    uv = uv + np.random.uniform(0, 1, (2, 1000))/100 #adding noise
    u, v = uv

    #partial derivatives (computed by hand) evaluated at the fixed points (when f and g equal 0)
    fu = 2 * b / (a + 1) - b
    fv = -((b / (a + 1)) ** 2)
    gu = 2 * (a + 1) / b
    gv = -1.0

    #defining the conditions
    trace = fu + gv
    det = fu*gv - fv*gu
    leftside = gv + (d * fu)
    rightside = 2 * ((d)**(1/2)) * ((det)**(1/2))


    #now we check if the conditions are met (also need d > 0):
    truth_array = (d > 0) & (trace < 0) & (det > 0) & (leftside > rightside) & (rightside > 0)

    return truth_array


#now we plot a heatmap 
#we let b = 1
a_vals = np.linspace(0, 1, 1000)
d_vals = np.linspace(0, 100, 1000)
#for each of these combinations of values, we want to check if we have a turing instability.

#create a mesh grid to compute the turing instability in the entire plane
mesh_a, mesh_d = np.meshgrid(a_vals, d_vals)
#1000 by 1000 arrays: 1000 arrays with 1000 elements each
b = 1
mask_turing = is_turing_instability(mesh_a, b, mesh_d)
print(mask_turing.shape)
print(mask_turing[100])


fig, ax = plt.subplots(1, 1)
plt.xlabel("a")
plt.ylabel("d")
plt.title("Turing Space")
cmap_red_green = ListedColormap(["#69dd5d", "#dd5d5d"])
plt.contourf(mesh_a, mesh_d, mask_turing, cmap = cmap_red_green)
#adding a legend based on the colors
stable_patch = mpatches.Patch(color="#69dd5d", label="Stable")
unstable_patch = mpatches.Patch(color="#dd5d5d", label="Unstable")
plt.legend(handles=[stable_patch, unstable_patch], loc="upper right")

plt.show()
