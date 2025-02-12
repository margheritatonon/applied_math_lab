import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from numpy import linalg


def is_turing_instability(a: np.array, b:float, d:np.array):
    """
    A function that checks if we have the necessary Turing instability conditions based on parameters a, b, d in the Gierer Meinhardt model.

    Parameters:
        a: np.array
        b: float
        d: np.array
        
    Returns:
        np.array
        An array with elements "True" if the combination of parameters forms the necessary conditions for Turing instability and elements "False" if otherwise.
    """
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
#print(mask_turing.shape)
#print(mask_turing[100])


def plot_turing_space():
    """
    Plots the Turing space based on previously defined parameters.
    """
    fig, ax = plt.subplots(1, 1)
    plt.xlabel("a", fontsize=19)
    plt.ylabel("d", fontsize=20)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title(f"Turing Space for b = {b}", fontsize=25)
    cmap_red_green = ListedColormap(["#69dd5d", "#dd5d5d"])
    plt.contourf(mesh_a, mesh_d, mask_turing, cmap = cmap_red_green)
    #adding a legend based on the colors
    stable_patch = mpatches.Patch(color="#69dd5d", label="Stable")
    unstable_patch = mpatches.Patch(color="#dd5d5d", label="Unstable")
    plt.legend(handles=[stable_patch, unstable_patch], loc="upper right")
    plt.show()

#plot_turing_space()

def find_leading_spatial_modes(a:float, b:float, d:float, length:float, number_of_modes:int):
    """
    Finds the leading spatial modes from mode 0 to mode N based on given parameters.
    Parameters:
        a: float
        b: float
        d: float
            Parameters in the Gierer-Meinhardt model
        length: float
            length of the 1D region we are observing
        number_of_modes: int
            The number of modes we are going to analyze for stability/instability
    Returns:
        A list with the leading spatial modes n.
    """
    #partial derivatives (computed by hand) evaluated at the fixed points (when f and g equal 0)
    fu = 2 * b / (a + 1) - b
    fv = -((b / (a + 1)) ** 2)
    gu = 2 * (a + 1) / b
    gv = -1.0

    jacobian = np.array([[fu, fv], [gu, gv]])

    range = np.arange(0, number_of_modes)
    max_real_temp_eigvals = []
    for n in range: #finding the temporal eigenvalues
        lambda_n = ((np.pi * (n+1)) / length) ** 2 #assuming dirichlet boundary conditions
        D_matrix = np.diag([1, d])
        A_n = jacobian - lambda_n * D_matrix
        egival1, eigval2 = np.linalg.eigvals(A_n)
        real1 = egival1.real
        real2 = eigval2.real
        max_eigval = max(real1, real2)
        max_real_temp_eigvals.append(max_eigval)
    
    sorted = np.argsort(max_real_temp_eigvals)[::-1] #from biggest to smallest
    #print((np.array(max_real_temp_eigvals) > 0).sum())
    unstable_modes_indices = sorted[np.array(max_real_temp_eigvals)[sorted] > 0] 
    return unstable_modes_indices

#question 3 from assignment
ex1 = find_leading_spatial_modes(0.4, 1, 30, 40, 10)
#print(f"Leading spatial modes for d = 30 are: {ex1}")
#print(f"Turing instability present: {is_turing_instability(0.4, 1, 30)}")
#n=4 is the leading spatial mode: find Fn and Gn

ex2 = find_leading_spatial_modes(0.4, 1, 20, 40, 10)
#print(f"Leading spatial modes for d = 20 are: {ex2}")
#print(f"Turing instability present: {is_turing_instability(0.4, 1, 20)}")

a = 0.1
ex2 = find_leading_spatial_modes(a, 1, 20, 40, 10)
#print(f"Leading spatial modes for a = {a} are: {ex2}")
#print(f"Turing instability present: {is_turing_instability(a, 1, 20)}") 


d = 60
ex2 = find_leading_spatial_modes(0.4, 1, d, 40, 10)
#print(f"Leading spatial modes for d = {d} are: {ex2}")
#print(f"Turing instability present: {is_turing_instability(0.4, 1, d)}") 

length = 15
ex2 = find_leading_spatial_modes(0.4, 1, 30, length, 50)
print(f"Leading spatial modes for L = {length} are: {ex2}")
print(f"Turing instability present: {is_turing_instability(0.4, 1, 30)}") 


#if __name__ == "__main__":
#    plot_turing_space()


#we could also try to animate this and plot for different values of b to see how the region evolves