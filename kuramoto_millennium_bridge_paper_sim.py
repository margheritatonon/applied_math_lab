import numpy as np
import matplotlib.pyplot as plt
from kuramoto_millennium_bridge import bridge_odes, order_parameter_r

#this script simulates the same numerical approach as seen in the paper 
# "Emergence of the London Millennium Bridge instability without synchronisation".

#defining parameters:
M = 1000  #bridge mass
D = 10    #damping
K = 500   #stiffness
F = 100   #amplitude of Fcos(theta_i(t))
beta = 0.01

#we initialize n_pedestrians as 2
n_pedestrians = 2
omegas = np.random.normal(0, 1, n_pedestrians)  #initial frequencies of pedestrians
thetas = np.random.uniform(0, 2 * np.pi, n_pedestrians) #initial phase of pedestrians (dispersed over whole circle, not concentrated)

t_add = 1000 #we will add a pedestrian every 1000 time
time = 1000 #total time that we let the model run for
dt = 0.01
time_steps = int(time/dt)
time_pts = np.linspace(0, time, time_steps)



def calculating_maxes(theta0, omega0, initialx1=0.0, initialx2=0.0, t_add = t_add, time = time, dt = dt, time_steps = time_steps, time_pts = time_pts):
    """
    Calculates the (maximum amplitudes and) maximum order parameter by adding a pedestrian every t_add time steps, 
    based on initial theta0 and omega0
    """
    #initializing the lists
    #max_amplitudes = []
    max_order_parameters = []
    max_r_in_interval = 0 #initializing

    for i, t in enumerate(time_pts):
        if i % t_add == 0:  #add a pedestrian
            max_order_parameters.append(max_r_in_interval)
            #print(max_r_in_interval)
            max_r_in_interval = 0 #we reset the r in the inveral to 0
            #also add a line to include the maximum amplitude
            omega0 = np.append(omega0, np.random.normal(0, 1))
            theta0 = np.append(theta0, np.random.uniform(0, 2 * np.pi))
            #print(theta0.shape)
            #print("\n")
        
        #computing odes and evolving the system
        x1_dot, x2_dot, thetas_dot = bridge_odes(t, initialx1, initialx2, theta0, omega0)
        initialx1 += x1_dot * dt
        initialx2 += x2_dot * dt
        theta0 += thetas_dot * dt
        theta0 = np.mod(theta0, 2 * np.pi)
        #print(theta0.shape)


        #computing max order parameter r
        r = order_parameter_r(theta0)
        max_r_in_interval = max(r, max_r_in_interval)

    return max_order_parameters

if __name__ == "__main__":
    max_r_values = calculating_maxes(thetas, omegas)
    pedestrian_counts = np.arange(2, 2 + len(max_r_values), 1)


    fig, axr = plt.subplots(1,1)
    axr.scatter(pedestrian_counts, max_r_values)
    axr.set_xlabel("Number of Pedestrians Added")
    axr.set_ylabel("Maximum Order Parameter (r)")
    axr.set_title("Maximum Order Parameter Under Gradual Pedestrian Addition")
    plt.show()

    print(calculating_maxes(thetas, omegas))
    print(len(calculating_maxes(thetas, omegas)))