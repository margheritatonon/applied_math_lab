import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from matplotlib.backend_bases import MouseEvent
from matplotlib.widgets import Slider
from scipy.spatial.distance import pdist, squareform


#defining the parameters
N = 300 #this needs to be 300
L = 25
density = N / (L**2)
v = 0.3 #speed
eta = 0.1 #noise amplitude
r = 1 #radius of neighbors

dt = 1

def initialize_things(L = L, N = N, v = v):
    """
    initializes positions, velocities, and orientations
    """
    initial_positions = np.random.uniform(0, L, (N, 2))
    initial_velocities = np.ones((N, 2)) * v
    initial_orientations = np.random.normal(0, 2*np.pi, N)
    return initial_positions, initial_velocities, initial_orientations

initial_positions, initial_velocities, initial_orientations = initialize_things()

def order_parameter_va(orientations):
    """
    Calculates the order parameter v_a based on the array of velocities/directions
    """
    v_x = np.cos(orientations)
    v_y = np.sin(orientations)
    va = np.sqrt((np.mean(v_x))**2 + (np.mean(v_y))**2)
    return va 

def update_efficient(num_iters, pos_0 = initial_positions, v = v, o_0 = initial_orientations, dt=dt, r = r, eta = eta):
    """
    Defines an (efficient) update rule and returns the positions and orientations at every iteration
    """
    all_pos = []
    all_os = []
    for k in range(num_iters):
        dist = pdist(pos_0)
        print(f"first dist shape = {dist.shape}")
        dist = squareform(dist)
        print(f"seocnd dist shape = {dist.shape}")
        neighbors = dist <= r

        print(f"dist shape = {dist.shape}")
        print(f"neighbors shape = {neighbors.shape}")
        print(neighbors)
        print(f"o0 shape = {o_0.shape}")
        
        mean_angle = neighbors @ o_0 / np.sum(neighbors, axis = 1)

        noise = np.random.uniform(-eta, eta, len(o_0))

        o_0 = mean_angle + noise
        o_0 = np.mod(o_0, 2 * np.pi)

        vel = v * np.array([np.cos(o_0), np.sin(o_0)])
        pos_0 = pos_0 + dt * vel.T
        #periodic boundary
        pos_0 = np.mod(pos_0, L)

        all_pos.append(pos_0)
        all_os.append(o_0)

    return np.array(all_pos), np.array(all_os)


def update_for(num_iters, pos_0 = initial_positions, v = v, o_0 = initial_orientations, dt = dt, r = r, eta = eta, L = L):
    """
    Defines the update rule for the position and orientation of the birds. 
    Returns lists of the positions and orientations at every iteration. 
    """
    pos_0 = pos_0.copy()
    o_0 = o_0.copy()

    all_positions = []
    all_orientations = []
    for j in range(num_iters):
        mean_orientations = []

        #computing the average velocity parameter over the region Ri
        #need to find the other dudes that are in a circle of radius r away from you and take their velocities
        #the positions of the birds have 2 coordinates: x and y
        for i in range(N):
            x_bird, y_bird = pos_0[i, :]
            close_birds_pos = []
            indices_close = []
            for k, ps in enumerate(pos_0):
                if (ps[0] - x_bird) ** 2 + (ps[1] - y_bird) ** 2 < r:
                    close_birds_pos.append((ps[0], ps[1]))
                    indices_close.append(k)
            
            #now we have an array with the POSITIONS of the birds close to R, but we need their orientation
            close_orientations = o_0[np.array(indices_close)]

            #so now we take the avg of these close orientations
            mean_orientation = np.mean(close_orientations) #this is the mean orientation for bird i.
            mean_orientations.append(mean_orientation)

        mean_orientations = np.array(mean_orientations)
        
        #now we can update the orientation
        noise = np.random.uniform(-eta, eta, (N))
        o_0 = mean_orientations + noise

        #updating position
        pos_0[:, 0] = pos_0[:, 0] + v * np.cos(o_0) * dt  
        pos_0[:, 1] = pos_0[:, 1] + v * np.sin(o_0) * dt #velocity v stays constant

        #implementing periodic boundary conditions:
        pos_0 = np.mod(pos_0, L)

        #appending to the big list
        all_positions.append(pos_0)
        all_orientations.append(o_0)

    return np.array(all_positions), np.array(all_orientations)

def get_coords(num_iters):
    """
    Returns the x and y coordinates of each of the birds after num_iters iterations
    """
    pos, o0 = update_efficient(num_iters)
    x, y = pos[:, 0], pos[:, 1]
    return x, y

def run_simulation(num_frames, L = L, N = N, v = v):
    """
    Plots the animation of the boids on an L times L square for num_frames frames
    """
    #figure, axis
    fig, ax = plt.subplots(1, 1)
    plt.subplots_adjust(bottom=0.25)

    #adding a slider for velocity
    ax_vel = plt.axes([0.2, 0.05, 0.65, 0.03]) 
    v0_min = 0
    v0_max = 1
    v0_step = 0.01
    slider_v0 = plt.Slider(ax_vel, "Velocity", v0_min, v0_max, valinit=v, valstep=v0_step)

    #slider for eta
    ax_eta = plt.axes([0.2, 0.1, 0.65, 0.03]) #for noise amplitude eta
    eta_min = 0
    eta_max = 1
    eta_step = 0.1
    slider_eta = plt.Slider(ax_eta, "Noise Amplitude", eta_min, eta_max, valinit=eta, valstep=eta_step)


    def update_sliders(_):
        """
        Function that connects slider value with animation
        """
        nonlocal v, pos, ors, ani
    	# Pause animation
        ani.event_source.stop()
    	# Update parameters with sliders
        v0 = slider_v0.val
        et = slider_eta.val
        pos, ors = update_efficient(num_frames, v=v0, eta = et)
    	# Reinitialize the animation
        #ani = animation.FuncAnimation(fig, update_q, frames=pos.shape[0], interval=100, blit=True)
        plot_q.set_UVC(v0 * np.cos(ors[0]), v0 * np.sin(ors[0]))
        ani.event_source.start()

    slider_v0.on_changed(update_sliders)
    slider_eta.on_changed(update_sliders)

    x_arr = np.linspace(0, L, N) 
    y_arr = np.linspace(0, L, N)
    vs = np.zeros(N)

    #not quiver, just points
    #(plot_an,) = ax.plot(x_arr, y_arr, marker = "o", linestyle="None") 
    plot_q = ax.quiver(x_arr, y_arr, np.cos(vs), np.sin(vs), angles = "xy")

    pos, ors = update_efficient(num_frames)
    print(f"ors.shape = {ors.shape}")

    """def update_an(frame):
        plot_an.set_xdata(pos[frame, :, 0])
        plot_an.set_ydata(pos[frame, :, 1]) 
        return plot_an,"""

    def update_q(frame):
        plot_q.set_offsets(pos[frame, :, :])
        plot_q.set_UVC(np.cos(ors[frame]), np.sin(ors[frame]))
        return plot_q,

    ani = animation.FuncAnimation(fig, update_q, frames=(ors.shape[0]), interval=100, blit=True)
    plt.xlabel("x")
    plt.ylabel("y")
    ax.set_xlim((0, L))
    ax.set_ylim((0, L))
    plt.show()


#control what plots are run
position = False
etavsorder = False
animatio = True

if __name__ == "__main__":

    #position plot - not sure this actually works well
    if position == True:
        iters = 5
        x, y = get_coords(iters)
        plt.scatter(np.array(x), np.array(y))
        plt.title(f"Birds after {iters} iterations")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

    #eta versus order parameter plot
    if etavsorder == True:
        etavals = np.linspace(0, 5, 20)
        orders = []
        for et in etavals:
            pos, orientation = update_efficient(100, v = v, eta = et)
            order = order_parameter_va(orientation)
            orders.append(order)
        plt.scatter(etavals, np.array(orders))
        plt.xlabel("eta")
        plt.ylabel("order parameter")
        plt.title("Noise (eta) Versus Order Parameter")
        plt.show()
    
    #animation
    if animatio == True:
        num_frames = 100
        run_simulation(num_frames)
        