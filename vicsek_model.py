import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from matplotlib.backend_bases import MouseEvent
from matplotlib.widgets import Slider


#defining the parameters
N = 50
L = 20
density = N / (L**2)
v = 5 #speed
eta = 0.5 #noise amplitude
r = 5 #radius of neighbors

dt = 0.01

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

def update(num_iters, pos_0 = initial_positions, v = v, o_0 = initial_orientations, dt=dt, r = r, eta = eta):
    #distance matrix - N by N
    #we want to compute the distance between each point to each point
    #so we have all of the points in pos_0 and we want their difference to each. so then each row i is the distance from pt i to all other points
    #use scipy.spatial.distance
    #and the neighbords is where the distance is less than the radius
    pass


def update_for(num_iters, pos_0 = initial_positions, v = v, o_0 = initial_orientations, dt = dt, r = r, eta = eta, L = L):
    """
    Defines the update rule for the position and orientation of the birds.
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
    pos, o0 = update_for(num_iters)
    x, y = pos[:, 0], pos[:, 1]
    return x, y

def run_simulation(num_frames, L = L, N = N, v = v):
    #figure, axis
    fig, ax = plt.subplots(1, 1)
    plt.subplots_adjust(bottom=0.25)

    #adding a slider
    ax_eta = plt.axes([0.2, 0.05, 0.65, 0.03])

    #slider for the velocity
    v0_min = 0
    v0_max = 20
    v0 = v
    v0_step = 0.5
    slider_v0 = plt.Slider(ax_eta, "Velocity", v0_min, v0_max, valinit=v0, valstep=v0_step)
    
    def update_slider_v0(_):
        nonlocal v0, pos, ors, ani
    	# Pause animation
        ani.event_source.stop()
    	# Update parameters with sliders
        v0 = slider_v0.val
        pos, ors = update_for(num_frames, v=v0)
    	# Reinitialize the animation
        #ani = animation.FuncAnimation(fig, update_q, frames=pos.shape[0], interval=100, blit=True)
        plot_q.set_UVC(v0 * np.cos(ors[0]), v0 * np.sin(ors[0]))
        ani.event_source.start()

    slider_v0.on_changed(update_slider_v0)

    x_arr = np.linspace(0, L, N) 
    y_arr = np.linspace(0, L, N)
    vs = np.zeros(N)

    #not quiver, just points
    #(plot_an,) = ax.plot(x_arr, y_arr, marker = "o", linestyle="None") 
    plot_q = ax.quiver(x_arr, y_arr, np.cos(vs), np.sin(vs), angles = "xy")

    pos, ors = update_for(num_frames)
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


position = False
etavsorder = False
animatio = True

if __name__ == "__main__":

    #position plot
    if position == True:
        iters = 10
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
            pos, orientation = update(100, v = v, eta = et)
            order = order_parameter_va(orientation)
            orders.append(order)
        plt.scatter(etavals, np.array(orders))
        plt.xlabel("eta")
        plt.ylabel("order parameter")
        plt.show()
    
    #animation
    if animatio == True:
        num_frames = 100
        run_simulation(num_frames)
        