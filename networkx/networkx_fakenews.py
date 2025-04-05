import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.animation as animation
from matplotlib.backend_bases import MouseEvent #for user interaction
from matplotlib.widgets import Slider

facebook = pd.read_csv("/Users/margheritatonon/applied_math_lab/networkx/facebook_combined.txt.gz", 
                       compression = "gzip", sep = " ", names = ["start_node", "end_node"])

G = nx.from_pandas_edgelist(facebook, "start_node", "end_node")
original_G = G.copy()

#creating subplots:
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
gs = fig.add_gridspec(3, 2, height_ratios=[5, 0.3, 0.3])
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])

slider_ax = fig.add_subplot(gs[1, :])
button_ax = fig.add_subplot(gs[2, :])
slider = Slider(slider_ax, "Number of Initial Spreaders", 1, 20, valinit=1, valstep=1)
num_spreaders = int(slider.val)

(plot_graph,) = ax1.plot([], []) #for the actual graph
line_I, = ax2.plot([], [], color="blue", label="Ignorant")
line_S, = ax2.plot([], [], color="red", label="Spreader")
line_R, = ax2.plot([], [], color="green", label="Stifler") #for the plot ov SIR over time

#the parameters:
beta = 0.8 #"infection" rate
gamma = 0.3 #"recovery" rate
pos = nx.spring_layout(G)
num_iters = 20
time_arr = np.linspace(0, num_iters, num_iters) #x axis of plot

def assign_values(my_list):
    """
    Function used in update rule that maps a list to values. 
    """
    mapping = {"S": 1, "I": 0, "R":0}
    return list(map(mapping.get, my_list))


#one iteration of the update rule:
def update_rule(G, beta, gamma):
    """
    Returns the new state of a graph given the current graph.
    """
    state = nx.get_node_attributes(G, "state")
    new_state = {}
    
    for node in G.nodes():
        #ignorant nodes: for every spreader node they have in their vicinity, they have a probability beta of becoming spreaders themselves
        if state[node] == "I":
            neighbors = list(G.neighbors(node))
            #now we access the state_dict 
            values = list(map(state.get, neighbors))

            #we apply a map to the values so we know what the probabilities we have are
            num_s = assign_values(values) #so now we have a list with all of the probabilities of becoming infected for all of the neighbors of a current node

            #now we take a count of the nonzero values to see how many nonzero probabilities we have
            #create an array with as many random numbers as susceptible neighbors and see if any of them are less than beta
            rand_arr = np.random.uniform(0, 1, sum(num_s))
            if np.any(rand_arr < beta):
                new_state[node] = "S"

        
        #spreader node: has a probability gamma of becoming a stifler for every spreader it has in its neighborhood
        if state[node] == "S":
            neighbors = list(G.neighbors(node))
            values = list(map(state.get, neighbors))
            num_s = assign_values(values)

            rand_arr = np.random.uniform(0, 1, sum(num_s)) #creating an array of random variables based on how many stiflers there are
            if np.any(rand_arr < gamma):
                new_state[node] = "R"
            else:
                new_state[node] = "S"
        elif state[node] == "R":
            new_state[node] = "R"

    return new_state


def all_iters(num_iters, beta, gamma):
    """
    Returns all of the states of the graph for the given number of iterations.
    """
    pos = nx.spring_layout(G)
    all_states = []
    for i in range(num_iters):
        new_state = update_rule(G, beta, gamma)
        all_states.append(new_state)
        state_dict.update(new_state)
        nx.set_node_attributes(G, state_dict, "state")
    return all_states


def extract_sir(all_states):
    """
    Returns a tuple of the number of ignorant, spreader, and stifler nodes for all states for all iterations
    """
    rss = [] #y
    iss = [] #y
    sss = [] #y
    for st in all_states:
        value_counts = Counter(st.values())
        print(value_counts) #the other ones that are not present here are I
        #4039 nodes in total

        #making sure it doesnt cause an error if there are no R or S nodes
        try:
            count_r = value_counts["R"]
        except:
            count_r = 0
        rss.append(count_r)

        try:
            count_s = value_counts["S"]
        except:
            count_s = 0
        sss.append(count_s)

        #calculating the number of ignorant (not present in new_state)
        total_count_rs = count_r + count_s 
        count_i = 4039 - total_count_rs
        iss.append(count_i)
    return rss, iss, sss 


def run_simulation(num_spreaders):
    """
    Runs the simulation.
    """
    global G, all_states, rss, iss, sss
    G = original_G.copy()
    nodes = G.nodes()
    state = {node: "I" for node in nodes} #giving ignorant to every node
    spreader_nodes = np.random.choice(nodes, size = num_spreaders, replace = False)
    for s in spreader_nodes:
        state[s] = "S"
    nx.set_node_attributes(G, state, "state")

    all_states = all_iters(num_iters, beta, gamma)
    rss, iss, sss = extract_sir(all_states=all_states)
    return G, all_states, rss, iss, sss

#ANIMATION:
def combined_animation(i, G, pos, x, iss, sss, rss):
    curr_state = all_states[i]
    i_list = [n for n, st in curr_state.items() if st == "I"]
    s_list = [n for n, st in curr_state.items() if st == "S"]
    r_list = [n for n, st in curr_state.items() if st == "R"]

    ax1.clear()
    nx.draw_networkx_edges(G, pos, ax=ax1, alpha=0.2)
    nx.draw_networkx_nodes(G, pos=pos, nodelist=i_list, node_color="blue", node_size=15, ax=ax1)
    nx.draw_networkx_nodes(G, pos=pos, nodelist=s_list, node_color="red", node_size=15, ax=ax1)
    nx.draw_networkx_nodes(G, pos=pos, nodelist=r_list, node_color="green", node_size=15, ax=ax1)
    ax1.set_title(f"Network at time {i}")

    ax2.clear()
    ax2.plot(x[:i], iss[:i], color="blue", label="Ignorant")
    ax2.plot(x[:i], sss[:i], color="red", label="Spreader")
    ax2.plot(x[:i], rss[:i], color="green", label="Stifler")
    ax2.set_xlim(0, num_iters)
    ax2.set_ylim(0, 4100)
    ax2.set_title("SIR Evolution Over Time")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Population")
    ax2.legend()

    return []


def start_animation(val):
    global ani
    num_spreaders = int(val)

    # Stop existing animation if running
    if ani is not None:
        ani.event_source.stop()

    run_simulation(num_spreaders)

    ani = animation.FuncAnimation(
        fig, combined_animation, frames=num_iters, interval=70, blit=False, repeat=False
    )
    fig.canvas.draw_idle()

slider.on_changed(start_animation)

plt.tight_layout()
plt.show()