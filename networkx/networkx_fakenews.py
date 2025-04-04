import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.animation as animation

facebook = pd.read_csv("/Users/margheritatonon/applied_math_lab/networkx/facebook_combined.txt.gz", 
                       compression = "gzip", sep = " ", names = ["start_node", "end_node"])

G = nx.from_pandas_edgelist(facebook, "start_node", "end_node")

plot_initial = False
if plot_initial == True:
    pos = nx.spring_layout(G)
    nx.draw(G, pos, width = 0.5, node_size = 10)
    plt.show()
    plt.close()

beta = 0.5 #"infection" rate
gamma = 0.3 #"recovery" rate


#giving ignorant to every node
state = []
nodes = G.nodes()
for node in nodes:
    state.append((node, "I"))
print(len(state))

random_node = np.random.choice(list(G.nodes))
print(f"the random spreader node is {random_node}")
state = [(node, "S") if node == random_node else (node, "I") for node, _ in state]
print(f"len state = {len(state)}")

#making a dictionary of the state
state_dict = dict(state)
print(f"len state_dict = {len(state_dict)}")

#updating the graph
nx.set_node_attributes(G, state_dict, "state")

#this assigns the initial state
ignorant_nodes = []
spreader_nodes = []
for key, label in state_dict.items():
    if label == "I":
        ignorant_nodes.append(key)
    else:
        spreader_nodes.append(key)
print(f"spreader node list = {spreader_nodes}")

first_updated_plot = False #this is the plot after 1 person is assigned the fake news
if first_updated_plot == True:
    pos = nx.spring_layout(G)
    nx.draw(G, pos = pos, nodelist = ignorant_nodes, node_color = "blue", node_size = 15)
    nx.draw(G, pos = pos, nodelist=spreader_nodes, node_color = "red", node_size = 15)
    plt.show()


#one iteration of the update rule:
def update_rule(G):
    """
    Returns the new state of a graph given the current graph.
    """
    state = nx.get_node_attributes(G, "state")
    new_state = {}

    def assign_values(my_list):
        mapping = {"S": 1, "I": 0, "R":0}
        return list(map(mapping.get, my_list))
    
    for node in G.nodes():
        #ignorant nodes: for every spreader node they have in their vicinity, they have a probability beta of becoming spreaders themselves
        if state_dict[node] == "I":
            neighbors = list(G.neighbors(node))
            #now we access the state_dict 
            values = list(map(state_dict.get, neighbors))

            #we apply a map to the values so we know what the probabilities we have are
            num_s = assign_values(values) #so now we have a list with all of the probabilities of becoming infected for all of the neighbors of a current node

            #now we take a count of the nonzero values to see how many nonzero probabilities we have
            #create an array with as many random numbers as susceptible neighbors and see if any of them are less than beta
            rand_arr = np.random.uniform(0, 1, sum(num_s))
            if np.any(rand_arr < beta):
                new_state[node] = "S"

        
        #spreader node: has a probability gamma of becoming a stifler for every spreader it has in its neighborhood
        if state_dict[node] == "S" or state_dict[node] == "R":
            neighbors = list(G.neighbors(node))
            values = list(map(state_dict.get, neighbors))
            #assign values so we know what probabilities we have
            num_s = assign_values(values)

            rand_arr = np.random.uniform(0, 1, sum(num_s))
            if np.any(rand_arr < gamma):
                new_state[node] = "R"
            else:
                if state_dict[node] == "S":
                    new_state[node] = "S"
                else:
                    new_state[node] = "R"

    return new_state

num_iters = 10

def all_iters(num_iters):
    """
    Returns all of the states of the graph for the given number of iterations.
    """
    pos = nx.spring_layout(G)
    all_states = []
    for i in range(num_iters):
        new_state = update_rule(G)
        all_states.append(new_state)
        state_dict.update(new_state)
    return all_states

time_arr = np.linspace(0, num_iters, num_iters) #x axis

def extract_sir(all_states):
    """
    Returns a tuple of the number of ignorant, spreader, and stifler nodes for all states for all iterations
    """
    rss = [] #y
    iss = [] #y
    sss = [] #y
    all_states = all_iters(num_iters)
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

all_states = all_iters(num_iters)
rss, iss, sss = extract_sir(all_states)

#now we can plot the graph of these versus iterations
static = False
if static == True:
    fig, ax = plt.subplots()
    ax.plot(time_arr, np.array(iss), label = "I", color = "blue")
    ax.plot(time_arr, np.array(rss), label = "R", color = "green")
    ax.plot(time_arr, np.array(sss), label = "S", color = "red")
    ax.legend()
    ax.set_title("Ignorant (I), Spreader (S), and Stifler (R) Evolution")
    ax.set_xlabel("Time")
    ax.set_ylabel("Population")
    plt.show()


#creating subplots:
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
(plot_graph,) = ax1.plot([], []) #for the actual graph
line_I, = ax2.plot([], [], color="blue", label="Ignorant")
line_S, = ax2.plot([], [], color="red", label="Spreader")
line_R, = ax2.plot([], [], color="green", label="Stifler") #for the plot ov SIR over time

#for the animation:
#need to at every time step save the network state but also the number of S, I, R for the other plot

def animate_SIR(i, x, ignorant, spreader, stifler):
    line_I.set_data(x[:i], ignorant[:i]) #it is complaining about unexpected argument color here
    line_S.set_data(x[:i], spreader[:i])
    line_R.set_data(x[:i], stifler[:i])
    return line_I, line_S, line_R,

ani2 = animation.FuncAnimation(fig, animate_SIR, fargs=(time_arr, iss, sss, rss), interval=200, blit=False, frames=num_iters)
ax2.legend()
ax2.set_title("Ignorant (I), Spreader (S), and Stifler (R) Evolution")
ax2.set_xlabel("Time")
ax2.set_ylabel("Population")


plt.tight_layout()
plt.show()